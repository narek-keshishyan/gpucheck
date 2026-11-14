"""Run engine: preflight, baseline phase, load phase, with a safety watchdog.

Time is driven by this module's own schedule: every iteration calls
adapter.sample() once, then sleeps until the next 1/hz slot (measured with the
injected clock). In the load phase the load burst runs first and the sample is
taken straight after it, so it reads the card under load. Sample timestamps (t_s) come from the adapter. A sample() that
raises is recorded as an all-None sample whose t_s is the previous sample's t_s
plus 1/hz (0.0 if there is none yet), so the series stays evenly spaced.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from .adapter import EventLogUnavailable, GpuInventory, TelemetryAdapter, TelemetrySample
from .checks import (
    PARTIAL, RAN, UNAVAILABLE, Check, CheckPlan, CheckUnavailable, driver_errors_report, nvlink_report, one_line,
)
from .preflight import Preflight, run_preflight
from .specs import GpuSpec

THERMAL_MARGIN_C = 5.0
UNKNOWN_GPU_THERMAL_LIMIT_C = 95.0  # abort limit when the model is not in the spec table
THERMAL_CONSECUTIVE = 3
ECC_JUMP = 100
MAX_CONSECUTIVE_SAMPLE_ERRORS = 10
INTERRUPTED = "interrupted"  # `aborted` when should_stop() ended the run


@dataclass(frozen=True)
class RunConfig:
    baseline_s: float = 60
    load_s: float = 600
    hz: float = 1.0


class LoadGenerator(Protocol):
    def start(self) -> None: ...

    def tick(self) -> float | None: ...

    def stop(self) -> None: ...


class NullLoad:
    def start(self) -> None:
        pass

    def tick(self) -> float | None:
        return None

    def stop(self) -> None:
        pass


@dataclass(frozen=True)
class RawRun:
    inventory: GpuInventory
    preflight: Preflight
    baseline: list[TelemetrySample]
    load: list[TelemetrySample]
    config: RunConfig
    aborted: str | None
    checks: dict = field(default_factory=dict)  # one entry per extra check asked for; empty by default


class _Watchdog:
    def __init__(self, spec: GpuSpec | None) -> None:
        self._limit = spec.max_temp_c + THERMAL_MARGIN_C if spec is not None else UNKNOWN_GPU_THERMAL_LIMIT_C
        self._hot = 0
        self._last_ecc: int | None = None

    def check(self, s: TelemetrySample) -> str | None:
        if s.temp_c is not None and s.temp_c >= self._limit:
            self._hot += 1
            if self._hot >= THERMAL_CONSECUTIVE:
                return "thermal"
        else:
            self._hot = 0
        if s.ecc_volatile is not None:
            if self._last_ecc is not None and s.ecc_volatile - self._last_ecc > ECC_JUMP:
                return "ecc"
            self._last_ecc = s.ecc_volatile
        return None


def _empty_sample(t_s: float) -> TelemetrySample:
    return TelemetrySample(
        t_s=t_s,
        temp_c=None,
        power_w=None,
        sm_clock_mhz=None,
        fan_pct=None,
        throttle_reasons=None,  # unknown, not "no throttling"
        ecc_volatile=None,
        achieved_tflops=None,
    )


MAX_CLEANUP_ERROR_LEN = 200


def warn_cleanup_failed(e: BaseException) -> None:
    """One stderr line for a load cleanup that failed; the run carries on.

    Never raises: it runs in a finally, so an error that cannot be printed or a broken stderr
    must not replace the finished run."""
    try:
        text = " ".join(f"{type(e).__name__}: {e}".split())[:MAX_CLEANUP_ERROR_LEN]
        print(f"warning: load cleanup failed ({text})", file=sys.stderr)
    except Exception:
        pass


def execute_run(
    adapter: TelemetryAdapter,
    load: LoadGenerator,
    config: RunConfig,
    spec: GpuSpec | None,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    plan: CheckPlan | None = None,
    checks: Sequence[Check] = (),
    before_load: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int, TelemetrySample], None] | None = None,
) -> RawRun:
    """Run one card. Everything after `sleep` is for the opt-in extras and is unused by default:

    plan: which extra checks were asked for. `checks` are the ones that run as their own phases
    after the load, in order. before_load is called once, just before the load starts (the
    all-cards-at-once mode waits there for the other cards). should_stop is polled every sample
    and ends the run as "interrupted". progress is called with (index, count, sample) for every
    load sample."""
    plan = plan or CheckPlan()
    inventory = adapter.inventory()
    preflight = run_preflight(adapter)
    baseline: list[TelemetrySample] = []
    loaded: list[TelemetrySample] = []
    period = 1.0 / config.hz
    state = {"last_t": None, "errors": 0}
    link: dict[str, int] = {}  # best PCIe generation and width seen while the load ran
    outcomes: dict = {}
    events: list[dict] = []
    log: dict = {"log": None, "status": None, "reason": None}  # the driver's event log, when listening
    links_before: dict = {}

    def read_links() -> dict | None:
        reader = getattr(adapter, "nvlink", None)
        if reader is None:
            return None
        try:
            return reader()
        except Exception:
            return None

    def close_log() -> None:
        if log["log"] is not None:
            try:
                log["log"].close()
            except Exception:
                pass
            log["log"] = None

    def drain(phase_name: str, t_s: float) -> None:
        """Move the events the driver has reported since the last sample into `events`."""
        if log["log"] is None:
            return
        try:
            new = log["log"].poll()
        except Exception as e:  # listening broke: keep what was heard and say it is partial
            log["status"], log["reason"] = PARTIAL, one_line(e)
            close_log()
            return
        events.extend({**e, "phase": phase_name, "t_s": t_s} for e in new)

    def finish() -> None:
        """Close what the extras opened and write their entries."""
        if plan.driver_errors:
            last = state["last_t"]
            drain("end", 0.0 if last is None else last)
            close_log()
            outcomes["driver_errors"] = driver_errors_report(log["status"] or RAN, events, log["reason"])
        if plan.links:
            outcomes["nvlink"] = nvlink_report(links_before.get("snapshot"), read_links())

    def result(aborted: str | None) -> RawRun:
        if aborted != "preflight_refused":
            finish()
        inv = inventory
        if "gen" in link and "width" in link:
            # A link drops to a lower speed at idle, so the idle reading from inventory() says
            # little; what the link reached under load is the one worth reporting.
            inv = replace(
                inv, pcie_gen_current=link["gen"], pcie_width_current=link["width"], pcie_link_under_load=True
            )
        return RawRun(inv, preflight, baseline, loaded, config, aborted, outcomes)

    def read_link() -> None:
        reader = getattr(adapter, "pcie_link", None)
        if reader is None:
            return
        try:
            gen, width = reader()
        except Exception:
            return
        if gen is not None and width is not None:
            link["gen"] = max(gen, link.get("gen", gen))
            link["width"] = max(width, link.get("width", width))

    if preflight.mode == "refuse":
        return result("preflight_refused")

    def take() -> TelemetrySample:
        """One sample, or an all-None one if the adapter raised."""
        try:
            s = adapter.sample()
            state["errors"] = 0
        except Exception:
            state["errors"] += 1
            last = state["last_t"]
            s = _empty_sample(0.0 if last is None else last + period)
        state["last_t"] = s.t_s
        return s

    def phase(count: int, out: list[TelemetrySample], watchdog: _Watchdog | None, tick: bool) -> str | None:
        start = clock()
        name = "load" if tick else "baseline"
        for i in range(count):
            if tick:
                # Sample straight after the burst. Sampling first would read the card after the
                # pause that ends each period: on cards that report instantaneous values (A100)
                # that showed idle power and a card several degrees cooler than under load.
                tflops = load.tick()
                s = take()
                if tflops is not None:
                    s = replace(s, achieved_tflops=tflops)
                    read_link()  # only when a burst really ran
            else:
                s = take()
            out.append(s)
            drain(name, s.t_s)
            if tick and progress is not None:
                progress(i, count, s)
            if watchdog is not None and (reason := watchdog.check(s)):
                return reason
            if state["errors"] >= MAX_CONSECUTIVE_SAMPLE_ERRORS:
                return "telemetry"
            if should_stop is not None and should_stop():
                return INTERRUPTED
            sleep(max(0.0, start + (i + 1) * period - clock()))
        return None

    def run_check(check: Check, watchdog: _Watchdog) -> str | None:
        """One extra check as its own phase: a slice of work, then a sample, until it is done.

        The card keeps being sampled so its temperature limit still stops the run, and so that
        driver events and memory-error counts during the check are recorded with it."""
        status, reason, aborted = RAN, None, None
        ecc: list[int] = []
        temps: list[float] = []
        try:
            check.start()
            while check.step():
                s = take()
                if s.ecc_volatile is not None:
                    ecc.append(s.ecc_volatile)
                if s.temp_c is not None:
                    temps.append(s.temp_c)
                drain("checks", s.t_s)
                aborted = watchdog.check(s)
                if aborted is None and state["errors"] >= MAX_CONSECUTIVE_SAMPLE_ERRORS:
                    aborted = "telemetry"
                if aborted is None and should_stop is not None and should_stop():
                    aborted = INTERRUPTED
                if aborted:
                    status = PARTIAL
                    break
        except CheckUnavailable as e:
            status, reason = UNAVAILABLE, " ".join(str(e).split())[:200]
        except Exception as e:  # a check that breaks must not lose the run that came before it
            status, reason = UNAVAILABLE, one_line(e)
        finally:
            try:
                check.stop()
            except Exception as e:
                warn_cleanup_failed(e)
        entry = {"status": status}
        if status != UNAVAILABLE:
            try:
                entry.update(check.result())
            except Exception as e:
                entry, reason = {"status": UNAVAILABLE}, one_line(e)
            if len(ecc) >= 2:
                entry["ecc_delta"] = ecc[-1] - ecc[0]
            if temps:
                entry["max_temp_c"] = max(temps)
        if reason:
            entry["reason"] = reason
        outcomes[check.name] = entry
        return aborted

    if plan.driver_errors:
        opener = getattr(adapter, "open_event_log", None)
        try:
            if opener is None:
                raise CheckUnavailable("this card's driver interface offers no event log")
            log["log"] = opener()
        except Exception as e:
            log["status"] = UNAVAILABLE
            known = isinstance(e, (CheckUnavailable, EventLogUnavailable))
            log["reason"] = " ".join(str(e).split())[:200] if known else one_line(e)
    if plan.links:
        links_before["snapshot"] = read_links()

    try:
        aborted = phase(round(config.baseline_s * config.hz), baseline, None, tick=False)
        if aborted:
            return result(aborted)
        try:
            if before_load is not None:
                before_load()
            load.start()
            aborted = phase(round(config.load_s * config.hz), loaded, _Watchdog(spec), tick=True)
        finally:
            try:
                load.stop()
            except Exception as e:  # a failed cleanup must not replace the samples already taken
                warn_cleanup_failed(e)
        guard = _Watchdog(spec)
        for check in checks:
            if aborted:
                break  # the run was stopped: the remaining checks are not run and are not listed
            aborted = run_check(check, guard)
        return result(aborted)
    finally:
        close_log()  # also on Ctrl-C or an unexpected error; a second close is a no-op
