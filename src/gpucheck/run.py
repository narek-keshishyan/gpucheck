"""Run engine: preflight, baseline phase, load phase, with a safety watchdog.

Time is driven by this module's own schedule: every iteration calls
adapter.sample() once, then sleeps until the next 1/hz slot (measured with the
injected clock). Sample timestamps (t_s) come from the adapter. A sample() that
raises is recorded as an all-None sample whose t_s is the previous sample's t_s
plus 1/hz (0.0 if there is none yet), so the series stays evenly spaced.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from .adapter import GpuInventory, TelemetryAdapter, TelemetrySample
from .preflight import Preflight, run_preflight
from .specs import GpuSpec

THERMAL_MARGIN_C = 5.0
UNKNOWN_GPU_THERMAL_LIMIT_C = 95.0  # abort limit when the model is not in the spec table
THERMAL_CONSECUTIVE = 3
ECC_JUMP = 100
MAX_CONSECUTIVE_SAMPLE_ERRORS = 10


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


def execute_run(
    adapter: TelemetryAdapter,
    load: LoadGenerator,
    config: RunConfig,
    spec: GpuSpec | None,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RawRun:
    inventory = adapter.inventory()
    preflight = run_preflight(adapter)
    baseline: list[TelemetrySample] = []
    loaded: list[TelemetrySample] = []
    period = 1.0 / config.hz
    state = {"last_t": None, "errors": 0}

    def result(aborted: str | None) -> RawRun:
        return RawRun(inventory, preflight, baseline, loaded, config, aborted)

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
        for i in range(count):
            s = take()
            if tick:
                tflops = load.tick()
                if tflops is not None:
                    s = replace(s, achieved_tflops=tflops)
            out.append(s)
            if watchdog is not None and (reason := watchdog.check(s)):
                return reason
            if state["errors"] >= MAX_CONSECUTIVE_SAMPLE_ERRORS:
                return "telemetry"
            sleep(max(0.0, start + (i + 1) * period - clock()))
        return None

    aborted = phase(round(config.baseline_s * config.hz), baseline, None, tick=False)
    if aborted:
        return result(aborted)
    try:
        load.start()
        aborted = phase(round(config.load_s * config.hz), loaded, _Watchdog(spec), tick=True)
    finally:
        try:
            load.stop()
        except Exception:  # a failed cleanup must not replace the samples already taken
            pass
    return result(aborted)
