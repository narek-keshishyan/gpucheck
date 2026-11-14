"""The optional extra checks: what a run may be asked to do beyond the standard load, and the
pure functions that turn what was observed into the result's "checks" block.

None of this runs in a default run. Each check is opt-in, runs outside the baseline and load
phases (or only listens while they run), and writes one entry under "checks" in result.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .adapter import TelemetrySample
from .coverage import NO_LINKS, NVLINK_COUNTERS, PARTIAL, RAN, UNAVAILABLE

__all__ = [
    "RAN", "PARTIAL", "UNAVAILABLE", "NO_LINKS", "Check", "CheckPlan", "CheckUnavailable",
    "add_transfer", "burn_in_report", "driver_errors_report", "nvlink_report", "one_line", "thin_load",
]

MAX_REASON_LEN = 200
MAX_EVENTS = 200  # events kept in a result; the count of all events is kept separately
BURN_IN_MIN_LOAD_S = 3600.0  # a load at least this long is a burn-in
STANDARD_LOAD_S = 600.0  # the standard run's load; a burn-in's first ten minutes
STANDARD_WINDOW_S = 120.0  # the final 20% of the standard load, where its throughput is read
MAX_STORED_LOAD_SAMPLES = 3600  # a longer load is stored as this many summaries at most


class CheckUnavailable(Exception):
    """A check cannot run here (no torch, no memory to test, an unsupported operation)."""


@dataclass(frozen=True)
class CheckPlan:
    """Which extra checks a run was asked for. The default asks for none."""

    driver_errors: bool = False
    memtest: bool = False
    answer: bool = False
    links: bool = False

    def any(self) -> bool:
        return self.driver_errors or self.memtest or self.answer or self.links


class Check(Protocol):
    """A check that runs as its own phase after the load, in slices of about a second so the
    run can keep sampling the card (and its temperature watchdog) between slices."""

    name: str

    def start(self) -> None: ...

    def step(self) -> bool:
        """Do one slice of work; False when there is nothing left to do."""
        ...

    def stop(self) -> None:
        """Release everything the check holds. Never raises."""
        ...

    def result(self) -> dict: ...


def one_line(e: BaseException) -> str:
    """An exception as a short single line, for a result's `reason`."""
    try:
        return " ".join(f"{type(e).__name__}: {e}".split())[:MAX_REASON_LEN]
    except Exception:
        return type(e).__name__


def driver_errors_report(status: str, events: list[dict], reason: str | None = None) -> dict:
    out: dict = {"status": status, "count": len(events), "events": events[:MAX_EVENTS]}
    if reason:
        out["reason"] = reason
    return out


def _counter_totals(snapshot: dict) -> dict[str, int]:
    """Each error counter summed over the links that report it; falls back to the card's own totals."""
    totals: dict[str, int] = {}
    for link in snapshot.get("links") or []:
        for name, value in (link.get("errors") or {}).items():
            if name in NVLINK_COUNTERS and isinstance(value, int):
                totals[name] = totals.get(name, 0) + value
    for name, value in (snapshot.get("error_totals") or {}).items():
        if name in NVLINK_COUNTERS and isinstance(value, int) and name not in totals:
            totals[name] = value
    return totals


def nvlink_report(before: dict | None, after: dict | None, transfer: dict | None = None) -> dict:
    """The "nvlink" entry from link snapshots taken before the baseline and after the run.

    A card without an active link reports NO_LINKS: nothing was checked, which is not a pass."""
    if before is None and after is None:
        return {"status": NO_LINKS}
    if before is None or after is None:
        return {"status": UNAVAILABLE, "reason": "the links could be read only once"}
    was = {link["link"]: bool(link.get("active")) for link in before.get("links") or []}
    now = {link["link"]: bool(link.get("active")) for link in after.get("links") or []}
    if not any(was.values()) and not any(now.values()):
        return {"status": NO_LINKS, "links": len(was)}
    dropped = sorted(i for i, up in was.items() if up and not now.get(i, False))
    b, a = _counter_totals(before), _counter_totals(after)
    # A counter that went down was reset by something else; its rise is then unknown, not zero.
    errors = {k: a[k] - b[k] for k in NVLINK_COUNTERS if k in a and k in b and a[k] >= b[k]}
    out: dict = {
        "status": RAN,
        "links": len(was),
        "active": sum(1 for up in now.values() if up),
        "dropped": len(dropped),
        "errors": errors,
    }
    if transfer is not None:
        out["transfer"] = transfer
    return out


def add_transfer(report: dict, before: dict | None, after: dict | None, transfer: dict) -> dict:
    """A card's "nvlink" entry with the card-to-card copies added.

    `before` and `after` are link snapshots taken around the copies: counter rises and links
    that went down during them are added to the run's own, and the rise of the card's NVLink
    data counter (KiB sent) is kept as the evidence that the copies travelled over the links."""
    out = {**report, "transfer": dict(transfer)}
    if before is None or after is None:
        return out
    b, a = _counter_totals(before), _counter_totals(after)
    errors = dict(report.get("errors") or {})
    for k in NVLINK_COUNTERS:
        if k in errors and k in a and k in b and a[k] >= b[k]:
            errors[k] += a[k] - b[k]
    was = {link["link"]: bool(link.get("active")) for link in before.get("links") or []}
    now = {link["link"]: bool(link.get("active")) for link in after.get("links") or []}
    out["errors"] = errors
    out["dropped"] = report.get("dropped", 0) + sum(1 for i, up in was.items() if up and not now.get(i, False))
    out["active"] = sum(1 for up in now.values() if up)
    sent_before, sent_after = before.get("tx_kib"), after.get("tx_kib")
    if isinstance(sent_before, int) and isinstance(sent_after, int) and sent_after >= sent_before:
        out["transfer"]["nvlink_kib"] = sent_after - sent_before
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def burn_in_report(load: list[TelemetrySample], load_s: float, hz: float) -> dict | None:
    """For a load of an hour or more: how long it really ran and whether throughput held.

    early_tflops is read where a standard run reads its throughput (the last two minutes of the
    first ten); late_tflops over the final 20% of the load, like metrics.sustained_tflops."""
    if load_s < BURN_IN_MIN_LOAD_S or not load:
        return None
    lo, hi = round((STANDARD_LOAD_S - STANDARD_WINDOW_S) * hz), round(STANDARD_LOAD_S * hz)
    early = _mean([s.achieved_tflops for s in load[lo:hi] if s.achieved_tflops is not None])
    tail = load[-max(1, round(0.2 * len(load))):]
    late = _mean([s.achieved_tflops for s in tail if s.achieved_tflops is not None])
    return {
        "planned_s": load_s,
        "completed_s": len(load) / hz,
        "early_tflops": early if len(load) >= hi else None,
        "late_tflops": late,
    }


def thin_load(load: list[dict], limit: int = MAX_STORED_LOAD_SAMPLES) -> tuple[list[dict], int]:
    """Summarise a long load's samples so a day of them fits in an upload.

    Returns (samples, bucket) where each stored sample covers `bucket` consecutive one-second
    samples. Up to `limit` samples nothing changes and bucket is 1. Beyond it each summary keeps
    what a reader of the chart must not miss: the highest temperature and fan speed, the lowest
    clock, every throttle reason seen, the latest error count, and the mean power and throughput.
    The measurements in "metrics" are computed from every sample before this is applied."""
    if len(load) <= limit:
        return load, 1
    bucket = -(-len(load) // limit)
    out = []
    for i in range(0, len(load), bucket):
        group = load[i:i + bucket]

        def pick(key, fn):
            values = [s[key] for s in group if s.get(key) is not None]
            return fn(values) if values else None

        reasons = [s["throttle_reasons"] for s in group if s.get("throttle_reasons") is not None]
        out.append({
            "t_s": group[-1]["t_s"],
            "temp_c": pick("temp_c", max),
            "power_w": pick("power_w", lambda v: sum(v) / len(v)),
            "sm_clock_mhz": pick("sm_clock_mhz", min),
            "fan_pct": pick("fan_pct", max),
            "throttle_reasons": sorted({r for rs in reasons for r in rs}) if reasons else None,
            "ecc_volatile": pick("ecc_volatile", max),
            "achieved_tflops": pick("achieved_tflops", lambda v: sum(v) / len(v)),
        })
    return out, bucket
