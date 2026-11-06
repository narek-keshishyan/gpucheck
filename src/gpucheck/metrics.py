"""Derived metrics computed from a RawRun.

Every metric whose inputs are all None is None and its metric-source field
name is recorded in `telemetry_gaps`; nothing is guessed. Load-phase sample
timestamps are absolute, so time_to_throttle_s is measured from the first
load sample.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from .adapter import TelemetrySample
from .run import RawRun
from .specs import GpuSpec

THERMAL_REASON = "thermal"
FINAL_FRACTION = 0.2


@dataclass(frozen=True)
class Metrics:
    idle_temp_c: float | None
    time_to_throttle_s: float | None
    steady_temp_c: float | None
    sustained_tflops: float | None
    tflops_ratio_vs_spec: float | None
    sm_clock_sigma: float | None
    power_sigma_w: float | None
    ecc_delta: int | None
    throttle_reasons_seen: list[str]
    telemetry_gaps: list[str]


def _present(values: Iterable[float | int | None]) -> list[float | int]:
    return [v for v in values if v is not None]


def _mean(values: Iterable[float | None]) -> float | None:
    vals = _present(values)
    return sum(vals) / len(vals) if vals else None


def _pstdev(values: Iterable[float | None]) -> float | None:
    vals = _present(values)
    return statistics.pstdev(vals) if len(vals) >= 2 else None


def derive(run: RawRun, spec: GpuSpec | None) -> Metrics:
    gaps: list[str] = []

    def gap(name: str) -> None:
        if name not in gaps:
            gaps.append(name)

    baseline = run.baseline
    load = run.load

    idle_temp = _mean(s.temp_c for s in baseline)
    if idle_temp is None:
        gap("temp_c")

    time_to_throttle: float | None = None
    known_reasons = [s for s in load if s.throttle_reasons is not None]
    if not known_reasons:
        gap("throttle_reasons")  # unknown throttle state is a gap, never "no throttling"
    for s in known_reasons:
        assert s.throttle_reasons is not None
        if THERMAL_REASON in s.throttle_reasons:
            time_to_throttle = s.t_s - load[0].t_s
            break

    window: list[TelemetrySample] = []
    if load:
        window = load[-max(1, round(FINAL_FRACTION * len(load))) :]

    steady_temp = _mean(s.temp_c for s in window)
    if steady_temp is None:
        gap("temp_c")

    sustained = _mean(s.achieved_tflops for s in window)
    if sustained is None:
        gap("achieved_tflops")
    ratio = sustained / spec.peak_bf16_tflops if sustained is not None and spec is not None else None

    sm_sigma = _pstdev(s.sm_clock_mhz for s in load)
    if sm_sigma is None:
        gap("sm_clock_mhz")
    power_sigma = _pstdev(s.power_w for s in load)
    if power_sigma is None:
        gap("power_w")

    all_ecc = _present(s.ecc_volatile for s in baseline + load)
    if not all_ecc:
        gap("ecc_volatile")
    ecc_delta: int | None = None
    load_ecc = _present(s.ecc_volatile for s in load)
    if load_ecc and len(all_ecc) >= 2:
        ecc_delta = load_ecc[-1] - all_ecc[0]

    reasons_seen = sorted({r for s in known_reasons for r in (s.throttle_reasons or ())})

    return Metrics(
        idle_temp_c=idle_temp,
        time_to_throttle_s=time_to_throttle,
        steady_temp_c=steady_temp,
        sustained_tflops=sustained,
        tflops_ratio_vs_spec=ratio,
        sm_clock_sigma=sm_sigma,
        power_sigma_w=power_sigma,
        ecc_delta=ecc_delta,
        throttle_reasons_seen=reasons_seen,
        telemetry_gaps=gaps,
    )
