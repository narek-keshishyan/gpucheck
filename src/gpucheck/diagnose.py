"""Diagnosis rule table: turn derived metrics into plain-language findings.

Rules are evaluated in a fixed order (MEMORY_ERRORS, MEMORY_NOT_CHECKED, HOT_UNDER_LOAD,
HOT_ENVIRONMENT, THERMAL_THROTTLING, POWER_CAPPED, UNDERPERFORMING_UNKNOWN,
INSUFFICIENT_EVIDENCE, PCIE_LINK_DEGRADED) and each rule fires at most once. Unknown throttle data
("throttle_reasons" in telemetry_gaps) is treated as unknown, never as "none". A rule whose required inputs are None does
not fire. Rules that compare against the datasheet need `spec`; they are
skipped when it is None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .adapter import GpuInventory
from .metrics import THERMAL_REASON, Metrics
from .preflight import Preflight
from .specs import GpuSpec

Severity = Literal["info", "warn", "bad"]
Confidence = Literal["low", "medium", "high"]

TIM_TIME_TO_THROTTLE_MAX_S = 180.0
TIM_STEADY_MARGIN_C = 3.0
TIM_IDLE_MAX_C = 45.0
TIM_HIGH_CONFIDENCE_IDLE_C = 40.0
HOT_IDLE_C = 50.0
POWER_LIMIT_FRACTION = 0.9
POWER_REASON_HEADROOM_C = 15.0
UNDERPERFORMING_RATIO = 0.9
# A card regulating at its stock power limit falls short of the datasheet peak without any fault,
# and by how much depends on the variant's limit. Measured on healthy cards: an A100 80GB at
# 400 W reached 0.86, two DGX Station A100s (275 W stock limit) 0.69 and 0.70, an A100 40GB held
# to 300 W 0.60. So this is only a floor for gross shortfalls; comparing a card with others of its
# model is the server's job, not this rule's.
UNDERPERFORMING_AT_POWER_LIMIT_RATIO = 0.4


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    confidence: Confidence
    message: str


def _g(value: float) -> str:
    return f"{value:g}"


def _throttled(m: Metrics) -> bool:
    return m.time_to_throttle_s is not None or THERMAL_REASON in m.throttle_reasons_seen


def _memory_errors(m: Metrics, inv: GpuInventory) -> Finding | None:
    parts: list[str] = []
    if m.ecc_delta is not None and m.ecc_delta > 0:
        parts.append(f"ECC error count rose by {m.ecc_delta} during the run")
    if inv.remapped_rows is not None and inv.remapped_rows > 0:
        parts.append(f"{inv.remapped_rows} memory row(s) were remapped")
    if inv.row_remap_failure:
        parts.append("memory row remapping has failed")
    if inv.ecc_aggregate_uncorrected is not None and inv.ecc_aggregate_uncorrected > 0:
        parts.append(f"{inv.ecc_aggregate_uncorrected} uncorrected memory error(s) were recorded over the card's lifetime")
    if not parts:
        return None
    sentence = " and ".join(parts)
    return Finding(
        "MEMORY_ERRORS",
        "bad",
        "high",
        sentence[0].upper() + sentence[1:] + ". The card has recorded memory errors.",
    )


def _memory_not_checked(m: Metrics, inv: GpuInventory) -> Finding | None:
    """Say so when the card exposes no memory-error counters at all."""
    if "ecc_volatile" not in m.telemetry_gaps:
        return None
    known = (inv.remapped_rows, inv.retired_pages, inv.ecc_aggregate_corrected, inv.ecc_aggregate_uncorrected)
    if any(v is not None for v in known):
        return None
    if _memory_errors(m, inv) is not None:
        return None
    return Finding(
        "MEMORY_NOT_CHECKED",
        "info",
        "high",
        "This card exposes no memory-error counters, so memory health was not checked.",
    )


def _pcie_link(inv: GpuInventory) -> Finding | None:
    vals = (inv.pcie_gen_current, inv.pcie_gen_max, inv.pcie_width_current, inv.pcie_width_max)
    if any(v is None for v in vals):
        return None
    gc, gm, wc, wm = vals
    if inv.pcie_link_under_load:
        if wc >= wm and gc >= gm:
            return None
        return Finding(
            "PCIE_LINK_DEGRADED",
            "info",
            "high",
            f"PCIe link ran at gen {gc} x{wc} under load; the card and slot support gen {gm} x{wm}. "
            "Check seating, riser and slot.",
        )
    # Read at idle: a link saves power by dropping to a lower generation, so only its width counts.
    if wc >= wm:
        return None
    return Finding(
        "PCIE_LINK_DEGRADED",
        "info",
        "high",
        f"PCIe link width is x{wc}; the card and slot support x{wm}. Check seating, riser and slot. "
        "The link was read at idle, where a lower speed is normal power saving, so its speed was not judged.",
    )


def _hot_under_load(m: Metrics, spec: GpuSpec | None) -> Finding | None:
    if spec is None or m.time_to_throttle_s is None or m.steady_temp_c is None or m.idle_temp_c is None:
        return None
    if m.idle_temp_c > TIM_IDLE_MAX_C:
        return None
    if m.time_to_throttle_s >= TIM_TIME_TO_THROTTLE_MAX_S:
        return None
    if m.steady_temp_c < spec.max_temp_c - TIM_STEADY_MARGIN_C:
        return None
    confidence: Confidence = "high" if m.idle_temp_c <= TIM_HIGH_CONFIDENCE_IDLE_C else "medium"
    return Finding(
        "HOT_UNDER_LOAD",
        "warn",
        confidence,
        f"Throttled after {_g(m.time_to_throttle_s)} s of load and ran at {_g(m.steady_temp_c)} °C "
        f"against a {_g(spec.max_temp_c)} °C limit, while idling at {_g(m.idle_temp_c)} °C. "
        "The card runs hot under load. Possible causes: degraded thermal paste or cooler "
        "contact, a failing fan, or restricted airflow.",
    )


def _hot_environment(m: Metrics) -> Finding | None:
    if m.idle_temp_c is None or m.idle_temp_c <= HOT_IDLE_C:
        return None
    return Finding(
        "HOT_ENVIRONMENT",
        "warn",
        "high",
        f"Idle temperature was {_g(m.idle_temp_c)} °C, above {_g(HOT_IDLE_C)} °C, so room "
        "temperature or airflow may be the cause of the heat.",
    )


def _power_capped(m: Metrics, inv: GpuInventory, spec: GpuSpec | None) -> Finding | None:
    limit_arm = (
        inv.power_limit_w is not None
        and inv.default_power_limit_w is not None
        and inv.power_limit_w < POWER_LIMIT_FRACTION * inv.default_power_limit_w
    )
    # A card at its stock limit shows the power-cap flag under full load: that is normal regulation.
    # So the flag alone counts only when the limits themselves could not be read.
    limits_unknown = inv.power_limit_w is None or inv.default_power_limit_w is None
    reason_arm = (
        limits_unknown
        and spec is not None
        and m.steady_temp_c is not None
        and any("power" in r for r in m.throttle_reasons_seen)
        and m.steady_temp_c < spec.max_temp_c - POWER_REASON_HEADROOM_C
    )
    if limit_arm:
        assert inv.power_limit_w is not None and inv.default_power_limit_w is not None
        return Finding(
            "POWER_CAPPED",
            "info",
            "high",
            f"Power limit is {_g(inv.power_limit_w)} W, below 90% of the "
            f"{_g(inv.default_power_limit_w)} W default, so the card is capped below normal power.",
        )
    if reason_arm:
        assert spec is not None and m.steady_temp_c is not None
        return Finding(
            "POWER_CAPPED",
            "info",
            "high",
            f"Power capping was seen while steady at {_g(m.steady_temp_c)} °C, well below the "
            f"{_g(spec.max_temp_c)} °C limit, so the power limit rather than heat held clocks down.",
        )
    return None


def _thermal_throttling(m: Metrics) -> Finding | None:
    """Throttling that neither HOT_UNDER_LOAD nor INSUFFICIENT_EVIDENCE explains."""
    if not _throttled(m):
        return None
    when = f" after {_g(m.time_to_throttle_s)} s of load" if m.time_to_throttle_s is not None else ""
    return Finding(
        "THERMAL_THROTTLING",
        "warn",
        "medium",
        f"The card thermal-throttled{when} even though it idled cool, so sustained "
        "clocks were limited by heat; check airflow and cooler.",
    )


def _at_stock_power_limit(inv: GpuInventory) -> bool:
    return (
        inv.power_limit_w is not None
        and inv.default_power_limit_w is not None
        and inv.power_limit_w >= POWER_LIMIT_FRACTION * inv.default_power_limit_w
    )


def _underperforming(m: Metrics, inv: GpuInventory, spec: GpuSpec | None) -> Finding | None:
    if spec is None or m.tflops_ratio_vs_spec is None:
        return None
    if m.throttle_reasons_seen:
        # Only the power flag, on a card at its stock limit: no other rule speaks for this card,
        # so a shortfall too large for normal power regulation must not pass without a word.
        only_power = not _throttled(m) and all("power" in r for r in m.throttle_reasons_seen)
        if not (only_power and _at_stock_power_limit(inv)):
            return None
        if m.tflops_ratio_vs_spec >= UNDERPERFORMING_AT_POWER_LIMIT_RATIO:
            return None
        return Finding(
            "UNDERPERFORMING_UNKNOWN",
            "warn",
            "low",
            f"Sustained throughput was {m.tflops_ratio_vs_spec:.1%} of the datasheet peak. The card was "
            "regulating at its normal power limit, which does not explain a shortfall this large, "
            "and the cause is unknown.",
        )
    if m.tflops_ratio_vs_spec >= UNDERPERFORMING_RATIO:
        return None
    if "throttle_reasons" in m.telemetry_gaps:
        context = "and throttle-reason data was unavailable"
    else:
        context = "with no throttling seen"
    return Finding(
        "UNDERPERFORMING_UNKNOWN",
        "warn",
        "low",
        f"Sustained throughput was {m.tflops_ratio_vs_spec:.1%} of the datasheet peak {context}, "
        "and the cause is unknown.",
    )


def _insufficient_evidence(m: Metrics, spec: GpuSpec | None) -> Finding | None:
    if not _throttled(m):
        return None
    if m.idle_temp_c is not None and m.idle_temp_c <= TIM_IDLE_MAX_C:
        return None
    if m.idle_temp_c is None:
        detail = "no idle baseline was recorded"
    else:
        detail = f"idle was {_g(m.idle_temp_c)} °C, above {_g(TIM_IDLE_MAX_C)} °C"
    return Finding(
        "INSUFFICIENT_EVIDENCE",
        "info",
        "low",
        f"Throttling was seen, but {detail}, so a cooler fault cannot be confirmed or ruled out.",
    )


def diagnose(
    metrics: Metrics,
    preflight: Preflight,
    inventory: GpuInventory,
    spec: GpuSpec | None,
) -> list[Finding]:
    """Return the findings for one card, in fixed rule order; empty means no findings."""
    del preflight  # the v1 rule table does not consult preflight
    tim = _hot_under_load(metrics, spec)
    insufficient = _insufficient_evidence(metrics, spec)
    throttling = _thermal_throttling(metrics) if tim is None and insufficient is None else None
    rules = (
        _memory_errors(metrics, inventory),
        _memory_not_checked(metrics, inventory),
        tim,
        _hot_environment(metrics),
        throttling,
        _power_capped(metrics, inventory, spec),
        _underperforming(metrics, inventory, spec),
        insufficient,
        _pcie_link(inventory),
    )
    return [f for f in rules if f is not None]
