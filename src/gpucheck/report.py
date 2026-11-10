"""Plain-text terminal report rendered from a result dict (stdlib only)."""

from __future__ import annotations

from typing import Any

NOT_COVERED = (
    "This is a pre-flight check, not proof of health. It does not test memory cells, "
    "silent data corruption, faults that appear only after hours of use, links between "
    "GPUs, or physical damage."
)

_MEASUREMENTS: list[tuple[str, str, str]] = [
    ("idle_temp_c", "Idle temperature", "C"),
    ("steady_temp_c", "Steady-state temperature", "C"),
    ("time_to_throttle_s", "Time to thermal throttle", "s"),
    ("sustained_tflops", "Sustained throughput", "TFLOPS"),
    ("tflops_ratio_vs_spec", "Throughput vs. datasheet", "x"),
    ("sm_clock_sigma", "SM clock variation (sigma)", "MHz"),
    ("power_sigma_w", "Power variation (sigma)", "W"),
    ("ecc_delta", "ECC errors during run", ""),
]

_ABORT_TEXT = {
    "thermal": "GPU temperature exceeded the safety limit",
    "ecc": "ECC errors jumped during the run",
    "telemetry": "telemetry became unreadable",
    "preflight_refused": "preflight refused the run",
}


def _fmt(value: Any, unit: str) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return f"{text} {unit}".rstrip()


def _pcie_link(inv: dict) -> str | None:
    keys = ("pcie_gen_current", "pcie_width_current", "pcie_gen_max", "pcie_width_max")
    if any(inv.get(k) is None for k in keys):
        return None
    return (
        f"gen {inv['pcie_gen_current']} x{inv['pcie_width_current']} "
        f"(max gen {inv['pcie_gen_max']} x{inv['pcie_width_max']})"
    )


def render_report(result: dict) -> str:
    inv = result.get("inventory") or {}
    pre = result.get("preflight") or {}
    met = result.get("metrics") or {}
    out: list[str] = ["gpucheck report", "=" * 15, ""]

    if result.get("aborted") and result["aborted"] != "preflight_refused":
        why = _ABORT_TEXT.get(result["aborted"], result["aborted"])
        out += [f"!! Run aborted early ({why}); showing what was collected.", ""]
    if pre.get("mode") == "measurements_only":
        out += [
            "NOTE: MIG/fractional GPU configuration - measurements only, no grade is given.",
            "",
        ]

    out.append("Inventory")
    out.append("---------")
    rows = [
        ("Model", inv.get("model_name")),
        ("Driver", inv.get("driver")),
        ("CUDA", inv.get("cuda")),
        ("ECC enabled", inv.get("ecc_enabled")),
        ("Power limit", _fmt(inv.get("power_limit_w"), "W")),
        ("Default power limit", _fmt(inv.get("default_power_limit_w"), "W")),
        ("MIG enabled", inv.get("mig_enabled")),
        ("Fractional sharing", inv.get("fractional")),
        ("Retired pages", inv.get("retired_pages")),
        ("Remapped rows", inv.get("remapped_rows")),
        ("VBIOS", inv.get("vbios_version")),
        ("PCIe link", _pcie_link(inv)),
        ("Lifetime ECC errors (corrected)", inv.get("ecc_aggregate_corrected")),
        ("Lifetime ECC errors (uncorrected)", inv.get("ecc_aggregate_uncorrected")),
    ]
    if inv.get("serial"):
        rows.append(("Serial", inv["serial"]))
    out += [f"  {k + ':':<36}{'n/a' if v is None else v}" for k, v in rows]
    for reason in pre.get("reasons") or []:
        out.append(f"  preflight: {reason}")
    out.append("")

    out.append("Measurements")
    out.append("------------")
    out += [f"  {label + ':':<30}{_fmt(met.get(key), unit)}" for key, label, unit in _MEASUREMENTS]
    seen = met.get("throttle_reasons_seen") or []
    if seen:
        seen_text = ", ".join(seen)
    elif "throttle_reasons" in (met.get("telemetry_gaps") or []):
        seen_text = "unknown"
    else:
        seen_text = "none"
    out.append(f"  {'Throttle reasons seen:':<30}{seen_text}")
    out.append("")

    out.append("Findings")
    out.append("--------")
    findings = result.get("findings") or []
    if findings:
        for f in findings:
            out.append(f"  [{f['severity'].upper()}] {f['code']} (confidence: {f['confidence']})")
            out.append(f"      {f['message']}")
    else:
        out.append("  none found by this test")
    out.append("")

    out.append("Telemetry gaps")
    out.append("--------------")
    gaps = list(met.get("telemetry_gaps") or [])
    for m in pre.get("missing_telemetry") or []:
        if m not in gaps:
            gaps.append(m)
    out.append("  " + (", ".join(gaps) if gaps else "none"))
    out.append("")

    out.append("Not covered")
    out.append("-----------")
    out.append("  " + NOT_COVERED)
    return "\n".join(out) + "\n"
