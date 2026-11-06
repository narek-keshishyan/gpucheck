"""Bundled published GPU specs and name-based lookup.

Values are the v1 seed set (SXM/PCIe split). Each row is pinned by the plan and
must be verified against the NVIDIA datasheet before launch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    key: str
    tdp_w: float
    peak_bf16_tflops: float
    max_temp_c: float
    boost_clock_mhz: float


_SPECS: dict[str, GpuSpec] = {
    s.key: s
    for s in (
        GpuSpec("rtx-4090", 450.0, 165.2, 90.0, 2520.0),  # pinned by plan; verify against datasheet before launch
        GpuSpec("rtx-3090", 350.0, 71.0, 93.0, 1695.0),  # pinned by plan; verify against datasheet before launch
        GpuSpec("a100-80gb-sxm", 400.0, 312.0, 85.0, 1410.0),  # pinned by plan; verify against datasheet before launch
        GpuSpec("a100-80gb-pcie", 300.0, 312.0, 85.0, 1410.0),  # pinned by plan; verify against datasheet before launch
        GpuSpec("h100-sxm", 700.0, 989.0, 83.0, 1980.0),  # pinned by plan; verify against datasheet before launch
        GpuSpec("h100-pcie", 350.0, 756.0, 83.0, 1755.0),  # pinned by plan; verify against datasheet before launch
    )
}

# "3090" or "4090" as a standalone number, not followed by "ti" (Ti variants are not in the table).
_RTX_3090 = re.compile(r"(?<!\d)3090(?!\d|ti)")
_RTX_4090 = re.compile(r"(?<!\d)4090(?!\d|ti)")


def lookup(model_name: str) -> GpuSpec | None:
    """Return the bundled spec matching an NVML-style model name, or None.

    Matching is case- and whitespace-insensitive.
    """
    s = "".join(model_name.lower().split())
    if "h100" in s:
        if "nvl" in s:  # H100 NVL is not in the v1 table; never silently match SXM
            return None
        return _SPECS["h100-pcie"] if "pcie" in s else _SPECS["h100-sxm"]
    if "a100" in s:
        if "pcie" in s:
            return _SPECS["a100-80gb-pcie"]
        if "80gb" in s:
            return _SPECS["a100-80gb-sxm"]
        return None
    if _RTX_4090.search(s):
        return _SPECS["rtx-4090"]
    if _RTX_3090.search(s):
        return _SPECS["rtx-3090"]
    return None
