"""Preflight policy: decide whether a card may be tested, and how.

Precedence when several rules apply: refuse beats measurements_only beats full.
`reasons` collects every rule that fired, in evaluation order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal

from .adapter import TelemetryAdapter, TelemetrySample

Mode = Literal["full", "measurements_only", "refuse"]

# TelemetrySample fields that are None by design at preflight, not missing hardware.
_EXPECTED_NONE = frozenset({"achieved_tflops"})


@dataclass(frozen=True)
class Preflight:
    mode: Mode
    reasons: list[str] = field(default_factory=list)
    missing_telemetry: list[str] = field(default_factory=list)


def run_preflight(adapter: TelemetryAdapter) -> Preflight:
    inv = adapter.inventory()
    busy = adapter.busy()
    shared = inv.mig_enabled or inv.fractional

    reasons: list[str] = []
    refuse = False
    measurements_only = False

    # Rule 1: a compute process on a full (non-shared) card refuses the run.
    if busy is True:
        if shared:
            reasons.append("busy neighbor on shared GPU (expected)")
        else:
            refuse = True
            reasons.append("GPU busy: compute process running on a full card")

    # Rule 2: shared configurations get measurements only.
    if inv.mig_enabled:
        measurements_only = True
        reasons.append("MIG enabled: measurements only")
    elif inv.fractional:
        measurements_only = True
        reasons.append("fractional sharing enabled: measurements only")

    # Rule 3: busy state cannot be read (process list hidden); proceed in full mode.
    if busy is None:
        reasons.append("busy-state unknown")

    sample = adapter.sample()
    missing = [
        f.name
        for f in fields(TelemetrySample)
        if getattr(sample, f.name) is None and f.name not in _EXPECTED_NONE
    ]

    if refuse:
        mode: Mode = "refuse"
    elif measurements_only:
        mode = "measurements_only"
    else:
        mode = "full"
    return Preflight(mode=mode, reasons=reasons, missing_telemetry=missing)
