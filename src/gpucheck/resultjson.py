"""Canonical result.json: build the result dict and serialize it reproducibly.

canonical_dumps is deterministic and idempotent: keys sorted, compact
separators, every float rounded to 4 decimal places, sets/frozensets emitted
as sorted lists, NaN/Infinity rejected (ValueError). Dataclass instances are
also accepted defensively and converted via dataclasses.asdict.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .diagnose import Finding
from .metrics import Metrics
from .run import RawRun

SCHEMA_VERSION = "1"
FLOAT_DP = 4


def _normalize(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(_normalize(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, float):
        # round(nan/inf) returns them unchanged; json.dumps(allow_nan=False) rejects.
        return round(obj, FLOAT_DP) + 0.0  # + 0.0 turns -0.0 into 0.0
    return obj


def build_result(run: RawRun, metrics: Metrics, findings: list[Finding]) -> dict:
    return _normalize(
        {
            "schema_version": SCHEMA_VERSION,
            "inventory": run.inventory,
            "preflight": run.preflight,
            "config": run.config,
            "metrics": metrics,
            "findings": list(findings),
            "samples": {"baseline": list(run.baseline), "load": list(run.load)},
            "aborted": run.aborted,
        }
    )


def canonical_dumps(d: dict) -> str:
    return json.dumps(
        _normalize(d), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def run_id(d: dict) -> str:
    return hashlib.sha256(canonical_dumps(d).encode()).hexdigest()[:16]
