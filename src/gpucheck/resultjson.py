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


# Per-card manifest fields: written to the inventory only when known, so a result
# without them stays byte-identical to one built before they existed.
MANIFEST_FIELDS = (
    "gpu_index",
    "vbios_version",
    "pcie_gen_current",
    "pcie_gen_max",
    "pcie_width_current",
    "pcie_width_max",
    "pcie_link_under_load",
    "ecc_aggregate_corrected",
    "ecc_aggregate_uncorrected",
    "serial",
)


def build_result(
    run: RawRun,
    metrics: Metrics,
    findings: list[Finding],
    challenge: dict | None = None,
    challenge_workload: dict | None = None,
    timing: dict | None = None,
    lot: dict | None = None,
) -> dict:
    result = _normalize(
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
    for key in MANIFEST_FIELDS:
        if result["inventory"].get(key) is None:
            result["inventory"].pop(key, None)
    # Both omitted -> output is byte-identical to v1 (run_ids must not shift).
    if challenge is not None:
        result["challenge"] = _normalize(
            {"challenge_id": challenge["challenge_id"], "nonce": challenge["nonce"]}
        )
    if challenge_workload is not None:
        result["config"]["challenge_workload"] = _normalize(challenge_workload)
    if timing is not None:
        result["timing"] = _normalize(timing)
    if lot is not None:
        result["lot"] = _normalize(lot)
    return result


def canonical_dumps(d: dict) -> str:
    return json.dumps(
        _normalize(d), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def run_id(d: dict) -> str:
    return hashlib.sha256(canonical_dumps(d).encode()).hexdigest()[:16]
