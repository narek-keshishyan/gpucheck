"""Helpers for testing several GPUs in one machine."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .adapter import GpuInventory, TelemetryAdapter
from .resultjson import canonical_dumps, run_id

MAX_LOT_CARDS = 16
MAX_LABEL_LEN = 80

TESTED, ABORTED, NOT_TESTED = "tested", "aborted", "not_tested"


def new_lot_id() -> str:
    return secrets.token_hex(16)


@dataclass(frozen=True)
class CardOutcome:
    """What happened to one card of a lot. `result` is None only when the card was not tested."""

    position: int
    inventory: GpuInventory | None
    outcome: str  # TESTED, ABORTED or NOT_TESTED
    reason: str | None
    result: dict | None


def _card_file(position: int) -> str:
    return f"gpu{position}.json"


def lot_summary(lot_id: str, label: str | None, created_utc: str, cards: list[CardOutcome]) -> dict:
    entries = []
    for c in cards:
        inv = c.inventory
        entries.append(
            {
                "position": c.position,
                "gpu_index": inv.gpu_index if inv else None,
                "model_name": inv.model_name if inv else None,
                "uuid_hash": inv.uuid_hash if inv else None,
                "serial": inv.serial if inv else None,
                "outcome": c.outcome,
                "reason": c.reason,
                "file": _card_file(c.position) if c.result is not None else None,
                "run_id": run_id(c.result) if c.result is not None else None,
                "findings": [f["code"] for f in c.result["findings"]] if c.result is not None else [],
            }
        )
    return {
        "lot_id": lot_id,
        "label": label,
        "size": len(cards),
        "created_utc": created_utc,
        "cards": entries,
        "uploaded": None,
    }


def write_lot(directory: Path, summary: dict, cards: list[CardOutcome]) -> None:
    """Write gpu<position>.json for every card with a result, then lot.json. Never overwrites a lot."""
    directory = Path(directory)
    if (directory / "lot.json").exists():
        raise FileExistsError(directory / "lot.json")
    directory.mkdir(parents=True, exist_ok=True)
    for c in cards:
        if c.result is not None:
            (directory / _card_file(c.position)).write_text(canonical_dumps(c.result))
    (directory / "lot.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def render_lot_table(summary: dict) -> str:
    out = ["Lot summary", "=" * 11]
    if summary.get("label"):
        out.append(f"Label: {summary['label']}")
    for c in summary["cards"]:
        gpu = "?" if c["gpu_index"] is None else c["gpu_index"]
        model = c["model_name"] or "unknown model"
        if c["serial"]:
            identity = c["serial"]
        elif c["uuid_hash"]:
            identity = c["uuid_hash"][:8]
        else:
            identity = "unknown"
        if c["outcome"] == TESTED:
            detail = ", ".join(c["findings"]) or "no findings"
        else:
            detail = c["reason"] or ""
        out.append(f"  #{c['position']}  GPU {gpu}  {model}  {identity}  {c['outcome']}  {detail}")
    return "\n".join(out) + "\n"


def lot_exit_code(cards: list[CardOutcome]) -> int:
    """0 when every card was tested, 1 when no card produced a result, otherwise 3."""
    if all(c.outcome == TESTED for c in cards):
        return 0
    if all(c.result is None for c in cards):
        return 1
    return 3


def parse_gpus(spec: str | None, count: int) -> list[int]:
    """Resolve a --gpus value to sorted, unique GPU indices. Raises ValueError with a one-line message."""
    if spec is None:
        return [0]
    if spec == "all":
        if count < 1:
            raise ValueError("no GPUs detected")
        indices = list(range(count))
    else:
        indices = []
        for part in spec.split(","):
            part = part.strip()
            if not part.isascii() or not part.isdigit():
                raise ValueError(f"expected 'all' or comma-separated GPU indices, got {spec!r}")
            i = int(part)
            if i >= count:
                raise ValueError(f"GPU {i} does not exist; this machine has {count} (indices 0-{count - 1})")
            indices.append(i)
        indices = sorted(set(indices))
    if len(indices) > MAX_LOT_CARDS:
        raise ValueError(f"at most {MAX_LOT_CARDS} GPUs can be tested at once, {len(indices)} selected")
    return indices


def settle(
    adapter: TelemetryAdapter,
    *,
    window_s: float = 30.0,
    max_drop_c: float = 1.0,
    cap_s: float = 300.0,
    period_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Wait until the idle temperature stops falling; return the seconds waited."""
    k = max(1, round(window_s / period_s))
    temps: list[float] = []
    elapsed = 0.0
    while True:
        try:
            temp = adapter.sample().temp_c
        except Exception:
            return elapsed
        if temp is None:
            return elapsed
        temps.append(temp)
        if len(temps) > k and temps[-k - 1] - temps[-1] < max_drop_c:
            return elapsed
        if elapsed >= cap_s:
            return elapsed
        sleep(period_s)
        elapsed += period_s
