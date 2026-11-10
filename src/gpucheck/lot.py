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
    gpu_index: int | None  # the index the card was selected by, kept even when it could not be read
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
                "gpu_index": c.gpu_index,
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
        raise FileExistsError(f"{directory / 'lot.json'} already exists")
    directory.mkdir(parents=True, exist_ok=True)
    for c in cards:
        if c.result is not None:
            (directory / _card_file(c.position)).write_text(canonical_dumps(c.result))
    (directory / "lot.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def untested_entries(summary: dict) -> list[dict]:
    """The cards that were not tested, in the shape the lot upload carries."""
    keys = ("position", "gpu_index", "model_name", "uuid_hash", "serial", "reason")
    return [{k: c[k] for k in keys} for c in summary["cards"] if c["outcome"] == NOT_TESTED]


def lot_request(summary: dict, results: list[dict]) -> dict:
    """The body of a lot upload: the lot, its per-card results and the cards that were not tested."""
    return {
        "lot": {"lot_id": summary["lot_id"], "size": summary["size"], "label": summary["label"]},
        "results": results,
        "untested": untested_entries(summary),
    }


def read_lot(directory: Path) -> tuple[dict, list[dict]]:
    """Read lot.json and the per-card results in position order. Raises ValueError with a one-line message."""
    directory = Path(directory)
    path = directory / "lot.json"

    def load(p: Path, what: str):
        try:
            return json.loads(p.read_text())
        except OSError as e:
            raise ValueError(f"cannot read {what} {p}: {e.strerror or e}") from None
        except ValueError:
            raise ValueError(f"{what} {p} is not valid JSON") from None

    summary = load(path, "lot")
    if not isinstance(summary, dict) or not summary.get("lot_id") or not isinstance(summary.get("cards"), list):
        raise ValueError(f"{path} is not a gpucheck lot")
    cards = summary["cards"]
    if not all(isinstance(c, dict) and isinstance(c.get("position"), int) for c in cards):
        raise ValueError(f"{path} is not a gpucheck lot")
    results = []
    for card in sorted(cards, key=lambda c: c["position"]):
        name = card.get("file")
        if not name:
            continue
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"{path} names an unusable card file {name!r}")
        result = load(directory / name, "card file")
        if not isinstance(result, dict):
            raise ValueError(f"card file {directory / name} is not a gpucheck result")
        if run_id(result) != card.get("run_id"):
            raise ValueError(f"card file {directory / name} was changed after the run; it will not be uploaded")
        results.append(result)
    return summary, results


def mark_uploaded(directory: Path, server: str, outcome) -> None:
    """Record a successful lot upload in lot.json (format as write_lot writes it).

    A retry of an upload the server already holds returns no delete tokens; the ones recorded earlier are kept."""
    path = Path(directory) / "lot.json"
    summary = json.loads(path.read_text())
    previous = {
        r.get("position"): r.get("delete_token")
        for r in ((summary.get("uploaded") or {}).get("runs") or [])
        if isinstance(r, dict) and r.get("delete_token")
    }
    summary["uploaded"] = {
        "server": server,
        "url": outcome.url,
        "runs": [
            {
                **{k: r.get(k) for k in ("position", "run_id", "url", "card_url", "delete_token")},
                "delete_token": r.get("delete_token") or previous.get(r.get("position")),
            }
            for r in outcome.runs
        ],
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


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
