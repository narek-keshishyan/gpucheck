"""Helpers for testing several GPUs in one machine."""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .adapter import GpuInventory, TelemetryAdapter
from .resultjson import canonical_dumps, run_id

MAX_LOT_CARDS = 16
MAX_LABEL_LEN = 80

_CARD_FILE = re.compile(r"gpu[0-9]+\.json")

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


def lot_summary(
    lot_id: str, label: str | None, created_utc: str, cards: list[CardOutcome], *, finished: bool = True
) -> dict:
    """lot.json's content. `finished` is False while the lot is still running (or was killed)."""
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
        "finished": finished,
    }


CARD_MODE = 0o666  # narrowed by the umask, like the single-run result file
SUMMARY_MODE = 0o600  # lot.json holds delete tokens: owner only, whatever the umask


def _write_new(path: Path, text: str, mode: int) -> None:
    """Create `path` and write `text`; FileExistsError if it already exists."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise


def _write_atomic(path: Path, text: str, mode: int) -> None:
    """Write via a temporary file in the same directory and an atomic rename, so a crash
    or kill leaves either the old file or the new one, never a partial one."""
    tmp = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    _write_new(tmp, text, mode)
    try:
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _summary_text(summary: dict) -> str:
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def write_card(directory: Path, card: CardOutcome) -> None:
    """Write gpu<position>.json for a card that has a result."""
    if card.result is not None:
        _write_atomic(Path(directory) / _card_file(card.position), canonical_dumps(card.result), CARD_MODE)


def create_summary(directory: Path, summary: dict) -> None:
    """Claim the directory for this run by creating lot.json; FileExistsError if another lot is there,
    including one whose run is still going."""
    _write_new(Path(directory) / "lot.json", _summary_text(summary), SUMMARY_MODE)


def write_summary(directory: Path, summary: dict) -> None:
    """Rewrite this run's own lot.json, which create_summary created before the first card ran."""
    _write_atomic(Path(directory) / "lot.json", _summary_text(summary), SUMMARY_MODE)


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
    invalid = ValueError(f"{path} is not a valid gpucheck lot")
    if not isinstance(summary, dict) or not summary.get("lot_id") or not isinstance(summary.get("cards"), list):
        raise invalid
    if type(summary.get("size")) is not int:
        raise invalid
    if not isinstance(summary.get("finished", True), bool):  # absent (older lots): finished
        raise invalid
    cards = summary["cards"]
    if not all(
        isinstance(c, dict) and type(c.get("position")) is int and isinstance(c.get("outcome"), str)
        for c in cards
    ):
        raise invalid
    results = []
    for card in sorted(cards, key=lambda c: c["position"]):
        name = card.get("file")
        if not name:
            continue
        card_path = directory / name if isinstance(name, str) else None
        if (
            card_path is None
            or not _CARD_FILE.fullmatch(name)
            or card_path.is_symlink()
            or not card_path.is_file()
        ):
            raise ValueError(f"{path} names a card file that is not a plain gpu<N>.json file in the lot: {name!r}")
        result = load(directory / name, "card file")
        if not isinstance(result, dict):
            raise ValueError(f"card file {directory / name} is not a gpucheck result")
        if run_id(result) != card.get("run_id"):
            raise ValueError(f"card file {directory / name} was changed after the run; it will not be uploaded")
        results.append(result)
    return summary, results


def mark_uploaded(directory: Path, server: str, outcome) -> dict:
    """Record a successful lot upload in lot.json (format as write_summary writes it); return what was written.

    A retry of an upload the server already holds returns no delete tokens; the ones recorded earlier are kept."""
    path = Path(directory) / "lot.json"
    summary = json.loads(path.read_text())
    fields = ("position", "run_id", "url", "card_url", "delete_token")
    old = summary.get("uploaded")
    record: dict[int, dict] = {}
    if isinstance(old, dict):
        if _same_server(old.get("server"), server):
            for r in old.get("runs") or []:
                if isinstance(r, dict) and type(r.get("position")) is int:
                    record[r["position"]] = {k: r.get(k) for k in fields}
        elif any(isinstance(r, dict) and r.get("delete_token") for r in old.get("runs") or []):
            summary["previous_uploads"] = [*(summary.get("previous_uploads") or []), old]
    for r in outcome.runs:
        if type(r.get("position")) is not int:  # a malformed response entry names no card
            continue
        entry = record.setdefault(r["position"], {k: None for k in fields})
        token = entry.get("delete_token")
        entry.update({k: r.get(k) for k in fields})
        entry["delete_token"] = r.get("delete_token") or token
    summary["uploaded"] = {
        "server": server,
        "url": outcome.url,
        "runs": sorted(record.values(), key=lambda e: e["position"]),
    }
    write_summary(directory, summary)
    return summary


def lot_finished(summary: dict) -> bool:
    """False only for a lot.json written while its run was still going (a killed run)."""
    return summary.get("finished", True) is not False


def _same_server(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and a.rstrip("/") == b.rstrip("/")


def previous_upload_with_tokens(summary: dict, server: str) -> str | None:
    """The server of an earlier upload that holds delete tokens and differs from `server`, else None."""
    old = summary.get("uploaded")
    if not isinstance(old, dict) or _same_server(old.get("server"), server):
        return None
    if any(isinstance(r, dict) and r.get("delete_token") for r in old.get("runs") or []):
        return old.get("server")
    return None


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
                if count < 1:
                    raise ValueError(f"GPU {i} does not exist; this machine reports no GPUs")
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
