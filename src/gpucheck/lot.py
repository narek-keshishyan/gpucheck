"""Helpers for testing several GPUs in one machine."""

from __future__ import annotations

MAX_LOT_CARDS = 16


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
