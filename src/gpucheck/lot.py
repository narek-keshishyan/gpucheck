"""Helpers for testing several GPUs in one machine."""

from __future__ import annotations

import time
from collections.abc import Callable

from .adapter import TelemetryAdapter

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
