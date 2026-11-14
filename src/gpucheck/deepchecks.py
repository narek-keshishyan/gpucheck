"""The extra checks that need PyTorch: they run as their own phases after the load.

torch is imported lazily, in start(), like the load generator: constructing a check touches
neither torch nor the GPU. Every check takes a `device` so its arithmetic can be exercised on a
CPU in tests; on a GPU it is "cuda:<index>".
"""

from __future__ import annotations

import time

from .checks import CheckUnavailable
from .load import pin_device_order
from .run import warn_cleanup_failed

STEP_S = 0.5  # work done between two telemetry samples
GIB = 1 << 30
MIB = 1 << 20


def _import_torch():
    pin_device_order()
    try:
        import torch
    except ImportError as e:
        raise CheckUnavailable("PyTorch is not installed") from e
    return torch


def _sync(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _release(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---- memory test ---------------------------------------------------------------------------

MEMTEST_FRACTION = 0.9  # of the memory that is free once the load has released its buffers
MEMTEST_RESERVE_BYTES = GIB  # always left free: the comparisons need working memory
MEMTEST_MIN_BYTES = 256 * MIB  # below this there is nothing worth calling a memory test
MEMTEST_CHUNK_BYTES = GIB
MEMTEST_SLICE_WORDS = 1 << 24  # 128 MiB compared at a time, so working memory stays small
WORD = 8  # bytes; the test writes 64-bit words

# (name, fixed 64-bit value or None). Solid and checkerboard patterns find stuck and coupled
# bits; "address" writes each word's own position, which finds addressing faults; "random"
# writes a reproducible pseudo-random stream.
MEMTEST_PATTERNS: tuple[tuple[str, int | None], ...] = (
    ("zeros", 0),
    ("ones", -1),
    ("checker_a", -0x5555555555555556),  # 0xAAAA... as a signed 64-bit value
    ("checker_5", 0x5555555555555555),
    ("address", None),
    ("random", None),
)


class TorchMemtest:
    """Write known patterns across most of the card's free memory, read them back, count mismatches.

    It runs after the load has freed its buffers and holds memory only between start() and
    stop(). What it cannot see: memory it could not allocate (the driver's own, and the reserve
    it leaves free), errors a card with ECC corrects on the way out (those show in the ECC
    counters, which the run records alongside), and faults these patterns do not provoke."""

    name = "memtest"

    def __init__(
        self,
        device_index: int = 0,
        *,
        device: str | None = None,
        budget_bytes: int | None = None,
        chunk_bytes: int = MEMTEST_CHUNK_BYTES,
        slice_words: int = MEMTEST_SLICE_WORDS,
        seed: int = 0x6770636B,
        step_s: float = STEP_S,
        min_bytes: int = MEMTEST_MIN_BYTES,
        reserve_bytes: int = MEMTEST_RESERVE_BYTES,
    ) -> None:
        self._device = device or f"cuda:{device_index}"
        self._budget = budget_bytes
        self._min = min_bytes
        self._reserve = reserve_bytes
        self._chunk_words = max(1, chunk_bytes // WORD)
        self._slice = max(1, slice_words)
        self._seed = seed
        self._step_s = step_s
        self._torch = None
        self._chunks: list = []
        self._total: int | None = None
        self._tested = 0
        self._done: list[str] = []
        self._mismatched: dict[str, int] = {}
        self._work = None
        self._t0: float | None = None
        self._seconds: float | None = None

    def start(self) -> None:
        t = self._torch = _import_torch()
        budget = self._budget
        if budget is None:
            try:
                free, self._total = t.cuda.mem_get_info(self._device)
            except Exception as e:
                raise CheckUnavailable(f"could not read the card's free memory ({e})") from e
            budget = min(int(MEMTEST_FRACTION * free), free - self._reserve)
        if budget < self._min:
            raise CheckUnavailable("not enough free memory on the card for a memory test")
        left = budget // WORD
        while left > 0:
            words = min(self._chunk_words, left)
            try:
                self._chunks.append(t.empty(words, dtype=t.int64, device=self._device))
            except RuntimeError:  # out of memory: test what was allocated so far
                break
            left -= words
        self._tested = sum(c.numel() for c in self._chunks) * WORD
        if self._tested < self._min:
            self.stop()
            raise CheckUnavailable("could not allocate enough of the card's memory for a memory test")
        self._t0 = time.perf_counter()
        self._work = self._units()

    def _slices(self, chunk):
        n = chunk.numel()
        for k, a in enumerate(range(0, n, self._slice)):
            yield k, a, min(n, a + self._slice)

    def _generator(self, c_i: int, k: int):
        g = self._torch.Generator(device=self._device)
        g.manual_seed(self._seed + (c_i << 24) + k)
        return g

    def _expected(self, name: str, c_i: int, k: int, a: int, b: int):
        """The words the `address` or `random` pattern puts in words a..b of chunk c_i."""
        t = self._torch
        if name == "address":
            base = c_i << 40
            return t.arange(base + a, base + b, dtype=t.int64, device=self._device)
        return t.empty(b - a, dtype=t.int64, device=self._device).random_(generator=self._generator(c_i, k))

    def _write(self, name: str, value: int | None, c_i: int, chunk) -> None:
        if value is not None:
            chunk.fill_(value)
            return
        for k, a, b in self._slices(chunk):
            chunk[a:b] = self._expected(name, c_i, k, a, b)

    def _verify(self, name: str, value: int | None, c_i: int, chunk) -> int:
        t = self._torch
        wrong = 0
        for k, a, b in self._slices(chunk):
            expected = value if value is not None else self._expected(name, c_i, k, a, b)
            wrong += int(t.count_nonzero(chunk[a:b] != expected))
        return wrong

    def _units(self):
        """One unit of work per yield: a chunk written, or a chunk read back and compared.

        Every chunk is written before any is read, so each pattern sits in the whole tested
        area while it is being checked."""
        for name, value in MEMTEST_PATTERNS:
            for c_i, chunk in enumerate(self._chunks):
                self._write(name, value, c_i, chunk)
                _sync(self._torch, self._device)
                yield
            for c_i, chunk in enumerate(self._chunks):
                wrong = self._verify(name, value, c_i, chunk)
                if wrong:
                    self._mismatched[name] = self._mismatched.get(name, 0) + wrong
                yield
            self._done.append(name)

    def step(self) -> bool:
        if self._work is None:
            return False
        end = time.perf_counter() + self._step_s
        while True:
            try:
                next(self._work)
            except StopIteration:
                self._work = None
                self._seconds = time.perf_counter() - (self._t0 or 0.0)
                return False
            if time.perf_counter() >= end:
                return True

    def stop(self) -> None:
        try:
            self._chunks = []
            self._work = None
            if self._torch is not None:
                _release(self._torch, self._device)
        except Exception as e:
            warn_cleanup_failed(e)

    def result(self) -> dict:
        seconds = self._seconds
        if seconds is None and self._t0 is not None:
            seconds = time.perf_counter() - self._t0
        return {
            "bytes_tested": self._tested,
            "bytes_total": self._total,
            "patterns": list(self._done),
            "mismatched_words": sum(self._mismatched.values()),
            "by_pattern": dict(self._mismatched),
            "seconds": seconds,
        }
