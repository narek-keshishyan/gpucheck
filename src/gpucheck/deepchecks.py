"""The extra checks that need PyTorch: they run as their own phases after the load.

torch is imported lazily, in start(), like the load generator: constructing a check touches
neither torch nor the GPU. Every check takes a `device` so its arithmetic can be exercised on a
CPU in tests; on a GPU it is "cuda:<index>".
"""

from __future__ import annotations

import random
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
        # random_() fills 63 bits (never the top one). Folding in a copy shifted left by one bit
        # makes every one of the 64 bits vary.
        drawn = t.empty(b - a, dtype=t.int64, device=self._device).random_(generator=self._generator(c_i, k))
        mixed = drawn << 1
        mixed ^= drawn
        return mixed

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


# ---- answer check --------------------------------------------------------------------------

ANSWER_S = 60.0
ANSWER_QUICK_S = 20.0
# (dtype, matrix size, nonzero entries per row). Each case is exact in its own number format:
# every partial sum of a product is a whole multiple, of at most `weight`, of one power of two,
# and `weight` is no more than the format holds exactly (2**24 for float32, 2**11 for float16,
# 2**8 for bfloat16). So the result is the same in whatever order, and at whatever precision
# from the operands' own upwards, the card adds the terms up. The half-precision cases keep
# the full size, so they go through the same large matrix-multiply routine as the load.
ANSWER_CASES: tuple[tuple[str, int, int], ...] = (
    ("float32", 4096, 4096),
    ("float16", 4096, 2048),
    ("bfloat16", 4096, 256),
    ("float64", 2048, 2048),
)
_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)  # row scales: powers of two, exact in every format
_SIGNED_SCALES = tuple(sign * scale for scale in _SCALES for sign in (1.0, -1.0))


def _hadamard(torch, n: int, dtype, device: str):
    """The n x n Sylvester Hadamard matrix (n a power of two): entries +1/-1, rows orthogonal."""
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


class TorchAnswerCheck:
    """Compute matrix products whose correct result is known in advance and count wrong ones.

    The operands are built from a Hadamard matrix, whose rows are orthogonal: with its rows
    shuffled and scaled by signed powers of two, the product of one such matrix with the
    transpose of another has exactly one known non-zero entry in each row and is zero
    everywhere else. Nothing is taken from a second computation on the same card: the expected
    matrix is written down from the shuffles alone.

    It catches a card that returns wrong numbers for these products during the minute it runs.
    It does not catch corruption that needs other operations, other data or more time to show,
    which by published accounts is most of it."""

    name = "answer"

    def __init__(
        self,
        device_index: int = 0,
        *,
        device: str | None = None,
        seconds: float = ANSWER_S,
        cases: tuple[tuple[str, int, int], ...] = ANSWER_CASES,
        seed: int = 0x6770636B,
        step_s: float = STEP_S,
        clock=time.perf_counter,
    ) -> None:
        self._device = device or f"cuda:{device_index}"
        self._seconds = seconds
        self._cases = list(cases)
        self._seed = seed
        self._step_s = step_s
        self._clock = clock
        self._torch = None
        self._base: dict[tuple, object] = {}
        self._counts: dict[tuple, list[int]] = {}
        self._skipped: list[str] = []
        self._i = 0
        self._rounds = 0
        self._t0: float | None = None
        self._elapsed: float | None = None
        self._wrong_elements = 0

    def start(self) -> None:
        t = self._torch = _import_torch()
        # The shuffles and scales are drawn, and the expected result worked out, in plain Python:
        # nothing of the bookkeeping runs on the card, and nothing runs as small CPU tensor
        # operations either. Those took 10 to 25 ms each on a rented machine whose container had
        # 32 of the host's 128 cores (PyTorch started a thread for every host core), which left
        # the card idle for most of the check.
        self._rng = random.Random(self._seed)
        for case in list(self._cases):
            name, n, weight = case
            try:
                dtype = getattr(t, name)
                block = _hadamard(t, weight, dtype, self._device)
                # Rows stay orthogonal, each with `weight` non-zero entries.
                self._base[case] = block if weight == n else t.kron(t.eye(n // weight, dtype=dtype, device=self._device), block)
                self._counts[case] = [0, 0]
            except Exception:  # a number format this card or build cannot hold
                self._skip(case)
        if not self._cases:
            raise CheckUnavailable("none of the answer check's number formats is usable on this card")
        _sync(t, self._device)
        self._t0 = self._clock()

    def _skip(self, case) -> None:
        self._cases.remove(case)
        self._base.pop(case, None)
        if case[0] not in self._skipped:
            self._skipped.append(case[0])

    def _shuffle(self, n: int) -> list[int]:
        order = list(range(n))
        self._rng.shuffle(order)
        return order

    def _product(self, case) -> int:
        """One product for `case`; the number of entries that differ from the known result."""
        t, dev = self._torch, self._device
        name, n, weight = case
        base = self._base[case]
        dtype = base.dtype
        pa, pb = self._shuffle(n), self._shuffle(n)
        sa = self._rng.choices(_SIGNED_SCALES, k=n)
        sb = self._rng.choices(_SIGNED_SCALES, k=n)
        # Row i of a is row pa[i] of the base; it meets the same base row in b at column k[i].
        where = [0] * n
        for column, row in enumerate(pb):
            where[row] = column
        k = [where[row] for row in pa]
        value = [float(weight) * sa[i] * sb[k[i]] for i in range(n)]

        def on_card(values, kind):
            return t.tensor(values, dtype=kind).to(dev)

        def scale(values):  # built as float32 (exact for powers of two), converted on the card
            return on_card(values, t.float32).to(dtype)

        a = base[on_card(pa, t.int64)] * scale(sa)[:, None]
        b = base[on_card(pb, t.int64)] * scale(sb)[:, None]
        c = a @ b.T
        expected = t.zeros((n, n), dtype=dtype, device=dev)
        expected[t.arange(n, device=dev), on_card(k, t.int64)] = scale(value)
        return int(t.count_nonzero(c != expected))

    def step(self) -> bool:
        if self._t0 is None or not self._cases:
            return False
        now = self._clock()
        if self._rounds >= 1 and now - self._t0 >= self._seconds:
            self._elapsed = now - self._t0
            return False
        end = now + self._step_s
        while self._cases:
            case = self._cases[self._i % len(self._cases)]
            try:
                wrong = self._product(case)
            except RuntimeError as e:
                if self._counts[case][0]:
                    # This format multiplied before, so this is the card or the driver failing, not
                    # an unsupported format. The run records the check as stopped early and keeps
                    # every product counted so far.
                    raise
                self._skip(case)  # never multiplied here: unsupported, which is not a wrong answer
                if not self._cases:
                    # Nothing left to compare. Say the check did not run, never that nothing was wrong.
                    raise CheckUnavailable(f"every number format failed to multiply ({e})") from e
                continue
            self._counts[case][0] += 1
            if wrong:
                self._counts[case][1] += 1
                self._wrong_elements += wrong
            self._i += 1
            if self._i % len(self._cases) == 0:
                self._rounds += 1  # every format has now been through at least this many products
            if self._clock() >= end:
                break
        return bool(self._cases)

    def stop(self) -> None:
        try:
            self._base = {}
            if self._torch is not None:
                _release(self._torch, self._device)
        except Exception as e:
            warn_cleanup_failed(e)

    def result(self) -> dict:
        elapsed = self._elapsed
        if elapsed is None and self._t0 is not None:
            elapsed = self._clock() - self._t0
        return {
            "products": sum(c[0] for c in self._counts.values()),
            "wrong_products": sum(c[1] for c in self._counts.values()),
            "wrong_elements": self._wrong_elements,
            "seconds": elapsed,
            "cases": [
                {"dtype": name, "n": n, "products": self._counts[(name, n, w)][0],
                 "wrong_products": self._counts[(name, n, w)][1]}
                for name, n, w in self._counts
                if self._counts[(name, n, w)][0]
            ],
            "skipped": list(self._skipped),
        }


# ---- card-to-card transfer -----------------------------------------------------------------

TRANSFER_BYTES = 256 * MIB


class TorchTransfer:
    """Copy pseudo-random data from one card to another and back, and compare.

    PyTorch and the driver choose the path the copy takes; this class does not know whether
    it was NVLink. The caller reads the cards' NVLink data counters around the copies for that.
    A short copy says nothing about how fast the link is or how it behaves under long traffic."""

    def __init__(self, nbytes: int = TRANSFER_BYTES, seed: int = 0x6770636B) -> None:
        self._words = max(1, nbytes // WORD)
        self._seed = seed

    def copy(self, src: int, dst: int) -> dict:
        t = _import_torch()
        a_dev, b_dev = f"cuda:{src}", f"cuda:{dst}"
        t0 = time.perf_counter()
        g = t.Generator(device=a_dev)
        g.manual_seed(self._seed + 64 * src + dst)
        sent = there = back = None
        try:
            sent = t.empty(self._words, dtype=t.int64, device=a_dev).random_(generator=g)
            there = sent.to(b_dev)
            back = there.to(a_dev)
            wrong = int(t.count_nonzero(sent != back))
            _sync(t, a_dev)
        finally:
            del sent, there, back
            try:
                t.cuda.empty_cache()
            except Exception as e:
                warn_cleanup_failed(e)
        return {"bytes": self._words * WORD, "mismatched_words": wrong, "seconds": time.perf_counter() - t0}


# ---- self-test -----------------------------------------------------------------------------

SELF_TEST_CASES = (("float32", 256, 256), ("float16", 256, 128), ("bfloat16", 256, 64), ("float64", 128, 128))


def self_test(device: str | None = None, seconds: float = 5.0) -> int:
    """Run the memory test and the answer check against the installed PyTorch and print what
    they report. `python -m gpucheck.deepchecks` runs it: on a CPU it uses small sizes and only
    shows that PyTorch accepts every call and that sound hardware gets no mismatch; on a GPU
    (`--device cuda:0`) it runs both checks at full size on that card, with no load before them.

    Returns 0 when both checks ran, used every number format and found nothing wrong."""
    import json

    torch = _import_torch()
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    on_gpu = device.startswith("cuda")
    checks = [
        TorchAnswerCheck(device=device, seconds=seconds, cases=ANSWER_CASES if on_gpu else SELF_TEST_CASES),
        TorchMemtest(device=device) if on_gpu else TorchMemtest(
            device=device, budget_bytes=64 * MIB, chunk_bytes=16 * MIB, min_bytes=MIB),
    ]
    ok = True
    for check in checks:
        try:
            check.start()
            while check.step():
                pass
            out = {"status": "ran", **check.result()}
        except Exception as e:
            out = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}
        finally:
            check.stop()
        print(json.dumps({check.name: out, "device": device, "torch": torch.__version__}, indent=2))
        ok = ok and out["status"] == "ran" and not out.get("mismatched_words") \
            and not out.get("wrong_products") and not out.get("skipped")
    print("self-test: every call was accepted and nothing differed" if ok else "self-test: see above")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    _p = argparse.ArgumentParser(prog="python -m gpucheck.deepchecks", description=self_test.__doc__)
    _p.add_argument("--device", default=None, help='"cpu" or "cuda:N" (default: cuda:0 when available, else cpu)')
    _p.add_argument("--seconds", type=float, default=5.0, help="how long the answer check runs")
    _a = _p.parse_args()
    sys.exit(self_test(_a.device, _a.seconds))
