"""Torch bf16 GEMM load generator. torch is imported lazily (optional extra)."""

from __future__ import annotations

import math
import time

MIN_N = 2048
MAX_N = 16384
MEM_FRACTION = 0.6
BYTES_PER_ELEM = 2  # bf16
TICK_S = 0.8


class LoadUnavailable(Exception):
    """torch or a CUDA device is not available."""


def pick_gemm_size(free_mem_bytes: int) -> int:
    """Largest n with 3 bf16 n*n matrices within 60% of free memory, clamped."""
    budget = int(MEM_FRACTION * max(free_mem_bytes, 0))
    n = math.isqrt(budget // (3 * BYTES_PER_ELEM))
    return max(MIN_N, min(MAX_N, n))


def tflops(n: int, iters: int, elapsed_s: float) -> float:
    return 2 * n**3 * iters / elapsed_s / 1e12


class TorchGemmLoad:
    def __init__(self) -> None:
        try:
            import torch
        except ImportError as e:
            raise LoadUnavailable("torch is not installed") from e
        if not torch.cuda.is_available():
            raise LoadUnavailable("CUDA is not available")
        self._torch = torch
        free, _total = torch.cuda.mem_get_info()
        self._n = pick_gemm_size(free)
        self._a = self._b = self._c = None

    def start(self) -> None:
        t = self._torch
        n = self._n
        self._a = t.randn(n, n, device="cuda", dtype=t.bfloat16)
        self._b = t.randn(n, n, device="cuda", dtype=t.bfloat16)
        self._c = t.empty(n, n, device="cuda", dtype=t.bfloat16)
        t.matmul(self._a, self._b, out=self._c)  # warmup
        t.cuda.synchronize()

    def tick(self) -> float | None:
        t = self._torch
        iters = 0
        t.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < TICK_S:
            t.matmul(self._a, self._b, out=self._c)
            iters += 1
        t.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return tflops(self._n, iters, elapsed) if iters else None

    def stop(self) -> None:
        self._a = self._b = self._c = None
        self._torch.cuda.empty_cache()
