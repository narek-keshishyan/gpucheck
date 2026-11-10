"""Torch bf16 GEMM load generator. torch is imported lazily (optional extra)."""

from __future__ import annotations

import math
import os
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


def apply_workload(base_n: int, offsets: list[int]) -> list[int]:
    """Resolve prescribed gemm_n offsets against an auto-sized base N.

    The base is first clamped into [MIN_N - min(offsets), MAX_N - max(offsets)]
    so that base + offset never needs clamping: the server verifies pairwise
    differences between the sizes, which clamping an individual value would break.
    """
    lo = MIN_N - min(offsets)
    hi = MAX_N - max(offsets)
    base = max(lo, min(hi, base_n))
    return [base + o for o in offsets]


def tflops(n: int, iters: int, elapsed_s: float) -> float:
    return 2 * n**3 * iters / elapsed_s / 1e12


class TorchGemmLoad:
    def __init__(self, workload: dict | None = None, device_index: int = 0) -> None:
        # Make torch's cuda:N the same physical card NVML index N reads; CUDA's
        # default "fastest first" order can differ from NVML's PCI bus order.
        # Must be set before torch initialises CUDA, so before the import.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        try:
            import torch
        except ImportError as e:
            raise LoadUnavailable("torch is not installed") from e
        try:
            if not torch.cuda.is_available():
                raise LoadUnavailable("CUDA is not available")
        except RuntimeError as e:
            raise LoadUnavailable(f"CUDA probe failed: {e}") from e
        self._torch = torch
        self._device = f"cuda:{device_index}"
        self._workload = dict(workload) if workload is not None else None
        self._ns: list[int] = []
        self._tick_i = 0
        self.resolved_workload: dict | None = None
        self._a = self._b = self._c = None

    def start(self) -> None:
        t = self._torch
        # Memory sizing creates this process's CUDA context, so it must not
        # happen in __init__: preflight reads the GPU's process list after the
        # load is constructed, and a context of our own makes the card look busy.
        try:
            free, _total = t.cuda.mem_get_info(self._device)
        except RuntimeError as e:
            raise LoadUnavailable(f"CUDA probe failed: {e}") from e
        n = pick_gemm_size(free)
        self._ns = [n]
        if self._workload is not None:
            self._ns = apply_workload(n, list(self._workload["gemm_n_offsets"]))
            self.resolved_workload = {
                "gemm_n": list(self._ns),
                "order_seed": self._workload["order_seed"],
            }
        # One flat buffer at the largest size; each tick views the first n*n
        # elements as a contiguous n x n matrix (no per-size reallocation).
        # Without a workload this is the plain single-size n x n path.
        m = max(self._ns)
        shape = (m * m,) if self.resolved_workload else (m, m)
        self._a = t.randn(*shape, device=self._device, dtype=t.bfloat16)
        self._b = t.randn(*shape, device=self._device, dtype=t.bfloat16)
        self._c = t.empty(*shape, device=self._device, dtype=t.bfloat16)
        a, b, c = self._views(self._ns[0])
        t.matmul(a, b, out=c)  # warmup
        t.cuda.synchronize(self._device)

    def _views(self, n: int):
        if not self.resolved_workload:
            return self._a, self._b, self._c
        k = n * n
        return (
            self._a[:k].view(n, n),
            self._b[:k].view(n, n),
            self._c[:k].view(n, n),
        )

    def tick(self) -> float | None:
        t = self._torch
        n = self._ns[self._tick_i % len(self._ns)]  # cycle prescribed sizes per tick
        self._tick_i += 1
        a, b, c = self._views(n)
        iters = 0
        t.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < TICK_S:
            t.matmul(a, b, out=c)
            # Wait for each kernel: matmul only enqueues, so without this the loop
            # would queue thousands of kernels in 0.8 s and the tick (and the
            # watchdog sampling between ticks) would stall for minutes.
            t.cuda.synchronize(self._device)
            iters += 1
        t.cuda.synchronize(self._device)
        elapsed = time.perf_counter() - t0
        return tflops(n, iters, elapsed) if iters else None

    def stop(self) -> None:
        self._a = self._b = self._c = None
        self._torch.cuda.empty_cache()
