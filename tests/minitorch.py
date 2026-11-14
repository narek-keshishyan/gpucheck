"""A tiny list-backed stand-in for the parts of PyTorch the extra checks use.

It exists because the development machine has no PyTorch. It implements each operation the
way PyTorch documents it, on Python numbers, so it checks the checks' arithmetic (patterns,
expected products, mismatch counting) but not that the real PyTorch accepts the calls: that is
what `python -m gpucheck.deepchecks` is for, on a machine that has PyTorch.

Only 1-D and 2-D tensors. A test can corrupt results through `faults`.
"""

from __future__ import annotations

import random as _random


class OutOfMemoryError(RuntimeError):
    pass


int64, float16, bfloat16, float32, float64 = "int64", "float16", "bfloat16", "float32", "float64"


class Tensor:
    def __init__(self, data: list, shape: tuple, dtype: str = float32, device: str = "cpu") -> None:
        self.data, self.shape, self.dtype, self.device = data, tuple(shape), dtype, device

    # -- shape helpers
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def _like(self, data, shape=None, dtype=None):
        return Tensor(data, self.shape if shape is None else shape, dtype or self.dtype, self.device)

    def _rows(self) -> list[list]:
        r, c = self.shape
        return [self.data[i * c:(i + 1) * c] for i in range(r)]

    @property
    def T(self):
        r, c = self.shape
        return self._like([self.data[i * c + j] for j in range(c) for i in range(r)], (c, r))

    # -- conversion
    def to(self, *args, device=None, dtype=None):
        for a in args:
            if a in (int64, float16, bfloat16, float32, float64):
                dtype = a
            else:
                device = a
        return Tensor(list(self.data), self.shape, dtype or self.dtype, str(device or self.device))

    def __int__(self) -> int:
        assert self.numel() == 1
        return int(self.data[0])

    # -- in-place fills
    def fill_(self, value):
        self.data = [value] * self.numel()
        return self

    def random_(self, generator=None):
        rng = generator.rng if generator is not None else _random
        self.data = [rng.getrandbits(63) for _ in range(self.numel())]
        return self

    # -- indexing
    def __getitem__(self, key):
        if isinstance(key, slice):
            assert len(self.shape) == 1
            part = self.data[key]
            return self._like(part, (len(part),))
        if isinstance(key, Tensor):
            if len(self.shape) == 1:
                return self._like([self.data[i] for i in key.data], (key.numel(),))
            rows = self._rows()
            return self._like([v for i in key.data for v in rows[i]], (key.numel(), self.shape[1]))
        if isinstance(key, tuple) and key == (slice(None), None):
            assert len(self.shape) == 1
            return self._like(list(self.data), (self.shape[0], 1))
        raise TypeError(f"unsupported index {key!r}")

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            start, stop, _ = key.indices(self.numel())
            assert value.numel() == stop - start
            self.data[start:stop] = list(value.data)
        elif isinstance(key, Tensor):
            for i, v in zip(key.data, value.data):
                self.data[i] = v
        elif isinstance(key, tuple) and all(isinstance(k, Tensor) for k in key):
            cols = self.shape[1]
            for i, j, v in zip(key[0].data, key[1].data, value.data):
                self.data[i * cols + j] = v
        else:
            raise TypeError(f"unsupported index {key!r}")

    # -- arithmetic
    def _zip(self, other, fn, dtype=None):
        if not isinstance(other, Tensor):
            return self._like([fn(a, other) for a in self.data], dtype=dtype)
        if self.shape == other.shape:
            return self._like([fn(a, b) for a, b in zip(self.data, other.data)], dtype=dtype)
        if len(self.shape) == 2 and other.shape == (self.shape[0], 1):  # a column broadcast over rows
            cols = self.shape[1]
            return self._like([fn(a, other.data[i // cols]) for i, a in enumerate(self.data)], dtype=dtype)
        raise ValueError(f"shapes {self.shape} and {other.shape} do not broadcast")

    def __mul__(self, other):
        return self._zip(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self._zip(other, lambda a, b: a - b)

    def __neg__(self):
        return self._like([-a for a in self.data])

    def __ne__(self, other):  # noqa: D105 - elementwise, like torch
        return self._zip(other, lambda a, b: a != b, dtype="bool")

    def __matmul__(self, other):
        (r, k), (k2, c) = self.shape, other.shape
        assert k == k2
        a, b = self._rows(), other.T._rows()
        out = [sum(x * y for x, y in zip(row, col)) for row in a for col in b]
        result = self._like(out, (r, c))
        for fault in faults:
            fault(result)
        return result


faults: list = []  # callables applied to every matmul result; tests use them to corrupt answers


class Generator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.rng = _random.Random(0)

    def manual_seed(self, seed: int):
        self.rng = _random.Random(seed)
        return self


def _shape(shape) -> tuple:
    return (shape,) if isinstance(shape, int) else tuple(shape)


def _count(shape) -> int:
    n = 1
    for d in _shape(shape):
        n *= d
    return n


class _Cuda:
    """Device memory for the fake "cuda:N" devices: allocations count against `free`."""

    def __init__(self) -> None:
        self.free, self.total = 8 << 30, 8 << 30
        self.synchronized = 0
        self.emptied = 0
        self.fail_alloc_after: int | None = None  # allocations that succeed before out-of-memory
        self._allocs = 0

    def mem_get_info(self, device):
        return self.free, self.total

    def synchronize(self, device=None) -> None:
        self.synchronized += 1

    def empty_cache(self) -> None:
        self.emptied += 1

    def alloc(self) -> None:
        if self.fail_alloc_after is not None and self._allocs >= self.fail_alloc_after:
            raise OutOfMemoryError("CUDA out of memory")
        self._allocs += 1


cuda = _Cuda()


def reset() -> None:
    global cuda
    cuda = _Cuda()
    faults.clear()


def empty(shape, dtype=float32, device="cpu") -> Tensor:
    if str(device).startswith("cuda"):
        cuda.alloc()
    return Tensor([0xDEAD] * _count(shape), _shape(shape), dtype, str(device))  # never what a pattern expects


def zeros(shape, dtype=float32, device="cpu") -> Tensor:
    return Tensor([0] * _count(shape), _shape(shape), dtype, str(device))


def ones(shape, dtype=float32, device="cpu") -> Tensor:
    return Tensor([1] * _count(shape), _shape(shape), dtype, str(device))


def eye(n, dtype=float32, device="cpu") -> Tensor:
    return Tensor([1 if i == j else 0 for i in range(n) for j in range(n)], (n, n), dtype, str(device))


def tensor(values, dtype=float32, device="cpu") -> Tensor:
    return Tensor(list(values), (len(values),), dtype, str(device))


def arange(start, end=None, dtype=int64, device="cpu") -> Tensor:
    if end is None:
        start, end = 0, start
    return Tensor(list(range(start, end)), (end - start,), dtype, str(device))


def randperm(n, generator=None) -> Tensor:
    order = list(range(n))
    generator.rng.shuffle(order)
    return Tensor(order, (n,), int64)


def randint(low, high, size, generator=None) -> Tensor:
    return Tensor([generator.rng.randrange(low, high) for _ in range(_count(size))], _shape(size), int64)


def cat(tensors, dim=0) -> Tensor:
    first = tensors[0]
    if dim == 0:
        return first._like([v for t in tensors for v in t.data], (sum(t.shape[0] for t in tensors), first.shape[1]))
    rows = [sum((t._rows()[i] for t in tensors), []) for i in range(first.shape[0])]
    return first._like([v for row in rows for v in row], (first.shape[0], sum(t.shape[1] for t in tensors)))


def kron(a: Tensor, b: Tensor) -> Tensor:
    (ar, ac), (br, bc) = a.shape, b.shape
    A, B = a._rows(), b._rows()
    out = [A[i // br][j // bc] * B[i % br][j % bc] for i in range(ar * br) for j in range(ac * bc)]
    return b._like(out, (ar * br, ac * bc))


def count_nonzero(t: Tensor) -> Tensor:
    return Tensor([sum(1 for v in t.data if v)], (1,), int64, t.device)
