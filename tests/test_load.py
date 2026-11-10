import argparse
import sys
import types

import pytest

from gpucheck import cli
from gpucheck.load import LoadUnavailable, TorchGemmLoad, pick_gemm_size, tflops
from gpucheck.run import NullLoad


@pytest.mark.parametrize(
    "free,expected",
    [
        (8 * 1024**3, 16384),  # n<=29308 -> clamped to max
        (1 * 1024**3, 10362),
        (64 * 1024**2, 2590),
        (41943040, 2048),  # raw exactly 2048
        (2684354560, 16384),  # raw exactly 16384
        (2684682250, 16384),  # raw 16385 -> clamped
        (10_000_000, 2048),  # raw 1000 -> clamped up
        (0, 2048),
        (1024, 2048),
        (10**15, 16384),
    ],
)
def test_pick_gemm_size(free, expected):
    assert pick_gemm_size(free) == expected


def test_pick_gemm_size_fits_budget_when_unclamped():
    free = 1024**3
    n = pick_gemm_size(free)
    assert 3 * 2 * n * n <= 0.6 * free < 3 * 2 * (n + 1) ** 2


def test_tflops_math():
    # 2 * 1000^3 * 10 / 2s = 1e10 FLOP/s = 0.01 TFLOPs
    assert tflops(1000, 10, 2.0) == pytest.approx(0.01)
    assert tflops(8192, 100, 0.8) == pytest.approx(2 * 8192**3 * 100 / 0.8 / 1e12)


def _stub_torch(monkeypatch, available=True, mem_info=None):
    cuda = types.SimpleNamespace(is_available=lambda: available)
    if mem_info is not None:
        cuda.mem_get_info = mem_info
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_load_unavailable_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(LoadUnavailable):
        TorchGemmLoad()


def test_load_unavailable_when_cuda_unavailable(monkeypatch):
    _stub_torch(monkeypatch, available=False)
    with pytest.raises(LoadUnavailable):
        TorchGemmLoad()


def test_load_unavailable_when_cuda_broken(monkeypatch):
    def boom(dev):
        raise RuntimeError("CUDA error")

    _stub_torch(monkeypatch, mem_info=boom)
    g = TorchGemmLoad()  # constructing must not probe CUDA
    with pytest.raises(LoadUnavailable):
        g.start()


def test_init_touches_no_cuda_memory_api(monkeypatch):
    # Regression: sizing in __init__ created a CUDA context, so preflight
    # (which runs after the load is constructed) saw gpucheck's own process
    # and refused every real run as "GPU busy".
    calls = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda dev: calls.append("mem") or (1024**3, 1024**3),
        synchronize=lambda dev: None,
        empty_cache=lambda: None,
    )
    fake = types.SimpleNamespace(
        cuda=cuda,
        bfloat16="bf16",
        randn=lambda *a, **k: object(),
        empty=lambda *a, **k: object(),
        matmul=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    g = TorchGemmLoad()
    assert calls == []
    g.start()
    assert calls == ["mem"]


def test_make_load_warns_and_falls_back(monkeypatch, capsys):
    def boom(*a, **k):
        raise LoadUnavailable("nope")

    monkeypatch.setattr(cli, "TorchGemmLoad", boom)
    load = cli._make_load(argparse.Namespace(fake=None))
    assert isinstance(load, NullLoad)
    c = capsys.readouterr()
    assert "no torch/CUDA" in c.out + c.err
    assert len((c.out + c.err).strip().splitlines()) == 1


def test_make_load_fake_is_null_without_warning(monkeypatch, capsys):
    def boom():
        raise AssertionError("must not be constructed")

    monkeypatch.setattr(cli, "TorchGemmLoad", boom)
    assert isinstance(cli._make_load(argparse.Namespace(fake="healthy")), NullLoad)
    c = capsys.readouterr()
    assert (c.out + c.err) == ""


def test_fake_quick_run_still_ok(tmp_path, capsys):
    rc = cli.main(["--fake", "healthy", "--offline", "--quick", "--json", str(tmp_path / "r.json")])
    assert rc == 0
    assert "no torch/CUDA" not in capsys.readouterr().err


def test_init_pins_pci_bus_device_order_before_torch_import(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    _stub_torch(monkeypatch, available=False)
    with pytest.raises(LoadUnavailable):
        TorchGemmLoad()
    import os

    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_tick_synchronizes_after_every_matmul(monkeypatch):
    events = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda dev: (1024**3, 1024**3),
        synchronize=lambda dev: events.append("sync"),
        empty_cache=lambda: None,
    )
    fake = types.SimpleNamespace(
        cuda=cuda,
        bfloat16="bf16",
        randn=lambda *a, **k: object(),
        empty=lambda *a, **k: object(),
        matmul=lambda *a, **k: events.append("mm"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    import gpucheck.load as load_mod

    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.9, 1.0])
    monkeypatch.setattr(load_mod, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    g = TorchGemmLoad()
    g.start()
    events.clear()
    assert g.tick() is not None
    # one sync before timing, then every matmul is followed by its own sync
    assert events[0] == "sync"
    body = events[1:]
    assert body.count("mm") == 3
    assert all(body[i + 1] == "sync" for i, e in enumerate(body) if e == "mm")


def test_load_targets_selected_device(monkeypatch):
    seen = []

    def mem(dev):
        seen.append(dev)
        return (1024**3, 1024**3)

    cuda = types.SimpleNamespace(is_available=lambda: True, mem_get_info=mem,
                                 synchronize=lambda dev: seen.append(dev), empty_cache=lambda: None)
    fake = types.SimpleNamespace(
        cuda=cuda, bfloat16="bf16",
        randn=lambda *a, **k: seen.append(k["device"]) or object(),
        empty=lambda *a, **k: seen.append(k["device"]) or object(),
        matmul=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    g = TorchGemmLoad(device_index=2)
    g.start()
    assert seen and set(seen) == {"cuda:2"}


def test_make_load_forwards_device_index(monkeypatch):
    made = []
    monkeypatch.setattr(cli, "TorchGemmLoad", lambda *a, **k: made.append((a, k)) or object())
    cli._make_load(argparse.Namespace(fake=None), device_index=2)
    assert made == [((), {"device_index": 2})]
    made.clear()
    w = {"gemm_n_offsets": [0], "order_seed": 1}
    cli._make_load(argparse.Namespace(fake=None), challenge={"workload": w}, device_index=2)
    assert made == [((w,), {"device_index": 2})]


def test_tick_stays_on_selected_device(monkeypatch):
    seen = []

    class T:
        def __getitem__(self, sl):
            return self

        def view(self, n, m):
            return self

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda dev: seen.append(dev) or (1024**3, 1024**3),
        synchronize=lambda dev: seen.append(dev),
        empty_cache=lambda: None,
    )
    fake = types.SimpleNamespace(
        cuda=cuda, bfloat16="bf16",
        randn=lambda *a, **k: seen.append(k["device"]) or T(),
        empty=lambda *a, **k: seen.append(k["device"]) or T(),
        matmul=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    import gpucheck.load as load_mod

    clock = [0.0]

    def pc():
        clock[0] += 0.5
        return clock[0]

    monkeypatch.setattr(load_mod.time, "perf_counter", pc)
    g = TorchGemmLoad(device_index=2)
    g.start()
    seen.clear()
    assert g.tick() is not None
    assert seen and set(seen) == {"cuda:2"}
