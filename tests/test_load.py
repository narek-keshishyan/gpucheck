import argparse

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


def test_load_unavailable_without_torch_cuda():
    with pytest.raises(LoadUnavailable):
        TorchGemmLoad()


def test_make_load_warns_and_falls_back(monkeypatch, capsys):
    def boom():
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
