import dataclasses

import pytest

from gpucheck.specs import GpuSpec, lookup


def test_lookup_h100_sxm():
    s = lookup("NVIDIA H100 80GB HBM3")
    assert s.key == "h100-sxm" and s.tdp_w == 700.0


def test_lookup_unknown_returns_none():
    assert lookup("NVIDIA B300 Mystery") is None


def test_spec_is_frozen_dataclass():
    s = lookup("NVIDIA H100 80GB HBM3")
    assert isinstance(s, GpuSpec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.tdp_w = 1.0


def test_lookup_h100_sxm_full_values():
    s = lookup("NVIDIA H100 80GB HBM3")
    assert s.peak_bf16_tflops == 989.0
    assert s.max_temp_c == 83.0
    assert s.boost_clock_mhz == 1980.0


@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA H100 PCIe",
        "NVIDIA H100 80GB PCIe",
        "nvidia h100pcie",
        "  NVIDIA   H100   PCIE  ",
    ],
)
def test_lookup_h100_pcie(name):
    s = lookup(name)
    assert s.key == "h100-pcie"
    assert s.tdp_w == 350.0 and s.peak_bf16_tflops == 756.0


@pytest.mark.parametrize(
    "name",
    ["NVIDIA A100 80GB PCIe", "NVIDIA A100-PCIE-80GB", "a100 pcie"],
)
def test_lookup_a100_pcie(name):
    assert lookup(name).key == "a100-80gb-pcie"


@pytest.mark.parametrize(
    "name",
    ["NVIDIA A100 80GB", "NVIDIA A100-SXM4-80GB", "a100 80gb sxm"],
)
def test_lookup_a100_sxm(name):
    s = lookup(name)
    assert s.key == "a100-80gb-sxm"
    assert s.tdp_w == 400.0


@pytest.mark.parametrize("name", ["NVIDIA H100 NVL", "nvidia h100nvl"])
def test_lookup_h100_nvl_is_none(name):
    assert lookup(name) is None


def test_lookup_a100_without_80gb_is_none():
    assert lookup("NVIDIA A100 40GB") is None


@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX 4090", "rtx 4090", "RTX4090"])
def test_lookup_rtx_4090(name):
    s = lookup(name)
    assert s.key == "rtx-4090"
    assert s.tdp_w == 450.0


@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX 3090", "rtx 3090"])
def test_lookup_rtx_3090(name):
    assert lookup(name).key == "rtx-3090"


def test_lookup_3090_ti_is_none():
    assert lookup("NVIDIA GeForce RTX 3090 Ti") is None
    assert lookup("NVIDIA GeForce RTX 3090Ti") is None


def test_lookup_4090_ti_is_none():
    assert lookup("NVIDIA GeForce RTX 4090 Ti") is None


def test_lookup_is_case_insensitive():
    assert lookup("nvidia h100 80gb hbm3").key == "h100-sxm"


def test_lookup_empty_string_is_none():
    assert lookup("") is None
