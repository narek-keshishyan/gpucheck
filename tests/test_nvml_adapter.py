"""NvmlAdapter against fake pynvml modules injected via sys.modules.

Three stubs: modern names (nvmlClocksEventReason* / ...ClocksEventReasons),
legacy names (nvmlClocksThrottleReason* / ...ClocksThrottleReasons), and one
missing every optional symbol. None of them may raise out of inventory()/sample().
"""

import sys
import types

import pytest

from gpucheck.adapter import NvmlAdapter
from gpucheck.diagnose import diagnose
from gpucheck.metrics import derive
from gpucheck.preflight import Preflight, run_preflight
from gpucheck.run import RawRun, RunConfig
from gpucheck.specs import lookup

SW_POWER_CAP, SW_THERMAL, HW_THERMAL, HW_POWER_BRAKE = 0x4, 0x20, 0x40, 0x80


def _base(mask=0, remap=(0, 0, False, False)):
    nv = types.SimpleNamespace()
    nv.nvmlInit = lambda: None
    nv.nvmlDeviceGetHandleByIndex = lambda i: "h"
    nv.nvmlDeviceGetUUID = lambda h: b"GPU-1234"
    nv.nvmlDeviceGetName = lambda h: b"NVIDIA H100 80GB HBM3"
    nv.nvmlSystemGetDriverVersion = lambda: "550.54.15"
    nv.nvmlSystemGetCudaDriverVersion = lambda: 12040
    nv.nvmlDeviceGetEccMode = lambda h: (1, 1)
    nv.nvmlDeviceGetPowerManagementLimit = lambda h: 700000
    nv.nvmlDeviceGetPowerManagementDefaultLimit = lambda h: 700000
    nv.nvmlDeviceGetMigMode = lambda h: (0, 0)
    nv.nvmlDeviceGetRetiredPages = lambda h, cause: []
    nv.nvmlDeviceGetRemappedRows = lambda h: remap
    nv.nvmlDeviceGetTemperature = lambda h, sensor: 61
    nv.nvmlDeviceGetPowerUsage = lambda h: 650000
    nv.nvmlDeviceGetClockInfo = lambda h, clk: 1980
    nv.nvmlDeviceGetFanSpeed = lambda h: 40
    nv.nvmlDeviceGetTotalEccErrors = lambda h, kind, counter: 0
    nv.nvmlDeviceGetComputeRunningProcesses = lambda h: []
    nv.NVML_TEMPERATURE_GPU = 0
    nv.NVML_CLOCK_SM = 1
    nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    nv.NVML_VOLATILE_ECC = 0
    nv.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS = 0
    return nv


def modern(mask=0, **kw):
    nv = _base(**kw)
    nv.nvmlClocksEventReasonSwPowerCap = SW_POWER_CAP
    nv.nvmlClocksEventReasonSwThermalSlowdown = SW_THERMAL
    nv.nvmlClocksEventReasonHwThermalSlowdown = HW_THERMAL
    nv.nvmlClocksEventReasonHwPowerBrakeSlowdown = HW_POWER_BRAKE
    nv.nvmlDeviceGetCurrentClocksEventReasons = lambda h: mask
    return nv


def legacy(mask=0, **kw):
    nv = _base(**kw)
    nv.nvmlClocksThrottleReasonSwPowerCap = SW_POWER_CAP
    nv.nvmlClocksThrottleReasonSwThermalSlowdown = SW_THERMAL
    nv.nvmlClocksThrottleReasonHwThermalSlowdown = HW_THERMAL
    nv.nvmlClocksThrottleReasonHwPowerBrakeSlowdown = HW_POWER_BRAKE
    nv.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: mask
    return nv


def missing():
    nv = _base()
    for name in list(vars(nv)):
        if name.startswith("NVML_"):
            delattr(nv, name)
    del nv.nvmlDeviceGetRemappedRows
    del nv.nvmlDeviceGetFanSpeed
    return nv


def adapter(monkeypatch, nv):
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    return NvmlAdapter()


@pytest.mark.parametrize("make", [modern, legacy])
def test_inventory_and_sample_both_naming_families(monkeypatch, make):
    a = adapter(monkeypatch, make(mask=HW_THERMAL | SW_POWER_CAP))
    inv = a.inventory()
    assert inv.model_name == "NVIDIA H100 80GB HBM3"
    assert inv.power_limit_w == 700.0 and inv.cuda == "12.4" and inv.ecc_enabled is True
    assert inv.remapped_rows == 0 and inv.retired_pages == 0
    s = a.sample()
    assert s.temp_c == 61.0 and s.power_w == 650.0 and s.sm_clock_mhz == 1980.0
    assert s.throttle_reasons == frozenset({"thermal", "power"})
    assert a.busy() is False


@pytest.mark.parametrize("make", [modern, legacy])
def test_no_throttle_mask_is_empty_set(monkeypatch, make):
    assert adapter(monkeypatch, make(mask=0)).sample().throttle_reasons == frozenset()


def test_missing_symbols_never_raise_and_throttle_is_unknown(monkeypatch):
    a = adapter(monkeypatch, missing())
    inv = a.inventory()
    assert inv.remapped_rows is None and inv.row_remap_pending is None
    s = a.sample()
    assert s.throttle_reasons is None  # unknown, not "no throttling"
    assert s.fan_pct is None
    pf = run_preflight(a)
    assert "throttle_reasons" in pf.missing_telemetry


def test_throttle_getter_raising_is_unknown(monkeypatch):
    nv = modern()

    def boom(h):
        raise RuntimeError("not supported")

    nv.nvmlDeviceGetCurrentClocksEventReasons = boom
    nv.nvmlDeviceGetCurrentClocksThrottleReasons = boom
    assert adapter(monkeypatch, nv).sample().throttle_reasons is None


def test_modern_getter_failing_falls_back_to_legacy_getter(monkeypatch):
    nv = legacy(mask=SW_THERMAL)

    def boom(h):
        raise RuntimeError("not supported")

    nv.nvmlDeviceGetCurrentClocksEventReasons = boom
    assert adapter(monkeypatch, nv).sample().throttle_reasons == frozenset({"thermal"})


def test_preflight_survives_sample_raising(monkeypatch):
    a = adapter(monkeypatch, modern())

    def boom():
        raise RuntimeError("nvml hiccup")

    monkeypatch.setattr(a, "sample", boom)
    pf = run_preflight(a)
    assert pf.mode == "full"
    assert "temp_c" in pf.missing_telemetry


def test_remapped_rows_counts_correctable_and_uncorrectable(monkeypatch):
    a = adapter(monkeypatch, modern(remap=(1, 2, True, False)))
    inv = a.inventory()
    assert inv.remapped_rows == 3
    assert inv.row_remap_pending is True and inv.row_remap_failure is False


def test_uncorrectable_remaps_fire_memory_wear(monkeypatch):
    a = adapter(monkeypatch, modern(remap=(0, 2, False, False)))
    inv = a.inventory()
    spec = lookup(inv.model_name)
    load = [a.sample() for _ in range(5)]
    run = RawRun(inv, Preflight("full"), [a.sample()], load, RunConfig(), None)
    codes = [f.code for f in diagnose(derive(run, spec), run.preflight, inv, spec)]
    assert "MEMORY_WEAR" in codes


def test_busy_none_when_process_query_fails(monkeypatch):
    nv = modern()

    def boom(h):
        raise RuntimeError("container")

    nv.nvmlDeviceGetComputeRunningProcesses = boom
    assert adapter(monkeypatch, nv).busy() is None
