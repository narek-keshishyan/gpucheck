"""NvmlAdapter against fake pynvml modules injected via sys.modules.

Three stubs: modern names (nvmlClocksEventReason* / ...ClocksEventReasons),
legacy names (nvmlClocksThrottleReason* / ...ClocksThrottleReasons), and one
missing every optional symbol. None of them may raise out of inventory()/sample().
"""

import sys
import types

import pytest

from gpucheck.adapter import NvmlAdapter, clean_serial
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


def test_uncorrectable_remaps_fire_memory_errors(monkeypatch):
    a = adapter(monkeypatch, modern(remap=(0, 2, False, False)))
    inv = a.inventory()
    spec = lookup(inv.model_name)
    load = [a.sample() for _ in range(5)]
    run = RawRun(inv, Preflight("full"), [a.sample()], load, RunConfig(), None)
    codes = [f.code for f in diagnose(derive(run, spec), run.preflight, inv, spec)]
    assert "MEMORY_ERRORS" in codes


def test_busy_none_when_process_query_fails(monkeypatch):
    nv = modern()

    def boom(h):
        raise RuntimeError("container")

    nv.nvmlDeviceGetComputeRunningProcesses = boom
    assert adapter(monkeypatch, nv).busy() is None


def _with_manifest(nv):
    nv.nvmlDeviceGetVbiosVersion = lambda h: b"96.00.89.00.01"
    nv.nvmlDeviceGetCurrPcieLinkGeneration = lambda h: 4
    nv.nvmlDeviceGetMaxPcieLinkGeneration = lambda h: 5
    nv.nvmlDeviceGetCurrPcieLinkWidth = lambda h: 8
    nv.nvmlDeviceGetMaxPcieLinkWidth = lambda h: 16
    nv.nvmlDeviceGetSerial = lambda h: b" 1650123456789 "
    nv.NVML_AGGREGATE_ECC = 1
    nv.NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
    nv.nvmlDeviceGetTotalEccErrors = lambda h, kind, counter: {(0, 1): 7, (1, 1): 2}.get((kind, counter), 0)
    return nv


def test_manifest_fields_collected(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _with_manifest(modern()))
    inv = NvmlAdapter(index=3).inventory()
    assert inv.gpu_index == 3 and inv.vbios_version == "96.00.89.00.01"
    assert (inv.pcie_gen_current, inv.pcie_gen_max, inv.pcie_width_current, inv.pcie_width_max) == (4, 5, 8, 16)
    assert (inv.ecc_aggregate_corrected, inv.ecc_aggregate_uncorrected) == (7, 2)
    assert inv.serial is None


def test_serial_read_only_on_opt_in(monkeypatch):
    nv = _with_manifest(modern())
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    assert NvmlAdapter(include_serial=True).inventory().serial == "1650123456789"

    calls = []
    nv.nvmlDeviceGetSerial = lambda h: calls.append(h) or b"1650123456789"
    assert NvmlAdapter().inventory().serial is None
    assert calls == []


def test_blank_vbios_is_unknown(monkeypatch):
    nv = _with_manifest(modern())
    nv.nvmlDeviceGetVbiosVersion = lambda h: b"   "
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    assert NvmlAdapter().inventory().vbios_version is None


def test_manifest_unknown_when_unsupported(monkeypatch):
    nv = missing()
    del nv.nvmlDeviceGetTotalEccErrors
    inv = adapter(monkeypatch, nv).inventory()
    assert inv.vbios_version is None and inv.pcie_width_current is None
    assert inv.ecc_aggregate_uncorrected is None and inv.gpu_index == 0


@pytest.mark.parametrize("raw,expected", [
    (b"0", None), ("N/A", None), ("  ", None), ("to be filled by o.e.m.", None), (None, None),
    ("x" * 100, "x" * 64), (" 165012 ", "165012"),
])
def test_clean_serial(raw, expected):
    assert clean_serial(raw) == expected


def test_non_numeric_link_and_ecc_values_are_unknown(monkeypatch):
    nv = _with_manifest(modern())
    nv.nvmlDeviceGetCurrPcieLinkWidth = lambda h: "x16"
    nv.nvmlDeviceGetMaxPcieLinkGeneration = lambda h: object()
    nv.nvmlDeviceGetTotalEccErrors = lambda h, kind, counter: "N/A"
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    inv = NvmlAdapter().inventory()
    assert inv.pcie_width_current is None and inv.pcie_gen_max is None
    assert (inv.pcie_gen_current, inv.pcie_width_max) == (4, 16)
    assert inv.ecc_aggregate_corrected is None and inv.ecc_aggregate_uncorrected is None


@pytest.mark.parametrize("remap", [("N/A", 2, False, False), (1, object(), False, False), (None, 0, False, False)])
def test_non_numeric_remapped_rows_are_unknown(monkeypatch, remap):
    inv = adapter(monkeypatch, modern(remap=remap)).inventory()
    assert inv.remapped_rows is None


def test_pcie_link_reads_the_current_generation_and_width(monkeypatch):
    nv = _with_manifest(modern())
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    a = NvmlAdapter()
    assert a.pcie_link() == (4, 8)
    nv.nvmlDeviceGetCurrPcieLinkGeneration = lambda h: 5  # the link speeds up under load
    assert a.pcie_link() == (5, 8)


def test_pcie_link_is_unknown_when_nvml_cannot_report_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", modern())  # no PCIe functions at all
    assert NvmlAdapter().pcie_link() == (None, None)


def test_importing_nvml_does_not_warn_that_the_package_is_deprecated():
    # Seen on every real run: the old "pynvml" distribution warns on import. The NVML bindings
    # must come from nvidia-ml-py, which provides the same module without the warning.
    import subprocess

    out = subprocess.run(
        [sys.executable, "-W", "error::FutureWarning", "-c", "import pynvml"], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-400:]


def test_an_old_pynvml_left_behind_by_an_upgrade_does_not_warn_through_gpucheck():
    # pip install -U leaves an already-installed "pynvml" distribution in place; its import warning
    # is nothing the person running a test can act on.
    import subprocess

    code = (
        "import warnings, gpucheck.adapter\n"
        "warnings.warn('The pynvml package is deprecated. Please install nvidia-ml-py instead.', FutureWarning)\n"
        "warnings.warn('something else', FutureWarning)\n"
    )
    out = subprocess.run([sys.executable, "-W", "error::FutureWarning", "-c", code], capture_output=True, text=True)
    assert out.returncode != 0 and "something else" in out.stderr and "pynvml package" not in out.stderr
