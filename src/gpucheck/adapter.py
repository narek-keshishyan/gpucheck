"""Telemetry types and the adapter protocol, plus the real NVML adapter.

All NVML access here is read-only: no NVML setter is ever called.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass
from typing import Protocol


class AdapterUnavailable(Exception):
    """Raised when no GPU or NVIDIA driver is available."""


@dataclass(frozen=True)
class GpuInventory:
    model_name: str
    uuid_hash: str
    driver: str
    cuda: str | None
    ecc_enabled: bool | None
    power_limit_w: float | None
    default_power_limit_w: float | None
    mig_enabled: bool
    fractional: bool
    retired_pages: int | None
    remapped_rows: int | None  # correctable + uncorrectable
    row_remap_pending: bool | None = None
    row_remap_failure: bool | None = None
    gpu_index: int | None = None
    vbios_version: str | None = None
    pcie_gen_current: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_current: int | None = None
    pcie_width_max: int | None = None
    ecc_aggregate_corrected: int | None = None  # lifetime counts
    ecc_aggregate_uncorrected: int | None = None
    serial: str | None = None  # read only on explicit opt-in


@dataclass(frozen=True)
class TelemetrySample:
    t_s: float
    temp_c: float | None
    power_w: float | None
    sm_clock_mhz: float | None
    fan_pct: float | None
    throttle_reasons: frozenset[str] | None  # None = NVML could not report reasons (unknown)
    ecc_volatile: int | None
    achieved_tflops: float | None


class TelemetryAdapter(Protocol):
    def inventory(self) -> GpuInventory: ...

    def sample(self) -> TelemetrySample: ...

    def busy(self) -> bool | None: ...


def hash_uuid(uuid: str) -> str:
    return hashlib.sha256(("gpucheck.v1." + uuid).encode()).hexdigest()[:16]


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


PLACEHOLDER_SERIALS = frozenset(
    {"0", "N/A", "NA", "NONE", "UNKNOWN", "NOT SPECIFIED", "TO BE FILLED BY O.E.M."}
)
_MAX_TEXT = 64


def clean_serial(raw: object) -> str | None:
    """Serial text, or None when empty or a known placeholder; at most 64 characters."""
    if raw is None:
        return None
    text = _text(raw).strip()
    if not text or text.upper() in PLACEHOLDER_SERIALS:
        return None
    return text[:_MAX_TEXT]


def _int(value: object) -> int | None:
    """An NVML count as an int; None when it is missing or cannot be converted."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _try(fn, *args):
    """Call an NVML query; None if the GPU does not support it."""
    try:
        return fn(*args)
    except Exception:
        return None


# NVML ABI values, used only when pynvml exposes neither naming family.
_ABI = {
    "SwPowerCap": 0x4,
    "SwThermalSlowdown": 0x20,
    "HwThermalSlowdown": 0x40,
    "HwPowerBrakeSlowdown": 0x80,
}


def _const(nv, *names: str, default: int) -> int:
    """First of `names` that pynvml defines, else `default`; never raises."""
    for name in names:
        value = getattr(nv, name, None)
        if isinstance(value, int):
            return value
    return default


def _reason_bit(nv, suffix: str) -> int:
    # pynvml renamed "ClocksThrottleReason" to "ClocksEventReason" (NVML 12.x); the
    # upper-case NVML_CLOCKS_THROTTLE_REASON_* spelling never existed in pynvml.
    return _const(
        nv,
        f"nvmlClocksEventReason{suffix}",
        f"nvmlClocksThrottleReason{suffix}",
        default=_ABI[suffix],
    )


def gpu_count() -> int:
    """Number of NVIDIA GPUs NVML can see; raises AdapterUnavailable when NVML cannot be loaded."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # pynvml deprecation notice
            import pynvml

        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetCount())
    except Exception as exc:  # ImportError, NVMLError, missing library...
        raise AdapterUnavailable(f"NVML unavailable: {exc}") from exc


class NvmlAdapter(TelemetryAdapter):
    """Reads one GPU (default index 0) through NVML (pynvml)."""

    def __init__(self, index: int = 0, include_serial: bool = False) -> None:
        self._index = index
        self._include_serial = include_serial
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)  # pynvml deprecation notice
                import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:  # ImportError, NVMLError, missing library...
            raise AdapterUnavailable(f"NVML unavailable: {exc}") from exc
        self._t0 = time.monotonic()

    def _q(self, name: str, *args):
        """Optional NVML query by name: None if pynvml lacks it or the GPU does not support it."""
        fn = getattr(self._nv, name, None)
        return None if fn is None else _try(fn, *args)

    def inventory(self) -> GpuInventory:
        nv, h = self._nv, self._h
        uuid = _text(nv.nvmlDeviceGetUUID(h))
        cuda_raw = self._q("nvmlSystemGetCudaDriverVersion")
        cuda = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}" if cuda_raw else None
        ecc = self._q("nvmlDeviceGetEccMode", h)
        limit = self._q("nvmlDeviceGetPowerManagementLimit", h)
        default = self._q("nvmlDeviceGetPowerManagementDefaultLimit", h)
        mig = self._q("nvmlDeviceGetMigMode", h)
        cause = _const(nv, "NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS", default=0)
        retired = self._q("nvmlDeviceGetRetiredPages", h, cause)
        # (correctable rows, uncorrectable rows, remap pending, remap failure occurred)
        remap = self._q("nvmlDeviceGetRemappedRows", h)
        remap_ok = isinstance(remap, (tuple, list)) and len(remap) >= 4
        rows = (_int(remap[0]), _int(remap[1])) if remap_ok else (None, None)
        vbios = self._q("nvmlDeviceGetVbiosVersion", h)
        vbios = _text(vbios).strip()[:_MAX_TEXT] or None if vbios is not None else None
        serial = clean_serial(self._q("nvmlDeviceGetSerial", h)) if self._include_serial else None
        aggregate = _const(nv, "NVML_AGGREGATE_ECC", default=1)
        ecc_corrected = self._q(
            "nvmlDeviceGetTotalEccErrors", h,
            _const(nv, "NVML_MEMORY_ERROR_TYPE_CORRECTED", default=0), aggregate,
        )
        ecc_uncorrected = self._q(
            "nvmlDeviceGetTotalEccErrors", h,
            _const(nv, "NVML_MEMORY_ERROR_TYPE_UNCORRECTED", default=1), aggregate,
        )
        link = {
            key: self._q(fn, h)
            for key, fn in (
                ("gen_cur", "nvmlDeviceGetCurrPcieLinkGeneration"),
                ("gen_max", "nvmlDeviceGetMaxPcieLinkGeneration"),
                ("width_cur", "nvmlDeviceGetCurrPcieLinkWidth"),
                ("width_max", "nvmlDeviceGetMaxPcieLinkWidth"),
            )
        }
        link = {k: _int(v) for k, v in link.items()}
        return GpuInventory(
            model_name=_text(nv.nvmlDeviceGetName(h)),
            uuid_hash=hash_uuid(uuid),
            driver=_text(nv.nvmlSystemGetDriverVersion()),
            cuda=cuda,
            ecc_enabled=bool(ecc[0]) if ecc else None,
            power_limit_w=limit / 1000.0 if limit is not None else None,
            default_power_limit_w=default / 1000.0 if default is not None else None,
            mig_enabled=bool(mig and mig[0]),
            fractional=False,
            retired_pages=len(retired) if retired is not None else None,
            remapped_rows=None if None in rows else rows[0] + rows[1],
            row_remap_pending=bool(remap[2]) if remap_ok else None,
            row_remap_failure=bool(remap[3]) if remap_ok else None,
            gpu_index=self._index,
            vbios_version=vbios,
            pcie_gen_current=link["gen_cur"],
            pcie_gen_max=link["gen_max"],
            pcie_width_current=link["width_cur"],
            pcie_width_max=link["width_max"],
            ecc_aggregate_corrected=_int(ecc_corrected),
            ecc_aggregate_uncorrected=_int(ecc_uncorrected),
            serial=serial,
        )

    def _throttle_mask(self) -> int | None:
        for name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            mask = self._q(name, self._h)
            if mask is not None:
                return int(mask)
        return None  # unknown: recorded as a telemetry gap, never as "no throttling"

    def sample(self) -> TelemetrySample:
        nv, h = self._nv, self._h
        temp = self._q("nvmlDeviceGetTemperature", h, _const(nv, "NVML_TEMPERATURE_GPU", default=0))
        power = self._q("nvmlDeviceGetPowerUsage", h)
        clock = self._q("nvmlDeviceGetClockInfo", h, _const(nv, "NVML_CLOCK_SM", default=1))
        fan = self._q("nvmlDeviceGetFanSpeed", h)
        mask = self._throttle_mask()
        ecc = self._q(
            "nvmlDeviceGetTotalEccErrors",
            h,
            _const(nv, "NVML_MEMORY_ERROR_TYPE_UNCORRECTED", default=1),
            _const(nv, "NVML_VOLATILE_ECC", default=0),
        )
        reasons: frozenset[str] | None = None
        if mask is not None:
            names = {
                "thermal": _reason_bit(nv, "HwThermalSlowdown") | _reason_bit(nv, "SwThermalSlowdown"),
                "power": _reason_bit(nv, "SwPowerCap") | _reason_bit(nv, "HwPowerBrakeSlowdown"),
            }
            reasons = frozenset(name for name, bits in names.items() if mask & bits)
        return TelemetrySample(
            t_s=time.monotonic() - self._t0,
            temp_c=float(temp) if temp is not None else None,
            power_w=power / 1000.0 if power is not None else None,
            sm_clock_mhz=float(clock) if clock is not None else None,
            fan_pct=float(fan) if fan is not None else None,
            throttle_reasons=reasons,
            ecc_volatile=int(ecc) if ecc is not None else None,
            achieved_tflops=None,  # supplied by the load generator, not NVML
        )

    def busy(self) -> bool | None:
        procs = self._q("nvmlDeviceGetComputeRunningProcesses", self._h)
        return None if procs is None else len(procs) > 0
