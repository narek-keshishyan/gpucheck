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
    remapped_rows: int | None


@dataclass(frozen=True)
class TelemetrySample:
    t_s: float
    temp_c: float | None
    power_w: float | None
    sm_clock_mhz: float | None
    fan_pct: float | None
    throttle_reasons: frozenset[str]
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


def _try(fn, *args):
    """Call an NVML query; None if the GPU does not support it."""
    try:
        return fn(*args)
    except Exception:
        return None


class NvmlAdapter(TelemetryAdapter):
    """Reads GPU 0 through NVML (pynvml)."""

    def __init__(self, index: int = 0) -> None:
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

    def inventory(self) -> GpuInventory:
        nv, h = self._nv, self._h
        uuid = _text(nv.nvmlDeviceGetUUID(h))
        cuda_raw = _try(nv.nvmlSystemGetCudaDriverVersion)
        cuda = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}" if cuda_raw else None
        ecc = _try(nv.nvmlDeviceGetEccMode, h)
        limit = _try(nv.nvmlDeviceGetPowerManagementLimit, h)
        default = _try(nv.nvmlDeviceGetPowerManagementDefaultLimit, h)
        mig = _try(nv.nvmlDeviceGetMigMode, h)
        retired = _try(nv.nvmlDeviceGetRetiredPages, h, nv.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS)
        remap = _try(nv.nvmlDeviceGetRemappedRows, h)
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
            remapped_rows=remap[0] if remap else None,
        )

    def sample(self) -> TelemetrySample:
        nv, h = self._nv, self._h
        temp = _try(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)
        power = _try(nv.nvmlDeviceGetPowerUsage, h)
        clock = _try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM)
        fan = _try(nv.nvmlDeviceGetFanSpeed, h)
        mask = _try(nv.nvmlDeviceGetCurrentClocksEventReasons, h) or 0
        ecc = _try(
            nv.nvmlDeviceGetTotalEccErrors,
            h,
            nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            nv.NVML_VOLATILE_ECC,
        )
        names = {
            "thermal": nv.NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL_SLOWDOWN
            | nv.NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL_SLOWDOWN,
            "power": nv.NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP
            | nv.NVML_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN,
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
        procs = _try(self._nv.nvmlDeviceGetComputeRunningProcesses, self._h)
        return None if procs is None else len(procs) > 0
