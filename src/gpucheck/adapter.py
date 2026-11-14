"""Telemetry types and the adapter protocol, plus the real NVML adapter.

All NVML access here is read-only: no NVML setter is ever called. Listening for driver error
events (open_event_log) registers this process as a listener; it changes nothing on the GPU.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass
from typing import Protocol


class AdapterUnavailable(Exception):
    """Raised when no GPU or NVIDIA driver is available."""


# An old "pynvml" distribution left in place by an upgrade warns on every import that it is
# deprecated. gpucheck depends on nvidia-ml-py, which provides the same module, so the warning
# is nothing the person running a test can act on.
warnings.filterwarnings("ignore", message="The pynvml package is deprecated", category=FutureWarning)


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
    pcie_link_under_load: bool | None = None  # True when the current link was read while the load ran
    ecc_aggregate_corrected: int | None = None  # lifetime counts
    ecc_aggregate_uncorrected: int | None = None
    serial: str | None = None  # read only on explicit opt-in


@dataclass(frozen=True, slots=True)  # slots: a day-long run holds 86,400 of these per card
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
    """An adapter may also offer pcie_link() -> (generation, width): the link as it is right now.
    The run reads it under load, because a link drops to a lower speed at idle to save power.

    For the opt-in extra checks it may also offer open_event_log() -> an object with poll() and
    close() (error events the driver reports for this card), nvlink() -> a snapshot of
    its NVLink links or None when it has none, and nvlink_peer(other) -> whether NVML reports
    NVLink peer-to-peer access between this card and another adapter's card."""

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


# Driver events that report an error, by the name gpucheck records and the NVML ABI bit. An Xid
# is not always a hardware fault: some numbers are raised by a fault in a program using the card.
# Clock, power-state and configuration-change events are not errors and are not listened for.
ERROR_EVENTS = (
    ("xid", "nvmlEventTypeXidCriticalError", 0x8),
    ("double_bit_ecc", "nvmlEventTypeDoubleBitEccError", 0x2),
    ("single_bit_ecc_storm", "nvmlEventTypeSingleBitEccErrorStorm", 0x200),
    ("dram_retirement", "nvmlEventTypeDramRetirementEvent", 0x400),
    ("dram_retirement_failure", "nvmlEventTypeDramRetirementFailure", 0x800),
    ("non_fatal_poison", "nvmlEventTypeNonFatalPoisonError", 0x1000),
    ("fatal_poison", "nvmlEventTypeFatalPoisonError", 0x2000),
    ("gpu_unavailable", "nvmlEventTypeGpuUnavailableError", 0x4000),
    ("gpu_recovery", "nvmlEventTypeGpuRecoveryAction", 0x8000),
)
EVENT_WAIT_MS = 1  # a wait of 0 is not documented as "return at once", so wait one millisecond
MAX_EVENTS_PER_POLL = 64
_NVML_ERROR_TIMEOUT = 10


class EventLogUnavailable(Exception):
    """The driver offers no usable event log for this card to this process."""


class NvmlEventLog:
    """Error events (Xid and memory errors) the driver reports for one card.

    The driver delivers only events that happen after registration, which is what a run wants:
    errors while it was testing. Registering is not a GPU setting and needs no root on a normal
    driver install; where the driver refuses (some containers and virtual GPUs), opening raises
    EventLogUnavailable and the run says the log was not available. gpucheck never falls back to
    reading the kernel log, which needs root."""

    def __init__(self, nv, handle) -> None:
        self._nv = nv
        self._kinds: dict[int, str] = {}
        wanted = 0
        for kind, name, abi in ERROR_EVENTS:
            bit = _const(nv, name, default=abi)
            self._kinds[bit] = kind
            wanted |= bit
        for name in ("nvmlEventSetCreate", "nvmlDeviceRegisterEvents", "nvmlEventSetWait_v2", "nvmlEventSetFree"):
            if getattr(nv, name, None) is None:
                raise EventLogUnavailable("the installed NVML bindings have no event functions")
        try:
            supported = int(nv.nvmlDeviceGetSupportedEventTypes(handle))
        except Exception as exc:
            raise EventLogUnavailable(f"the driver does not report which events this card supports ({exc})") from exc
        mask = supported & wanted
        if not mask:
            raise EventLogUnavailable("the driver reports no error events for this card")
        self._set = None
        self._pending: list[dict] = []
        try:
            self._set = nv.nvmlEventSetCreate()
            nv.nvmlDeviceRegisterEvents(handle, mask, self._set)
        except Exception as exc:
            self.close()
            raise EventLogUnavailable(f"the driver refused to register for error events ({exc})") from exc
        try:
            # One wait now: a driver library too old to have this call fails here, where it can
            # be reported as "not available", not at the first sample as a log that broke.
            self._pending = self._waiting(1)
        except Exception as exc:
            self.close()
            raise EventLogUnavailable(f"the driver cannot be asked for error events ({exc})") from exc

    def _timed_out(self, exc: Exception) -> bool:
        timeout = _const(self._nv, "NVML_ERROR_TIMEOUT", default=_NVML_ERROR_TIMEOUT)
        return getattr(exc, "value", None) == timeout

    def poll(self) -> list[dict]:
        """The events reported since the last poll, oldest first. Raises if the log stops working.

        NVML reports the most recent Xid number for every Xid event that was waiting, so when
        several arrive between two polls their count is right but their numbers may repeat."""
        if self._set is None:
            return []
        out, self._pending = self._pending, []
        return out + self._waiting(MAX_EVENTS_PER_POLL - len(out))

    def _waiting(self, limit: int) -> list[dict]:
        out: list[dict] = []
        for _ in range(limit):
            try:
                data = self._nv.nvmlEventSetWait_v2(self._set, EVENT_WAIT_MS)
            except Exception as exc:
                if self._timed_out(exc):
                    break  # nothing more waiting
                raise
            kind = self._kinds.get(_int(getattr(data, "eventType", None)) or 0, "other")
            event: dict = {"kind": kind}
            if kind == "xid":
                event["xid"] = _int(getattr(data, "eventData", None))
            out.append(event)
        return out

    def close(self) -> None:
        if self._set is not None:
            try:
                self._nv.nvmlEventSetFree(self._set)
            except Exception:
                pass
            self._set = None


# NVLink error counters by the name gpucheck records, the NVML enum name and its ABI value.
_NVLINK_ERRORS = (
    ("replay", "NVML_NVLINK_ERROR_DL_REPLAY", 0),
    ("recovery", "NVML_NVLINK_ERROR_DL_RECOVERY", 1),
    ("crc_flit", "NVML_NVLINK_ERROR_DL_CRC_FLIT", 2),
    ("crc_data", "NVML_NVLINK_ERROR_DL_CRC_DATA", 3),
)
# Card-wide totals of the same counters as NVML field ids, for drivers that no longer answer the
# per-link query, and the data counters (KiB) summed over every link.
_NVLINK_TOTAL_FIELDS = (
    ("replay", "NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL", 52),
    ("recovery", "NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL", 59),
    ("crc_flit", "NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL", 38),
    ("crc_data", "NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL", 45),
)
_NVLINK_DATA_FIELDS = (
    ("tx_kib", "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX", 138),
    ("rx_kib", "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX", 139),
)
_NVLINK_COUNT_FIELD = (("count", "NVML_FI_DEV_NVLINK_LINK_COUNT", 91),)
_ALL_LINKS = 0xFFFFFFFF  # scope id meaning "summed over every link" (documented for the data counters)
_MAX_NVLINKS = 36
# nvmlFieldValue_t.valueType -> the union member holding the value
_FIELD_MEMBERS = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal", 5: "siVal", 6: "usVal"}


def _field_int(value) -> int | None:
    """A whole number from one nvmlFieldValue_t; None when NVML could not read the field."""
    if _int(getattr(value, "nvmlReturn", 1)) != 0:
        return None
    member = _FIELD_MEMBERS.get(_int(getattr(value, "valueType", None)))
    union = getattr(value, "value", None)
    return _int(getattr(union, member, None)) if member and union is not None else None


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
        gen_cur, width_cur = self.pcie_link()
        link = {
            "gen_cur": gen_cur,
            "gen_max": _int(self._q("nvmlDeviceGetMaxPcieLinkGeneration", h)),
            "width_cur": width_cur,
            "width_max": _int(self._q("nvmlDeviceGetMaxPcieLinkWidth", h)),
        }
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

    def pcie_link(self) -> tuple[int | None, int | None]:
        """The PCIe link's current (generation, width); None for a value NVML cannot report."""
        h = self._h
        return (
            _int(self._q("nvmlDeviceGetCurrPcieLinkGeneration", h)),
            _int(self._q("nvmlDeviceGetCurrPcieLinkWidth", h)),
        )

    def open_event_log(self) -> NvmlEventLog:
        """Start listening for error events on this card; EventLogUnavailable if it cannot."""
        return NvmlEventLog(self._nv, self._h)

    def _fields(self, wanted: tuple, scope: int = 0) -> dict[str, int]:
        """NVML field values by name, for the ones the card answers."""
        nv = self._nv
        ids = [(_const(nv, name, default=abi), scope) for _, name, abi in wanted]
        values = self._q("nvmlDeviceGetFieldValues", self._h, ids)
        out: dict[str, int] = {}
        if values is None:
            return out
        for (key, _, _), value in zip(wanted, values):
            n = _field_int(value)
            if n is not None:
                out[key] = n
        return out

    def nvlink(self) -> dict | None:
        """This card's NVLink links as they are now, or None when it has none.

        {"links": [{"link": i, "active": bool, "errors": {counter: count}}], "error_totals": {...},
        "tx_kib": n, "rx_kib": n}; a value the driver does not report is absent."""
        nv, h = self._nv, self._h
        links = []
        limit = min(_const(nv, "NVML_NVLINK_MAX_LINKS", default=_MAX_NVLINKS), _MAX_NVLINKS)
        # Where the card says how many links it has, look no further: a driver that answers
        # "inactive" for a link that does not exist would otherwise inflate the count.
        count = self._fields(_NVLINK_COUNT_FIELD).get("count")
        if count is not None:
            limit = min(limit, count)
        for i in range(limit):
            state = self._q("nvmlDeviceGetNvLinkState", h, i)
            if state is None:
                continue  # no such link on this card
            errors = {}
            for key, name, abi in _NVLINK_ERRORS:
                n = _int(self._q("nvmlDeviceGetNvLinkErrorCounter", h, i, _const(nv, name, default=abi)))
                if n is not None:
                    errors[key] = n
            links.append({"link": i, "active": bool(_int(state)), "errors": errors})
        if not links:
            return None
        return {
            "links": links,
            "error_totals": self._fields(_NVLINK_TOTAL_FIELDS),
            **self._fields(_NVLINK_DATA_FIELDS, _ALL_LINKS),
        }

    def nvlink_peer(self, other) -> bool | None:
        """Whether NVML reports NVLink peer-to-peer access between this card and `other`'s; None if unknown."""
        handle = getattr(other, "_h", None)
        if handle is None:
            return None
        index = _const(self._nv, "NVML_P2P_CAPS_INDEX_NVLINK", default=2)
        status = self._q("nvmlDeviceGetP2PStatus", self._h, handle, index)
        return None if status is None else _int(status) == _const(self._nv, "NVML_P2P_STATUS_OK", default=0)

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
