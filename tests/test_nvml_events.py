"""The driver event log and the NVLink readers against fake pynvml modules.

The names and numbers here are the ones in the installed nvidia-ml-py (13.x): the event
functions, nvmlEventTypeXidCriticalError = 8, NVML_ERROR_TIMEOUT = 10, the NVLink getters."""

import sys
import types
import warnings

import pytest

from gpucheck.adapter import EventLogUnavailable, NvmlAdapter, NvmlEventLog
from test_nvml_adapter import modern

XID, DBE, PSTATE, CLOCK = 0x8, 0x2, 0x4, 0x10


class Timeout(Exception):
    value = 10


class Lost(Exception):
    value = 15


def with_events(queue, supported=XID | DBE | PSTATE | CLOCK):
    nv = modern()
    nv.calls = []
    nv.nvmlEventTypeXidCriticalError = XID
    nv.nvmlEventTypeDoubleBitEccError = DBE
    nv.NVML_ERROR_TIMEOUT = 10
    nv.nvmlDeviceGetSupportedEventTypes = lambda h: supported
    nv.nvmlEventSetCreate = lambda: nv.calls.append("create") or "set"

    def register(h, mask, s):
        nv.calls.append(("register", h, mask, s))

    def wait(s, timeout_ms):
        nv.calls.append(("wait", timeout_ms))
        if not queue:
            raise Timeout()
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return types.SimpleNamespace(eventType=item[0], eventData=item[1])

    nv.nvmlDeviceRegisterEvents = register
    nv.nvmlEventSetWait_v2 = wait
    nv.nvmlEventSetFree = lambda s: nv.calls.append("free")
    return nv


def adapter(monkeypatch, nv):
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    return NvmlAdapter()


def test_registers_only_for_error_events_the_card_supports(monkeypatch):
    nv = with_events([])
    adapter(monkeypatch, nv).open_event_log()
    # never power-state or clock events; then one wait, to see that the driver can be asked at all
    assert nv.calls == ["create", ("register", "h", XID | DBE, "set"), ("wait", 1)]


def test_poll_returns_the_events_waiting_and_stops_at_the_timeout(monkeypatch):
    nv = with_events([(XID, 79), (DBE, 0), (0x9999, 1)])
    log = adapter(monkeypatch, nv).open_event_log()
    assert log.poll() == [{"kind": "xid", "xid": 79}, {"kind": "double_bit_ecc"}, {"kind": "other"}]
    assert log.poll() == []
    assert all(c == ("wait", 1) for c in nv.calls if isinstance(c, tuple) and c[0] == "wait")  # never a wait of 0


def test_poll_never_drains_more_than_64_events_at_once(monkeypatch):
    nv = with_events([(XID, 13)] * 100)
    log = adapter(monkeypatch, nv).open_event_log()
    assert len(log.poll()) == 64 and len(log.poll()) == 36


def test_an_error_other_than_a_timeout_is_raised(monkeypatch):
    queue = []
    log = adapter(monkeypatch, with_events(queue)).open_event_log()
    queue.append(Lost("GPU is lost"))
    with pytest.raises(Lost):
        log.poll()


def test_a_driver_that_cannot_be_asked_for_events_is_unavailable_from_the_start(monkeypatch):
    nv = with_events([Lost("Function Not Found")])  # an old driver library without this call
    with pytest.raises(EventLogUnavailable, match="cannot be asked for error events"):
        adapter(monkeypatch, nv).open_event_log()
    assert nv.calls[-1] == "free"


def test_an_event_that_arrives_while_the_log_is_opened_is_not_lost(monkeypatch):
    log = adapter(monkeypatch, with_events([(XID, 79), (XID, 48)])).open_event_log()
    assert log.poll() == [{"kind": "xid", "xid": 79}, {"kind": "xid", "xid": 48}]


def test_close_frees_the_event_set_once(monkeypatch):
    nv = with_events([])
    log = adapter(monkeypatch, nv).open_event_log()
    log.close()
    log.close()
    assert nv.calls.count("free") == 1 and log.poll() == []


def test_a_card_that_supports_no_error_event_is_unavailable(monkeypatch):
    a = adapter(monkeypatch, with_events([], supported=PSTATE | CLOCK))
    with pytest.raises(EventLogUnavailable, match="reports no error events"):
        a.open_event_log()


def test_a_driver_that_refuses_registration_is_unavailable_and_leaks_no_event_set(monkeypatch):
    nv = with_events([])

    def refuse(h, mask, s):
        raise RuntimeError("Not Supported")

    nv.nvmlDeviceRegisterEvents = refuse
    with pytest.raises(EventLogUnavailable, match="refused to register"):
        adapter(monkeypatch, nv).open_event_log()
    assert nv.calls == ["create", "free"]


def test_bindings_without_event_functions_are_unavailable(monkeypatch):
    with pytest.raises(EventLogUnavailable, match="no event functions"):
        adapter(monkeypatch, modern()).open_event_log()


def test_event_bits_fall_back_to_the_abi_values_when_the_bindings_do_not_name_them(monkeypatch):
    nv = with_events([(XID, 62)])
    del nv.nvmlEventTypeXidCriticalError, nv.nvmlEventTypeDoubleBitEccError, nv.NVML_ERROR_TIMEOUT
    log = adapter(monkeypatch, nv).open_event_log()
    assert log.poll() == [{"kind": "xid", "xid": 62}]


def test_the_installed_bindings_have_the_names_the_event_log_uses():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pynvml
    for name in ("nvmlEventSetCreate", "nvmlDeviceRegisterEvents", "nvmlDeviceGetSupportedEventTypes",
                 "nvmlEventSetWait_v2", "nvmlEventSetFree", "nvmlDeviceGetNvLinkState",
                 "nvmlDeviceGetNvLinkErrorCounter", "nvmlDeviceGetFieldValues", "nvmlDeviceGetP2PStatus"):
        assert callable(getattr(pynvml, name)), name
    assert pynvml.nvmlEventTypeXidCriticalError == 0x8 and pynvml.NVML_ERROR_TIMEOUT == 10
    assert pynvml.NVMLError(10).value == 10  # how a timeout is recognised
    assert pynvml.NVML_NVLINK_ERROR_DL_REPLAY == 0 and pynvml.NVML_P2P_CAPS_INDEX_NVLINK == 2
    assert pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX == 138
    assert pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG == 3


def test_no_setter_or_reset_is_named_anywhere_in_the_adapter():
    import re
    from pathlib import Path

    import gpucheck.adapter as mod

    source = Path(mod.__file__).read_text()
    called = set(re.findall(r'"(nvml[A-Za-z0-9_]+)"', source)) | set(re.findall(r"nv\.(nvml[A-Za-z0-9_]+)\(", source))
    assert called and not [n for n in called if re.search(r"Set[A-Z]|Reset|Freeze|Clear", n.replace("EventSet", ""))]


# ---- NVLink --------------------------------------------------------------------------------


def with_nvlink(states, counters=None, fields=None):
    """states: {link index: active}; counters: {(link, counter id): value}; fields: {field id: value}."""
    nv = modern()

    def state(h, i):
        if i not in states:
            raise RuntimeError("Invalid Argument")
        return states[i]

    def counter(h, i, which):
        if counters is None or (i, which) not in counters:
            raise RuntimeError("Not Supported")
        return counters[(i, which)]

    nv.scopes = {}

    def field_values(h, ids):
        out = []
        for field_id, scope in ids:
            nv.scopes[field_id] = scope
            ok = fields is not None and field_id in fields
            out.append(types.SimpleNamespace(
                nvmlReturn=0 if ok else 3, valueType=3,
                value=types.SimpleNamespace(ullVal=fields[field_id] if ok else 0)))
        return out

    nv.nvmlDeviceGetNvLinkState = state
    nv.nvmlDeviceGetNvLinkErrorCounter = counter
    nv.nvmlDeviceGetFieldValues = field_values
    return nv


def test_a_card_without_nvlink_reports_none(monkeypatch):
    assert adapter(monkeypatch, modern()).nvlink() is None
    assert adapter(monkeypatch, with_nvlink({})).nvlink() is None


def test_links_states_and_per_link_error_counters_are_read(monkeypatch):
    nv = with_nvlink({0: 1, 1: 1, 3: 0}, counters={(0, 0): 2, (0, 1): 0, (0, 2): 7, (0, 3): 0, (1, 0): 1})
    snap = adapter(monkeypatch, nv).nvlink()
    assert snap["links"] == [
        {"link": 0, "active": True, "errors": {"replay": 2, "recovery": 0, "crc_flit": 7, "crc_data": 0}},
        {"link": 1, "active": True, "errors": {"replay": 1}},
        {"link": 3, "active": False, "errors": {}},
    ]
    assert snap["error_totals"] == {} and "tx_kib" not in snap


def test_card_wide_totals_and_data_counters_come_from_field_values(monkeypatch):
    nv = with_nvlink({0: 1}, fields={52: 4, 59: 1, 38: 0, 45: 0, 138: 1000, 139: 2000})
    snap = adapter(monkeypatch, nv).nvlink()
    assert snap["error_totals"] == {"replay": 4, "recovery": 1, "crc_flit": 0, "crc_data": 0}
    assert snap["tx_kib"] == 1000 and snap["rx_kib"] == 2000
    assert snap["links"] == [{"link": 0, "active": True, "errors": {}}]
    # "every link" is a scope NVML documents for the data counters only; the totals take none.
    assert nv.scopes[138] == nv.scopes[139] == 0xFFFFFFFF
    assert nv.scopes[52] == nv.scopes[59] == nv.scopes[38] == nv.scopes[45] == nv.scopes[91] == 0


def test_the_card_s_own_link_count_bounds_the_links_that_are_read(monkeypatch):
    # A driver that answers "inactive" for links that do not exist must not inflate the count.
    nv = with_nvlink({i: (1 if i < 4 else 0) for i in range(36)}, fields={91: 4})
    snap = adapter(monkeypatch, nv).nvlink()
    assert [link["link"] for link in snap["links"]] == [0, 1, 2, 3]
    unbounded = adapter(monkeypatch, with_nvlink({i: (1 if i < 4 else 0) for i in range(6)})).nvlink()
    assert len(unbounded["links"]) == 6  # no count reported: every link that answers is listed


def test_nvlink_peer_asks_nvml_for_nvlink_p2p_between_the_two_cards(monkeypatch):
    nv = with_nvlink({0: 1})
    asked = []
    nv.nvmlDeviceGetP2PStatus = lambda a, b, index: asked.append((a, b, index)) or 0
    a = adapter(monkeypatch, nv)
    assert a.nvlink_peer(a) is True and asked == [("h", "h", 2)]
    nv.nvmlDeviceGetP2PStatus = lambda a, b, index: 4
    assert a.nvlink_peer(a) is False
    del nv.nvmlDeviceGetP2PStatus
    assert a.nvlink_peer(a) is None and a.nvlink_peer(object()) is None
