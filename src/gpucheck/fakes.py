"""Scripted fake GPUs for tests and the CLI --fake flag.

Each profile is a generator of TelemetrySample. After the scripted phase the
generator keeps yielding the final steady state, so it can be sampled forever.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

from .adapter import EventLogUnavailable, GpuInventory, TelemetryAdapter, TelemetrySample, hash_uuid
from .checks import CheckUnavailable

MODEL = "NVIDIA H100 80GB HBM3"
FAKE_UUID = "GPU-00000000-0000-0000-0000-000000000000"

HEALTHY_TFLOPS = 950.0
DEGRADED_TFLOPS = 600.0


def _inventory(**overrides) -> GpuInventory:
    base = dict(
        model_name=MODEL,
        uuid_hash=hash_uuid(FAKE_UUID),
        driver="550.54.15",
        cuda="12.4",
        ecc_enabled=True,
        power_limit_w=700.0,
        default_power_limit_w=700.0,
        mig_enabled=False,
        fractional=False,
        retired_pages=0,
        remapped_rows=0,
    )
    base.update(overrides)
    return GpuInventory(**base)


def _sample(t: float, temp: float, **kw) -> TelemetrySample:
    fields = dict(
        t_s=t,
        temp_c=temp,
        power_w=650.0,
        sm_clock_mhz=1980.0,
        fan_pct=None,
        throttle_reasons=frozenset(),
        ecc_volatile=0,
        achieved_tflops=HEALTHY_TFLOPS,
    )
    fields.update(kw)
    return TelemetrySample(**fields)


def _wobble(i: int) -> float:
    return (0.0, 0.5, 1.0, 0.5)[i % 4] - 0.5  # deterministic, in [-0.5, 0.5]


IDLE_SAMPLES = 61  # preflight sample + the longest (60 s) baseline


def _healthy() -> Iterator[TelemetrySample]:
    i = 0
    while True:
        if i < IDLE_SAMPLES:  # cool, low-power idle through preflight and baseline
            yield _sample(float(i), 35.0 + _wobble(i), power_w=80.0, sm_clock_mhz=210.0)
        else:
            warm = min(i - IDLE_SAMPLES, 20) / 20  # ramp 35 -> 62 C over 20 s, then steady
            yield _sample(float(i), 35.0 + 27.0 * warm + _wobble(i))
        i += 1


def _hot_env() -> Iterator[TelemetrySample]:
    i = 0
    while True:
        yield _sample(float(i), 55.0 + _wobble(i), power_w=80.0, sm_clock_mhz=210.0)
        i += 1


def _tim_degraded() -> Iterator[TelemetrySample]:
    i = 0
    while True:
        temp = min(87.0, 35.0 + i * 0.52)  # 87 C at sample 100
        throttled = temp > 83.0
        if temp <= 80.0:
            tflops = HEALTHY_TFLOPS
        else:
            tflops = HEALTHY_TFLOPS - (HEALTHY_TFLOPS - DEGRADED_TFLOPS) * (temp - 80.0) / 7.0
        yield _sample(
            float(i),
            temp,
            sm_clock_mhz=1980.0 if not throttled else 1500.0,
            throttle_reasons=frozenset({"thermal"}) if throttled else frozenset(),
            achieved_tflops=tflops,
        )
        i += 1


def _power_capped() -> Iterator[TelemetrySample]:
    i = 0
    while True:
        yield _sample(
            float(i),
            55.0 + 8.0 * min(i, 20) / 20 + _wobble(i),
            power_w=400.0,
            sm_clock_mhz=1400.0,
            throttle_reasons=frozenset({"power"}),
            achieved_tflops=700.0,
        )
        i += 1


def _overheat() -> Iterator[TelemetrySample]:
    for sample in _healthy():
        yield replace(sample, temp_c=200.0)


SAG_AFTER_S = 3000  # the "sag" card slows down this long into its load
SAG_TFLOPS = 800.0


def _sag() -> Iterator[TelemetrySample]:
    """Fine through a standard run, then loses throughput: only a burn-in sees it."""
    for i, sample in enumerate(_healthy()):
        late = i - IDLE_SAMPLES >= SAG_AFTER_S
        yield replace(sample, achieved_tflops=SAG_TFLOPS) if late else sample


_PROFILES: dict[str, tuple[Callable[[], Iterator[TelemetrySample]], dict]] = {
    "healthy": (_healthy, {}),
    "tim_degraded": (_tim_degraded, {}),
    "hot_env": (_hot_env, {}),
    "power_capped": (_power_capped, {"power_limit_w": 400.0, "default_power_limit_w": 700.0}),
    "mig": (_healthy, {"mig_enabled": True}),
    "busy": (_healthy, {}),
    "overheat": (_overheat, {}),
    # Cards for the opt-in extra checks. Each failing card is healthy in a standard run.
    "xid": (_healthy, {}),
    "no_event_log": (_healthy, {}),
    "bad_memory": (_healthy, {}),
    "wrong_answers": (_healthy, {}),
    "sag": (_sag, {}),
    "nvlink": (_healthy, {}),
    "nvlink_errors": (_healthy, {}),
    "nvlink_idle": (_healthy, {}),
    "nvlink_no_counter": (_healthy, {}),
}

_BUSY_PROFILES = frozenset({"busy"})

GIB = 1 << 30
FAKE_MEMORY_BYTES = 80 * GIB
FAKE_PATTERNS = ["zeros", "ones", "checker_a", "checker_5", "address", "random"]
XID_AT_POLL = 90  # the "xid" card reports its events at this poll: under load in any run length


def _nvlink_snapshot(active: list[bool], errors: dict[str, int], tx_kib: int = 0) -> dict:
    links = [{"link": i, "active": up, "errors": dict(errors) if i == 0 else {k: 0 for k in errors}}
             for i, up in enumerate(active)]
    return {"links": links, "error_totals": dict(errors), "tx_kib": tx_kib, "rx_kib": tx_kib}


_ZERO = {"replay": 0, "recovery": 0, "crc_flit": 0, "crc_data": 0}

# What each card does in the extra checks; a card not listed here behaves like "healthy".
_EXTRAS: dict[str, dict] = {
    "xid": {"events": {XID_AT_POLL: [{"kind": "xid", "xid": 79}, {"kind": "double_bit_ecc"}]}},
    "no_event_log": {"event_log": False},
    "bad_memory": {"memtest": {"mismatched_words": 3, "by_pattern": {"checker_a": 2, "random": 1}}},
    "wrong_answers": {"answer": {"wrong_products": 2, "wrong_elements": 17}},
    "nvlink": {"nvlink": [_nvlink_snapshot([True] * 4, _ZERO), _nvlink_snapshot([True] * 4, _ZERO)]},
    "nvlink_errors": {
        "nvlink": [
            _nvlink_snapshot([True] * 4, _ZERO),
            _nvlink_snapshot([True, True, True, False], {"replay": 5, "recovery": 1, "crc_flit": 12, "crc_data": 0}),
        ],
        "transfer_mismatch": 4,
    },
    "nvlink_idle": {"nvlink": [_nvlink_snapshot([False] * 3, _ZERO), _nvlink_snapshot([False] * 3, _ZERO)]},
    # Links are fine, but copies to and from this card do not show on its NVLink data counter.
    "nvlink_no_counter": {"nvlink": [_nvlink_snapshot([True] * 4, _ZERO), _nvlink_snapshot([True] * 4, _ZERO)],
                          "transfer_off_nvlink": True},
}


class FakeEventLog:
    """Scripted driver events: {poll index: [events]}; the driver reports nothing at other polls."""

    def __init__(self, script: dict[int, list[dict]]) -> None:
        self._script = script
        self._polls = 0
        self.closed = False

    def poll(self) -> list[dict]:
        events = self._script.get(self._polls, [])
        self._polls += 1
        return [dict(e) for e in events]

    def close(self) -> None:
        self.closed = True


class FakeCheck:
    """A check phase that takes `steps` slices of work and then reports `outcome`."""

    def __init__(self, name: str, outcome: dict, steps: int = 3, unavailable: str | None = None) -> None:
        self.name = name
        self._outcome = outcome
        self._left = steps
        self._unavailable = unavailable
        self.stopped = False

    def start(self) -> None:
        if self._unavailable:
            raise CheckUnavailable(self._unavailable)

    def step(self) -> bool:
        self._left -= 1
        return self._left >= 0

    def stop(self) -> None:
        self.stopped = True

    def result(self) -> dict:
        return dict(self._outcome)


def fake_memtest(extras: dict) -> FakeCheck:
    tested = int(0.9 * FAKE_MEMORY_BYTES)
    outcome = {
        "bytes_tested": tested, "bytes_total": FAKE_MEMORY_BYTES, "patterns": list(FAKE_PATTERNS),
        "mismatched_words": 0, "by_pattern": {}, "seconds": 20.0,
    }
    outcome.update(extras.get("memtest", {}))
    return FakeCheck("memtest", outcome, steps=2 * len(FAKE_PATTERNS))


def fake_answer(extras: dict) -> FakeCheck:
    outcome = {"products": 400, "wrong_products": 0, "wrong_elements": 0, "seconds": 60.0, "skipped": []}
    outcome.update(extras.get("answer", {}))
    return FakeCheck("answer", outcome, steps=4)


TRANSFER_BYTES = 256 << 20


def fake_transfer(a: "FakeAdapter", b: "FakeAdapter") -> dict:
    """One simulated copy from card a to card b and back; it shows on both cards' data counters."""
    for card in (a, b):
        if not card.extras.get("transfer_off_nvlink"):
            card.transferred_kib += TRANSFER_BYTES // 1024
    return {
        "bytes": TRANSFER_BYTES,
        "mismatched_words": max(a.extras.get("transfer_mismatch", 0), b.extras.get("transfer_mismatch", 0)),
        "seconds": 0.1,
    }


class FakeAdapter:
    def __init__(
        self, inventory: GpuInventory, samples: Iterator[TelemetrySample], busy: bool = False,
        extras: dict | None = None,
    ) -> None:
        self._inventory = inventory
        self._samples = samples
        self._busy = busy
        self.extras = extras or {}
        self._nvlink_reads = 0
        self.transferred_kib = 0  # test data the transfer check moved over this card's links

    def open_event_log(self) -> FakeEventLog:
        if self.extras.get("event_log") is False:
            raise EventLogUnavailable("the driver reports no error events for this card")
        return FakeEventLog(self.extras.get("events", {}))

    def nvlink(self) -> dict | None:
        """The first read is the state before the run; every later read the state after it."""
        script = self.extras.get("nvlink")
        if not script:
            return None
        snapshot = script[min(self._nvlink_reads, len(script) - 1)]
        self._nvlink_reads += 1
        return {**snapshot, "tx_kib": snapshot["tx_kib"] + self.transferred_kib,
                "rx_kib": snapshot["rx_kib"] + self.transferred_kib}

    def nvlink_peer(self, other) -> bool | None:
        return bool(self.extras.get("nvlink")) and bool(getattr(other, "extras", {}).get("nvlink"))

    def inventory(self) -> GpuInventory:
        return self._inventory

    def sample(self) -> TelemetrySample:
        return next(self._samples)

    def busy(self) -> bool | None:
        return self._busy


def make_fake(profile: str) -> TelemetryAdapter:
    try:
        factory, inv = _PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown fake profile {profile!r}; choose from {sorted(_PROFILES)}") from None
    return FakeAdapter(_inventory(**inv), factory(), busy=profile in _BUSY_PROFILES, extras=_EXTRAS.get(profile))


def make_fakes(spec: str) -> list[TelemetryAdapter]:
    """One fake card per comma-separated profile name; several cards get distinct indices and UUIDs."""
    profiles = [p.strip() for p in spec.split(",")]
    if len(profiles) == 1:
        return [make_fake(profiles[0])]
    cards: list[TelemetryAdapter] = []
    for i, profile in enumerate(profiles):
        try:
            factory, inv = _PROFILES[profile]
        except KeyError:
            raise ValueError(f"unknown fake profile {profile!r}; choose from {sorted(_PROFILES)}") from None
        inventory = _inventory(
            **{**inv, "gpu_index": i, "uuid_hash": hash_uuid(f"GPU-00000000-0000-0000-0000-{i:012d}")}
        )
        cards.append(FakeAdapter(inventory, factory(), busy=profile in _BUSY_PROFILES, extras=_EXTRAS.get(profile)))
    return cards
