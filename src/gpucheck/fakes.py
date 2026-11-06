"""Scripted fake GPUs for tests and the CLI --fake flag.

Each profile is a generator of TelemetrySample. After the scripted phase the
generator keeps yielding the final steady state, so it can be sampled forever.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .adapter import GpuInventory, TelemetryAdapter, TelemetrySample, hash_uuid

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


_PROFILES: dict[str, tuple[Callable[[], Iterator[TelemetrySample]], dict]] = {
    "healthy": (_healthy, {}),
    "tim_degraded": (_tim_degraded, {}),
    "hot_env": (_hot_env, {}),
    "power_capped": (_power_capped, {"power_limit_w": 400.0, "default_power_limit_w": 700.0}),
    "mig": (_healthy, {"mig_enabled": True}),
}


class FakeAdapter:
    def __init__(self, inventory: GpuInventory, samples: Iterator[TelemetrySample]) -> None:
        self._inventory = inventory
        self._samples = samples

    def inventory(self) -> GpuInventory:
        return self._inventory

    def sample(self) -> TelemetrySample:
        return next(self._samples)

    def busy(self) -> bool | None:
        return False


def make_fake(profile: str) -> TelemetryAdapter:
    try:
        factory, inv = _PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown fake profile {profile!r}; choose from {sorted(_PROFILES)}") from None
    return FakeAdapter(_inventory(**inv), factory())
