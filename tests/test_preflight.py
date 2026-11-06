from dataclasses import replace

from gpucheck.fakes import make_fake
from gpucheck.preflight import Preflight, run_preflight


class StubAdapter:
    """Wraps a fake and overrides busy() and selected inventory fields."""

    def __init__(self, base, *, busy, **inventory_overrides):
        self._base = base
        self._busy = busy
        self._overrides = inventory_overrides

    def inventory(self):
        return replace(self._base.inventory(), **self._overrides)

    def sample(self):
        return self._base.sample()

    def busy(self):
        return self._busy


def test_mig_gets_measurements_only():
    assert run_preflight(make_fake("mig")).mode == "measurements_only"


def test_mig_reason_is_recorded():
    result = run_preflight(make_fake("mig"))
    assert any("MIG" in r for r in result.reasons)


def test_busy_full_card_refuses():
    adapter = StubAdapter(make_fake("healthy"), busy=True)
    result = run_preflight(adapter)
    assert result.mode == "refuse"
    assert any("busy" in r.lower() for r in result.reasons)


def test_busy_shared_gpu_does_not_refuse():
    adapter = StubAdapter(make_fake("healthy"), busy=True, mig_enabled=True)
    result = run_preflight(adapter)
    assert result.mode == "measurements_only"
    assert any("neighbor" in r for r in result.reasons)


def test_fractional_gets_measurements_only():
    adapter = StubAdapter(make_fake("healthy"), busy=False, fractional=True)
    result = run_preflight(adapter)
    assert result.mode == "measurements_only"
    assert any("fractional" in r for r in result.reasons)


def test_busy_unknown_stays_full_with_reason():
    adapter = StubAdapter(make_fake("healthy"), busy=None)
    result = run_preflight(adapter)
    assert result.mode == "full"
    assert result.reasons == ["busy-state unknown"]


def test_idle_healthy_card_is_full_with_no_reasons():
    result = run_preflight(make_fake("healthy"))
    assert result == Preflight(mode="full", reasons=[], missing_telemetry=["fan_pct"])


def test_missing_telemetry_excludes_achieved_tflops():
    result = run_preflight(make_fake("healthy"))
    assert "achieved_tflops" not in result.missing_telemetry
    assert "fan_pct" in result.missing_telemetry


def test_reasons_follow_evaluation_order():
    adapter = StubAdapter(make_fake("healthy"), busy=None, mig_enabled=True)
    result = run_preflight(adapter)
    assert result.mode == "measurements_only"
    assert len(result.reasons) == 2
    assert "MIG" in result.reasons[0]
    assert result.reasons[1] == "busy-state unknown"
