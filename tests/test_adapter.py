import pytest

from gpucheck.adapter import AdapterUnavailable, NvmlAdapter
from gpucheck.fakes import make_fake


def test_fake_healthy_profile_samples_monotonic_time():
    a = make_fake("healthy")
    assert a.inventory().model_name == "NVIDIA H100 80GB HBM3"
    s1, s2 = a.sample(), a.sample()
    assert s2.t_s > s1.t_s and s1.throttle_reasons == frozenset()


def test_nvml_adapter_raises_adapter_unavailable_without_gpu():
    with pytest.raises(AdapterUnavailable):
        NvmlAdapter()  # this dev machine has no NVIDIA driver


def test_tim_degraded_heats_and_throttles():
    a = make_fake("tim_degraded")
    samples = [a.sample() for _ in range(300)]
    assert samples[0].temp_c < 40
    assert max(s.temp_c for s in samples[:120]) >= 87
    assert samples[-1].temp_c == 87 and "thermal" in samples[-1].throttle_reasons
    for s in samples:
        assert ("thermal" in s.throttle_reasons) == (s.temp_c > 83)
    assert samples[-1].achieved_tflops == pytest.approx(600.0)


def test_healthy_steady_state_indefinitely():
    a = make_fake("healthy")
    samples = [a.sample() for _ in range(500)]
    assert samples[-1].achieved_tflops == pytest.approx(950.0)
    assert all(not s.throttle_reasons and s.ecc_volatile == 0 for s in samples)
    assert [s.t_s for s in samples] == sorted(s.t_s for s in samples)


def test_other_profiles():
    assert make_fake("hot_env").sample().temp_c == pytest.approx(55.0, abs=3)
    inv = make_fake("power_capped").inventory()
    assert (inv.power_limit_w, inv.default_power_limit_w) == (400.0, 700.0)
    s = make_fake("power_capped").sample()
    assert "power" in s.throttle_reasons and s.temp_c <= 65
    assert make_fake("mig").inventory().mig_enabled is True
    assert make_fake("healthy").inventory().mig_enabled is False


def test_uuid_hash_formula_and_unknown_profile():
    import hashlib

    inv = make_fake("healthy").inventory()
    assert len(inv.uuid_hash) == 16
    assert all(
        make_fake(p).inventory().uuid_hash
        == hashlib.sha256(("gpucheck.v1." + "GPU-00000000-0000-0000-0000-000000000000").encode()).hexdigest()[:16]
        for p in ("healthy", "mig")
    )
    with pytest.raises(ValueError):
        make_fake("nope")
