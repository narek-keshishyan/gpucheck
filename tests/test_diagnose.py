from gpucheck.adapter import GpuInventory
from gpucheck.diagnose import Finding, diagnose
from gpucheck.metrics import Metrics
from gpucheck.preflight import Preflight
from gpucheck.specs import GpuSpec

SPEC = GpuSpec("test-spec", 450.0, 19.0, 83.0, 2500.0)  # max_temp_c = 83


def inv(power_limit=450.0, default=450.0, remapped=0):
    return GpuInventory(
        model_name="Test GPU",
        uuid_hash="0" * 16,
        driver="550.00",
        cuda="12.4",
        ecc_enabled=True,
        power_limit_w=power_limit,
        default_power_limit_w=default,
        mig_enabled=False,
        fractional=False,
        retired_pages=0,
        remapped_rows=remapped,
    )


def healthy(**overrides):
    base = dict(
        idle_temp_c=35.0,
        time_to_throttle_s=None,
        steady_temp_c=70.0,
        sustained_tflops=18.0,
        tflops_ratio_vs_spec=0.95,
        sm_clock_sigma=10.0,
        power_sigma_w=5.0,
        ecc_delta=0,
        throttle_reasons_seen=[],
        telemetry_gaps=[],
    )
    base.update(overrides)
    return Metrics(**base)


PF = Preflight(mode="full")


def codes(findings):
    return [f.code for f in findings]


def find(findings, code):
    matches = [f for f in findings if f.code == code]
    assert len(matches) == 1, f"expected one {code}, got {codes(findings)}"
    return matches[0]


def tim_metrics(**overrides):
    # Throttled at 120 s, steady 86.5 C (>= max 83 - 3), idle 38 C: TIM fires with high confidence.
    base = dict(
        time_to_throttle_s=120.0,
        steady_temp_c=86.5,
        idle_temp_c=38.0,
        throttle_reasons_seen=["thermal"],
    )
    base.update(overrides)
    return healthy(**base)


# --- healthy card ---------------------------------------------------------


def test_healthy_card_gives_no_findings():
    assert diagnose(healthy(), PF, inv(), SPEC) == []


# --- MEMORY_WEAR (bad, high) ----------------------------------------------


def test_memory_wear_fires_on_ecc_delta_above_zero():
    f = find(diagnose(healthy(ecc_delta=1), PF, inv(), SPEC), "MEMORY_WEAR")
    assert (f.severity, f.confidence) == ("bad", "high")


def test_memory_wear_not_on_ecc_delta_zero():
    assert "MEMORY_WEAR" not in codes(diagnose(healthy(ecc_delta=0), PF, inv(), SPEC))


def test_memory_wear_fires_on_remapped_rows_above_zero():
    f = find(diagnose(healthy(), PF, inv(remapped=1), SPEC), "MEMORY_WEAR")
    assert (f.severity, f.confidence) == ("bad", "high")


def test_memory_wear_not_on_remapped_rows_zero():
    assert "MEMORY_WEAR" not in codes(diagnose(healthy(), PF, inv(remapped=0), SPEC))


def test_memory_wear_runs_without_spec():
    assert "MEMORY_WEAR" in codes(diagnose(healthy(ecc_delta=2), PF, inv(), None))


def test_memory_wear_not_when_inputs_none():
    m = healthy(ecc_delta=None)
    assert "MEMORY_WEAR" not in codes(diagnose(m, PF, inv(remapped=None), SPEC))


# --- TIM_DEGRADATION (bad) and its confidence ----------------------------


def test_tim_fires_at_boundaries_with_medium_confidence():
    # time 179.9 (< 180), steady 80.0 (= max - 3), idle 45.0 (<= 45)
    m = tim_metrics(time_to_throttle_s=179.9, steady_temp_c=80.0, idle_temp_c=45.0)
    f = find(diagnose(m, PF, inv(), SPEC), "TIM_DEGRADATION")
    assert (f.severity, f.confidence) == ("bad", "medium")


def test_tim_high_confidence_at_idle_40():
    m = tim_metrics(idle_temp_c=40.0)
    f = find(diagnose(m, PF, inv(), SPEC), "TIM_DEGRADATION")
    assert f.confidence == "high"


def test_tim_medium_confidence_just_above_idle_40():
    m = tim_metrics(idle_temp_c=40.1)
    f = find(diagnose(m, PF, inv(), SPEC), "TIM_DEGRADATION")
    assert f.confidence == "medium"


def test_tim_not_at_time_to_throttle_180():
    m = tim_metrics(time_to_throttle_s=180.0)
    assert "TIM_DEGRADATION" not in codes(diagnose(m, PF, inv(), SPEC))


def test_tim_not_at_steady_just_below_max_minus_3():
    m = tim_metrics(steady_temp_c=79.9)
    assert "TIM_DEGRADATION" not in codes(diagnose(m, PF, inv(), SPEC))


def test_tim_not_at_idle_45_1_and_emits_insufficient_instead():
    m = tim_metrics(idle_temp_c=45.1)
    out = diagnose(m, PF, inv(), SPEC)
    assert "TIM_DEGRADATION" not in codes(out)
    f = find(out, "INSUFFICIENT_EVIDENCE")
    assert (f.severity, f.confidence) == ("info", "low")


def test_tim_skipped_without_spec():
    m = tim_metrics()
    assert "TIM_DEGRADATION" not in codes(diagnose(m, PF, inv(), None))


def test_tim_message_has_key_numbers():
    f = find(diagnose(tim_metrics(), PF, inv(), SPEC), "TIM_DEGRADATION")
    for token in ("120", "86.5", "83", "38"):
        assert token in f.message


def test_no_tim_claim_without_idle_baseline():
    m = tim_metrics(idle_temp_c=None)
    out = diagnose(m, PF, inv(), SPEC)
    assert "TIM_DEGRADATION" not in codes(out)
    assert "INSUFFICIENT_EVIDENCE" in codes(out)


def test_thermal_reason_alone_counts_as_throttling_for_insufficient():
    m = healthy(
        time_to_throttle_s=None,
        steady_temp_c=85.0,
        idle_temp_c=None,
        throttle_reasons_seen=["thermal"],
    )
    assert "INSUFFICIENT_EVIDENCE" in codes(diagnose(m, PF, inv(), SPEC))


def test_no_insufficient_when_not_throttled():
    m = healthy(idle_temp_c=None)
    assert "INSUFFICIENT_EVIDENCE" not in codes(diagnose(m, PF, inv(), SPEC))


# --- HOT_ENVIRONMENT (warn, high) -----------------------------------------


def test_hot_environment_fires_above_50():
    f = find(diagnose(healthy(idle_temp_c=50.1), PF, inv(), SPEC), "HOT_ENVIRONMENT")
    assert (f.severity, f.confidence) == ("warn", "high")


def test_hot_environment_not_at_50():
    assert "HOT_ENVIRONMENT" not in codes(diagnose(healthy(idle_temp_c=50.0), PF, inv(), SPEC))


def test_hot_environment_not_when_idle_none():
    assert "HOT_ENVIRONMENT" not in codes(diagnose(healthy(idle_temp_c=None), PF, inv(), SPEC))


# --- POWER_CAPPED (info, high) --------------------------------------------


def test_power_capped_limit_arm_fires_below_90_percent_of_default():
    f = find(diagnose(healthy(), PF, inv(power_limit=404.9, default=450.0), SPEC), "POWER_CAPPED")
    assert (f.severity, f.confidence) == ("info", "high")


def test_power_capped_limit_arm_not_at_90_percent_of_default():
    assert "POWER_CAPPED" not in codes(diagnose(healthy(), PF, inv(power_limit=405.0, default=450.0), SPEC))


def test_power_capped_limit_arm_runs_without_spec():
    assert "POWER_CAPPED" in codes(diagnose(healthy(), PF, inv(power_limit=300.0, default=450.0), None))


def test_power_capped_limit_arm_skipped_when_limits_unknown():
    assert "POWER_CAPPED" not in codes(diagnose(healthy(), PF, inv(power_limit=None, default=450.0), SPEC))
    assert "POWER_CAPPED" not in codes(diagnose(healthy(), PF, inv(power_limit=300.0, default=None), SPEC))


def test_power_capped_reason_arm_fires_below_max_minus_15():
    # SPEC max is 83 C, so the reason arm needs steady < 68 C.
    m = healthy(steady_temp_c=67.9, throttle_reasons_seen=["power"])
    f = find(diagnose(m, PF, inv(), SPEC), "POWER_CAPPED")
    assert (f.severity, f.confidence) == ("info", "high")


def test_power_capped_reason_arm_matches_any_power_reason():
    m = healthy(steady_temp_c=60.0, throttle_reasons_seen=["sw_power_cap"])
    assert "POWER_CAPPED" in codes(diagnose(m, PF, inv(), SPEC))


def test_power_capped_reason_arm_not_at_max_minus_15():
    m = healthy(steady_temp_c=68.0, throttle_reasons_seen=["power"])
    assert "POWER_CAPPED" not in codes(diagnose(m, PF, inv(), SPEC))


def test_power_capped_reason_arm_skipped_without_spec():
    m = healthy(steady_temp_c=60.0, throttle_reasons_seen=["power"])
    assert "POWER_CAPPED" not in codes(diagnose(m, PF, inv(), None))


# --- UNDERPERFORMING_UNKNOWN (warn, low) ----------------------------------


def test_underperforming_fires_below_0_9_with_no_throttle_reasons():
    f = find(diagnose(healthy(tflops_ratio_vs_spec=0.899), PF, inv(), SPEC), "UNDERPERFORMING_UNKNOWN")
    assert (f.severity, f.confidence) == ("warn", "low")


def test_underperforming_not_at_0_9():
    assert "UNDERPERFORMING_UNKNOWN" not in codes(diagnose(healthy(tflops_ratio_vs_spec=0.9), PF, inv(), SPEC))


def test_underperforming_not_when_throttle_reasons_seen():
    m = healthy(tflops_ratio_vs_spec=0.5, throttle_reasons_seen=["sw_power_cap"], steady_temp_c=70.0)
    assert "UNDERPERFORMING_UNKNOWN" not in codes(diagnose(m, PF, inv(), SPEC))


def test_underperforming_skipped_without_spec():
    m = healthy(tflops_ratio_vs_spec=0.5)
    assert "UNDERPERFORMING_UNKNOWN" not in codes(diagnose(m, PF, inv(), None))


# --- ordering and multiple findings ---------------------------------------


def test_findings_emitted_in_deterministic_order():
    m = tim_metrics(ecc_delta=1, throttle_reasons_seen=["thermal", "power"], steady_temp_c=86.5)
    out = diagnose(m, PF, inv(power_limit=300.0, default=450.0), SPEC)
    assert codes(out) == ["MEMORY_WEAR", "TIM_DEGRADATION", "POWER_CAPPED"]


def test_insufficient_evidence_comes_last():
    m = healthy(
        idle_temp_c=None,
        time_to_throttle_s=150.0,
        steady_temp_c=85.0,
        ecc_delta=1,
        throttle_reasons_seen=["thermal"],
        tflops_ratio_vs_spec=0.5,
    )
    out = diagnose(m, PF, inv(), SPEC)
    assert codes(out) == ["MEMORY_WEAR", "INSUFFICIENT_EVIDENCE"]


def test_findings_are_frozen_dataclass_instances():
    out = diagnose(healthy(ecc_delta=1), PF, inv(), SPEC)
    assert isinstance(out[0], Finding)
    assert isinstance(out[0].message, str) and out[0].message.endswith(".")
