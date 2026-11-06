import pytest

from gpucheck.adapter import GpuInventory, TelemetrySample
from gpucheck.metrics import derive
from gpucheck.preflight import Preflight
from gpucheck.run import RawRun, RunConfig
from gpucheck.specs import GpuSpec

INV = GpuInventory(
    model_name="Test GPU",
    uuid_hash="0" * 16,
    driver="550.00",
    cuda="12.4",
    ecc_enabled=True,
    power_limit_w=450.0,
    default_power_limit_w=450.0,
    mig_enabled=False,
    fractional=False,
    retired_pages=0,
    remapped_rows=0,
)
SPEC = GpuSpec("test-spec", 450.0, 19.0, 90.0, 2500.0)


def s(t, temp=None, power=None, clock=None, reasons=(), ecc=None, tflops=None):
    return TelemetrySample(
        t_s=float(t),
        temp_c=temp,
        power_w=power,
        sm_clock_mhz=clock,
        fan_pct=None,
        throttle_reasons=frozenset(reasons),
        ecc_volatile=ecc,
        achieved_tflops=tflops,
    )


def make_run(baseline=(), load=()):
    return RawRun(
        inventory=INV,
        preflight=Preflight(mode="full"),
        baseline=list(baseline),
        load=list(load),
        config=RunConfig(baseline_s=len(baseline), load_s=len(load)),
        aborted=None,
    )


def load_series(n, start=60, **fields_by_index):
    """n load samples at t = start, start+1, ...; fields_by_index maps name -> list of n values."""
    out = []
    for i in range(n):
        kwargs = {k: v[i] for k, v in fields_by_index.items()}
        out.append(s(start + i, **kwargs))
    return out


def test_idle_temp_is_mean_of_baseline():
    baseline = [s(0, temp=40.0), s(1, temp=42.0), s(2, temp=44.0)]
    load = [s(3, temp=70.0), s(4, temp=72.0)]
    m = derive(make_run(baseline, load), SPEC)
    assert m.idle_temp_c == 42.0
    assert "temp_c" not in m.telemetry_gaps


def test_time_to_throttle_is_relative_to_load_start():
    # Baseline t=0..59, load starts at absolute t=60. First thermal flag at t=240 -> 180 s into load.
    baseline = [s(t, temp=40.0) for t in range(60)]
    load = [
        s(60, temp=70.0),
        s(61, temp=72.0, reasons={"power"}),  # power cap alone must not count
        s(180, temp=80.0),
        s(240, temp=85.0, reasons={"thermal"}),
        s(241, temp=86.0, reasons={"thermal", "power"}),
    ]
    m = derive(make_run(baseline, load), SPEC)
    assert m.time_to_throttle_s == 180.0


def test_time_to_throttle_none_without_thermal_flag():
    load = [s(60, temp=70.0, reasons={"power"}), s(61, temp=71.0, reasons={"power"})]
    m = derive(make_run(load=load), SPEC)
    assert m.time_to_throttle_s is None
    assert m.throttle_reasons_seen == ["power"]


def test_throttle_reasons_seen_is_sorted_union_over_load_only():
    baseline = [s(0, reasons={"thermal"})]  # baseline reasons are not load-phase reasons
    load = [
        s(1, reasons={"power", "thermal"}),
        s(2, reasons=set()),
        s(3, reasons={"power"}),
    ]
    m = derive(make_run(baseline, load), SPEC)
    assert m.throttle_reasons_seen == ["power", "thermal"]


def test_all_none_temps_yield_gaps_without_crashing():
    baseline = [s(t) for t in range(3)]
    load = [s(t) for t in range(3, 13)]
    m = derive(make_run(baseline, load), SPEC)
    assert m.idle_temp_c is None
    assert m.steady_temp_c is None
    assert m.time_to_throttle_s is None
    assert "temp_c" in m.telemetry_gaps


def test_empty_load_is_all_none_with_gaps_and_no_crash():
    baseline = [s(t, temp=41.0, ecc=7) for t in range(3)]
    m = derive(make_run(baseline, []), SPEC)
    assert m.idle_temp_c == 41.0
    assert m.time_to_throttle_s is None
    assert m.steady_temp_c is None
    assert m.sustained_tflops is None
    assert m.tflops_ratio_vs_spec is None
    assert m.sm_clock_sigma is None
    assert m.power_sigma_w is None
    assert m.ecc_delta is None
    assert m.throttle_reasons_seen == []
    for gap in ("temp_c", "achieved_tflops", "sm_clock_mhz", "power_w"):
        assert gap in m.telemetry_gaps


def test_empty_baseline_and_load_does_not_crash():
    m = derive(make_run([], []), SPEC)
    assert m.idle_temp_c is None
    assert "temp_c" in m.telemetry_gaps


def test_sustained_window_math_on_hand_computed_series():
    # 10 load samples -> final window = round(0.2 * 10) = 2 samples (indices 8, 9).
    n = 10
    m = derive(
        make_run(load=load_series(n, temp=[float(60 + i) for i in range(n)], tflops=[float(i + 1) for i in range(n)])),
        SPEC,
    )
    assert m.steady_temp_c == pytest.approx((68.0 + 69.0) / 2)  # 68.5
    assert m.sustained_tflops == pytest.approx((9.0 + 10.0) / 2)  # 9.5
    assert m.tflops_ratio_vs_spec == pytest.approx(9.5 / 19.0)  # 0.5


@pytest.mark.parametrize("n, window", [(1, 1), (3, 1), (7, 1), (10, 2), (12, 2), (20, 4)])
def test_final_window_size_is_round_20_percent(n, window):
    m = derive(make_run(load=load_series(n, tflops=[float(i) for i in range(n)])), SPEC)
    expected = sum(range(n - window, n)) / window
    assert m.sustained_tflops == pytest.approx(expected)


def test_window_means_skip_none_values():
    # 10 samples -> window is the last two. tflops [None, 8.0] -> 8.0; temps [None, 70.0] -> 70.0.
    tflops = [None] * 8 + [None, 8.0]
    temps = [None] * 8 + [None, 70.0]
    m = derive(make_run(load=load_series(10, tflops=tflops, temp=temps)), SPEC)
    assert m.sustained_tflops == 8.0
    assert m.steady_temp_c == 70.0
    assert "achieved_tflops" not in m.telemetry_gaps


def test_sustained_all_none_is_gap_and_ratio_none():
    m = derive(make_run(load=load_series(5, tflops=[None] * 5)), SPEC)
    assert m.sustained_tflops is None
    assert m.tflops_ratio_vs_spec is None
    assert "achieved_tflops" in m.telemetry_gaps


def test_ratio_none_without_spec_but_sustained_still_computed():
    m = derive(make_run(load=load_series(5, tflops=[2.0] * 5)), None)
    assert m.sustained_tflops == 2.0
    assert m.tflops_ratio_vs_spec is None


def test_sigma_is_population_stddev_over_load():
    m = derive(
        make_run(load=load_series(2, clock=[1000.0, 1200.0], power=[100.0, 100.0])),
        SPEC,
    )
    assert m.sm_clock_sigma == pytest.approx(100.0)
    assert m.power_sigma_w == pytest.approx(0.0)


def test_sigma_ignores_none_values():
    m = derive(make_run(load=load_series(3, clock=[1000.0, None, 1200.0])), SPEC)
    assert m.sm_clock_sigma == pytest.approx(100.0)


def test_sigma_with_fewer_than_two_values_is_none_and_gap():
    m = derive(make_run(load=load_series(3, clock=[1000.0, None, None], power=[None, None, None])), SPEC)
    assert m.sm_clock_sigma is None
    assert m.power_sigma_w is None
    assert "sm_clock_mhz" in m.telemetry_gaps
    assert "power_w" in m.telemetry_gaps


def test_ecc_delta_with_gaps_in_readings():
    # Combined baseline+load non-None: 100 (baseline, first), 130, 150 (load, last).
    baseline = [s(0, ecc=None), s(1, ecc=100)]
    load = load_series(4, start=2, ecc=[None, 130, None, 150])
    m = derive(make_run(baseline, load), SPEC)
    assert m.ecc_delta == 50
    assert "ecc_volatile" not in m.telemetry_gaps


def test_ecc_delta_none_with_single_reading():
    m = derive(make_run([s(0, ecc=5)], [s(1, ecc=None)]), SPEC)
    assert m.ecc_delta is None
    assert "ecc_volatile" not in m.telemetry_gaps


def test_ecc_delta_none_when_load_has_no_reading():
    baseline = [s(0, ecc=1), s(1, ecc=2)]
    m = derive(make_run(baseline, load_series(2, start=2, ecc=[None, None])), SPEC)
    assert m.ecc_delta is None


def test_ecc_gap_only_when_all_readings_none():
    m = derive(make_run([s(0)], load_series(3, start=1)), SPEC)
    assert m.ecc_delta is None
    assert "ecc_volatile" in m.telemetry_gaps


def test_no_duplicate_gap_entries():
    m = derive(make_run([s(0)], load_series(3, start=1)), SPEC)
    assert len(m.telemetry_gaps) == len(set(m.telemetry_gaps))


def _unknown(t, **kw):
    from dataclasses import replace

    return replace(s(t, **kw), throttle_reasons=None)


def test_unknown_throttle_reasons_are_a_gap_not_empty():
    load = [_unknown(60 + i, temp=70.0, tflops=10.0) for i in range(5)]
    m = derive(make_run([s(0, temp=40.0)], load), SPEC)
    assert "throttle_reasons" in m.telemetry_gaps
    assert m.time_to_throttle_s is None and m.throttle_reasons_seen == []


def test_partially_unknown_throttle_reasons_use_known_samples():
    load = [_unknown(60, temp=70.0), s(61, temp=71.0, reasons=["thermal"])]
    m = derive(make_run([s(0, temp=40.0)], load), SPEC)
    assert "throttle_reasons" not in m.telemetry_gaps
    assert m.time_to_throttle_s == 1.0 and m.throttle_reasons_seen == ["thermal"]
