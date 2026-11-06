import json

import pytest

from gpucheck.adapter import GpuInventory, TelemetrySample
from gpucheck.diagnose import Finding
from gpucheck.metrics import Metrics
from gpucheck.preflight import Preflight
from gpucheck.resultjson import build_result, canonical_dumps, run_id
from gpucheck.run import RawRun, RunConfig

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


def _s(t, reasons=()):
    return TelemetrySample(
        t_s=float(t),
        temp_c=40.123456,
        power_w=100.0,
        sm_clock_mhz=1500.0,
        fan_pct=None,
        throttle_reasons=frozenset(reasons),
        ecc_volatile=0,
        achieved_tflops=None,
    )


sample_run = RawRun(
    inventory=INV,
    preflight=Preflight(mode="full"),
    baseline=[_s(0), _s(1)],
    load=[_s(2, ["thermal", "power"])],
    config=RunConfig(baseline_s=2, load_s=1, hz=1.0),
    aborted=None,
)
sample_metrics = Metrics(
    idle_temp_c=40.12345,
    time_to_throttle_s=None,
    steady_temp_c=None,
    sustained_tflops=None,
    tflops_ratio_vs_spec=None,
    sm_clock_sigma=None,
    power_sigma_w=None,
    ecc_delta=None,
    throttle_reasons_seen=["power", "thermal"],
    telemetry_gaps=["fan_pct"],
)


def test_canonical_dumps_is_order_and_float_stable():
    assert canonical_dumps({"b": 1.00004999, "a": [{"y": 2, "x": 1}]}) == canonical_dumps({"a": [{"x": 1, "y": 2}], "b": 1.0})


def test_build_result_round_trips_and_has_schema_version():
    d = build_result(sample_run, sample_metrics, [])
    assert d["schema_version"] == "1" and json.loads(canonical_dumps(d))


def test_build_result_shape():
    f = Finding("x", "warn", "high", "msg")
    d = build_result(sample_run, sample_metrics, [f])
    assert set(d) == {"schema_version", "inventory", "preflight", "config", "metrics",
                      "findings", "samples", "aborted"}
    assert set(d["samples"]) == {"baseline", "load"}
    assert len(d["samples"]["baseline"]) == 2
    assert d["findings"] == [{"code": "x", "severity": "warn", "confidence": "high", "message": "msg"}]
    assert d["aborted"] is None


def test_canonical_dumps_idempotent():
    s1 = canonical_dumps(build_result(sample_run, sample_metrics, []))
    assert canonical_dumps(json.loads(s1)) == s1


def test_frozenset_becomes_sorted_list_and_compact():
    out = canonical_dumps({"r": frozenset({"thermal", "power", "idle"})})
    assert out == '{"r":["idle","power","thermal"]}'
    d = build_result(sample_run, sample_metrics, [])
    assert d["samples"]["load"][0]["throttle_reasons"] == ["power", "thermal"]


def test_floats_rounded_to_4dp():
    assert canonical_dumps({"a": 40.123456}) == '{"a":40.1235}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(bad):
    with pytest.raises(ValueError):
        canonical_dumps({"a": bad})


def test_run_id_stable_across_key_order():
    a = {"b": 1.00001, "a": {"y": 1, "x": [2, 3]}}
    b = {"a": {"x": [2, 3], "y": 1}, "b": 1.0}
    assert run_id(a) == run_id(b)
    assert len(run_id(a)) == 16
    assert run_id(a) != run_id({"a": 2})


def test_negative_zero_canonicalized():
    assert canonical_dumps({"a": -0.00001}) == canonical_dumps({"a": 0.0})
