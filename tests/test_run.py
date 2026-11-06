import pytest
from dataclasses import replace

from gpucheck.fakes import make_fake
from gpucheck.run import NullLoad, RawRun, RunConfig, execute_run
from gpucheck.specs import lookup

SPEC = lookup("NVIDIA H100 80GB HBM3")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.5
        return self.now


def run(adapter, load=None, config=None, spec=SPEC):
    return execute_run(
        adapter,
        load or NullLoad(),
        config or RunConfig(baseline_s=5, load_s=10),
        spec,
        clock=FakeClock(),
        sleep=lambda s: None,
    )


class ScriptedAdapter:
    """Delegates inventory/busy to a fake; sample() is overridden by `fn(call_index)`."""

    def __init__(self, fn, base="healthy"):
        self._base = make_fake(base)
        self._fn = fn
        self.calls = 0

    def inventory(self):
        return self._base.inventory()

    def busy(self):
        return False

    def sample(self):
        i = self.calls
        self.calls += 1
        return self._fn(i, self._base)


class RecordingLoad:
    def __init__(self):
        self.events = []

    def start(self):
        self.events.append("start")

    def tick(self):
        self.events.append("tick")
        return 42.0

    def stop(self):
        self.events.append("stop")


def test_collects_expected_sample_counts():
    r = run(make_fake("healthy"))
    assert isinstance(r, RawRun)
    assert len(r.baseline) == 5 and len(r.load) == 10 and r.aborted is None
    assert r.preflight.mode == "full" and r.config.load_s == 10


def test_load_generator_lifecycle_and_tflops_override():
    load = RecordingLoad()
    r = run(make_fake("healthy"), load)
    assert load.events == ["start"] + ["tick"] * 10 + ["stop"]
    assert all(s.achieved_tflops == 42.0 for s in r.load)


def test_tim_degraded_does_not_abort():
    r = run(make_fake("tim_degraded"), config=RunConfig(baseline_s=5, load_s=120))
    assert r.aborted is None and len(r.load) == 120


def test_watchdog_aborts_on_thermal_runaway():
    def fn(i, base):
        s = base.sample()
        return replace(s, temp_c=40.0 if i < 8 else 90.0)  # past 83 + 5 from call 8 on

    load = RecordingLoad()
    r = run(ScriptedAdapter(fn), load)
    assert r.aborted == "thermal"
    assert 0 < len(r.load) < 10
    assert all(s.temp_c == 90.0 for s in r.load[-3:])
    assert load.events[-1] == "stop"


def test_thermal_needs_three_consecutive():
    def fn(i, base):
        s = base.sample()
        return replace(s, temp_c=90.0 if i % 3 else 40.0)  # never 3 in a row

    assert run(ScriptedAdapter(fn)).aborted is None


def test_unknown_gpu_thermal_falls_back_to_95c():
    r = run(ScriptedAdapter(lambda i, b: replace(b.sample(), temp_c=120.0)), spec=None)
    assert r.aborted == "thermal" and len(r.load) == 3


def test_unknown_gpu_below_95c_does_not_abort():
    r = run(ScriptedAdapter(lambda i, b: replace(b.sample(), temp_c=94.9)), spec=None)
    assert r.aborted is None and len(r.load) == 10


def test_unknown_gpu_ecc_abort_still_runs():
    def fn(i, base):
        return replace(base.sample(), ecc_volatile=0 if i < 9 else 500)

    r = run(ScriptedAdapter(fn), spec=None)
    assert r.aborted == "ecc"


def test_watchdog_aborts_on_ecc_growth():
    def fn(i, base):
        s = base.sample()
        return replace(s, ecc_volatile=0 if i < 9 else 500)

    r = run(ScriptedAdapter(fn))
    assert r.aborted == "ecc"
    assert len(r.load) < 10


def test_sampling_survives_adapter_exceptions():
    def fn(i, base):
        if i in (7, 8):  # call 0 is preflight; these land in the load phase
            raise RuntimeError("nvml hiccup")
        return base.sample()

    r = run(ScriptedAdapter(fn))
    assert r.aborted is None and len(r.load) == 10
    bad = [s for s in r.baseline + r.load if s.temp_c is None]
    assert len(bad) == 2
    assert bad[0].power_w is None and bad[0].throttle_reasons is None  # unknown, not empty
    assert bad[0].ecc_volatile is None and bad[0].achieved_tflops is None


def test_dead_adapter_aborts_telemetry():
    def fn(i, base):
        if i == 0:
            return base.sample()  # preflight succeeds
        raise RuntimeError("gone")

    load = RecordingLoad()
    r = run(ScriptedAdapter(fn), load, RunConfig(baseline_s=100, load_s=100))
    assert r.aborted == "telemetry"
    assert len(r.baseline) == 10 and r.load == []
    assert "start" not in load.events


def test_load_stopped_when_adapter_aborts_run():
    load = RecordingLoad()
    r = run(ScriptedAdapter(lambda i, b: (_ for _ in ()).throw(RuntimeError()) if i else b.sample()), load)
    assert r.aborted == "telemetry"
    assert load.events.count("stop") == load.events.count("start")


class BusyAdapter(ScriptedAdapter):
    def busy(self):
        return True


def test_preflight_refused_runs_nothing():
    load = RecordingLoad()
    r = run(BusyAdapter(lambda i, b: b.sample()), load)
    assert r.preflight.mode == "refuse"
    assert r.aborted == "preflight_refused"
    assert r.baseline == [] and r.load == []
    assert load.events == []


def test_measurements_only_still_runs_load_phase():
    load = RecordingLoad()
    r = run(make_fake("mig"), load)
    assert r.preflight.mode == "measurements_only"
    assert r.aborted is None and len(r.baseline) == 5 and len(r.load) == 10
    assert load.events == ["start"] + ["tick"] * 10 + ["stop"]


class StepClock:
    def __init__(self, step):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def _sleeps(step):
    sleeps = []
    execute_run(
        make_fake("healthy"),
        NullLoad(),
        RunConfig(baseline_s=1, load_s=1, hz=2.0),
        SPEC,
        clock=StepClock(step),
        sleep=sleeps.append,
    )
    return sleeps


def test_pacing_sleeps_follow_schedule_at_2hz():
    # clock reads 0.2 per call: phase start, then one read per iteration; period 0.5
    # baseline start=0.2: 0.2+0.5-0.4=0.3, 0.2+1.0-0.6=0.6; load start=0.8: 0.3, 0.6
    assert _sleeps(0.2) == [pytest.approx(x) for x in (0.3, 0.6, 0.3, 0.6)]


def test_pacing_never_sleeps_negative_when_behind():
    assert _sleeps(10.0) == [0.0, 0.0, 0.0, 0.0]
