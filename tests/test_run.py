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


def test_no_watchdog_without_spec():
    r = run(ScriptedAdapter(lambda i, b: replace(b.sample(), temp_c=120.0)), spec=None)
    assert r.aborted is None and len(r.load) == 10


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
    assert bad[0].power_w is None and bad[0].throttle_reasons == frozenset()
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
