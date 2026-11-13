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


class BrokenStopLoad(RecordingLoad):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def stop(self):
        super().stop()
        raise RuntimeError(self.message)


def test_failing_load_cleanup_is_reported_and_run_is_kept(capsys):
    load = BrokenStopLoad("CUDA error:\n  an illegal   memory access")
    r = run(make_fake("healthy"), load)
    assert len(r.load) == 10 and r.aborted is None
    assert capsys.readouterr().err == (
        "warning: load cleanup failed (RuntimeError: CUDA error: an illegal memory access)\n"
    )


def test_load_cleanup_warning_is_at_most_200_characters(capsys):
    r = run(make_fake("healthy"), BrokenStopLoad("x " * 500))
    assert r.aborted is None
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    inner = err[len("warning: load cleanup failed ("):-len(")\n")]
    assert len(inner) == 200 and inner.startswith("RuntimeError: x x")


class _Unprintable(Exception):
    def __str__(self):
        raise RuntimeError("cannot render")


class _UnprintableStop(RecordingLoad):
    def stop(self):
        raise _Unprintable()


class _BrokenStderr:
    def write(self, text):
        raise OSError("stderr is closed")

    def flush(self):
        raise OSError("stderr is closed")


def test_cleanup_warning_with_an_unprintable_error_keeps_the_run():
    r = run(make_fake("healthy"), _UnprintableStop())
    assert len(r.load) == 10 and r.aborted is None


def test_cleanup_warning_on_a_broken_stderr_keeps_the_run(monkeypatch):
    monkeypatch.setattr("sys.stderr", _BrokenStderr())
    r = run(make_fake("healthy"), BrokenStopLoad("CUDA error"))
    assert len(r.load) == 10 and r.aborted is None


def _event_run(config, sample=None):
    """Run with every sample, tick and sleep appended, in order, to one list."""
    events = []

    class Load:
        def start(self):
            events.append("start")

        def tick(self):
            events.append("tick")
            return 42.0

        def stop(self):
            events.append("stop")

    def fn(i, base):
        events.append("sample")
        return sample(i, base) if sample else base.sample()

    r = execute_run(
        ScriptedAdapter(fn), Load(), config, SPEC,
        clock=FakeClock(), sleep=lambda s: events.append("sleep"),
    )
    return r, events


def test_load_samples_are_taken_right_after_the_load_burst_not_after_the_pause():
    # Found on a real A100: a sample taken after the pause between bursts reads idle power and a
    # cooler card, so each load-phase sample must follow its burst with no sleep in between.
    _, events = _event_run(RunConfig(baseline_s=1, load_s=3))
    assert events[events.index("start"):] == [
        "start",
        "tick", "sample", "sleep",
        "tick", "sample", "sleep",
        "tick", "sample", "sleep",
        "stop",
    ]


def test_baseline_samples_are_still_one_per_period_with_no_load():
    _, events = _event_run(RunConfig(baseline_s=2, load_s=1))
    # the first sample is preflight's
    assert events[: events.index("start")] == ["sample", "sample", "sleep", "sample", "sleep"]


class LinkAdapter(ScriptedAdapter):
    """A card whose PCIe link is slow at idle (what inventory() sees) and faster while loaded."""

    def __init__(self, links, idle=(1, 8)):
        super().__init__(lambda i, base: base.sample())
        self._links = list(links)
        self._idle = idle
        self.link_reads = 0

    def inventory(self):
        return replace(
            self._base.inventory(),
            pcie_gen_current=self._idle[0], pcie_width_current=self._idle[1], pcie_gen_max=4, pcie_width_max=16,
        )

    def pcie_link(self):
        link = self._links[min(self.link_reads, len(self._links) - 1)]
        self.link_reads += 1
        if isinstance(link, Exception):
            raise link
        return link


def _link(inventory):
    return (inventory.pcie_gen_current, inventory.pcie_width_current, inventory.pcie_link_under_load)


def test_pcie_link_is_read_under_load_and_replaces_the_idle_reading():
    r = run(LinkAdapter([(4, 16)]), load=RecordingLoad())
    assert _link(r.inventory) == (4, 16, True)
    assert (r.inventory.pcie_gen_max, r.inventory.pcie_width_max) == (4, 16)


def test_pcie_link_under_load_keeps_the_best_speed_and_width_seen():
    r = run(LinkAdapter([(2, 8), (4, 8), (3, 8)]), load=RecordingLoad())
    assert _link(r.inventory) == (4, 8, True)


def test_pcie_link_is_left_as_read_at_idle_when_no_load_ran():
    adapter = LinkAdapter([(4, 16)])
    r = run(adapter)  # NullLoad: tick() returns None
    assert _link(r.inventory) == (1, 8, None) and adapter.link_reads == 0


def test_pcie_link_read_that_fails_or_is_unknown_leaves_the_idle_reading():
    assert _link(run(LinkAdapter([RuntimeError("nvml")]), load=RecordingLoad()).inventory) == (1, 8, None)
    assert _link(run(LinkAdapter([(None, 16)]), load=RecordingLoad()).inventory) == (1, 8, None)


def test_adapter_without_a_link_reader_still_runs_under_load():
    r = run(make_fake("healthy"), load=RecordingLoad())
    assert r.aborted is None and r.inventory.pcie_link_under_load is None
