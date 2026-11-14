"""The run engine's opt-in extras: the driver event log, the check phases after the load, and
the hooks the all-cards-at-once mode uses. A plain run must not touch any of them."""

from dataclasses import replace

import pytest

from gpucheck.checks import CheckPlan, CheckUnavailable
from gpucheck.fakes import FakeCheck, FakeEventLog, fake_answer, fake_memtest, make_fake
from gpucheck.run import INTERRUPTED, NullLoad, RunConfig, execute_run
from gpucheck.specs import lookup

SPEC = lookup("NVIDIA H100 80GB HBM3")


def run(adapter, **kw):
    config = kw.pop("config", RunConfig(baseline_s=5, load_s=100))
    return execute_run(adapter, kw.pop("load", NullLoad()), config, SPEC,
                       clock=lambda: 0.0, sleep=lambda s: None, **kw)


class Spy:
    """Wraps a fake card and fails the test if a plain run touches an extra."""

    def __init__(self, base="healthy"):
        self._base = make_fake(base)
        self.touched = []

    def inventory(self):
        return self._base.inventory()

    def sample(self):
        return self._base.sample()

    def busy(self):
        return False

    def open_event_log(self):
        self.touched.append("open_event_log")
        return self._base.open_event_log()

    def nvlink(self):
        self.touched.append("nvlink")
        return self._base.nvlink()


def test_a_plain_run_touches_no_extra_and_reports_no_checks():
    spy = Spy("xid")
    r = run(spy)
    assert spy.touched == [] and r.checks == {} and r.aborted is None


def test_driver_events_under_load_are_recorded_with_their_phase():
    r = run(make_fake("xid"), plan=CheckPlan(driver_errors=True))
    log = r.checks["driver_errors"]
    assert log["status"] == "ran" and log["count"] == 2
    assert [(e["kind"], e.get("xid"), e["phase"]) for e in log["events"]] == [
        ("xid", 79, "load"), ("double_bit_ecc", None, "load")]
    assert log["events"][0]["t_s"] == r.load[90 - 5].t_s  # the sample at which it was heard


def test_a_quiet_log_reports_no_events():
    r = run(make_fake("healthy"), plan=CheckPlan(driver_errors=True))
    assert r.checks == {"driver_errors": {"status": "ran", "count": 0, "events": []}}


def test_a_card_whose_driver_offers_no_log_says_unavailable_and_still_runs():
    r = run(make_fake("no_event_log"), plan=CheckPlan(driver_errors=True))
    assert r.aborted is None and len(r.load) == 100
    assert r.checks["driver_errors"] == {
        "status": "unavailable", "count": 0, "events": [],
        "reason": "the driver reports no error events for this card",
    }


def test_an_adapter_without_an_event_log_says_unavailable():
    class Bare:
        def __init__(self):
            self._b = make_fake("healthy")
            self.inventory, self.sample, self.busy = self._b.inventory, self._b.sample, self._b.busy

    r = run(Bare(), plan=CheckPlan(driver_errors=True))
    assert r.checks["driver_errors"]["status"] == "unavailable"


class _Card:
    def __init__(self, log, base="healthy"):
        self._b = make_fake(base)
        self.inventory, self.sample, self.busy = self._b.inventory, self._b.sample, self._b.busy
        self._log = log

    def open_event_log(self):
        return self._log


def test_a_log_that_breaks_partway_keeps_what_it_heard_and_says_partial():
    class Breaking(FakeEventLog):
        def poll(self):
            if self._polls == 50:
                raise RuntimeError("GPU is lost\nmore")
            return super().poll()

    log = Breaking({10: [{"kind": "xid", "xid": 31}]})
    r = run(_Card(log), plan=CheckPlan(driver_errors=True))
    out = r.checks["driver_errors"]
    assert out["status"] == "partial" and out["count"] == 1 and out["reason"] == "RuntimeError: GPU is lost more"
    assert log.closed and r.aborted is None and len(r.load) == 100


def test_the_log_is_closed_when_the_run_ends_and_when_it_is_interrupted():
    log = FakeEventLog({})
    run(_Card(log), plan=CheckPlan(driver_errors=True))
    assert log.closed

    class Interrupting(FakeEventLog):
        def poll(self):
            raise KeyboardInterrupt

    log = Interrupting({})
    with pytest.raises(KeyboardInterrupt):
        run(_Card(log), plan=CheckPlan(driver_errors=True))
    assert log.closed


def test_events_at_the_very_end_are_not_lost():
    log = FakeEventLog({105: [{"kind": "xid", "xid": 48}]})  # the poll after the last sample
    r = run(_Card(log), plan=CheckPlan(driver_errors=True))
    assert [e["phase"] for e in r.checks["driver_errors"]["events"]] == ["end"]


def test_no_more_than_200_events_are_stored_but_all_are_counted():
    log = FakeEventLog({i: [{"kind": "xid", "xid": 13}] * 5 for i in range(100)})
    out = run(_Card(log), plan=CheckPlan(driver_errors=True)).checks["driver_errors"]
    assert out["count"] == 500 and len(out["events"]) == 200


def test_a_refused_card_reports_no_checks():
    busy = make_fake("busy")
    r = run(busy, plan=CheckPlan(driver_errors=True, links=True), checks=[fake_memtest({})])
    assert r.aborted == "preflight_refused" and r.checks == {}


# ---- check phases after the load ----------------------------------------------------------


class Recorder(FakeCheck):
    def __init__(self, name, log, **kw):
        super().__init__(name, {"n": 1}, **kw)
        self._log = log

    def start(self):
        self._log.append(f"{self.name}:start")
        super().start()

    def stop(self):
        self._log.append(f"{self.name}:stop")
        super().stop()


class LoggingLoad(NullLoad):
    def __init__(self, log):
        self._log = log

    def start(self):
        self._log.append("load:start")

    def stop(self):
        self._log.append("load:stop")


def test_checks_run_after_the_load_has_released_its_memory_in_the_order_given():
    log = []
    r = run(make_fake("healthy"), load=LoggingLoad(log),
            checks=[Recorder("answer", log), Recorder("memtest", log)])
    assert log == ["load:start", "load:stop", "answer:start", "answer:stop", "memtest:start", "memtest:stop"]
    assert list(r.checks) == ["answer", "memtest"]
    assert r.checks["answer"] == {"status": "ran", "n": 1, "ecc_delta": 0, "max_temp_c": 62.5}
    assert len(r.load) == 100  # the phases add no load samples


def test_a_check_that_cannot_run_is_recorded_and_the_next_one_still_runs():
    a = FakeCheck("memtest", {}, unavailable="not enough free memory\nto test")
    b = FakeCheck("answer", {"products": 5})
    r = run(make_fake("healthy"), checks=[a, b])
    assert r.checks["memtest"] == {"status": "unavailable", "reason": "not enough free memory to test"}
    assert r.checks["answer"]["status"] == "ran" and a.stopped and b.stopped and r.aborted is None


def test_a_check_that_crashes_keeps_the_run_and_what_it_had_already_found():
    class Crashing(FakeCheck):
        def step(self):
            if self._left == 1:
                raise RuntimeError("CUDA error: device-side assert")
            return super().step()

    c = Crashing("memtest", {"mismatched_words": 7, "bytes_tested": 1 << 30})
    r = run(make_fake("healthy"), checks=[c])
    out = r.checks["memtest"]
    assert out["status"] == "partial" and out["reason"] == "RuntimeError: CUDA error: device-side assert"
    assert out["mismatched_words"] == 7 and out["ecc_delta"] == 0  # not turned into "did not run"
    assert c.stopped and len(r.load) == 100 and r.aborted is None


def test_a_check_that_cannot_even_start_is_not_run():
    class NoStart(FakeCheck):
        def start(self):
            raise RuntimeError("CUDA out of memory")

    r = run(make_fake("healthy"), checks=[NoStart("memtest", {"mismatched_words": 7})])
    assert r.checks["memtest"] == {"status": "unavailable", "reason": "RuntimeError: CUDA out of memory"}


def test_a_check_whose_cleanup_fails_warns_and_keeps_its_result(capsys):
    class BadStop(FakeCheck):
        def stop(self):
            raise RuntimeError("free failed")

    r = run(make_fake("healthy"), checks=[BadStop("answer", {"products": 3})])
    assert r.checks["answer"]["products"] == 3
    assert "load cleanup failed" in capsys.readouterr().err


class Scripted:
    def __init__(self, fn, base="healthy"):
        self._b = make_fake(base)
        self._fn = fn
        self.calls = 0
        self.inventory, self.busy = self._b.inventory, self._b.busy

    def sample(self):
        i = self.calls
        self.calls += 1
        return self._fn(i, self._b.sample())


def test_the_temperature_limit_still_stops_the_run_during_a_check():
    # 1 preflight + 5 baseline + 100 load samples, then the check's samples run hot.
    hot = Scripted(lambda i, s: replace(s, temp_c=120.0) if i >= 106 else s)
    first, second = FakeCheck("answer", {"products": 9}, steps=50), FakeCheck("memtest", {})
    r = run(hot, checks=[first, second])
    assert r.aborted == "thermal"
    assert r.checks["answer"]["status"] == "partial" and r.checks["answer"]["max_temp_c"] == 120.0
    assert "memtest" not in r.checks and first.stopped  # the check after it never started


def test_memory_errors_counted_during_a_check_are_recorded_with_it():
    rising = Scripted(lambda i, s: replace(s, ecc_volatile=max(0, i - 106)))
    r = run(rising, checks=[FakeCheck("memtest", {}, steps=4)])
    assert r.checks["memtest"]["ecc_delta"] == 3


def test_checks_do_not_run_after_an_aborted_load():
    c = FakeCheck("memtest", {})
    r = run(make_fake("overheat"), checks=[c], plan=CheckPlan(driver_errors=True))
    assert r.aborted == "thermal" and "memtest" not in r.checks
    assert r.checks["driver_errors"]["status"] == "ran"  # listening still ended cleanly


# ---- hooks ---------------------------------------------------------------------------------


def test_before_load_runs_once_between_the_baseline_and_the_load():
    seen = []
    adapter = Scripted(lambda i, s: s)
    run(adapter, before_load=lambda: seen.append(adapter.calls))
    assert seen == [6]  # preflight + 5 baseline samples, no load sample yet


def test_should_stop_ends_the_run_as_interrupted_and_stops_the_load():
    log = []
    n = iter(range(1000))
    r = run(make_fake("healthy"), load=LoggingLoad(log), should_stop=lambda: next(n) >= 20)
    assert r.aborted == INTERRUPTED and log == ["load:start", "load:stop"]
    assert len(r.baseline) == 5 and len(r.load) == 16


def test_progress_is_called_for_every_load_sample():
    seen = []
    run(make_fake("healthy"), progress=lambda i, n, s: seen.append((i, n, s.achieved_tflops)))
    assert len(seen) == 100 and seen[0][:2] == (0, 100) and seen[-1][0] == 99


def test_links_are_read_before_and_after_and_a_card_without_them_reports_nothing_checked():
    assert run(make_fake("healthy"), plan=CheckPlan(links=True)).checks == {"nvlink": {"status": "no_links"}}
    out = run(make_fake("nvlink_errors"), plan=CheckPlan(links=True)).checks["nvlink"]
    assert out == {"status": "ran", "links": 4, "active": 3, "dropped": 1,
                   "errors": {"replay": 5, "recovery": 1, "crc_flit": 12, "crc_data": 0}}
    assert run(make_fake("nvlink_idle"), plan=CheckPlan(links=True)).checks["nvlink"] == {"status": "no_links", "links": 3}
