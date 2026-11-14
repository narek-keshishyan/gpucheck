"""Burn-in: a load of hours, stored as summaries, with its own drift check."""

import json
import signal

import pytest

from gpucheck import cli
from gpucheck.adapter import TelemetrySample
from gpucheck.checks import MAX_STORED_LOAD_SAMPLES, burn_in_report, thin_load
from gpucheck.cli import main
from gpucheck.fakes import make_fakes
from gpucheck.report import NOT_COVERED
from gpucheck.resultjson import canonical_dumps


def sample(t, tflops=950.0, **kw):
    fields = dict(t_s=float(t), temp_c=60.0, power_w=650.0, sm_clock_mhz=1980.0, fan_pct=None,
                  throttle_reasons=frozenset(), ecc_volatile=0, achieved_tflops=tflops)
    fields.update(kw)
    return TelemetrySample(**fields)


def stored(t, **kw):
    d = dict(t_s=float(t), temp_c=60.0, power_w=650.0, sm_clock_mhz=1980.0, fan_pct=None,
             throttle_reasons=[], ecc_volatile=0, achieved_tflops=950.0)
    d.update(kw)
    return d


# ---- how a long load is stored ------------------------------------------------------------


def test_a_load_of_up_to_an_hour_is_stored_sample_for_sample():
    load = [stored(i) for i in range(MAX_STORED_LOAD_SAMPLES)]
    out, bucket = thin_load(load)
    assert out is load and bucket == 1


@pytest.mark.parametrize("hours,bucket", [(1.5, 2), (2, 2), (24, 24), (72, 72)])
def test_a_longer_load_is_stored_as_at_most_3600_summaries(hours, bucket):
    n = round(hours * 3600)
    out, got = thin_load([stored(i) for i in range(n)])
    assert got == bucket and len(out) == -(-n // bucket) <= MAX_STORED_LOAD_SAMPLES
    assert out[0]["t_s"] == bucket - 1 and out[-1]["t_s"] == n - 1  # each summary ends its span


def test_a_summary_keeps_the_worst_of_what_it_covers():
    group = [
        stored(0, temp_c=60.0, sm_clock_mhz=1980.0, power_w=600.0, achieved_tflops=900.0),
        stored(1, temp_c=91.0, sm_clock_mhz=1200.0, power_w=700.0, achieved_tflops=1000.0,
               throttle_reasons=["thermal"], ecc_volatile=3, fan_pct=80.0),
        stored(2, temp_c=None, sm_clock_mhz=None, power_w=None, achieved_tflops=None, throttle_reasons=None),
    ]
    (s,), bucket = thin_load(group, limit=1)
    assert bucket == 3
    assert s == {"t_s": 2.0, "temp_c": 91.0, "sm_clock_mhz": 1200.0, "power_w": 650.0, "fan_pct": 80.0,
                 "throttle_reasons": ["thermal"], "ecc_volatile": 3, "achieved_tflops": 950.0}


def test_a_span_with_no_reading_stays_unknown_not_zero():
    (s,), _ = thin_load([stored(i, temp_c=None, throttle_reasons=None, achieved_tflops=None) for i in range(4)], limit=1)
    assert s["temp_c"] is None and s["throttle_reasons"] is None and s["achieved_tflops"] is None


def test_a_brief_throttle_in_a_day_long_run_is_still_in_the_stored_series():
    load = [stored(i) for i in range(86400)]
    load[40000] = stored(40000, temp_c=95.0, sm_clock_mhz=900.0, throttle_reasons=["thermal"])
    out, _ = thin_load(load)
    hit = [s for s in out if s["throttle_reasons"] == ["thermal"]]
    assert len(hit) == 1 and hit[0]["temp_c"] == 95.0 and hit[0]["sm_clock_mhz"] == 900.0


# ---- the burn-in entry --------------------------------------------------------------------


def test_only_a_load_of_an_hour_or_more_is_a_burn_in():
    load = [sample(i) for i in range(1800)]
    assert burn_in_report(load, 1800, 1.0) is None and burn_in_report([], 7200, 1.0) is None


def test_early_throughput_is_read_where_a_standard_run_reads_it():
    load = [sample(i, tflops=100.0 if 480 <= i < 600 else 500.0) for i in range(7200)]
    r = burn_in_report(load, 7200, 1.0)
    assert r == {"planned_s": 7200, "completed_s": 7200.0, "early_tflops": 100.0, "late_tflops": 500.0}


def test_a_burn_in_stopped_early_reports_how_long_it_really_ran():
    r = burn_in_report([sample(i) for i in range(500)], 86400, 1.0)
    assert r["completed_s"] == 500.0 and r["early_tflops"] is None and r["late_tflops"] == 950.0


# ---- from the command line ----------------------------------------------------------------


def burn(tmp_path, fake, hours, *flags, name="r.json"):
    out = tmp_path / name
    rc = main(["--fake", fake, "--offline", "--burn-in", str(hours), "--json", str(out), *flags])
    return rc, json.loads(out.read_text())


def codes(result):
    return [f["code"] for f in result["findings"]]


def test_burn_in_passing_card(tmp_path, capsys):
    rc, result = burn(tmp_path, "healthy", 24)
    assert rc == 0 and codes(result) == []
    assert result["config"] == {"baseline_s": 60, "load_s": 86400, "hz": 1.0}
    assert result["checks"]["burn_in"] == {
        "planned_s": 86400, "completed_s": 86400.0, "early_tflops": 950.0, "late_tflops": 950.0}
    assert len(result["samples"]["load"]) == 3600 and result["samples"]["load_bucket_s"] == 24.0
    assert len(result["samples"]["baseline"]) == 60
    out = capsys.readouterr().out
    assert "Burn-in: the load ran for 24 h 0 min; throughput over its final part was 100.0% of its first ten minutes" in out
    assert "faults that appear only after more than 24 hours of use" in out
    assert "Stored samples: each load sample in the result file summarises 24 s" in out


def test_burn_in_failing_card_is_fine_in_a_standard_run(tmp_path, capsys):
    out = tmp_path / "std.json"
    assert main(["--fake", "sag", "--offline", "--json", str(out)]) == 0
    assert codes(json.loads(out.read_text())) == []  # ten minutes do not show it
    rc, result = burn(tmp_path, "sag", 2)
    assert rc == 0 and "THROUGHPUT_FELL" in codes(result)
    finding = next(f for f in result["findings"] if f["code"] == "THROUGHPUT_FELL")
    assert finding["severity"] == "warn" and "15.8% lower than in its first ten minutes" in finding["message"]


def test_measurements_come_from_every_sample_not_from_the_stored_summaries(tmp_path):
    _, result = burn(tmp_path, "sag", 2)
    assert result["metrics"]["sustained_tflops"] == 800.0  # the final 20% of all 7200 samples
    assert result["samples"]["load_bucket_s"] == 2.0 and len(result["samples"]["load"]) == 3600


def test_a_three_day_burn_in_fits_one_upload_and_sixteen_fit_a_lot(tmp_path):
    out = tmp_path / "r.json"
    main(["--fake", "healthy", "--offline", "--burn-in", "72", "--json", str(out),
          "--driver-errors", "--memtest", "--answer-check"])
    size = len(canonical_dumps(json.loads(out.read_text())).encode())
    assert size < 2 * 1024 * 1024 * 0.5  # the server takes 2 MB a run
    assert 16 * size < 16 * 1024 * 1024 * 0.75  # and 16 MB a lot


def test_a_standard_and_a_long_run_are_stored_exactly_as_before(tmp_path):
    for flags in ([], ["--long"]):
        out = tmp_path / "r.json"
        main(["--fake", "healthy", "--offline", "--json", str(out), *flags])
        result = json.loads(out.read_text())
        assert "checks" not in result and "load_bucket_s" not in result["samples"]
        assert len(result["samples"]["load"]) == result["config"]["load_s"]


def test_burn_in_prints_a_progress_line_every_ten_minutes(tmp_path, capsys):
    burn(tmp_path, "healthy", 1)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("burn-in:")]
    assert len(lines) == 5  # at 10, 20, 30, 40 and 50 minutes; the end needs no line
    assert lines[0] == "burn-in: 0 h 10 min of 1 h 0 min of load done, 62 C, 950.0 TFLOPS"


def test_a_plain_run_prints_no_progress_line(tmp_path, capsys):
    main(["--fake", "healthy", "--offline", "--long", "--json", str(tmp_path / "r.json")])
    assert "burn-in" not in capsys.readouterr().out.lower().replace("burn-in:", "burn-in").split("not covered")[0]


@pytest.mark.parametrize("value", ["0.5", "0", "-3", "73", "nan", "inf"])
def test_burn_in_hours_must_be_in_range(value, capsys):
    with pytest.raises(SystemExit):
        main(["--fake", "healthy", "--offline", "--burn-in", value])
    assert "--burn-in must be from 1 to 72 hours" in capsys.readouterr().err


@pytest.mark.parametrize("other", ["--quick", "--long"])
def test_burn_in_excludes_the_other_lengths(other, capsys):
    with pytest.raises(SystemExit):
        main(["--fake", "healthy", "--offline", "--burn-in", "2", other])
    assert "not allowed with argument" in capsys.readouterr().err


def test_burn_in_cannot_be_certified(capsys):
    with pytest.raises(SystemExit):
        main(["--fake", "healthy", "--burn-in", "2", "--certify"])
    assert "--certify cannot be combined with --burn-in" in capsys.readouterr().err


def test_fractional_hours_are_allowed(tmp_path):
    _, result = burn(tmp_path, "healthy", 1.5)
    assert result["config"]["load_s"] == 5400 and result["samples"]["load_bucket_s"] == 2.0


class _InterruptAt:
    """A simulated card on which Ctrl-C is pressed when sample `n` is taken."""

    def __init__(self, n, times=1):
        self._card = make_fakes("healthy")[0]
        self._n, self._times, self.calls = n, times, 0
        self.inventory, self.busy, self.extras = self._card.inventory, self._card.busy, {}

    def sample(self):
        self.calls += 1
        if self.calls == self._n:
            for _ in range(self._times):  # a second Ctrl-C arrives before the run reaches its next sample
                signal.raise_signal(signal.SIGINT)
        return self._card.sample()


def test_ctrl_c_during_a_burn_in_keeps_what_was_measured(tmp_path, monkeypatch, capsys):
    card = _InterruptAt(5000)
    monkeypatch.setattr(cli, "make_fakes", lambda spec: [card])
    before = signal.getsignal(signal.SIGINT)
    out = tmp_path / "r.json"
    rc = main(["--fake", "healthy", "--burn-in", "24", "--json", str(out), "--yes-upload"])
    assert rc == 130 and signal.getsignal(signal.SIGINT) is before
    result = json.loads(out.read_text())
    assert result["aborted"] == "interrupted"
    assert 4900 < result["checks"]["burn_in"]["completed_s"] < 5000
    captured = capsys.readouterr()
    assert "stopping at the next sample and keeping what was measured" in captured.err
    assert "Interrupted: not uploaded." in captured.out and "Run aborted early (the run was interrupted)" in captured.out
    assert "faults that appear only after more than 1 hour of use" in captured.out  # what it really ran


def test_a_second_ctrl_c_quits_at_once_without_a_result(tmp_path, monkeypatch):
    card = _InterruptAt(5000, times=2)
    monkeypatch.setattr(cli, "make_fakes", lambda spec: [card])
    before = signal.getsignal(signal.SIGINT)
    out = tmp_path / "r.json"
    with pytest.raises(KeyboardInterrupt):
        main(["--fake", "healthy", "--offline", "--burn-in", "24", "--json", str(out)])
    assert not out.exists() and signal.getsignal(signal.SIGINT) is before


def test_ctrl_c_in_a_plain_run_is_unchanged(tmp_path, monkeypatch):
    card = _InterruptAt(100)
    monkeypatch.setattr(cli, "make_fakes", lambda spec: [card])
    out = tmp_path / "r.json"
    with pytest.raises(KeyboardInterrupt):
        main(["--fake", "healthy", "--offline", "--json", str(out)])
    assert not out.exists()


def test_a_burn_in_lot_summarises_every_card_and_says_what_no_card_covered(tmp_path, capsys):
    d = tmp_path / "lot"
    rc = main(["--fake", "healthy,sag", "--offline", "--burn-in", "1", "--lot-dir", str(d)])
    assert rc == 0
    summary = json.loads((d / "lot.json").read_text())
    assert [c["findings"] for c in summary["cards"]] == [[], ["UNDERPERFORMING_UNKNOWN", "THROUGHPUT_FELL"]]
    out = capsys.readouterr().out
    assert "GPU 1 burn-in: 0 h 10 min of 1 h 0 min of load done" in out
    assert NOT_COVERED not in out.split("Lot summary")[1]
    assert "No run listed here tests memory cells, silent data corruption, links between GPUs, or physical damage." in out
