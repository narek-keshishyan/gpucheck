import json
from dataclasses import replace

from gpucheck import cli
from gpucheck.adapter import AdapterUnavailable
from gpucheck.cli import main
from gpucheck.fakes import make_fake
from gpucheck.report import NOT_COVERED
from gpucheck.run import NullLoad, RunConfig
from gpucheck.upload import UploadOutcome


def test_offline_fake_run_writes_json_and_report(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = main(["--fake", "healthy", "--offline", "--quick", "--json", str(out)])
    assert rc == 0 and out.exists()
    text = capsys.readouterr().out
    for section in ("Inventory", "Measurements", "Findings", "Telemetry gaps"):
        assert section in text
    assert json.loads(out.read_text())["schema_version"] == "1"


def test_no_gpu_is_friendly(capsys):
    assert main(["--offline"]) == 2
    c = capsys.readouterr()
    assert "Traceback" not in c.out + c.err
    assert len((c.out + c.err).strip().splitlines()) == 1


def test_mig_profile_prints_measurements_only_banner(tmp_path, capsys):
    rc = main(["--fake", "mig", "--offline", "--quick", "--json", str(tmp_path / "r.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "measurements only" in out.lower() and "no grade" in out.lower()


def test_tim_profile_report_names_hot_under_load(tmp_path, capsys):
    main(["--fake", "tim_degraded", "--offline", "--quick", "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "HOT_UNDER_LOAD" in out and "runs hot under load" in out


class _Busy:
    def __init__(self):
        self._b = make_fake("healthy")

    def inventory(self):
        return self._b.inventory()

    def sample(self):
        return self._b.sample()

    def busy(self):
        return True


def test_refusal_returns_1_and_prints_reason(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_make_adapter", lambda args: _Busy())
    rc = main(["--fake", "healthy", "--offline", "--quick", "--json", str(tmp_path / "r.json")])
    assert rc == 1
    assert "GPU busy" in capsys.readouterr().out


class _Hot:
    def __init__(self):
        self._b = make_fake("healthy")

    def inventory(self):
        return self._b.inventory()

    def busy(self):
        return False

    def sample(self):
        return replace(self._b.sample(), temp_c=200.0)


def test_watchdog_abort_returns_3_and_writes_partial_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_make_adapter", lambda args: _Hot())
    out = tmp_path / "r.json"
    rc = main(["--fake", "healthy", "--offline", "--quick", "--json", str(out)])
    assert rc == 3
    assert json.loads(out.read_text())["aborted"] == "thermal"
    assert "Inventory" in capsys.readouterr().out


def test_quick_and_long_durations(tmp_path):
    def cfg(*flags):
        out = tmp_path / "r.json"
        main(["--fake", "healthy", "--offline", *flags, "--json", str(out)])
        return json.loads(out.read_text())["config"]

    q = cfg("--quick")
    assert (q["baseline_s"], q["load_s"], q["hz"]) == (30, 150, 1.0)
    d = cfg()
    assert (d["baseline_s"], d["load_s"]) == (60, 600)
    lg = cfg("--long")
    assert (lg["baseline_s"], lg["load_s"]) == (60, 1800)


def test_bad_fake_profile_is_friendly(capsys):
    assert main(["--fake", "nope", "--offline"]) == 2
    c = capsys.readouterr()
    assert "Traceback" not in c.out + c.err
    assert len((c.out + c.err).strip().splitlines()) == 1


def test_unrelated_value_error_propagates(monkeypatch):
    import pytest

    def boom(*a, **k):
        raise ValueError("real bug")

    monkeypatch.setattr(cli, "execute_run", boom)
    with pytest.raises(ValueError):
        main(["--fake", "healthy", "--offline", "--quick"])


def test_report_carries_not_covered_note(tmp_path, capsys):
    main(["--fake", "healthy", "--offline", "--quick", "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert NOT_COVERED in out
    assert "none found by this test" in out


def test_not_covered_text_is_exact():
    assert NOT_COVERED == (
        "This is a pre-flight check, not proof of health. It does not test memory cells, "
        "silent data corruption, faults that appear only after hours of use, links between "
        "GPUs, or physical damage."
    )


def test_grade_line_is_relative_to_reference_median():
    out = UploadOutcome(True, None, {"letter": "B", "ratio": 0.941, "n": 12, "low_confidence": True}, None, None)
    assert cli.format_grade(out, "NVIDIA H100") == (
        "Grade B — 94.1% of the median clean reference run for NVIDIA H100 (n=12, low confidence)"
    )


def test_result_records_start_and_end_times(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    calls = []

    def now():
        calls.append(1)
        return datetime(2026, 11, 10, 12, 0 if len(calls) == 1 else 11, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(cli, "_now", now)
    out = tmp_path / "r.json"
    main(["--fake", "healthy", "--offline", "--quick", "--json", str(out)])
    assert json.loads(out.read_text())["timing"] == {
        "started_utc": "2026-11-10T12:00:00Z", "ended_utc": "2026-11-10T12:11:00Z", "settle_s": None}


def _real_path(monkeypatch, count):
    made = []
    monkeypatch.setattr(cli, "gpu_count", lambda: count)
    monkeypatch.setattr(cli, "NvmlAdapter",
                        lambda index=0, include_serial=False: made.append((index, include_serial)) or make_fake("healthy"))
    monkeypatch.setattr(cli, "_make_load",
                        lambda args, challenge=None, device_index=0: made.append(("load", device_index)) or NullLoad())
    monkeypatch.setattr(cli, "_config", lambda args: RunConfig(baseline_s=2, load_s=2, hz=1000.0))
    return made


def test_gpus_flag_selects_index_and_device(monkeypatch, tmp_path):
    made = _real_path(monkeypatch, 4)
    rc = main(["--gpus", "1", "--include-serial", "--offline", "--json", str(tmp_path / "r.json")])
    assert rc == 0 and made == [(1, True), ("load", 1)]


def test_notice_when_more_gpus_exist(monkeypatch, tmp_path, capsys):
    _real_path(monkeypatch, 4)
    main(["--offline", "--json", str(tmp_path / "r.json")])
    assert "4 GPUs detected; testing GPU 0 only. Use --gpus all to test every card." in capsys.readouterr().out


def test_gpus_all_on_one_gpu_is_a_single_run(monkeypatch, tmp_path):
    made = _real_path(monkeypatch, 1)
    out = tmp_path / "r.json"
    assert main(["--gpus", "all", "--offline", "--json", str(out)]) == 0
    assert out.exists() and made == [(0, False), ("load", 0)]


def test_bad_gpus_value_is_friendly(monkeypatch, capsys):
    _real_path(monkeypatch, 4)
    assert main(["--gpus", "9", "--offline"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("gpucheck: --gpus:") and "Traceback" not in err


def test_zero_gpus_without_gpus_flag_exits_2(monkeypatch, capsys):
    def no_device(index=0, include_serial=False):
        raise AdapterUnavailable("no device")

    monkeypatch.setattr(cli, "gpu_count", lambda: 0)
    monkeypatch.setattr(cli, "NvmlAdapter", no_device)
    assert main(["--offline"]) == 2
    c = capsys.readouterr()
    assert "Traceback" not in c.out + c.err
    assert len((c.out + c.err).strip().splitlines()) == 1


def test_fake_list_with_gpus_runs_the_selected_card(tmp_path):
    out = tmp_path / "r.json"
    rc = main(["--fake", "healthy,tim_degraded", "--gpus", "1", "--offline", "--quick", "--json", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["inventory"]["gpu_index"] == 1


def test_fake_list_selecting_several_cards_is_not_available_yet(capsys):
    assert main(["--fake", "healthy,healthy", "--offline"]) == 1
    assert "gpucheck: multi-GPU runs are not available yet" in capsys.readouterr().err
