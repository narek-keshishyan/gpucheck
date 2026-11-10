import json
from dataclasses import replace

from gpucheck import cli
from gpucheck.cli import main
from gpucheck.fakes import make_fake


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
