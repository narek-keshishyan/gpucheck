"""The opt-in extra checks from the command line, on simulated cards: each has a card that
passes and one that fails, and a plain run is unchanged by all of them."""

import json

import pytest

from gpucheck import cli
from gpucheck.cli import main
from gpucheck.report import NOT_COVERED
from gpucheck.resultjson import canonical_dumps


def single(tmp_path, fake, *flags, name="r.json"):
    out = tmp_path / name
    rc = main(["--fake", fake, "--offline", "--quick", "--json", str(out), *flags])
    return rc, json.loads(out.read_text())


def codes(result):
    return [f["code"] for f in result["findings"]]


def lot(tmp_path, fake, *flags, name="lot"):
    d = tmp_path / name
    rc = main(["--fake", fake, "--offline", "--quick", "--lot-dir", str(d), *flags])
    summary = json.loads((d / "lot.json").read_text())
    cards = [json.loads((d / c["file"]).read_text()) for c in summary["cards"] if c["file"]]
    return rc, summary, cards


# ---- a plain run is what it was -----------------------------------------------------------


def test_a_plain_run_has_no_checks_block_and_calls_the_engine_as_before(tmp_path, monkeypatch, capsys):
    seen = []
    real = cli.execute_run
    monkeypatch.setattr(cli, "execute_run", lambda *a, **kw: seen.append(sorted(kw)) or real(*a, **kw))
    rc, result = single(tmp_path, "xid")  # a card that would report events if anyone listened
    assert rc == 0 and seen == [["clock", "sleep"]]
    assert "checks" not in result and "load_bucket_s" not in result["samples"] and codes(result) == []
    out = capsys.readouterr().out
    assert "Extra checks" not in out and NOT_COVERED in out


def test_a_plain_result_is_byte_identical_whether_or_not_the_extras_exist(tmp_path):
    from gpucheck.diagnose import diagnose
    from gpucheck.fakes import make_fake
    from gpucheck.metrics import derive
    from gpucheck.resultjson import build_result
    from gpucheck.run import NullLoad, RunConfig, execute_run
    from gpucheck.specs import lookup

    def build(**kw):
        a = make_fake("healthy")
        spec = lookup(a.inventory().model_name)
        run = execute_run(a, NullLoad(), RunConfig(30, 150), spec, clock=lambda: 0.0, sleep=lambda s: None)
        m = derive(run, spec)
        return canonical_dumps(build_result(run, m, diagnose(m, run.preflight, run.inventory, spec, **kw), **kw))

    plain = build()
    assert plain == build(checks=None) == build(checks={})
    assert set(json.loads(plain)) == {
        "schema_version", "inventory", "preflight", "config", "metrics", "findings", "samples", "aborted"}


# ---- driver error log ---------------------------------------------------------------------


def test_driver_errors_passing_card(tmp_path, capsys):
    rc, result = single(tmp_path, "healthy", "--driver-errors")
    assert rc == 0 and codes(result) == []
    assert result["checks"] == {"driver_errors": {"status": "ran", "count": 0, "events": []}}
    out = capsys.readouterr().out
    assert "Driver error log: listened for hardware error events from the driver for the whole run; none was reported" in out
    assert NOT_COVERED in out  # listening for driver errors narrows none of the "not covered" items


def test_driver_errors_failing_card(tmp_path, capsys):
    rc, result = single(tmp_path, "xid", "--driver-errors")
    assert rc == 0 and codes(result) == ["DRIVER_ERRORS"]
    finding = result["findings"][0]
    assert finding["severity"] == "bad" and "Xid 79 x1, double-bit ECC error x1" in finding["message"]
    events = result["checks"]["driver_errors"]["events"]
    assert [(e["kind"], e["phase"]) for e in events] == [("xid", "load"), ("double_bit_ecc", "load")]
    assert "[BAD] DRIVER_ERRORS" in capsys.readouterr().out


def test_driver_errors_unavailable_is_said_not_passed(tmp_path, capsys):
    rc, result = single(tmp_path, "no_event_log", "--driver-errors")
    assert rc == 0 and codes(result) == ["CHECK_NOT_RUN"]
    assert result["checks"]["driver_errors"]["status"] == "unavailable"
    assert "driver error log (could not be run)" in result["findings"][0]["message"]
    assert "could not be started on this system, so driver errors were not checked" in capsys.readouterr().out


def test_driver_errors_in_a_lot_are_per_card(tmp_path, capsys):
    rc, summary, cards = lot(tmp_path, "healthy,xid", "--driver-errors")
    assert rc == 0
    assert [c["findings"] for c in summary["cards"]] == [[], ["DRIVER_ERRORS"]]
    assert [c["checks"]["driver_errors"]["count"] for c in cards] == [0, 2]
    assert NOT_COVERED in capsys.readouterr().out


def test_upload_saved_cannot_be_combined_with_an_extra_check(capsys):
    with pytest.raises(SystemExit):
        main(["--upload-saved", "x.json", "--driver-errors"])
    assert "--upload-saved cannot be combined with --driver-errors" in capsys.readouterr().err


# ---- memory test --------------------------------------------------------------------------


def test_memtest_passing_card(tmp_path, capsys):
    rc, result = single(tmp_path, "healthy", "--memtest")
    assert rc == 0 and codes(result) == []
    m = result["checks"]["memtest"]
    assert m["status"] == "ran" and m["mismatched_words"] == 0 and len(m["patterns"]) == 6
    out = capsys.readouterr().out
    assert "Memory test: wrote 6 patterns to 72 GiB (about 90% of the card's memory) and read them back: no mismatch" in out
    assert "It does not test silent data corruption, faults that appear only after hours of use, links between GPUs, or physical damage." in out
    assert "The memory test wrote patterns to about 90% of the card's memory and read them back" in out
    assert "does not test memory cells" not in out


def test_memtest_failing_card(tmp_path, capsys):
    rc, result = single(tmp_path, "bad_memory", "--memtest")
    assert rc == 0 and codes(result) == ["MEMTEST_MISMATCH"]
    assert result["findings"][0]["severity"] == "bad" and "3 words that differed" in result["findings"][0]["message"]
    assert result["checks"]["memtest"]["by_pattern"] == {"checker_a": 2, "random": 1}
    assert "read them back: 3 words differed" in capsys.readouterr().out


def test_a_run_without_the_memory_test_still_says_memory_cells_are_not_tested(tmp_path, capsys):
    single(tmp_path, "bad_memory", "--driver-errors")
    assert NOT_COVERED in capsys.readouterr().out


def test_memtest_does_not_change_the_load_measurement(tmp_path):
    _, plain = single(tmp_path, "healthy", name="a.json")
    _, tested = single(tmp_path, "healthy", "--memtest", name="b.json")
    for key in ("metrics", "samples", "config", "preflight", "inventory"):
        assert plain[key] == tested[key]


def test_the_real_memory_test_is_built_for_the_selected_card(monkeypatch):
    import argparse

    made = []
    monkeypatch.setattr(cli, "TorchMemtest", lambda index: made.append(index) or "memtest")
    args = argparse.Namespace(fake=None, memtest=True, driver_errors=False)
    assert cli._make_checks(args, object(), 3) == ["memtest"] and made == [3]
    assert sorted(cli._extras(args, object(), 3)) == ["checks", "plan"]
    assert cli._extras(argparse.Namespace(fake=None, memtest=False, driver_errors=False), object(), 3) == {}


def test_memtest_without_pytorch_is_reported_as_not_run(tmp_path, monkeypatch, capsys):
    import sys

    from gpucheck.deepchecks import TorchMemtest

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(cli, "fake_memtest", lambda extras: TorchMemtest(0))
    rc, result = single(tmp_path, "healthy", "--memtest")
    assert rc == 0 and codes(result) == ["CHECK_NOT_RUN"]
    assert result["checks"]["memtest"] == {"status": "unavailable", "reason": "PyTorch is not installed"}
    out = capsys.readouterr().out
    assert "Memory test: could not be run, so memory cells were not tested" in out and NOT_COVERED in out
