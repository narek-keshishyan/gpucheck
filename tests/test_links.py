"""NVLink: link state and error counters around the run, and the card-to-card copy in an
all-at-once lot. A card without active links reports nothing rather than a pass."""

import json
import sys

import pytest

import minitorch
from gpucheck import cli
from gpucheck.checks import add_transfer, nvlink_report
from gpucheck.cli import main
from gpucheck.deepchecks import TorchTransfer
from gpucheck.report import NOT_COVERED

ZERO = {"replay": 0, "recovery": 0, "crc_flit": 0, "crc_data": 0}


def snap(active, errors=ZERO, totals=None, tx=None):
    out = {"links": [{"link": i, "active": up, "errors": dict(errors) if i == 0 else {k: 0 for k in errors}}
                     for i, up in enumerate(active)],
           "error_totals": dict(totals or {})}
    if tx is not None:
        out["tx_kib"] = out["rx_kib"] = tx
    return out


# ---- the report built from two snapshots ---------------------------------------------------


def test_no_links_at_all_reports_nothing_checked():
    assert nvlink_report(None, None) == {"status": "no_links"}


def test_links_that_are_all_inactive_report_nothing_checked():
    assert nvlink_report(snap([False, False]), snap([False, False])) == {"status": "no_links", "links": 2}


def test_clean_links():
    assert nvlink_report(snap([True] * 4), snap([True] * 4)) == {
        "status": "ran", "links": 4, "active": 4, "dropped": 0, "errors": ZERO}


def test_counter_rises_are_summed_over_links_and_a_dropped_link_is_counted():
    before = snap([True, True, True], {"replay": 10, "recovery": 0, "crc_flit": 3, "crc_data": 0})
    after = snap([True, False, True], {"replay": 14, "recovery": 2, "crc_flit": 3, "crc_data": 0})
    assert nvlink_report(before, after) == {
        "status": "ran", "links": 3, "active": 2, "dropped": 1,
        "errors": {"replay": 4, "recovery": 2, "crc_flit": 0, "crc_data": 0}}


def test_card_wide_totals_are_used_when_the_links_report_no_counters():
    before = snap([True, True], {}, totals={"replay": 1, "crc_flit": 5})
    after = snap([True, True], {}, totals={"replay": 3, "crc_flit": 5})
    assert nvlink_report(before, after)["errors"] == {"replay": 2, "crc_flit": 0}


def test_a_counter_that_went_down_is_unknown_not_a_negative_or_zero_rise():
    out = nvlink_report(snap([True], {"replay": 9, "recovery": 0}), snap([True], {"replay": 2, "recovery": 0}))
    assert out["errors"] == {"recovery": 0}


def test_counters_nobody_reports_leave_errors_empty():
    assert nvlink_report(snap([True], {}), snap([True], {}))["errors"] == {}


def test_a_single_reading_is_unavailable():
    assert nvlink_report(snap([True]), None)["status"] == "unavailable"
    assert nvlink_report(None, snap([True]))["status"] == "unavailable"


def test_a_transfer_adds_what_happened_during_the_copies():
    report = nvlink_report(snap([True] * 2, tx=100), snap([True] * 2, tx=100))
    before = snap([True, True], {"replay": 0, "recovery": 0, "crc_flit": 0, "crc_data": 0}, tx=100)
    after = snap([True, False], {"replay": 2, "recovery": 0, "crc_flit": 0, "crc_data": 0}, tx=100 + 4096)
    out = add_transfer(report, before, after, {"peers": 1, "bytes": 1 << 28, "mismatched_words": 0})
    assert out["errors"]["replay"] == 2 and out["dropped"] == 1 and out["active"] == 1
    assert out["transfer"] == {"peers": 1, "bytes": 1 << 28, "mismatched_words": 0, "nvlink_kib": 4096}
    assert "transfer" not in report  # the run's own entry is not changed in place


def test_a_transfer_without_readable_data_counters_claims_no_nvlink_traffic():
    report = nvlink_report(snap([True]), snap([True]))
    out = add_transfer(report, snap([True]), snap([True]), {"peers": 1, "bytes": 8, "mismatched_words": 0})
    assert "nvlink_kib" not in out["transfer"]
    assert "nvlink_kib" not in add_transfer(report, None, None, {"peers": 1, "bytes": 8, "mismatched_words": 0})["transfer"]


# ---- from the command line: one card, or one after another --------------------------------


def single(tmp_path, fake, *flags):
    out = tmp_path / "r.json"
    rc = main(["--fake", fake, "--offline", "--quick", "--json", str(out), *flags])
    return rc, json.loads(out.read_text())


def codes(result):
    return [f["code"] for f in result["findings"]]


def test_links_passing_card(tmp_path, capsys):
    rc, result = single(tmp_path, "nvlink", "--links")
    assert rc == 0 and codes(result) == []
    assert result["checks"]["nvlink"] == {"status": "ran", "links": 4, "active": 4, "dropped": 0, "errors": ZERO}
    out = capsys.readouterr().out
    assert "NVLink: 4 links, 4 active; error counters did not rise" in out
    assert "data transfer over links between GPUs, or physical damage." in out
    assert "It read the state and error counters of this card's NVLink links before and after the load." in out


def test_links_failing_card(tmp_path):
    rc, result = single(tmp_path, "nvlink_errors", "--links")
    assert rc == 0 and codes(result) == ["NVLINK_ERRORS", "NVLINK_LINKS_INACTIVE"]
    assert result["checks"]["nvlink"]["errors"] == {"replay": 5, "recovery": 1, "crc_flit": 12, "crc_data": 0}


@pytest.mark.parametrize("fake", ["healthy", "nvlink_idle"])
def test_a_card_without_active_links_reports_nothing_and_no_pass(tmp_path, capsys, fake):
    rc, result = single(tmp_path, fake, "--links")
    assert rc == 0 and codes(result) == []
    assert result["checks"]["nvlink"]["status"] == "no_links"
    out = capsys.readouterr().out
    assert "NVLink: this card has no active NVLink link, so there was nothing to check" in out
    assert NOT_COVERED in out


def test_without_the_flag_links_are_not_read(tmp_path):
    _, result = single(tmp_path, "nvlink_errors")
    assert "checks" not in result and codes(result) == []


def test_a_sequential_lot_reads_links_but_copies_nothing(tmp_path, capsys):
    d = tmp_path / "lot"
    main(["--fake", "nvlink,nvlink", "--offline", "--quick", "--links", "--lot-dir", str(d)])
    r = json.loads((d / "gpu0.json").read_text())
    assert "transfer" not in r["checks"]["nvlink"]
    assert "Copying test data" not in capsys.readouterr().out


# ---- all at once: the copy -----------------------------------------------------------------


def together(tmp_path, fake, *flags):
    d = tmp_path / "lot"
    rc = main(["--fake", fake, "--offline", "--quick", "--together", "--links", "--lot-dir", str(d), *flags])
    summary = json.loads((d / "lot.json").read_text())
    return rc, summary, {c["position"]: json.loads((d / c["file"]).read_text()) for c in summary["cards"] if c["file"]}


def test_linked_cards_copy_test_data_between_every_pair(tmp_path, capsys):
    rc, summary, cards = together(tmp_path, "nvlink,nvlink,nvlink")
    assert rc == 0 and [c["findings"] for c in summary["cards"]] == [[], [], []]
    for r in cards.values():
        assert r["checks"]["nvlink"]["transfer"] == {
            "peers": 2, "peers_with_mismatch": 0, "bytes": 2 * (256 << 20), "mismatched_words": 0,
            "nvlink_kib": 2 * (256 << 10)}
    out = capsys.readouterr().out
    assert "Copying test data between 3 linked cards (3 pairs)..." in out
    assert "copied 0.5 GiB of test data to and from 2 other cards: it came back unchanged; the NVLink data counters rose during the copies" in out
    assert "It read this card's NVLink links and copied test data between it and 2 other cards" in out
    assert "links between GPUs," not in out.split("Lot summary")[0].split("Not covered")[1]


def test_a_copy_that_comes_back_changed_is_a_finding_on_both_cards_and_points_at_the_common_one(tmp_path):
    rc, summary, cards = together(tmp_path, "nvlink,nvlink,nvlink_errors")
    assert [("LINK_TRANSFER_MISMATCH" in c["findings"]) for c in summary["cards"]] == [True, True, True]
    t = [cards[i]["checks"]["nvlink"]["transfer"] for i in range(3)]
    assert [x["peers_with_mismatch"] for x in t] == [1, 1, 2] and [x["mismatched_words"] for x in t] == [4, 4, 8]


def test_cards_without_links_are_left_out_and_report_nothing(tmp_path, capsys):
    rc, summary, cards = together(tmp_path, "nvlink,healthy,nvlink,nvlink_idle")
    assert cards[0]["checks"]["nvlink"]["transfer"]["peers"] == 1
    for pos in (1, 3):
        assert cards[pos]["checks"]["nvlink"]["status"] == "no_links" and "transfer" not in cards[pos]["checks"]["nvlink"]
    assert "Copying test data between 2 linked cards (1 pair)..." in capsys.readouterr().out


def test_one_linked_card_alone_has_nobody_to_copy_to(tmp_path, capsys):
    rc, _, cards = together(tmp_path, "nvlink,healthy")
    assert "transfer" not in cards[0]["checks"]["nvlink"]
    assert "Copying test data" not in capsys.readouterr().out


def test_a_copy_that_does_not_show_on_the_nvlink_counters_does_not_claim_the_links_were_used(tmp_path, capsys):
    rc, _, cards = together(tmp_path, "nvlink_no_counter,nvlink_no_counter")
    assert cards[0]["checks"]["nvlink"]["transfer"]["nvlink_kib"] == 0
    out = capsys.readouterr().out
    assert "the NVLink data counters did not show the copies, so they may have travelled another way" in out
    assert "data transfer over links between GPUs, or physical damage." in out
    assert "copied test data between it and" not in out


def test_pairs_nvml_does_not_report_as_linked_are_not_copied_between(tmp_path, monkeypatch, capsys):
    from gpucheck.fakes import FakeAdapter

    monkeypatch.setattr(FakeAdapter, "nvlink_peer", lambda self, other: None)
    rc, _, cards = together(tmp_path, "nvlink,nvlink")
    assert all("transfer" not in r["checks"]["nvlink"] for r in cards.values())
    assert "Copying test data" not in capsys.readouterr().out


def test_a_copy_that_fails_is_warned_about_and_not_reported_as_checked(tmp_path, monkeypatch, capsys):
    def boom(a, b):
        raise RuntimeError("CUDA error: peer mapping resources exhausted")

    monkeypatch.setattr(cli, "fake_transfer", boom)
    rc, _, cards = together(tmp_path, "nvlink,nvlink")
    assert rc == 0 and all("transfer" not in r["checks"]["nvlink"] for r in cards.values())
    assert "warning: the copy between GPU 0 and GPU 1 failed (RuntimeError: CUDA error" in capsys.readouterr().err


def test_an_aborted_card_takes_no_part_in_the_copies(tmp_path, monkeypatch):
    from gpucheck import fakes

    monkeypatch.setitem(fakes._EXTRAS, "overheat", fakes._EXTRAS["nvlink"])
    rc, summary, cards = together(tmp_path, "nvlink,overheat,nvlink")
    assert summary["cards"][1]["outcome"] == "aborted"
    assert cards[0]["checks"]["nvlink"]["transfer"]["peers"] == 1
    assert "transfer" not in cards[1]["checks"].get("nvlink", {})


def test_without_links_an_all_at_once_lot_copies_nothing(tmp_path, capsys):
    d = tmp_path / "lot"
    main(["--fake", "nvlink,nvlink", "--offline", "--quick", "--together", "--lot-dir", str(d)])
    assert "nvlink" not in json.loads((d / "gpu0.json").read_text())["checks"]


# ---- the copy itself, on the PyTorch stand-in ----------------------------------------------


@pytest.fixture
def torch(monkeypatch):
    minitorch.reset()
    monkeypatch.setitem(sys.modules, "torch", minitorch)
    return minitorch


def test_copy_goes_there_and_back_and_compares(torch):
    out = TorchTransfer(nbytes=800).copy(1, 3)
    assert out["bytes"] == 800 and out["mismatched_words"] == 0 and out["seconds"] >= 0
    assert torch.cuda.emptied == 1


def test_copy_counts_words_that_came_back_changed(torch):
    real = minitorch.Tensor.to
    hops = []

    def to(self, *args, **kw):
        out = real(self, *args, **kw)
        hops.append(out.device)
        if len(hops) == 2:  # the copy back
            out.data[3] ^= 1
            out.data[9] ^= 1
        return out

    minitorch.Tensor.to = to
    try:
        out = TorchTransfer(nbytes=800).copy(0, 1)
    finally:
        minitorch.Tensor.to = real
    assert out["mismatched_words"] == 2 and hops == ["cuda:1", "cuda:0"]
