"""Findings from the opt-in extra checks. Without a checks block the rule table is the old one."""

from gpucheck.diagnose import THROUGHPUT_FELL_RATIO, diagnose
from gpucheck.fakes import make_fake
from gpucheck.metrics import derive
from gpucheck.run import NullLoad, RunConfig, execute_run
from gpucheck.specs import lookup

GIB = 1 << 30
SPEC = lookup("NVIDIA H100 80GB HBM3")


def findings(checks):
    a = make_fake("healthy")
    run = execute_run(a, NullLoad(), RunConfig(30, 150), SPEC, clock=lambda: 0.0, sleep=lambda s: None)
    return diagnose(derive(run, SPEC), run.preflight, run.inventory, SPEC, checks)


def codes(checks):
    return [f.code for f in findings(checks)]


def test_no_checks_block_gives_the_old_rule_table():
    assert codes(None) == [] and codes({}) == []


def test_clean_checks_give_no_finding():
    assert codes({
        "driver_errors": {"status": "ran", "count": 0, "events": []},
        "memtest": {"status": "ran", "bytes_tested": 72 * GIB, "bytes_total": 80 * GIB, "mismatched_words": 0, "ecc_delta": 0},
        "answer": {"status": "ran", "products": 400, "wrong_products": 0},
        "nvlink": {"status": "ran", "links": 4, "active": 4, "dropped": 0, "errors": {"replay": 0}},
        "burn_in": {"completed_s": 86400, "early_tflops": 950.0, "late_tflops": 948.0},
        "together": {"cards": 4},
    }) == []


def test_memtest_mismatch():
    (f,) = findings({"memtest": {"status": "ran", "bytes_tested": 72 * GIB, "mismatched_words": 1}})
    assert (f.code, f.severity, f.confidence) == ("MEMTEST_MISMATCH", "bad", "high")
    assert f.message == "The memory test read back 1 word that differed from what it wrote (72.0 GiB tested)."


def test_a_memtest_that_stopped_early_still_reports_the_mismatches_it_saw():
    got = codes({"memtest": {"status": "partial", "bytes_tested": GIB, "mismatched_words": 5}})
    assert got == ["MEMTEST_MISMATCH", "CHECK_NOT_RUN"]


def test_wrong_answers():
    (f,) = findings({"answer": {"status": "ran", "products": 400, "wrong_products": 2}})
    assert (f.code, f.severity) == ("WRONG_ANSWERS", "bad")
    assert "got 2 of 400 products wrong" in f.message


def test_ecc_errors_during_a_check_are_memory_errors():
    (f,) = findings({"memtest": {"status": "ran", "bytes_tested": GIB, "mismatched_words": 0, "ecc_delta": 4}})
    assert f.code == "MEMORY_ERRORS"
    assert f.message.startswith("ECC error count rose by 4 during the memory test.")


def test_nvlink_errors_and_inactive_links():
    fs = findings({"nvlink": {"status": "ran", "links": 4, "active": 3, "dropped": 1,
                              "errors": {"replay": 5, "recovery": 0, "crc_flit": 12}}})
    assert [f.code for f in fs] == ["NVLINK_ERRORS", "NVLINK_LINKS_INACTIVE"]
    assert fs[0].severity == "warn" and fs[0].message == (
        "NVLink error counters rose during the run (replay +5, crc flit +12) and 1 NVLink link that was "
        "active before the run had gone down after it."
    )
    assert fs[1].severity == "info" and fs[1].message.startswith("1 of this card's 4 NVLink links were inactive.")


def test_a_card_without_links_gets_no_link_finding():
    assert codes({"nvlink": {"status": "no_links"}}) == []
    assert codes({"nvlink": {"status": "no_links", "links": 3}}) == []


def test_transfer_mismatch():
    (f,) = findings({"nvlink": {"status": "ran", "links": 2, "active": 2, "errors": {},
                                "transfer": {"peers": 1, "bytes": GIB, "mismatched_words": 4, "nvlink_kib": 9}}})
    assert (f.code, f.severity) == ("LINK_TRANSFER_MISMATCH", "bad")
    assert f.message.startswith("Test data copied between this card and 1 other card came back changed (4 words differed). ")
    (g,) = findings({"nvlink": {"status": "ran", "links": 2, "active": 2, "errors": {}, "transfer": {
        "peers": 3, "peers_with_mismatch": 1, "bytes": GIB, "mismatched_words": 4, "nvlink_kib": 9}}})
    assert "(4 words differed, in the copies with 1 of them)" in g.message


def test_throughput_that_fell_over_a_burn_in():
    late = 950.0 * THROUGHPUT_FELL_RATIO - 1
    (f,) = findings({"burn_in": {"completed_s": 86400, "early_tflops": 950.0, "late_tflops": late}})
    assert (f.code, f.severity, f.confidence) == ("THROUGHPUT_FELL", "warn", "low")
    assert "5.1% lower than in its first ten minutes" in f.message
    assert codes({"burn_in": {"completed_s": 86400, "early_tflops": 950.0, "late_tflops": 950.0 * THROUGHPUT_FELL_RATIO}}) == []
    assert codes({"burn_in": {"completed_s": 300, "early_tflops": None, "late_tflops": 10.0}}) == []


def test_every_check_that_did_not_finish_is_named_once():
    (f,) = findings({"driver_errors": {"status": "partial", "events": []}, "memtest": {"status": "unavailable"},
                     "answer": {"status": "unavailable"}, "nvlink": {"status": "unavailable"}})
    assert f.code == "CHECK_NOT_RUN" and f.severity == "info"
    assert f.message.endswith(
        "driver error log (stopped before finishing), memory test (could not be run), "
        "answer check (could not be run), NVLink links (could not be run)."
    )


def test_an_xid_alone_is_a_warning_that_says_it_may_not_be_the_hardware():
    # Seen on a rented A100 (14 November 2026): a deliberate fault in another process on the card
    # was reported as Xid 43, which is a program fault, not a hardware one.
    (f,) = findings({"driver_errors": {"status": "ran", "events": [{"kind": "xid", "xid": 43, "phase": "load"}]}})
    assert (f.code, f.severity, f.confidence) == ("DRIVER_ERRORS", "warn", "high")
    assert f.message == (
        "The driver reported 1 error event for this card during the run: Xid 43 x1. An Xid can come from the "
        "hardware, the driver or a program using the card; NVIDIA's Xid list says which causes each number has."
    )
    assert "hardware error" not in f.message


def test_a_memory_error_event_is_bad_whatever_else_was_reported():
    (f,) = findings({"driver_errors": {"status": "ran", "events": [
        {"kind": "xid", "xid": 48}, {"kind": "double_bit_ecc"}]}})
    assert f.severity == "bad" and f.message == (
        "The driver reported 2 error events for this card during the run: Xid 48 x1, double-bit ECC error x1."
    )
