"""The per-result "not covered" note and check lines, including for hostile input (the server
shows this text on public pages for whatever an uploader sent)."""

import pytest

from gpucheck.coverage import (
    NOT_COVERED, check_lines, checks_run, narrowed, not_covered, not_covered_many,
)

GIB = 1 << 30
MEMTEST = {"status": "ran", "bytes_tested": 72 * GIB, "bytes_total": 80 * GIB, "patterns": ["a"] * 6,
           "mismatched_words": 0}
ANSWER = {"status": "ran", "products": 400, "wrong_products": 0}
LINKS = {"status": "ran", "links": 4, "active": 4, "dropped": 0,
         "errors": {"replay": 0, "recovery": 0, "crc_flit": 0, "crc_data": 0}}
TRANSFER = {"peers": 3, "bytes": 3 * GIB, "mismatched_words": 0, "nvlink_kib": 4096}


def result(**checks):
    return {"config": {"load_s": 86400}, "checks": checks}


@pytest.mark.parametrize("r", [{}, {"checks": {}}, {"checks": None}, {"checks": []}, None, "x", {"checks": {"memtest": 3}}])
def test_a_result_without_extra_checks_gets_exactly_the_old_note(r):
    assert not_covered(r) == NOT_COVERED
    assert check_lines(r) == [] and narrowed(r) == [] and checks_run(r) == []


def test_a_memory_test_removes_memory_cells_and_says_what_it_reached():
    text = not_covered(result(memtest=MEMTEST))
    assert "does not test memory cells" not in text and "memory cells" not in text
    assert "wrote patterns to about 90% of the card's memory" in text
    assert "cannot rule out faults that these patterns do not provoke" in text
    assert "silent data corruption, faults that appear only after hours of use" in text


def test_a_memory_test_that_did_not_finish_still_says_memory_cells_are_not_tested():
    for status in ("partial", "unavailable"):
        assert not_covered(result(memtest={**MEMTEST, "status": status})) == NOT_COVERED


def test_an_answer_check_is_worded_as_catching_some_corruption_never_as_ruling_it_out():
    text = not_covered(result(answer=ANSWER))
    assert "It does not test memory cells, faults that appear" in text
    assert text.endswith("it can catch some silent data corruption and cannot rule it out.")


def test_a_burn_in_names_the_hours_it_really_ran():
    text = not_covered(result(burn_in={"completed_s": 86400.0}))
    assert "faults that appear only after more than 24 hours of use" in text
    aborted = not_covered(result(burn_in={"completed_s": 2 * 3600 + 5}))
    assert "after more than 2 hours of use" in aborted
    assert "after more than 1 hour of use" in not_covered(result(burn_in={"completed_s": 3700}))
    assert not_covered(result(burn_in={"completed_s": 3000})) == NOT_COVERED


def test_burn_in_hours_never_exceed_the_planned_load():
    r = {"config": {"load_s": 7200}, "checks": {"burn_in": {"completed_s": 9e6}}}
    assert "after more than 2 hours of use" in not_covered(r)


def test_links_that_were_read_but_not_used_still_leave_transfer_untested():
    text = not_covered(result(nvlink=LINKS))
    assert "data transfer over links between GPUs" in text
    assert "It read the state and error counters of this card's NVLink links before and after the load." in text


def test_a_transfer_seen_on_the_nvlink_counters_covers_the_links():
    text = not_covered(result(nvlink={**LINKS, "transfer": TRANSFER}))
    assert "links between GPUs," not in text
    assert "copied test data between it and 3 other cards" in text
    unseen = not_covered(result(nvlink={**LINKS, "transfer": {**TRANSFER, "nvlink_kib": 0}}))
    assert "data transfer over links between GPUs" in unseen


@pytest.mark.parametrize("nvlink", [{"status": "no_links"}, {"status": "unavailable"}, {**LINKS, "active": 0}])
def test_a_card_without_usable_links_reports_no_link_coverage(nvlink):
    assert not_covered(result(nvlink=nvlink)) == NOT_COVERED
    assert "links" not in narrowed(result(nvlink=nvlink))


def test_everything_at_once_leaves_only_physical_damage_and_long_use():
    text = not_covered(result(memtest=MEMTEST, answer=ANSWER, burn_in={"completed_s": 86400},
                              nvlink={**LINKS, "transfer": TRANSFER}))
    assert "It does not test faults that appear only after more than 24 hours of use or physical damage." in text


def test_narrowed_and_checks_run():
    r = result(memtest=MEMTEST, answer=ANSWER, driver_errors={"status": "ran", "events": []},
               together={"cards": 4}, nvlink=LINKS, burn_in={"completed_s": 7200})
    assert narrowed(r) == ["memory", "sdc", "hours", "links"]
    assert checks_run(r) == ["driver_errors", "memtest", "answer", "burn_in", "together", "nvlink"]


def test_many_runs_note_is_the_old_one_when_nothing_was_narrowed():
    assert not_covered_many([]) == NOT_COVERED
    assert not_covered_many([[], []]) == NOT_COVERED


def test_many_runs_note_lists_only_what_no_run_tested():
    text = not_covered_many([["memory", "sdc"], []])
    assert "No run listed here tests faults that appear only after hours of use, links between GPUs, or physical damage." in text
    assert "each full report says what that run did and did not cover" in text
    assert "memory cells" not in text
    assert "No run listed here tests physical damage." in not_covered_many([["memory", "sdc", "hours", "links", "junk"]])


def test_check_lines_say_what_each_check_did():
    r = result(
        together={"cards": 4},
        driver_errors={"status": "ran", "events": [{"kind": "xid", "xid": 79}, {"kind": "xid", "xid": 79},
                                                    {"kind": "double_bit_ecc"}, {"kind": "<script>"}]},
        memtest={**MEMTEST, "mismatched_words": 3},
        answer={**ANSWER, "wrong_products": 2},
        burn_in={"completed_s": 86400.0, "early_tflops": 950.0, "late_tflops": 800.0},
        nvlink={**LINKS, "active": 3, "dropped": 1, "errors": {"replay": 5, "crc_flit": 0},
                "transfer": {**TRANSFER, "mismatched_words": 4}},
    )
    lines = dict(check_lines(r))
    assert lines["Loaded together"] == "this card was loaded at the same time as 3 other cards in the same machine"
    assert lines["Driver error log"].endswith("4 reported (Xid 79 x2, double-bit ECC error x1, other event x1)")
    assert lines["Memory test"] == (
        "wrote 6 patterns to 72 GiB (about 90% of the card's memory) and read them back: 3 words differed"
    )
    assert lines["Answer check"] == "compared 400 computed products with known results: 2 wrong"
    assert lines["Burn-in"] == "the load ran for 24 h 0 min; throughput over its final part was 84.2% of its first ten minutes"
    assert lines["NVLink"] == (
        "4 links, 3 active; 1 link went down during the run; error counters rose: replay +5; "
        "copied 3 GiB of test data to and from 3 other cards: 4 words came back changed; "
        "the NVLink data counters rose during the copies"
    )
    assert "<script>" not in str(check_lines(r))


def test_check_lines_say_when_a_check_could_not_run():
    r = result(driver_errors={"status": "unavailable", "reason": "<b>x</b>"}, memtest={"status": "unavailable"},
               answer={"status": "unavailable"}, nvlink={"status": "no_links"})
    lines = dict(check_lines(r))
    assert lines["Driver error log"] == "could not be started on this system, so driver errors were not checked"
    assert lines["Memory test"] == "could not be run, so memory cells were not tested"
    assert lines["Answer check"] == "could not be run, so computed results were not checked"
    assert lines["NVLink"] == "this card has no active NVLink link, so there was nothing to check"
    assert "<b>" not in str(lines)


def test_a_clean_log_and_a_partial_one():
    assert dict(check_lines(result(driver_errors={"status": "ran", "events": []})))["Driver error log"] == (
        "listened for hardware error events from the driver for the whole run; none was reported"
    )
    part = dict(check_lines(result(driver_errors={"status": "partial", "events": []})))["Driver error log"]
    assert "for part of the run (listening stopped early)" in part


HOSTILE = [
    {"status": "ran", "bytes_tested": float("nan"), "bytes_total": "80"},
    {"status": "ran", "bytes_tested": -5, "bytes_total": 1},
    {"status": "ran", "bytes_tested": 10**400, "bytes_total": 1},
    {"status": "ran", "bytes_tested": True, "products": True, "links": True, "active": True},
    {"status": ["ran"], "products": {"a": 1}, "events": "xid", "cards": [4]},
    {"status": "ran", "products": 1.5, "wrong_products": 99, "links": 99999, "active": 2, "errors": [1]},
    {"status": "ran", "events": [None, 3, {"kind": 7, "xid": "79"}, {"kind": "xid", "xid": -1}], "transfer": "x"},
    {"status": "ran", "completed_s": float("inf"), "early_tflops": 0, "late_tflops": "9", "cards": 10**9},
]


@pytest.mark.parametrize("junk", HOSTILE)
def test_hostile_values_never_raise_and_never_claim_coverage(junk):
    r = {"config": {"load_s": "x"}, "checks": {name: junk for name in
                                              ("driver_errors", "memtest", "answer", "burn_in", "together", "nvlink")}}
    text = not_covered(r)
    assert text.startswith("This is a pre-flight check, not proof of health.")
    assert "memory cells" in text and "silent data corruption" in text and "links between GPUs" in text
    for label, line in check_lines(r):
        assert isinstance(label, str) and isinstance(line, str)
    narrowed(r), checks_run(r)


def test_memory_percent_is_dropped_when_the_total_is_smaller_than_what_was_tested():
    r = result(memtest={**MEMTEST, "bytes_total": 1})
    assert "wrote patterns to 72 GiB of the card's memory" in not_covered(r)
