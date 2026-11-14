"""What one result did and did not cover, in words built only from the result's own numbers.

The server imports this module and shows its text on public pages, where a result is whatever an
uploader sent. So every function here reads the result defensively: a value of the wrong type or
out of range counts as absent, and no free text from the result is ever echoed. A result without
a "checks" block (every result written before the extra checks existed, and every default run)
gets exactly NOT_COVERED and no check lines.
"""

from __future__ import annotations

import math
from typing import Any

NOT_COVERED = (
    "This is a pre-flight check, not proof of health. It does not test memory cells, "
    "silent data corruption, faults that appear only after hours of use, links between "
    "GPUs, or physical damage."
)
_BASE = "This is a pre-flight check, not proof of health."

RAN = "ran"  # the check ran to its end
PARTIAL = "partial"  # it ran but stopped before its end
UNAVAILABLE = "unavailable"  # it could not be started
NO_LINKS = "no_links"  # NVLink only: the card has no active link, so there was nothing to check

# The checks a result may carry, in the order they are listed.
CHECK_NAMES = ("driver_errors", "memtest", "answer", "burn_in", "together", "nvlink")
CHECK_LABELS = {
    "driver_errors": "Driver error log",
    "memtest": "Memory test",
    "answer": "Answer check",
    "burn_in": "Burn-in",
    "together": "Loaded together",
    "nvlink": "NVLink",
}
# The default "not covered" items a check narrows.
NARROWS = {"memtest": "memory", "answer": "sdc", "burn_in": "hours", "nvlink": "links"}

# Driver event kinds gpucheck records; any other kind in a result is shown only as "other".
EVENT_KINDS = {
    "xid": "Xid",
    "double_bit_ecc": "double-bit ECC error",
    "single_bit_ecc_storm": "single-bit ECC error storm",
    "dram_retirement": "memory retirement",
    "dram_retirement_failure": "memory retirement failure",
    "non_fatal_poison": "non-fatal poison error",
    "fatal_poison": "fatal poison error",
    "gpu_unavailable": "GPU unavailable",
    "gpu_recovery": "GPU recovery action",
}
NVLINK_COUNTERS = ("replay", "recovery", "crc_flit", "crc_data")
ANSWER_FORMATS = ("float32", "float16", "bfloat16", "float64")  # the answer check's number formats
_COUNTER_WORDS = {"replay": "replay", "recovery": "recovery", "crc_flit": "CRC (flit)", "crc_data": "CRC (data)"}
MAX_EVENTS_SHOWN = 6
GIB = 1 << 30


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def count(v: Any, hi: float = 1e15) -> int | None:
    """A whole, non-negative number of at most `hi`, else None."""
    f = _num(v)
    if f is None or f < 0 or f > hi or f != int(f):
        return None
    return int(f)


def checks_of(result: Any) -> dict:
    c = result.get("checks") if isinstance(result, dict) else None
    return c if isinstance(c, dict) else {}


def check(result: Any, name: str) -> dict:
    v = checks_of(result).get(name)
    return v if isinstance(v, dict) else {}


def status(result: Any, name: str) -> str | None:
    s = check(result, name).get("status")
    return s if isinstance(s, str) else None


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _gib(n: int) -> str:
    if n < GIB:
        return f"{n / (1 << 20):.0f} MiB"
    return f"{n / GIB:.1f}".rstrip("0").rstrip(".") + " GiB"


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60} h {minutes % 60} min"


# ---- facts read from a result -------------------------------------------------------------


def memtest_facts(result: Any) -> dict | None:
    """Bytes tested, share of the card's memory and words that differed, for a memory test that ran."""
    c = check(result, "memtest")
    if c.get("status") not in (RAN, PARTIAL):
        return None
    tested = count(c.get("bytes_tested"))
    if not tested:
        return None
    total = count(c.get("bytes_total"))
    percent = None
    if total and tested <= total:
        percent = int(100 * tested / total)
    patterns = c.get("patterns")
    return {
        "complete": c["status"] == RAN,
        "bytes": tested,
        "percent": percent if percent and percent >= 1 else None,
        "patterns": len(patterns) if isinstance(patterns, list) and len(patterns) <= 64 else None,
        "mismatched": count(c.get("mismatched_words")),
    }


def answer_facts(result: Any) -> dict | None:
    c = check(result, "answer")
    if c.get("status") not in (RAN, PARTIAL):
        return None
    products = count(c.get("products"))
    if not products:
        return None
    wrong = count(c.get("wrong_products"))
    skipped = c.get("skipped")
    skipped = [name for name in ANSWER_FORMATS if name in skipped] if isinstance(skipped, list) else []
    return {"complete": c["status"] == RAN, "products": products, "skipped": skipped,
            "wrong": wrong if wrong is not None and wrong <= products else None}


def load_hours(result: Any) -> int:
    """Whole hours of load a burn-in completed (0 for any other run), never more than was planned."""
    done = _num(check(result, "burn_in").get("completed_s"))
    cfg = result.get("config") if isinstance(result, dict) else None
    planned = _num(cfg.get("load_s")) if isinstance(cfg, dict) else None
    if done is None or planned is None or done <= 0:
        return 0
    return int(min(done, planned) // 3600)


def together_others(result: Any) -> int | None:
    """How many other cards were loaded at the same time as this one, or None when it ran alone."""
    cards = count(check(result, "together").get("cards"), hi=64)
    return cards - 1 if cards and cards >= 2 else None


def driver_events(result: Any) -> list[tuple[str, int]] | None:
    """(what, how many) for the driver events a result carries; None when the log did not run."""
    c = check(result, "driver_errors")
    if c.get("status") not in (RAN, PARTIAL):
        return None
    tally: dict[str, int] = {}
    events = c.get("events")
    for e in events[:1000] if isinstance(events, list) else []:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        xid = count(e.get("xid"), hi=9999)
        if kind == "xid" and xid is not None:
            name = f"Xid {xid}"
        elif isinstance(kind, str) and kind in EVENT_KINDS:
            name = EVENT_KINDS[kind]
        else:
            name = "other event"
        tally[name] = tally.get(name, 0) + 1
    return sorted(tally.items())


def nvlink_facts(result: Any) -> dict | None:
    """Links, counter rises and transfer outcome for a card whose NVLink links were read."""
    c = check(result, "nvlink")
    if c.get("status") != RAN:
        return None
    links, active = count(c.get("links"), hi=64), count(c.get("active"), hi=64)
    if not links or active is None or active > links or active < 1:
        return None
    errors = c.get("errors") if isinstance(c.get("errors"), dict) else {}
    rises = {k: n for k in NVLINK_COUNTERS if (n := count(errors.get(k))) is not None}
    t = c.get("transfer") if isinstance(c.get("transfer"), dict) else {}
    transfer = None
    peers, moved = count(t.get("peers"), hi=64), count(t.get("bytes"))
    if peers and moved:
        bad_peers = count(t.get("peers_with_mismatch"), hi=64)
        transfer = {"peers": peers, "bytes": moved, "mismatched": count(t.get("mismatched_words")),
                    "bad_peers": bad_peers if bad_peers is not None and bad_peers <= peers else None,
                    "over_nvlink": (count(t.get("nvlink_kib")) or 0) > 0}
    return {"links": links, "active": active, "dropped": count(c.get("dropped"), hi=64) or 0,
            "rises": rises, "counters_read": bool(rises), "transfer": transfer}


def narrowed(result: Any) -> list[str]:
    """Which of the default "not covered" items this result narrows: memory, sdc, hours, links."""
    out = []
    m = memtest_facts(result)
    if m and m["complete"]:
        out.append("memory")
    a = answer_facts(result)
    if a and a["complete"]:
        out.append("sdc")
    if load_hours(result) >= 1:
        out.append("hours")
    if nvlink_facts(result):
        out.append("links")
    return out


def checks_run(result: Any) -> list[str]:
    """Names of the extra checks this result reports as run, at least in part."""
    out = []
    for name in CHECK_NAMES:
        if name == "burn_in":
            ran = load_hours(result) >= 1
        elif name == "together":
            ran = together_others(result) is not None
        elif name == "driver_errors":
            ran = driver_events(result) is not None
        elif name == "memtest":
            ran = memtest_facts(result) is not None
        elif name == "answer":
            ran = answer_facts(result) is not None
        else:
            ran = nvlink_facts(result) is not None
        if ran:
            out.append(name)
    return out


# ---- sentences ----------------------------------------------------------------------------


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


def not_covered(result: Any) -> str:
    """The "not covered" note for one result. Exactly NOT_COVERED when no extra check ran."""
    items: list[str] = []
    extra: list[str] = []

    m = memtest_facts(result)
    if m and m["complete"]:
        share = f"about {m['percent']}% of the card's memory" if m["percent"] else f"{_gib(m['bytes'])} of the card's memory"
        extra.append(
            f"The memory test wrote patterns to {share} and read them back; it does not test the rest "
            "of the memory, and it cannot rule out faults that these patterns do not provoke."
        )
    else:
        items.append("memory cells")

    a = answer_facts(result)
    if a and a["complete"]:
        extra.append(
            f"The answer check compared {_plural(a['products'], 'computed product')} with known results; "
            "it can catch some silent data corruption and cannot rule it out."
        )
    else:
        items.append("silent data corruption")

    hours = load_hours(result)
    if hours >= 1:
        items.append(f"faults that appear only after more than {_plural(hours, 'hour')} of use")
    else:
        items.append("faults that appear only after hours of use")

    n = nvlink_facts(result)
    if n and n["transfer"] and n["transfer"]["over_nvlink"]:
        extra.append(
            f"It read this card's NVLink links and copied test data between it and "
            f"{_plural(n['transfer']['peers'], 'other card')}; it does not test links to cards that were "
            "not selected, or faults that a short copy does not provoke."
        )
    elif n:
        items.append("data transfer over links between GPUs")
        what = "state and error counters" if n["counters_read"] else "state"
        extra.append(f"It read the {what} of this card's NVLink links before and after the load.")
    else:
        items.append("links between GPUs")

    items.append("physical damage")
    return " ".join([_BASE, f"It does not test {_join(items)}.", *extra])


_ITEM_WORDS = {
    "memory": "memory cells",
    "sdc": "silent data corruption",
    "hours": "faults that appear only after hours of use",
    "links": "links between GPUs",
}


def not_covered_many(narrowed_per_run: list[list[str]]) -> str:
    """The note for a page or table listing several runs: true for every one of them.

    `narrowed_per_run` holds narrowed() for each run. With no run narrowing anything it is
    exactly NOT_COVERED."""
    some = {k for run in narrowed_per_run for k in run if k in _ITEM_WORDS}
    if not some:
        return NOT_COVERED
    items = [words for key, words in _ITEM_WORDS.items() if key not in some] + ["physical damage"]
    return (
        f"{_BASE} No run listed here tests {_join(items)}. Other limits differ from run to run: "
        "each full report says what that run did and did not cover."
    )


def check_lines(result: Any) -> list[tuple[str, str]]:
    """(label, sentence) for every extra check the result mentions, including ones that could not run."""
    out: list[tuple[str, str]] = []
    present = checks_of(result)

    others = together_others(result)
    if others is not None:
        out.append((CHECK_LABELS["together"],
                    f"this card was loaded at the same time as {_plural(others, 'other card')} in the same machine"))

    if isinstance(present.get("driver_errors"), dict):
        events = driver_events(result)
        if events is None:
            text = "could not be started on this system, so driver errors were not checked"
        else:
            total = sum(n for _, n in events)
            whole = status(result, "driver_errors") == RAN
            span = "for the whole run" if whole else "for part of the run (listening stopped early)"
            if not total:
                text = f"listened for error events from the driver {span}; none was reported"
            else:
                shown = ", ".join(f"{name} x{n}" for name, n in events[:MAX_EVENTS_SHOWN])
                more = ", and others" if len(events) > MAX_EVENTS_SHOWN else ""
                text = (f"listened for error events from the driver {span}; "
                        f"{total} reported ({shown}{more})")
        out.append((CHECK_LABELS["driver_errors"], text))

    if isinstance(present.get("memtest"), dict):
        m = memtest_facts(result)
        if m is None:
            text = "could not be run, so memory cells were not tested"
        else:
            share = f" (about {m['percent']}% of the card's memory)" if m["percent"] else ""
            patterns = _plural(m["patterns"], "pattern") if m["patterns"] else "patterns"
            if m["mismatched"] is None:
                outcome = "the number of mismatches was not reported"
            elif m["mismatched"] == 0:
                outcome = "no mismatch"
            else:
                outcome = f"{_plural(m['mismatched'], 'word')} differed"
            early = "" if m["complete"] else "; it stopped before finishing"
            text = f"wrote {patterns} to {_gib(m['bytes'])}{share} and read them back: {outcome}{early}"
        out.append((CHECK_LABELS["memtest"], text))

    if isinstance(present.get("answer"), dict):
        a = answer_facts(result)
        if a is None:
            text = "could not be run, so computed results were not checked"
        else:
            if a["wrong"] is None:
                outcome = "the number of wrong results was not reported"
            elif a["wrong"] == 0:
                outcome = "none was wrong"
            else:
                outcome = f"{a['wrong']} wrong"
            early = "" if a["complete"] else "; it stopped before finishing"
            unused = f"; not used, because the card could not multiply them: {', '.join(a['skipped'])}" if a["skipped"] else ""
            text = f"compared {_plural(a['products'], 'computed product')} with known results: {outcome}{early}{unused}"
        out.append((CHECK_LABELS["answer"], text))

    b = check(result, "burn_in")
    done = _num(b.get("completed_s"))
    if done is not None and 0 < done <= 1e7:
        text = f"the load ran for {_duration(done)}"
        early, late = _num(b.get("early_tflops")), _num(b.get("late_tflops"))
        if early and late is not None and 0 < early <= 1e6 and 0 <= late <= 1e6:
            text += f"; throughput over its final part was {100 * late / early:.1f}% of its first ten minutes"
        out.append((CHECK_LABELS["burn_in"], text))

    if isinstance(present.get("nvlink"), dict):
        n = nvlink_facts(result)
        if n is None:
            if status(result, "nvlink") == NO_LINKS:
                text = "this card has no active NVLink link, so there was nothing to check"
            else:
                text = "the links could not be read, so they were not checked"
        else:
            parts = [f"{_plural(n['links'], 'link')}, {n['active']} active"]
            if n["dropped"]:
                parts.append(f"{_plural(n['dropped'], 'link')} went down during the run")
            if not n["counters_read"]:
                parts.append("error counters could not be read")
            else:
                risen = [f"{_COUNTER_WORDS[k]} +{v}" for k, v in n["rises"].items() if v]
                parts.append("error counters rose: " + ", ".join(risen) if risen else "error counters did not rise")
            t = n["transfer"]
            if t:
                if t["mismatched"] is None:
                    outcome = "the comparison was not reported"
                elif t["mismatched"] == 0:
                    outcome = "it came back unchanged"
                else:
                    outcome = f"{_plural(t['mismatched'], 'word')} came back changed"
                    if t["bad_peers"]:
                        outcome += f" (in the copies with {t['bad_peers']} of them)"
                how = ("the NVLink data counters rose during the copies" if t["over_nvlink"]
                       else "the NVLink data counters did not show the copies, so they may have travelled another way")
                parts.append(
                    f"copied {_gib(t['bytes'])} of test data to and from {_plural(t['peers'], 'other card')}: "
                    f"{outcome}; {how}"
                )
            text = "; ".join(parts)
        out.append((CHECK_LABELS["nvlink"], text))
    return out
