"""All cards at once: a lot whose cards are loaded at the same time."""

import json
import threading

import pytest

from gpucheck import cli
from gpucheck.cli import main
from gpucheck.fakes import make_fakes
from gpucheck.run import NullLoad


def together(tmp_path, fake, *flags, name="lot"):
    d = tmp_path / name
    rc = main(["--fake", fake, "--offline", "--quick", "--together", "--lot-dir", str(d), *flags])
    summary = json.loads((d / "lot.json").read_text())
    cards = {c["position"]: json.loads((d / c["file"]).read_text()) for c in summary["cards"] if c["file"]}
    return rc, summary, cards


def test_every_card_is_marked_as_measured_together_and_measurements_only(tmp_path, capsys):
    rc, summary, cards = together(tmp_path, "healthy,healthy,healthy")
    assert rc == 0 and summary["together"] is True and summary["finished"] is True
    assert [c["outcome"] for c in summary["cards"]] == ["tested"] * 3
    for pos, r in cards.items():
        assert r["checks"]["together"] == {"cards": 3}
        assert r["preflight"]["mode"] == "measurements_only"
        assert r["preflight"]["reasons"] == ["loaded together with 2 other cards: measurements only"]
        assert r["lot"]["position"] == pos and r["timing"]["settle_s"] is None
        assert len(r["samples"]["baseline"]) == 30 and len(r["samples"]["load"]) == 150
    out = capsys.readouterr().out
    assert "Testing 3 GPUs at the same time (GPU 0, 1, 2)..." in out
    assert "NOTE: loaded together with other cards - measurements only, no grade is given." in out
    assert "Loaded together: this card was loaded at the same time as 2 other cards in the same machine" in out
    assert "The cards were loaded at the same time: measurements only, not graded." in out


def test_a_sequential_lot_is_not_marked(tmp_path):
    d = tmp_path / "lot"
    main(["--fake", "healthy,healthy", "--offline", "--quick", "--lot-dir", str(d)])
    assert "together" not in json.loads((d / "lot.json").read_text())
    r = json.loads((d / "gpu1.json").read_text())
    assert "checks" not in r and r["preflight"]["mode"] == "full"


class _Card:
    """A simulated card that counts its samples; `log` is shared by every card of the lot."""

    def __init__(self, card, log, index):
        self._card, self._log, self.index = card, log, index
        self.samples = 0
        self.inventory, self.busy = card.inventory, card.busy
        self.extras = card.extras

    def sample(self):
        self.samples += 1
        return self._card.sample()


class _Load(NullLoad):
    def __init__(self, card, cards, log):
        self._card, self._cards, self._log = card, cards, log

    def start(self):
        # 1 preflight sample + 30 baseline samples: has every card finished its idle baseline?
        self._log.append(("start", self._card.index, [c.samples for c in self._cards]))

    def tick(self):
        self._log.append(("tick", self._card.index))
        return None


def _instrument(monkeypatch, spec):
    log = []
    cards = [_Card(c, log, i) for i, c in enumerate(make_fakes(spec))]
    monkeypatch.setattr(cli, "make_fakes", lambda s: cards)
    monkeypatch.setattr(cli, "_make_load", lambda args, challenge=None, device_index=0: _Load(cards[device_index], cards, log))
    return cards, log


def test_no_load_starts_until_every_card_has_finished_its_idle_baseline(tmp_path, monkeypatch):
    cards, log = _instrument(monkeypatch, "healthy,healthy,healthy,healthy")
    rc, _, _ = together(tmp_path, "healthy,healthy,healthy,healthy")
    starts = [e for e in log if e[0] == "start"]
    assert rc == 0 and len(starts) == 4
    for _, _, samples in starts:
        assert all(n >= 31 for n in samples)  # nobody is still idling while another card is loaded
    first_tick = next(i for i, e in enumerate(log) if e[0] == "tick")
    assert first_tick >= 1  # at least one start precedes any work


def test_a_card_that_cannot_be_tested_does_not_hold_the_others_up(tmp_path, monkeypatch):
    cards, log = _instrument(monkeypatch, "healthy,busy,healthy")
    rc, summary, results = together(tmp_path, "healthy,busy,healthy")
    assert rc == 3
    assert [c["outcome"] for c in summary["cards"]] == ["tested", "not_tested", "tested"]
    assert summary["cards"][1]["reason"] == "GPU busy: compute process running on a full card"
    assert [r["checks"]["together"] for r in results.values()] == [{"cards": 2}, {"cards": 2}]
    assert sorted(e[1] for e in log if e[0] == "start") == [0, 2]


def test_a_card_left_on_its_own_is_an_ordinary_run(tmp_path):
    rc, summary, results = together(tmp_path, "healthy,busy,busy")
    assert rc == 3
    (r,) = results.values()
    assert "checks" not in r and r["preflight"]["mode"] == "full" and r["preflight"]["reasons"] == []


def test_a_card_that_overheats_stops_alone_and_keeps_its_partial_result(tmp_path):
    rc, summary, results = together(tmp_path, "healthy,overheat,healthy")
    assert rc == 3
    assert [(c["outcome"], c["reason"]) for c in summary["cards"]] == [
        ("tested", None), ("aborted", "thermal"), ("tested", None)]
    assert results[1]["aborted"] == "thermal" and results[1]["checks"]["together"] == {"cards": 3}
    assert len(results[0]["samples"]["load"]) == 150


def test_a_mig_card_keeps_its_own_reason_too(tmp_path):
    _, _, results = together(tmp_path, "healthy,mig")
    assert results[1]["preflight"]["mode"] == "measurements_only"
    assert results[1]["preflight"]["reasons"] == [
        "MIG enabled: measurements only", "loaded together with 1 other card: measurements only"]


def test_a_card_that_cannot_be_read_or_whose_load_cannot_be_built_is_recorded(tmp_path, monkeypatch):
    cards, _ = _instrument(monkeypatch, "healthy,healthy,healthy,healthy")

    def broken():
        raise RuntimeError("GPU is lost")

    cards[1].inventory = broken
    real = cli._make_load

    def make_load(args, challenge=None, device_index=0):
        if device_index == 2:
            raise RuntimeError("CUDA error: out of memory\nwhile building")
        return real(args, challenge, device_index)

    monkeypatch.setattr(cli, "_make_load", make_load)
    rc, summary, results = together(tmp_path, "healthy,healthy,healthy,healthy")
    assert rc == 3
    assert [c["outcome"] for c in summary["cards"]] == ["tested", "not_tested", "not_tested", "tested"]
    assert summary["cards"][1]["reason"] == "could not read the card: GPU is lost"
    assert summary["cards"][2]["reason"] == "run failed: RuntimeError: CUDA error: out of memory while building"
    assert sorted(results) == [0, 3] and results[0]["checks"]["together"] == {"cards": 2}


def test_the_extra_checks_run_on_every_card(tmp_path):
    rc, summary, results = together(tmp_path, "healthy,xid,bad_memory", "--driver-errors", "--memtest", "--answer-check")
    assert rc == 0
    assert [c["findings"] for c in summary["cards"]] == [[], ["DRIVER_ERRORS"], ["MEMTEST_MISMATCH"]]
    assert all(sorted(r["checks"]) == ["answer", "driver_errors", "memtest", "together"] for r in results.values())


def test_together_needs_at_least_two_cards(tmp_path, capsys):
    assert main(["--fake", "healthy", "--offline", "--quick", "--together"]) == 1
    assert "--together needs at least two GPUs" in capsys.readouterr().err
    assert main(["--fake", "healthy,healthy", "--offline", "--quick", "--together", "--gpus", "1"]) == 1


def test_together_cannot_be_certified(tmp_path, capsys):
    rc = main(["--fake", "healthy,healthy", "--together", "--certify", "--lot-dir", str(tmp_path / "lot")])
    assert rc == 1 and "--certify cannot be combined with a lot" in capsys.readouterr().err


def test_the_lot_request_carries_the_marker_in_every_result(tmp_path, capsys):
    d = tmp_path / "lot"
    main(["--fake", "healthy,healthy", "--quick", "--together", "--dry-run", "--lot-dir", str(d)])
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(body["lot"]) == {"lot_id", "size", "label"}  # the lot header is what it always was
    for r in body["results"]:
        assert r["checks"]["together"] == {"cards": 2} and r["preflight"]["mode"] == "measurements_only"
        assert set(r["lot"]) == {"lot_id", "size", "label", "position"}


class _Blocking(_Card):
    """Blocks at sample `at` until `go` is set, so a test can interrupt at a known point."""

    def __init__(self, card, index, at, reached, go):
        super().__init__(card, [], index)
        self._at, self._reached, self._go = at, reached, go

    def sample(self):
        if self.samples == self._at:
            self._reached.release()
            self._go.wait(10)
        return super().sample()


def _interrupt_when_blocked(monkeypatch, at):
    reached, go = threading.Semaphore(0), threading.Event()
    cards = [_Blocking(c, i, at, reached, go) for i, c in enumerate(make_fakes("healthy,healthy"))]
    monkeypatch.setattr(cli, "make_fakes", lambda s: cards)
    real, calls = cli._join, []

    def join(threads, timeout_s=None):
        calls.append(1)
        if len(calls) == 1:
            for _ in cards:
                reached.acquire(timeout=10)  # every card is now waiting at the same sample
            raise KeyboardInterrupt
        go.set()
        real(threads, timeout_s)

    monkeypatch.setattr(cli, "_join", join)


def test_ctrl_c_under_load_keeps_what_every_card_measured(tmp_path, monkeypatch, capsys):
    _interrupt_when_blocked(monkeypatch, at=100)  # 1 preflight + 30 baseline + 69 load samples in
    rc, summary, results = together(tmp_path, "healthy,healthy", "--yes-upload")
    assert rc == 130 and summary["finished"] is True
    assert [(c["outcome"], c["reason"]) for c in summary["cards"]] == [("aborted", "interrupted")] * 2
    for r in results.values():
        assert r["aborted"] == "interrupted" and 60 <= len(r["samples"]["load"]) <= 80
        assert r["checks"]["together"] == {"cards": 2}
    assert "Interrupted: stopping every card at its next sample" in capsys.readouterr().out


def test_ctrl_c_before_any_load_leaves_the_cards_not_tested(tmp_path, monkeypatch):
    _interrupt_when_blocked(monkeypatch, at=5)
    rc, summary, results = together(tmp_path, "healthy,healthy")
    assert rc == 130 and results == {}
    assert [(c["outcome"], c["reason"]) for c in summary["cards"]] == [("not_tested", "interrupted")] * 2


def test_a_second_ctrl_c_keeps_nothing_and_still_writes_the_summary(tmp_path, monkeypatch):
    def join(threads, timeout_s=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_join", join)
    rc, summary, results = together(tmp_path, "healthy,healthy")
    assert rc == 130 and results == {}
    assert [c["reason"] for c in summary["cards"]] == ["interrupted", "interrupted"]


def test_start_gate_opens_when_everyone_has_arrived_or_withdrawn():
    gate = cli._StartGate(3)
    passed = []

    def arrive(i):
        gate.arrive()
        passed.append(i)

    threads = [threading.Thread(target=arrive, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    threads[0].join(0.1)
    assert passed == []  # two of three are there: still waiting
    gate.withdraw()
    for t in threads:
        t.join(5)
    assert sorted(passed) == [0, 1] and gate.arrived == 2


def test_ctrl_c_while_cards_wait_to_start_does_not_start_any_load(tmp_path, monkeypatch):
    cards, log = _instrument(monkeypatch, "healthy,healthy")
    waiting, go = threading.Event(), threading.Event()
    real_sample = cards[1].sample

    def slow_baseline():  # card 1 is still idling while card 0 waits at the gate
        if cards[1].samples == 20:
            waiting.set()
            go.wait(10)
        return real_sample()

    cards[1].sample = slow_baseline
    real, calls = cli._join, []

    def join(threads, timeout_s=None):
        calls.append(1)
        if len(calls) == 1:
            waiting.wait(10)
            while cards[0].samples < 31:  # card 0 has finished its baseline and is at the gate
                pass
            raise KeyboardInterrupt
        go.set()
        real(threads, timeout_s)

    monkeypatch.setattr(cli, "_join", join)
    rc, summary, results = together(tmp_path, "healthy,healthy")
    assert rc == 130 and results == {} and [e for e in log if e[0] == "start"] == []
    assert [(c["outcome"], c["reason"]) for c in summary["cards"]] == [("not_tested", "interrupted")] * 2


def test_a_card_whose_load_fails_to_start_is_not_counted_as_loaded_together(tmp_path, monkeypatch):
    cards, log = _instrument(monkeypatch, "healthy,healthy,healthy")
    real = cli._make_load

    class NoStart(NullLoad):
        def start(self):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cli, "_make_load", lambda args, challenge=None, device_index=0:
                        NoStart() if device_index == 1 else real(args, challenge, device_index))
    rc, summary, results = together(tmp_path, "healthy,healthy,healthy")
    assert [c["outcome"] for c in summary["cards"]] == ["tested", "not_tested", "tested"]
    assert [r["checks"]["together"] for r in results.values()] == [{"cards": 2}, {"cards": 2}]


def test_a_card_that_never_stops_does_not_lose_what_the_others_measured(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "STOP_WAIT_S", 0.3)
    cards = make_fakes("healthy,healthy")
    hung, release = threading.Event(), threading.Event()
    real_sample = cards[1].sample
    seen = [0]

    def hangs():  # a driver call that never returns, 20 samples into the load
        seen[0] += 1
        if seen[0] == 51:
            hung.set()
            release.wait(30)
        return real_sample()

    cards[1].sample = hangs
    monkeypatch.setattr(cli, "make_fakes", lambda s: cards)
    real, calls = cli._join, []

    def join(threads, timeout_s=None):
        calls.append(1)
        if len(calls) == 1:
            hung.wait(10)
            threads[0].wait(10)  # card 0 runs to its end
            raise KeyboardInterrupt
        real(threads, timeout_s)

    monkeypatch.setattr(cli, "_join", join)
    try:
        rc, summary, results = together(tmp_path, "healthy,healthy")
    finally:
        release.set()
    assert rc == 130 and sorted(results) == [0]
    assert [(c["outcome"], c["reason"]) for c in summary["cards"]] == [
        ("tested", None), ("not_tested", "did not stop after the interrupt")]
    assert len(results[0]["samples"]["load"]) == 150
    # It ran alongside the card that hung, so it is still marked and can never be a reference.
    assert results[0]["checks"]["together"] == {"cards": 2} and results[0]["preflight"]["mode"] == "measurements_only"


def test_cards_that_are_ready_do_not_wait_for_ever_for_one_that_is_not():
    gate = cli._StartGate(2, wait_s=0.2)
    t0 = cli.time.monotonic()
    gate.arrive()  # the other card never arrives and never withdraws
    assert 0.15 < cli.time.monotonic() - t0 < 5 and gate.arrived == 1


def test_a_real_ctrl_c_keeps_every_card_even_when_it_lands_while_waiting_for_the_first(tmp_path, monkeypatch):
    # Found on a rented two-card machine: Ctrl-C arrived while the main thread was waiting for
    # card 0, Python marked that still-running thread as finished, and its measurements were lost.
    import os
    import signal
    import time

    cards = make_fakes("healthy,healthy")
    under_load = threading.Event()

    def slow(card):
        real = card.sample
        seen = [0]

        def sample():
            seen[0] += 1
            if seen[0] > 40:  # under load: every sample takes real time, as a real load burst does
                under_load.set()
                time.sleep(0.05)
            return real()

        card.sample = sample

    for card in cards:
        slow(card)
    monkeypatch.setattr(cli, "make_fakes", lambda s: cards)

    def press_ctrl_c():
        under_load.wait(10)
        time.sleep(0.3)  # the main thread is by now waiting for the cards
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=press_ctrl_c, daemon=True).start()
    rc, summary, results = together(tmp_path, "healthy,healthy")
    assert rc == 130
    assert [(c["outcome"], c["reason"]) for c in summary["cards"]] == [("aborted", "interrupted")] * 2
    assert sorted(results) == [0, 1] and all(r["samples"]["load"] for r in results.values())
