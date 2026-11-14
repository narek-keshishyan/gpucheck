"""The memory test's logic on a list-backed stand-in for PyTorch (see minitorch.py)."""

import sys

import pytest

import minitorch
from gpucheck.checks import CheckUnavailable
from gpucheck.deepchecks import MEMTEST_PATTERNS, TorchMemtest
from gpucheck.fakes import make_fake
from gpucheck.run import NullLoad, RunConfig, execute_run

NAMES = [name for name, _ in MEMTEST_PATTERNS]


@pytest.fixture(autouse=True)
def torch(monkeypatch):
    minitorch.reset()
    monkeypatch.setitem(sys.modules, "torch", minitorch)
    return minitorch


def memtest(**kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("budget_bytes", 4096)
    kw.setdefault("chunk_bytes", 1024)  # 128 words a chunk
    kw.setdefault("slice_words", 50)  # so a chunk is compared in three slices
    kw.setdefault("min_bytes", 64)
    kw.setdefault("step_s", 0.0)  # one unit of work per step
    return TorchMemtest(**kw)


def run_to_end(check):
    check.start()
    steps = 0
    while check.step():
        steps += 1
    return steps


def test_sound_memory_passes_every_pattern():
    m = memtest()
    steps = run_to_end(m)
    r = m.result()
    assert r["bytes_tested"] == 4096 and r["patterns"] == NAMES and r["mismatched_words"] == 0
    assert r["by_pattern"] == {} and r["seconds"] >= 0
    assert steps == len(NAMES) * 2 * 4  # every chunk written, then every chunk read, per pattern


def test_patterns_are_the_documented_bit_patterns():
    values = dict(MEMTEST_PATTERNS)
    assert values["zeros"] == 0 and values["ones"] & (2**64 - 1) == 2**64 - 1
    assert values["checker_a"] & (2**64 - 1) == 0xAAAAAAAAAAAAAAAA
    assert values["checker_5"] == 0x5555555555555555
    assert values["address"] is None and values["random"] is None


@pytest.mark.parametrize("pattern", NAMES)
def test_a_word_that_changes_after_it_is_written_is_counted(pattern):
    m = memtest()
    real = m._write

    def write_then_corrupt(name, value, c_i, chunk):
        real(name, value, c_i, chunk)
        if name == pattern and c_i == 2:
            chunk.data[77] ^= 1 << 40  # one stuck bit, in the second slice of the third chunk

    m._write = write_then_corrupt
    run_to_end(m)
    r = m.result()
    assert r["mismatched_words"] == 1 and r["by_pattern"] == {pattern: 1}
    assert r["patterns"] == NAMES  # the test goes on after a mismatch


def test_address_pattern_differs_between_chunks_and_words():
    m = memtest()
    m.start()
    seen = set()
    for c_i in range(4):
        seen |= set(m._expected("address", c_i, 0, 0, 128).data)
    assert len(seen) == 4 * 128


def test_random_pattern_is_reproducible_and_differs_between_slices():
    m = memtest()
    m.start()
    a = m._expected("random", 1, 0, 0, 50).data
    assert a == m._expected("random", 1, 0, 0, 50).data
    assert a != m._expected("random", 1, 1, 50, 100).data and a != m._expected("random", 2, 0, 0, 50).data


def test_memory_that_was_never_written_does_not_pass():
    m = memtest()
    m._write = lambda name, value, c_i, chunk: None  # writes are lost
    run_to_end(m)
    assert m.result()["mismatched_words"] == len(NAMES) * 512


def test_budget_is_most_of_free_memory_less_a_reserve(torch):
    torch.cuda.free, torch.cuda.total = 100_000, 160_000
    m = memtest(device=None, budget_bytes=None, reserve_bytes=20_000, chunk_bytes=8000)
    m.start()
    assert m.result()["bytes_tested"] == 80_000 and m.result()["bytes_total"] == 160_000  # min(90%, free - reserve)
    torch.cuda.free = 1_000_000
    m = memtest(device=None, budget_bytes=None, reserve_bytes=20_000, chunk_bytes=80_000)
    m.start()
    assert m.result()["bytes_tested"] == 900_000


def test_too_little_free_memory_is_unavailable(torch):
    torch.cuda.free = 2 << 30  # with the real 1 GiB reserve and 90% rule this is plenty...
    torch.cuda.free = 1 << 29  # ...but half a gigabyte leaves nothing after the reserve
    with pytest.raises(CheckUnavailable, match="not enough free memory"):
        TorchMemtest(0).start()


def test_out_of_memory_partway_tests_what_was_allocated(torch):
    torch.cuda.fail_alloc_after = 3
    m = memtest(device="cuda:0", budget_bytes=4096)
    m.start()
    torch.cuda.fail_alloc_after = None  # the comparisons' working memory comes out of the reserve
    while m.step():
        pass
    assert m.result()["bytes_tested"] == 3 * 1024 and m.result()["mismatched_words"] == 0


def test_out_of_memory_at_once_is_unavailable_and_holds_nothing(torch):
    torch.cuda.fail_alloc_after = 0
    m = memtest(device="cuda:0")
    with pytest.raises(CheckUnavailable, match="could not allocate"):
        m.start()
    assert m._chunks == []


def test_stop_releases_the_memory(torch):
    m = memtest(device="cuda:1")
    m.start()
    m.step()
    m.stop()
    assert m._chunks == [] and torch.cuda.emptied == 1 and m.step() is False
    assert m.result()["patterns"] == []  # stopped before the first pattern was complete


def test_without_pytorch_the_check_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(CheckUnavailable, match="PyTorch is not installed"):
        TorchMemtest(0).start()


def test_constructing_the_check_touches_neither_pytorch_nor_the_card(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    TorchMemtest(3)  # no import, no allocation


def test_unreadable_free_memory_is_unavailable(torch):
    def boom(device):
        raise RuntimeError("CUDA error")

    torch.cuda.mem_get_info = boom
    with pytest.raises(CheckUnavailable, match="could not read the card's free memory"):
        TorchMemtest(0).start()


def test_in_a_run_the_memory_test_is_its_own_phase_after_the_load():
    m = memtest()
    r = execute_run(make_fake("healthy"), NullLoad(), RunConfig(5, 20), None,
                    clock=lambda: 0.0, sleep=lambda s: None, checks=[m])
    out = r.checks["memtest"]
    assert out["status"] == "ran" and out["bytes_tested"] == 4096 and out["mismatched_words"] == 0
    assert m._chunks == [] and len(r.load) == 20


def test_the_random_pattern_uses_all_64_bits():
    m = memtest()
    m.start()
    words = m._expected("random", 0, 0, 0, 50).data
    assert any(w < 0 for w in words) and any(w >= 0 for w in words)  # the top bit varies too
    assert all(-(1 << 63) <= w < (1 << 63) for w in words)
