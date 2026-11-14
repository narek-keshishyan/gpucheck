"""The PyTorch checks against the real PyTorch, on a CPU. Skipped where PyTorch is not installed
(the default development environment); run with `uv run --with torch pytest tests/test_deepchecks_real_torch.py`.

The other tests use a list-backed stand-in, which cannot show that PyTorch accepts the calls or
that each number format really is exact. These do, for the CPU. A GPU's own routines are covered
only by the rented-hardware procedure."""

import pytest

torch = pytest.importorskip("torch")

from gpucheck.deepchecks import ANSWER_CASES, MIB, SELF_TEST_CASES, TorchAnswerCheck, TorchMemtest, self_test  # noqa: E402


def finish(check):
    check.start()
    while check.step():
        pass
    out = check.result()
    check.stop()
    return out


def test_self_test_passes_on_a_cpu(capsys):
    assert self_test("cpu", seconds=1.0) == 0
    assert "every call was accepted and nothing differed" in capsys.readouterr().out


@pytest.mark.parametrize("case", SELF_TEST_CASES + (("float16", 512, 128), ("bfloat16", 512, 64)))
def test_every_number_format_is_exact_on_a_working_machine(case):
    r = finish(TorchAnswerCheck(device="cpu", cases=(case,), seconds=0.5))
    assert r["skipped"] == [] and r["products"] >= 1 and r["wrong_products"] == 0


@pytest.mark.parametrize("case", ANSWER_CASES)
def test_the_full_size_cases_are_exact_too(case):
    r = finish(TorchAnswerCheck(device="cpu", cases=(case,), seconds=0.0))
    assert r["skipped"] == [] and r["products"] >= 1 and r["wrong_products"] == 0


def test_one_wrong_entry_in_a_real_product_is_counted(monkeypatch):
    real, calls = torch.Tensor.__matmul__, []

    def faulty(self, other):
        out = real(self, other)
        calls.append(1)
        if len(calls) == 2:
            out[3, 7] += 1
        return out

    monkeypatch.setattr(torch.Tensor, "__matmul__", faulty)
    r = finish(TorchAnswerCheck(device="cpu", cases=(("float32", 256, 256), ("bfloat16", 256, 64)), seconds=0.0))
    assert r["wrong_products"] == 1 and r["wrong_elements"] == 1


def test_sound_memory_reads_back_every_pattern():
    r = finish(TorchMemtest(device="cpu", budget_bytes=8 * MIB, chunk_bytes=2 * MIB, slice_words=1000,
                            min_bytes=MIB, step_s=0.0))
    assert r["bytes_tested"] == 8 * MIB and len(r["patterns"]) == 6 and r["mismatched_words"] == 0


def test_a_bit_that_flips_after_the_write_is_counted_in_every_pattern():
    m = TorchMemtest(device="cpu", budget_bytes=8 * MIB, chunk_bytes=2 * MIB, min_bytes=MIB, step_s=0.0)
    write = m._write

    def write_then_flip(name, value, c_i, chunk):
        write(name, value, c_i, chunk)
        if c_i == 1:
            chunk[12345] ^= 1 << 17

    m._write = write_then_flip
    r = finish(m)
    assert r["by_pattern"] == {name: 1 for name in r["patterns"]} and len(r["patterns"]) == 6
