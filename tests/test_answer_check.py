"""The answer check's arithmetic on a list-backed stand-in for PyTorch (see minitorch.py)."""

import sys

import pytest

import minitorch
from gpucheck.checks import CheckUnavailable
from gpucheck.deepchecks import ANSWER_CASES, TorchAnswerCheck, _hadamard
from gpucheck.fakes import make_fake
from gpucheck.run import NullLoad, RunConfig, execute_run

SMALL = (("float32", 16, 16), ("float16", 16, 8), ("bfloat16", 16, 4), ("float64", 8, 8))


@pytest.fixture(autouse=True)
def torch(monkeypatch):
    minitorch.reset()
    monkeypatch.setitem(sys.modules, "torch", minitorch)
    return minitorch


class Clock:
    def __init__(self, tick=1.0):
        self.now, self.tick = 0.0, tick

    def __call__(self):
        self.now += self.tick
        return self.now


def check(**kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("cases", SMALL)
    kw.setdefault("seconds", 30.0)
    kw.setdefault("clock", Clock())
    return TorchAnswerCheck(**kw)


def run_to_end(c):
    c.start()
    while c.step():
        pass
    return c.result()


def test_hadamard_rows_are_orthogonal_with_entries_of_plus_and_minus_one(torch):
    h = _hadamard(torch, 16, "float32", "cpu")
    assert h.shape == (16, 16) and set(h.data) == {1, -1}
    gram = (h @ h.T).data
    assert [gram[i * 16 + j] for i in range(16) for j in range(16)] == [16 if i == j else 0 for i in range(16) for j in range(16)]


def test_a_sound_card_gets_every_product_right():
    r = run_to_end(check())
    assert r["products"] >= len(SMALL) and r["wrong_products"] == 0 and r["wrong_elements"] == 0
    assert [c["dtype"] for c in r["cases"]] == ["float32", "float16", "bfloat16", "float64"]
    assert all(c["products"] >= 1 for c in r["cases"]) and r["skipped"] == []


def test_the_known_result_has_one_entry_per_row_and_it_is_not_trivially_zero(torch):
    c = check()
    c.start()
    seen = []
    torch.faults.append(lambda result: seen.append(result))
    assert c._product(SMALL[0]) == 0
    (product,) = seen
    rows = product._rows()
    assert all(sum(1 for v in row if v) == 1 for row in rows)  # exactly one non-zero entry a row
    values = {abs(v) for row in rows for v in row if v}
    assert values <= {16 * s for s in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16)} and len(values) > 1


def test_every_partial_sum_is_exact_in_the_case_s_own_number_format():
    # A product's terms are +/-(scale a)*(scale b), `weight` of them non-zero per entry, so a
    # partial sum is at most weight * 16 and is a multiple of 1/16: whole multiples of a power of
    # two, at most `weight` of them. These are the limits each format holds exactly.
    exact = {"float32": 2**24, "float16": 2**11, "bfloat16": 2**8, "float64": 2**53}
    largest = {"float32": 3.4e38, "float16": 65504, "bfloat16": 3.4e38, "float64": 1.7e308}
    for name, n, weight in ANSWER_CASES:
        assert weight <= exact[name] and n % weight == 0 and weight & (weight - 1) == 0
        assert weight * 16 <= largest[name]


def test_a_single_wrong_entry_is_counted(torch):
    def corrupt(result):
        result.data[5] += 1  # one entry of every product comes back wrong

    torch.faults.append(corrupt)
    r = run_to_end(check())
    assert r["wrong_products"] == r["products"] > 0 and r["wrong_elements"] == r["products"]


def test_a_fault_in_one_number_format_is_attributed_to_it(torch):
    def corrupt(result):
        if result.dtype == "bfloat16":
            result.data[0] = float("nan")  # NaN is never equal to the known result

    torch.faults.append(corrupt)
    r = run_to_end(check())
    by = {c["dtype"]: c for c in r["cases"]}
    assert by["bfloat16"]["wrong_products"] == by["bfloat16"]["products"] > 0
    assert by["float32"]["wrong_products"] == 0 and r["wrong_products"] == by["bfloat16"]["products"]


def test_an_intermittent_fault_is_counted_only_when_it_happens(torch):
    calls = []

    def sometimes(result):
        calls.append(1)
        if len(calls) == 3:
            result.data[7] = -result.data[7] if result.data[7] else 1

    torch.faults.append(sometimes)
    r = run_to_end(check())
    assert r["wrong_products"] == 1 and r["products"] > 3


def test_it_runs_for_the_time_asked_and_at_least_once_through_every_format():
    clock = Clock(tick=1.0)
    c = check(seconds=10.0, clock=clock, step_s=2.0)
    r = run_to_end(c)
    assert 10.0 <= r["seconds"] <= 16.0
    short = run_to_end(check(seconds=0.0))
    assert all(case["products"] >= 1 for case in short["cases"])  # never zero products


def test_products_differ_from_one_to_the_next(torch):
    c = check()
    c.start()
    seen = []
    torch.faults.append(lambda result: seen.append(tuple(result.data)))
    c._product(SMALL[0])
    c._product(SMALL[0])
    assert seen[0] != seen[1]


def test_a_number_format_the_card_cannot_multiply_is_skipped_not_counted_wrong(torch):
    real = minitorch.Tensor.__matmul__

    def matmul(self, other):
        if self.dtype == "float16":
            raise RuntimeError('"addmm" not implemented for Half')
        return real(self, other)

    minitorch.Tensor.__matmul__ = matmul
    try:
        r = run_to_end(check())
    finally:
        minitorch.Tensor.__matmul__ = real
    assert r["skipped"] == ["float16"] and r["wrong_products"] == 0
    assert [c["dtype"] for c in r["cases"]] == ["float32", "bfloat16", "float64"]


def test_when_every_format_fails_to_multiply_the_check_did_not_run(torch):
    real = minitorch.Tensor.__matmul__

    def matmul(self, other):
        raise RuntimeError("CUDA error: unspecified launch failure")

    minitorch.Tensor.__matmul__ = matmul
    try:
        c = check()
        c.start()
        with pytest.raises(CheckUnavailable, match="every number format failed to multiply"):
            c.step()
    finally:
        minitorch.Tensor.__matmul__ = real
    r = execute_run(make_fake("healthy"), NullLoad(), RunConfig(5, 20), None,
                    clock=lambda: 0.0, sleep=lambda s: None, checks=[_AlwaysFails()])
    assert r.checks["answer"]["status"] == "unavailable"


class _AlwaysFails(TorchAnswerCheck):
    def __init__(self):
        super().__init__(device="cpu", cases=SMALL, clock=Clock())

    def _product(self, case):
        raise RuntimeError("CUDA error")


def test_no_usable_number_format_is_unavailable(torch, monkeypatch):
    monkeypatch.setattr(minitorch, "ones", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("unsupported")))
    with pytest.raises(CheckUnavailable, match="number formats"):
        check().start()


def test_without_pytorch_the_check_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    TorchAnswerCheck(2)  # constructing it needs no PyTorch
    with pytest.raises(CheckUnavailable, match="PyTorch is not installed"):
        TorchAnswerCheck(2).start()


def test_stop_releases_what_it_holds(torch):
    c = check(device="cuda:0")
    c.start()
    c.step()
    c.stop()
    assert c._base == {} and torch.cuda.emptied == 1


def test_in_a_run_the_answer_check_is_a_phase_after_the_load():
    c = check()
    r = execute_run(make_fake("healthy"), NullLoad(), RunConfig(5, 20), None,
                    clock=lambda: 0.0, sleep=lambda s: None, checks=[c])
    out = r.checks["answer"]
    assert out["status"] == "ran" and out["products"] >= 4 and out["wrong_products"] == 0


def test_self_test_runs_both_checks_and_reports_what_they_found(torch, capsys, monkeypatch):
    from gpucheck import deepchecks

    torch.__version__ = "stand-in"
    torch.cuda.is_available = lambda: False
    monkeypatch.setattr(deepchecks, "SELF_TEST_CASES", SMALL)
    monkeypatch.setattr(deepchecks, "MIB", 64)  # the stand-in holds numbers in Python lists
    assert deepchecks.self_test(seconds=0.0) == 0
    out = capsys.readouterr().out
    assert '"answer"' in out and '"memtest"' in out and '"device": "cpu"' in out
    assert "every call was accepted and nothing differed" in out

    torch.faults.append(lambda result: result.data.__setitem__(0, 12345))
    assert deepchecks.self_test(seconds=0.0) == 1
    assert "self-test: see above" in capsys.readouterr().out
