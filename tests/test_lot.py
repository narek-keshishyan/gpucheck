import pytest

from gpucheck.fakes import make_fake
from gpucheck.lot import parse_gpus, settle


@pytest.mark.parametrize("spec,count,expected", [
    (None, 4, [0]), (None, 0, [0]), ("all", 4, [0, 1, 2, 3]), ("2", 4, [2]), ("0,2", 4, [0, 2]), ("2,0,2", 4, [0, 2]),
    ("all", 1, [0]),
])
def test_parse_gpus(spec, count, expected):
    assert parse_gpus(spec, count) == expected


@pytest.mark.parametrize("spec,count", [("4", 4), ("-1", 4), ("a", 4), ("", 4), ("0,,1", 4), ("all", 0)])
def test_parse_gpus_rejects_bad_input(spec, count):
    with pytest.raises(ValueError):
        parse_gpus(spec, count)


def test_parse_gpus_rejects_more_than_16():
    with pytest.raises(ValueError, match="16"):
        parse_gpus("all", 17)


class Temps:
    def __init__(self, fn):
        self._base, self._fn, self.i = make_fake("healthy"), fn, 0

    def sample(self):
        from dataclasses import replace

        s = replace(self._base.sample(), temp_c=self._fn(self.i))
        self.i += 1
        return s


def _settle(fn):
    slept = []
    return settle(Temps(fn), sleep=slept.append), slept


def test_settle_returns_after_one_window_when_flat():
    waited, slept = _settle(lambda i: 40.0)
    assert waited == 30.0 and slept == [1.0] * 30


def test_settle_waits_while_the_card_is_still_cooling():
    waited, _ = _settle(lambda i: max(40.0, 70.0 - 0.5 * i))
    assert waited == 89.0


def test_settle_gives_up_at_the_cap():
    waited, _ = _settle(lambda i: 500.0 - 0.1 * i)
    assert waited == 300.0


def test_settle_returns_at_once_when_temperature_is_unreadable():
    assert _settle(lambda i: None)[0] == 0.0


def test_settle_survives_a_failing_sample():
    class Dead:
        def sample(self):
            raise RuntimeError("fell off the bus")

    assert settle(Dead(), sleep=lambda s: None) == 0.0
