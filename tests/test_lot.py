import pytest

from gpucheck.lot import parse_gpus


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
