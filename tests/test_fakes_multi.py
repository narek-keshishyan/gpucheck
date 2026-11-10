import pytest

from gpucheck.adapter import hash_uuid
from gpucheck.fakes import FAKE_UUID, make_fake, make_fakes
from gpucheck.run import NullLoad, RunConfig, execute_run
from gpucheck.specs import lookup


def test_make_fakes_gives_distinct_cards():
    invs = [c.inventory() for c in make_fakes("healthy,tim_degraded,healthy")]
    assert [i.gpu_index for i in invs] == [0, 1, 2]
    assert len({i.uuid_hash for i in invs}) == 3


def test_single_profile_is_the_original_fake():
    (only,) = make_fakes("healthy")
    assert only.inventory() == make_fake("healthy").inventory()
    assert only.inventory().uuid_hash == hash_uuid(FAKE_UUID) and only.inventory().gpu_index is None


def test_busy_profile_reports_busy():
    assert make_fakes("busy")[0].busy() is True


def test_overheat_profile_aborts_on_thermal():
    a = make_fakes("overheat")[0]
    t = [0.0]
    run = execute_run(a, NullLoad(), RunConfig(5, 10, 1.0), lookup(a.inventory().model_name),
                      clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    assert run.aborted == "thermal"


def test_unknown_profile_in_list_is_named():
    with pytest.raises(ValueError, match="nope"):
        make_fakes("healthy,nope")
