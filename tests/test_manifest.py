from dataclasses import replace

from gpucheck.diagnose import diagnose
from gpucheck.fakes import make_fake
from gpucheck.metrics import derive
from gpucheck.resultjson import MANIFEST_FIELDS, build_result
from gpucheck.run import NullLoad, RunConfig, execute_run
from gpucheck.specs import lookup


def fake_run(**inventory):
    a = make_fake("healthy")
    spec = lookup(a.inventory().model_name)
    t = [0.0]
    run = execute_run(a, NullLoad(), RunConfig(30, 150, 1.0), spec,
                      clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    run = replace(run, inventory=replace(run.inventory, **inventory))
    m = derive(run, spec)
    return run, m, diagnose(m, run.preflight, run.inventory, spec)


def test_unknown_manifest_fields_are_omitted():
    inv = build_result(*fake_run())["inventory"]
    assert not set(MANIFEST_FIELDS) & set(inv)


def test_known_manifest_fields_are_written():
    inv = build_result(*fake_run(gpu_index=2, vbios_version="1.2", serial="S1"))["inventory"]
    assert inv["gpu_index"] == 2 and inv["vbios_version"] == "1.2" and inv["serial"] == "S1"
    assert "pcie_width_max" not in inv


LINK = dict(pcie_gen_max=4, pcie_width_max=16, pcie_width_current=8)


def test_link_read_under_load_is_recorded_and_shown_as_such():
    from gpucheck.report import render_report

    result = build_result(*fake_run(pcie_gen_current=4, pcie_link_under_load=True, **LINK))
    assert result["inventory"]["pcie_link_under_load"] is True
    assert "gen 4 x8 under load (max gen 4 x16)" in render_report(result)


def test_link_read_at_idle_is_shown_as_such_and_not_marked():
    from gpucheck.report import render_report

    result = build_result(*fake_run(pcie_gen_current=1, **LINK))
    assert "pcie_link_under_load" not in result["inventory"]
    assert "gen 1 x8 at idle (max gen 4 x16)" in render_report(result)
