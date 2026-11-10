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
