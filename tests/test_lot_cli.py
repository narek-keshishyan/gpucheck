import json
import re

import pytest

from gpucheck import cli
from gpucheck.cli import main
from gpucheck.fakes import make_fakes
from gpucheck.report import NOT_COVERED
from gpucheck.resultjson import run_id


def lot(tmp_path, fake, *flags):
    d = tmp_path / "lot"
    rc = main(["--fake", fake, "--offline", "--quick", "--lot-dir", str(d), *flags])
    summary = json.loads((d / "lot.json").read_text()) if (d / "lot.json").exists() else None
    return rc, d, summary


def test_lot_writes_per_card_results_and_a_summary(tmp_path, capsys):
    rc, d, s = lot(tmp_path, "tim_degraded,healthy", "--label", "Rack 4 / chassis 2")
    assert rc == 0
    assert re.fullmatch(r"[0-9a-f]{32}", s["lot_id"])
    assert s["label"] == "Rack 4 / chassis 2" and s["size"] == 2 and s["uploaded"] is None
    assert [c["outcome"] for c in s["cards"]] == ["tested", "tested"]
    assert [c["file"] for c in s["cards"]] == ["gpu0.json", "gpu1.json"]
    r0, r1 = (json.loads((d / f).read_text()) for f in ("gpu0.json", "gpu1.json"))
    assert r0["lot"] == {"lot_id": s["lot_id"], "position": 0, "size": 2, "label": "Rack 4 / chassis 2"}
    assert r1["lot"]["position"] == 1 and r1["inventory"]["gpu_index"] == 1
    assert s["cards"][0]["run_id"] == run_id(r0) and s["cards"][0]["uuid_hash"] == r0["inventory"]["uuid_hash"]
    assert "HOT_UNDER_LOAD" in s["cards"][0]["findings"]
    assert r0["timing"]["settle_s"] is None and r1["timing"]["settle_s"] == 30.0
    out = capsys.readouterr().out
    assert "Lot summary" in out and NOT_COVERED in out and str(d) in out


def test_refused_card_does_not_stop_the_lot(tmp_path):
    rc, d, s = lot(tmp_path, "healthy,busy,healthy")
    assert rc == 3
    assert [c["outcome"] for c in s["cards"]] == ["tested", "not_tested", "tested"]
    mid = s["cards"][1]
    assert mid["file"] is None and mid["run_id"] is None and "GPU busy" in mid["reason"]
    assert mid["uuid_hash"] and not (d / "gpu1.json").exists()


def test_aborted_card_keeps_its_partial_result(tmp_path):
    rc, d, s = lot(tmp_path, "healthy,overheat")
    assert rc == 3
    assert s["cards"][1]["outcome"] == "aborted" and s["cards"][1]["reason"] == "thermal"
    assert json.loads((d / "gpu1.json").read_text())["aborted"] == "thermal"


def test_all_cards_refused_exits_1_and_still_writes_the_summary(tmp_path):
    rc, _, s = lot(tmp_path, "busy,busy")
    assert rc == 1 and [c["outcome"] for c in s["cards"]] == ["not_tested", "not_tested"]


def test_card_that_cannot_be_read_is_recorded_and_skipped(tmp_path, monkeypatch):
    class Dead:
        def inventory(self):
            raise RuntimeError("GPU has fallen off the bus")

        def sample(self):
            raise RuntimeError("GPU has fallen off the bus")

        def busy(self):
            return None

    good = make_fakes("healthy,healthy")
    monkeypatch.setattr(cli, "make_fakes", lambda spec: [good[0], Dead(), good[1]])
    rc, _, s = lot(tmp_path, "healthy,healthy,healthy")
    assert rc == 3
    dead = s["cards"][1]
    assert dead["outcome"] == "not_tested" and "fallen off the bus" in dead["reason"]
    assert dead["uuid_hash"] is None and s["cards"][2]["outcome"] == "tested"


def test_existing_lot_directory_is_not_overwritten(tmp_path, capsys):
    lot(tmp_path, "healthy,healthy")
    before = (tmp_path / "lot" / "lot.json").read_text()
    rc, _, _ = lot(tmp_path, "healthy,healthy")
    assert rc == 1 and (tmp_path / "lot" / "lot.json").read_text() == before
    assert "already contains a lot" in capsys.readouterr().err


def test_certify_cannot_be_combined_with_several_cards(capsys):
    assert main(["--fake", "healthy,healthy", "--certify"]) == 1
    assert "--certify cannot be combined with a multi-GPU run" in capsys.readouterr().err


def test_label_longer_than_80_is_rejected():
    with pytest.raises(SystemExit):
        main(["--fake", "healthy,healthy", "--offline", "--label", "x" * 81])


def test_real_path_records_an_adapter_that_cannot_be_created(tmp_path, monkeypatch):
    from dataclasses import replace

    from gpucheck.adapter import AdapterUnavailable
    from gpucheck.fakes import make_fake
    from gpucheck.run import NullLoad, RunConfig

    made = []

    def nvml(index=0, include_serial=False):
        if index == 1:
            raise AdapterUnavailable("NVML unavailable: GPU is lost")
        card = make_fake("healthy")
        card._inventory = replace(card._inventory, gpu_index=index)
        return card

    monkeypatch.setattr(cli, "gpu_count", lambda: 3)
    monkeypatch.setattr(cli, "NvmlAdapter", nvml)
    monkeypatch.setattr(cli, "_make_load",
                        lambda args, challenge=None, device_index=0: made.append(("load", device_index)) or NullLoad())
    monkeypatch.setattr(cli, "_config", lambda args: RunConfig(baseline_s=2, load_s=2, hz=1000.0))
    monkeypatch.setattr(cli, "settle", lambda adapter, **kw: made.append("settle") or 12.0)
    d = tmp_path / "lot"
    rc = main(["--gpus", "all", "--offline", "--lot-dir", str(d)])
    s = json.loads((d / "lot.json").read_text())
    assert rc == 3
    assert [c["outcome"] for c in s["cards"]] == ["tested", "not_tested", "tested"]
    assert "GPU is lost" in s["cards"][1]["reason"] and s["cards"][1]["uuid_hash"] is None
    assert made == [("load", 0), "settle", ("load", 2)]
    assert json.loads((d / "gpu2.json").read_text())["timing"]["settle_s"] == 12.0
