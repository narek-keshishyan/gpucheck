import json

import pytest

from gpucheck import cli
from gpucheck.cli import main
from test_lot_upload import LOT_OK, server  # noqa: F401  (server is a fixture)

RUN_OK = {"run_id": "abc", "url": "https://x.test/r/abc", "grade": None, "n": 0}


def _offline_lot(tmp_path, fake="healthy,healthy"):
    d = tmp_path / "lot"
    main(["--fake", fake, "--offline", "--quick", "--lot-dir", str(d)])
    return d


def _single(tmp_path):
    out = tmp_path / "r.json"
    main(["--fake", "healthy", "--offline", "--quick", "--json", str(out)])
    return out


def test_upload_saved_lot(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    lot_id = json.loads((d / "lot.json").read_text())["lot_id"]
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    path, _, raw = server["requests"][0]
    assert path == "/v1/lots" and json.loads(raw)["lot"]["lot_id"] == lot_id
    assert json.loads((d / "lot.json").read_text())["uploaded"]["url"] == "https://x.test/l/abc"
    out = capsys.readouterr().out
    assert "Lot page: https://x.test/l/abc" in out and "Delete tokens were saved" in out


def test_upload_saved_single_result_sends_the_exact_bytes(server, tmp_path, capsys):
    out = _single(tmp_path)
    server["body"] = json.dumps(RUN_OK).encode()
    assert main(["--upload-saved", str(out), "--server", server["url"]]) == 0
    path, _, raw = server["requests"][0]
    assert path == "/v1/runs" and raw == out.read_text().encode()
    assert "Result page: https://x.test/r/abc" in capsys.readouterr().out


def test_upload_saved_missing_path_is_friendly(tmp_path, capsys):
    assert main(["--upload-saved", str(tmp_path / "nope")]) == 1
    c = capsys.readouterr()
    assert "Traceback" not in c.out + c.err and len((c.out + c.err).strip().splitlines()) == 1


def test_upload_saved_failure_exits_1(server, tmp_path, capsys):
    server["status"] = 500
    d = _offline_lot(tmp_path)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 1
    c = capsys.readouterr()
    assert "Traceback" not in c.out + c.err
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


@pytest.mark.parametrize(
    "flag",
    [["--offline"], ["--dry-run"], ["--certify"], ["--fake", "healthy"], ["--gpus", "all"],
     ["--quick"], ["--long"], ["--label", "x"], ["--lot-dir", "d"], ["--include-serial"]],
)
def test_upload_saved_rejects_run_flags(tmp_path, flag):
    with pytest.raises(SystemExit):
        main(["--upload-saved", str(tmp_path), *flag])


def test_edited_card_file_is_refused(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    p = d / "gpu0.json"
    data = json.loads(p.read_text())
    data["inventory"]["model_name"] = "edited"
    p.write_text(json.dumps(data))
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 1
    c = capsys.readouterr()
    assert "gpu0.json" in c.out + c.err and len((c.out + c.err).strip().splitlines()) == 1
    assert server["requests"] == []


def test_missing_card_file_is_refused(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    (d / "gpu1.json").unlink()
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 1
    c = capsys.readouterr()
    assert "gpu1.json" in c.out + c.err and "Traceback" not in c.out + c.err
    assert server["requests"] == []


def test_lot_without_any_result_is_not_uploaded(server, tmp_path):
    d = _offline_lot(tmp_path, "busy,busy")
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 1
    assert server["requests"] == []


def test_not_a_result_file_is_refused(server, tmp_path, capsys):
    p = tmp_path / "x.json"
    p.write_text('{"hello": 1}')
    assert main(["--upload-saved", str(p), "--server", server["url"]]) == 1
    assert len(capsys.readouterr().err.strip().splitlines()) == 1
    p.write_text("not json")
    assert main(["--upload-saved", str(p), "--server", server["url"]]) == 1
    assert server["requests"] == []


def test_api_key_is_sent_for_lot_and_single(server, tmp_path):
    assert main(["--upload-saved", str(_offline_lot(tmp_path)), "--server", server["url"], "--api-key", "gpk_k"]) == 0
    server["body"] = json.dumps(RUN_OK).encode()
    assert main(["--upload-saved", str(_single(tmp_path)), "--server", server["url"], "--api-key", "gpk_k"]) == 0
    assert [r[1]["Authorization"] for r in server["requests"]] == ["Bearer gpk_k"] * 2


def test_reupload_keeps_recorded_delete_tokens(server, tmp_path):
    d = _offline_lot(tmp_path)
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    retry = {**LOT_OK, "runs": [{**LOT_OK["runs"][0], "delete_token": None}]}
    server["body"] = json.dumps(retry).encode()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    assert len(server["requests"]) == 2
    runs = json.loads((d / "lot.json").read_text())["uploaded"]["runs"]
    assert runs[0]["delete_token"] == "tok0"


def test_upload_saved_needs_no_gpu(server, tmp_path, monkeypatch):
    d = _offline_lot(tmp_path)

    def boom(*a, **k):
        raise AssertionError("touched the GPU")

    monkeypatch.setattr(cli, "gpu_count", boom)
    monkeypatch.setattr(cli, "NvmlAdapter", boom)
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0


def _edit_lot(d, fn):
    p = d / "lot.json"
    data = json.loads(p.read_text())
    fn(data)
    p.write_text(json.dumps(data))


def test_empty_response_never_erases_recorded_tokens(server, tmp_path):
    d = _offline_lot(tmp_path)
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    first = json.loads((d / "lot.json").read_text())["uploaded"]["runs"]
    assert first[0]["delete_token"] == "tok0"
    server["body"] = json.dumps({"lot_id": "x", "url": "https://x.test/l/abc", "runs": []}).encode()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    assert json.loads((d / "lot.json").read_text())["uploaded"]["runs"] == first


def test_other_server_keeps_previous_record(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    a = server["url"]
    assert main(["--upload-saved", str(d), "--server", a]) == 0
    capsys.readouterr()
    server["body"] = json.dumps({**LOT_OK, "url": "https://b.test/l/abc", "runs": [
        {**LOT_OK["runs"][0], "delete_token": None}]}).encode()
    assert main(["--upload-saved", str(d), "--server", a + "/b"]) == 0
    out = capsys.readouterr().out
    assert (
        f"note: this lot was uploaded to {a} before; its delete tokens for that server "
        'will be kept in lot.json under "previous_uploads".'
    ) in out
    lot = json.loads((d / "lot.json").read_text())
    assert lot["uploaded"]["server"] == a + "/b"
    assert lot["uploaded"]["runs"][0]["delete_token"] is None
    (prev,) = lot["previous_uploads"]
    assert prev["server"] == a and prev["runs"][0]["delete_token"] == "tok0"


@pytest.mark.parametrize("name", ["../x.json", "/etc/hosts", "lot.json", "gpu0.txt", "sub/gpu0.json"])
def test_unsafe_card_file_name_is_refused(server, tmp_path, capsys, name):
    d = _offline_lot(tmp_path)
    _edit_lot(d, lambda s: s["cards"][0].__setitem__("file", name))
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 1
    c = capsys.readouterr()
    assert name in c.err and len(c.err.strip().splitlines()) == 1
    assert server["requests"] == []


def test_symlinked_card_file_is_refused(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes((d / "gpu0.json").read_bytes())
    (d / "gpu0.json").unlink()
    try:
        (d / "gpu0.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 1
    assert "gpu0.json" in capsys.readouterr().err
    assert server["requests"] == []


@pytest.mark.parametrize("damage", [
    lambda s: s.pop("size"),
    lambda s: s.__setitem__("size", "2"),
    lambda s: s["cards"][0].pop("outcome"),
    lambda s: s["cards"].append("x"),
    lambda s: s.__setitem__("finished", "no"),
])
def test_damaged_lot_json_is_refused_clearly(server, tmp_path, capsys, damage):
    d = _offline_lot(tmp_path)
    _edit_lot(d, damage)
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 1
    err = capsys.readouterr().err
    assert "not a valid" in err and "lot.json" in err and len(err.strip().splitlines()) == 1
    assert server["requests"] == []


def test_failed_run_upload_then_upload_saved_records_tokens(server, tmp_path):
    d = tmp_path / "lot"
    server["status"] = 500
    main(["--fake", "healthy,healthy", "--quick", "--lot-dir", str(d), "--yes-upload", "--server", server["url"]])
    assert json.loads((d / "lot.json").read_text())["uploaded"] is None
    server["status"] = 200
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    runs = json.loads((d / "lot.json").read_text())["uploaded"]["runs"]
    assert runs[0]["delete_token"] == "tok0"


class _Killed(BaseException):
    pass


def test_killed_lot_can_be_uploaded_with_the_cards_it_reached(server, tmp_path, monkeypatch, capsys):
    real = cli.execute_run
    calls = []

    def dies_on_second_card(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise _Killed()
        return real(*a, **kw)

    monkeypatch.setattr(cli, "execute_run", dies_on_second_card)
    d = tmp_path / "lot"
    with pytest.raises(_Killed):
        main(["--fake", "healthy,healthy,healthy", "--offline", "--quick", "--lot-dir", str(d)])
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    body = json.loads(server["requests"][0][2])
    assert [r["lot"]["position"] for r in body["results"]] == [0]
    assert [u["position"] for u in body["untested"]] == [1, 2] and body["lot"]["size"] == 3
    assert "finished" not in body["lot"]
    out = capsys.readouterr().out
    assert "note: this lot did not finish; cards that were not reached are recorded as not tested" in out
    assert sorted(p.name for p in d.iterdir()) == ["gpu0.json", "lot.json"]


def test_lot_json_without_finished_is_a_finished_lot(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    _edit_lot(d, lambda s: s.pop("finished"))
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    assert "did not finish" not in capsys.readouterr().out


def test_finished_lot_has_no_note(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    capsys.readouterr()
    assert main(["--upload-saved", str(d), "--server", server["url"]]) == 0
    assert "did not finish" not in capsys.readouterr().out


def test_a_card_file_of_a_lot_is_not_uploaded_on_its_own(server, tmp_path, capsys):
    d = _offline_lot(tmp_path)
    capsys.readouterr()
    assert main(["--upload-saved", str(d / "gpu0.json"), "--server", server["url"]]) == 1
    err = capsys.readouterr().err
    assert err.strip() == (
        f"gpucheck: {d / 'gpu0.json'} is part of a lot; upload its directory instead: gpucheck --upload-saved {d}"
    )
    assert server["requests"] == []


def test_upload_saved_refuses_a_key_over_plain_http(tmp_path, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr("gpucheck.upload._open", lambda *a, **k: calls.append(a) or pytest.fail("network call"))
    d = _offline_lot(tmp_path)
    single = _single(tmp_path)
    capsys.readouterr()
    for path in (d, single):
        before = {p.name: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
        assert main(["--upload-saved", str(path), "--server", "http://example.invalid", "--api-key", "gpk_k"]) == 1
        assert {p.name: p.read_bytes() for p in path.parent.iterdir() if p.is_file()} == before
        err = capsys.readouterr().err
        assert "gpucheck: refusing to send the API key over plain http to example.invalid; use an https server URL" in err
        assert "gpk_k" not in err
    assert calls == []
