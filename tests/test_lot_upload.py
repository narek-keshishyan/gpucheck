import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gpucheck.cli import main

LOT_OK = {"lot_id": "x", "url": "https://x.test/l/abc", "runs": [
    {"position": 0, "run_id": "r0", "url": "https://x.test/r/r0", "card_url": "https://x.test/g/c0",
     "grade": None, "n": 0, "reason": "insufficient reference (n=0)", "delete_token": "tok0"}]}


@pytest.fixture
def server():
    state = {"status": 200, "body": json.dumps(LOT_OK).encode(), "requests": []}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            state["requests"].append(
                (self.path, dict(self.headers), self.rfile.read(n))
            )
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            for k, v in state.get("headers", {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(state["body"])

        do_GET = do_POST  # a followed redirect shows up as a second recorded request

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}"
    yield state
    srv.shutdown()
    srv.server_close()



def run_lot(tmp_path, server, fake, *flags):
    d = tmp_path / "lot"
    rc = main(["--fake", fake, "--quick", "--lot-dir", str(d), "--yes-upload", "--server", server["url"], *flags])
    return rc, d


def test_lot_is_uploaded_in_one_request(server, tmp_path, capsys):
    rc, d = run_lot(tmp_path, server, "healthy,busy", "--api-key", "gpk_secret")
    assert rc == 3 and len(server["requests"]) == 1
    path, headers, raw = server["requests"][0]
    body = json.loads(raw)
    summary = json.loads((d / "lot.json").read_text())
    assert path == "/v1/lots" and headers["Authorization"] == "Bearer gpk_secret"
    assert body["lot"] == {"lot_id": summary["lot_id"], "size": 2, "label": None}
    assert [r["lot"]["position"] for r in body["results"]] == [0]
    assert body["results"][0] == json.loads((d / "gpu0.json").read_text())
    (u,) = body["untested"]
    assert u["position"] == 1 and "GPU busy" in u["reason"] and set(u) == {
        "position", "gpu_index", "model_name", "uuid_hash", "serial", "reason"}
    out = capsys.readouterr().out
    assert "https://x.test/l/abc" in out and "https://x.test/g/c0" in out
    assert summary["uploaded"]["url"] == "https://x.test/l/abc"
    assert summary["uploaded"]["runs"][0]["delete_token"] == "tok0"


@pytest.mark.parametrize("status", [200, 500])
def test_api_key_never_reaches_disk_or_terminal(server, tmp_path, capsys, status):
    server["status"] = status
    _, d = run_lot(tmp_path, server, "healthy,healthy", "--api-key", "gpk_secret")
    c = capsys.readouterr()
    assert "gpk_secret" not in c.out + c.err
    assert all("gpk_secret" not in p.read_text() for p in d.iterdir())


def test_api_key_from_environment(server, tmp_path, monkeypatch):
    monkeypatch.setenv("GPUCHECK_API_KEY", "gpk_env")
    run_lot(tmp_path, server, "healthy,healthy")
    assert server["requests"][0][1]["Authorization"] == "Bearer gpk_env"


@pytest.mark.parametrize("status", [409, 413, 500])
def test_failed_lot_upload_keeps_files_and_names_the_retry(server, tmp_path, capsys, status):
    server["status"] = status
    rc, d = run_lot(tmp_path, server, "healthy,healthy")
    assert rc == 0 and json.loads((d / "lot.json").read_text())["uploaded"] is None
    out = capsys.readouterr().out
    assert "upload failed" in out and f"--upload-saved {d}" in out


def test_account_upload_says_it_cannot_be_self_deleted(server, tmp_path, capsys):
    body = {**LOT_OK, "runs": [{**LOT_OK["runs"][0], "delete_token": None}]}
    server["body"] = json.dumps(body).encode()
    run_lot(tmp_path, server, "healthy,healthy", "--api-key", "gpk_secret")
    out = capsys.readouterr().out
    assert "Account uploads cannot be self-deleted." in out and "Delete token" not in out


def test_dry_run_and_offline_send_nothing_and_dry_run_hides_the_key(server, tmp_path, capsys):
    run_lot(tmp_path, server, "healthy,healthy", "--dry-run", "--api-key", "gpk_secret")
    out = capsys.readouterr().out
    assert '"untested"' in out and "gpk_secret" not in out and server["requests"] == []
    run_lot(tmp_path / "b", server, "healthy,healthy", "--offline")
    assert server["requests"] == []


def test_lot_401_names_the_key(server, tmp_path, capsys):
    server["status"] = 401
    run_lot(tmp_path, server, "healthy,healthy", "--api-key", "gpk_bad")
    out = capsys.readouterr().out
    assert "API key was rejected by the server" in out and "gpk_bad" not in out


def test_non_interactive_lot_needs_yes_upload(server, tmp_path, capsys):
    main(["--fake", "healthy,healthy", "--quick", "--lot-dir", str(tmp_path / "l"), "--server", server["url"]])
    out = capsys.readouterr().out
    assert server["requests"] == []
    assert f"gpucheck --upload-saved {tmp_path / 'l'}" in out and "re-run" not in out


def test_lot_upload_refuses_redirects(server, tmp_path, capsys):
    server["status"] = 302
    server["headers"] = {"Location": server["url"] + "/elsewhere"}
    server["body"] = b'{"runs": []}'
    _, d = run_lot(tmp_path, server, "healthy,healthy", "--api-key", "gpk_secret")
    out = capsys.readouterr().out
    assert len(server["requests"]) == 1
    assert "server redirected the upload (HTTP 302); check the --server URL" in out
    assert json.loads((d / "lot.json").read_text())["uploaded"] is None


def test_lot_upload_without_runs_list_is_still_a_success(server, tmp_path, capsys):
    server["body"] = json.dumps({"lot_id": "x", "url": "https://x.test/l/abc"}).encode()
    _, d = run_lot(tmp_path, server, "healthy,healthy")
    out = capsys.readouterr().out
    assert "upload failed" not in out and "https://x.test/l/abc" in out
    assert json.loads((d / "lot.json").read_text())["uploaded"]["runs"] == []


@pytest.mark.parametrize("error", [OSError("disk full"), TypeError("unexpected response")])
def test_delete_tokens_are_printed_when_lot_json_cannot_be_updated(server, tmp_path, capsys, monkeypatch, error):
    def boom(*a, **k):
        raise error
    monkeypatch.setattr("gpucheck.cli.mark_uploaded", boom)
    run_lot(tmp_path, server, "healthy,healthy")
    out = capsys.readouterr().out
    assert "GPU 0 delete token: tok0" in out and "lot.json" in out
    assert "Delete tokens were saved" not in out


def test_aborted_and_busy_cards_are_split_between_results_and_untested(server, tmp_path):
    run_lot(tmp_path, server, "healthy,overheat,busy")
    body = json.loads(server["requests"][0][2])
    assert [r["lot"]["position"] for r in body["results"]] == [0, 1]
    assert body["results"][1]["aborted"] == "thermal"
    assert [u["position"] for u in body["untested"]] == [2] and body["lot"]["size"] == 3


def test_nothing_is_uploaded_when_no_card_produced_a_result(server, tmp_path):
    rc, _ = run_lot(tmp_path, server, "busy,busy")
    assert rc == 1 and server["requests"] == []


def test_empty_key_argument_falls_back_to_environment(server, tmp_path, monkeypatch):
    monkeypatch.setenv("GPUCHECK_API_KEY", "gpk_env")
    run_lot(tmp_path, server, "healthy,healthy", "--api-key", "")
    assert server["requests"][0][1]["Authorization"] == "Bearer gpk_env"


def test_no_authorization_header_when_both_keys_are_empty(server, tmp_path, monkeypatch):
    monkeypatch.setenv("GPUCHECK_API_KEY", "")
    run_lot(tmp_path, server, "healthy,healthy", "--api-key", "")
    assert "Authorization" not in server["requests"][0][1]


ALREADY = "The server returned no delete tokens: this was already uploaded."


def test_malformed_position_in_the_response_keeps_the_other_tokens(server, tmp_path, capsys):
    odd = {"position": [1], "run_id": "r1", "url": "https://x.test/r/r1", "delete_token": "tok1"}
    server["body"] = json.dumps({**LOT_OK, "runs": [LOT_OK["runs"][0], odd]}).encode()
    rc, d = run_lot(tmp_path, server, "healthy,healthy")
    c = capsys.readouterr()
    assert rc == 0 and "Traceback" not in c.out + c.err
    runs = json.loads((d / "lot.json").read_text())["uploaded"]["runs"]
    assert [r["position"] for r in runs] == [0] and runs[0]["delete_token"] == "tok0"


def test_anonymous_lot_without_tokens_says_it_was_already_uploaded(server, tmp_path, capsys):
    server["body"] = json.dumps({**LOT_OK, "runs": [{**LOT_OK["runs"][0], "delete_token": None}]}).encode()
    run_lot(tmp_path, server, "healthy,healthy")
    out = capsys.readouterr().out
    assert ALREADY in out and "Account uploads" not in out


def test_account_lot_upload_does_not_say_already_uploaded(server, tmp_path, capsys):
    server["body"] = json.dumps({**LOT_OK, "runs": [{**LOT_OK["runs"][0], "delete_token": None}]}).encode()
    run_lot(tmp_path, server, "healthy,healthy", "--api-key", "gpk_secret")
    assert ALREADY not in capsys.readouterr().out


def test_ctrl_c_during_the_lot_upload_keeps_the_exit_code(server, tmp_path, capsys, monkeypatch):
    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("gpucheck.cli.upload_lot", interrupted)
    rc, d = run_lot(tmp_path, server, "healthy,busy")
    assert rc == 3
    out = capsys.readouterr().out
    assert f"upload interrupted; the lot is saved in {d}. Retry with: gpucheck --upload-saved {d}" in out
    assert json.loads((d / "lot.json").read_text())["uploaded"] is None


HTTP_REFUSAL = (
    "gpucheck: refusing to send the API key over plain http to example.invalid; use an https server URL"
)


def _no_network(monkeypatch):
    calls = []
    monkeypatch.setattr("gpucheck.upload._open", lambda *a, **k: calls.append(a) or pytest.fail("network call"))
    return calls


def test_lot_run_refuses_a_key_over_plain_http(tmp_path, capsys, monkeypatch):
    calls = _no_network(monkeypatch)
    d = tmp_path / "lot"
    rc = main(["--fake", "healthy,healthy", "--quick", "--lot-dir", str(d), "--yes-upload",
               "--server", "http://example.invalid", "--api-key", "gpk_secret"])
    c = capsys.readouterr()
    assert rc == 0 and calls == [] and HTTP_REFUSAL in c.err
    assert f"gpucheck --upload-saved {d}" in c.out and "gpk_secret" not in c.out + c.err
    assert json.loads((d / "lot.json").read_text())["uploaded"] is None
