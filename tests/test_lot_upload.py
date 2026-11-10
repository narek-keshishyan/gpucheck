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
            self.end_headers()
            self.wfile.write(state["body"])

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


def test_api_key_never_reaches_disk_or_terminal(server, tmp_path, capsys):
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
    assert server["requests"] == [] and "--yes-upload" in capsys.readouterr().out
