import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gpucheck.cli import main
from gpucheck.resultjson import canonical_dumps, run_id
from gpucheck.upload import UploadOutcome, upload

OK_BODY = {
    "run_id": "abc",
    "url": "https://x.test/r/abc",
    "grade": {"letter": "A"},
    "n": 12,
    "delete_token": "tok123",
}


@pytest.fixture
def server():
    state = {"status": 200, "body": json.dumps(OK_BODY).encode(), "requests": []}

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


def _dead_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


RESULT = {"schema_version": "1", "x": 1.5}


def test_upload_happy_path(server):
    o = upload(RESULT, server["url"])
    assert o == UploadOutcome(True, OK_BODY["url"], OK_BODY["grade"], "tok123", None, 12)
    path, headers, body = server["requests"][0]
    assert path == "/v1/runs"
    assert body == canonical_dumps(RESULT).encode()
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Run-Id"] == run_id(RESULT)


def test_upload_tolerates_missing_keys(server):
    server["body"] = b"{}"
    o = upload(RESULT, server["url"])
    assert o.ok and o.url is None and o.grade is None and o.delete_token is None


def test_upload_500(server):
    server["status"] = 500
    o = upload(RESULT, server["url"])
    assert not o.ok and o.error and "\n" not in o.error


def test_upload_connection_refused():
    o = upload(RESULT, _dead_url(), timeout_s=2)
    assert not o.ok and o.error and "\n" not in o.error


def test_upload_bad_json(server):
    server["body"] = b"<html>nope"
    o = upload(RESULT, server["url"])
    assert not o.ok and o.error


def _run(tmp_path, *extra):
    out = tmp_path / "r.json"
    rc = main(["--fake", "healthy", "--quick", "--json", str(out), *extra])
    return rc, out


def test_cli_yes_upload_prints_url_and_token(server, tmp_path, capsys):
    rc, out = _run(tmp_path, "--yes-upload", "--server", server["url"])
    assert rc == 0
    text = capsys.readouterr().out
    assert OK_BODY["url"] in text and "tok123" in text
    assert "save this token to delete the page later" in text
    assert server["requests"][0][2] == out.read_text().encode()


def test_cli_null_grade_prints_insufficient_reference(server, tmp_path, capsys):
    server["body"] = json.dumps({**OK_BODY, "grade": None}).encode()
    _run(tmp_path, "--yes-upload", "--server", server["url"])
    assert "insufficient reference (n=12)" in capsys.readouterr().out


def test_upload_failure_preserves_local_result(tmp_path, capsys):
    rc, out = _run(tmp_path, "--yes-upload", "--server", _dead_url())
    assert rc == 0 and out.exists()
    assert json.loads(out.read_text())["schema_version"] == "1"
    assert "upload failed" in capsys.readouterr().out.lower()


def test_dry_run_sends_nothing(server, tmp_path, capsys):
    rc, out = _run(tmp_path, "--dry-run", "--yes-upload", "--server", server["url"])
    assert rc == 0
    assert server["requests"] == []
    printed = capsys.readouterr().out
    assert canonical_dumps(json.loads(out.read_text())) in printed


def test_non_tty_no_prompt_no_upload(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    rc, _ = _run(tmp_path, "--server", server["url"])
    assert rc == 0 and server["requests"] == []
    assert "--yes-upload" in capsys.readouterr().out


def test_tty_prompt_yes_uploads(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": (print(prompt, end=""), "y")[1])
    rc, _ = _run(tmp_path, "--server", server["url"])
    assert rc == 0 and len(server["requests"]) == 1
    assert f"Upload this result to {server['url']}? [y/N] " in capsys.readouterr().out


def test_tty_prompt_default_no(server, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    rc, _ = _run(tmp_path, "--server", server["url"])
    assert rc == 0 and server["requests"] == []


def test_offline_skips_upload_silently(server, tmp_path, capsys):
    rc, _ = _run(tmp_path, "--offline", "--yes-upload", "--server", server["url"])
    assert rc == 0 and server["requests"] == []
    out = capsys.readouterr().out.replace(str(tmp_path), "")
    assert "upload" not in out.lower()


# --- grade line formatting ---------------------------------------------------

from gpucheck.cli import format_grade  # noqa: E402

MODEL = "NVIDIA H100 80GB HBM3"


def test_format_graded_low_confidence():
    g = {"letter": "B", "ratio": 0.94123, "n": 12, "low_confidence": True}
    assert format_grade(UploadOutcome(True, None, g, None, None, 12), MODEL) == (
        "Grade B — delivers 94.1% of a healthy NVIDIA H100 80GB HBM3 (n=12, low confidence)"
    )


def test_format_graded_confident():
    g = {"letter": "A", "ratio": 0.99, "n": 40, "low_confidence": False}
    assert format_grade(UploadOutcome(True, None, g, None, None, 40), MODEL) == (
        "Grade A — delivers 99.0% of a healthy NVIDIA H100 80GB HBM3 (n=40)"
    )


@pytest.mark.parametrize(
    "reason",
    ["insufficient reference (n=3)", "measurements only", "no throughput measured", "unknown model"],
)
def test_format_null_grade_uses_reason_verbatim(reason):
    o = UploadOutcome(True, None, None, None, None, 3, reason)
    assert format_grade(o, MODEL) == f"Grade: {reason}"


def test_format_null_grade_without_reason_falls_back_to_n():
    assert format_grade(UploadOutcome(True, None, None, None, None, 4), MODEL) == (
        "Grade: insufficient reference (n=4)"
    )


def test_format_nothing_known():
    assert format_grade(UploadOutcome(True, None, None, None, None, None), MODEL) is None


def test_upload_parses_reason(server):
    server["body"] = json.dumps({**OK_BODY, "grade": None, "reason": "measurements only"}).encode()
    assert upload(RESULT, server["url"]).reason == "measurements only"


def test_cli_prints_formatted_grade(server, tmp_path, capsys):
    g = {"letter": "B", "ratio": 0.941, "n": 12, "low_confidence": True}
    server["body"] = json.dumps({**OK_BODY, "grade": g}).encode()
    _run(tmp_path, "--yes-upload", "--server", server["url"])
    assert "Grade B — delivers 94.1% of a healthy NVIDIA H100 80GB HBM3 (n=12, low confidence)" in (
        capsys.readouterr().out
    )


# --- placeholder default server ----------------------------------------------


def test_default_placeholder_server_skips_upload_with_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)

    def no_prompt(prompt=""):
        raise AssertionError("must not prompt")

    monkeypatch.setattr("builtins.input", no_prompt)
    rc, _ = _run(tmp_path, "--yes-upload")
    assert rc == 0
    out = capsys.readouterr().out
    assert "no public server is configured yet; pass --server URL" in out
    assert "upload failed" not in out.lower()


# --- prompt interrupted -------------------------------------------------------


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_prompt_eof_or_interrupt_means_no(server, tmp_path, capsys, monkeypatch, exc):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)

    def raise_(prompt=""):
        raise exc

    monkeypatch.setattr("builtins.input", raise_)
    rc, out = _run(tmp_path, "--server", server["url"])
    assert rc == 0 and server["requests"] == [] and out.exists()
    c = capsys.readouterr()
    assert "Traceback" not in c.out + c.err
