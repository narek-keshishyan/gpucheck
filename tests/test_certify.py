import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gpucheck import cli
from gpucheck.cli import main
from gpucheck.fakes import make_fake
from gpucheck.load import LoadUnavailable, apply_workload, pick_gemm_size
from gpucheck.upload import ChallengeUnavailable, fetch_challenge

CHALLENGE = {
    "challenge_id": "ch1",
    "nonce": "n0nce",
    "workload": {"gemm_n_offsets": [-64, 0, 16, 128], "order_seed": 77},
    "expires_at": "2026-11-07T00:00:00Z",
}


@pytest.fixture
def server():
    state = {"status": 200, "challenge": CHALLENGE, "requests": []}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            state["requests"].append((self.path, body))
            if self.path == "/v1/challenges":
                status, out = state["status"], state["challenge"]
            else:
                status, out = 200, {"url": "https://x.test/r/1", "grade": {"letter": "A"}, "n": 3}
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}"
    yield state
    srv.shutdown()
    srv.server_close()


def test_fetch_challenge_happy(server):
    assert fetch_challenge(server["url"]) == CHALLENGE
    assert server["requests"][0][0] == "/v1/challenges"


def test_fetch_challenge_404_distinct_message(server):
    server["status"] = 404
    with pytest.raises(ChallengeUnavailable, match="certification disabled"):
        fetch_challenge(server["url"])


def test_fetch_challenge_429_and_network(server):
    server["status"] = 429
    with pytest.raises(ChallengeUnavailable, match="429"):
        fetch_challenge(server["url"])
    with pytest.raises(ChallengeUnavailable) as ei:
        fetch_challenge("http://127.0.0.1:1", timeout_s=1)
    assert "disabled" not in str(ei.value)


def test_fetch_challenge_malformed(server):
    server["challenge"] = {"nonce": "x"}
    with pytest.raises(ChallengeUnavailable):
        fetch_challenge(server["url"])


def test_apply_workload_plain():
    assert apply_workload(8192, [-64, 0, 16, 128]) == [8128, 8192, 8208, 8320]


def test_apply_workload_clamps_base_not_individuals():
    offs = [-256, 0, 8, 256]
    lo = apply_workload(2048, offs)
    assert lo == [2048 + 256 - 256, 2048 + 256, 2048 + 256 + 8, 2048 + 256 + 256]
    assert min(lo) == 2048
    hi = apply_workload(16384, offs)
    assert max(hi) == 16384
    # pairwise differences always preserved
    assert [x - lo[0] for x in lo] == [o - offs[0] for o in offs]
    assert [x - hi[0] for x in hi] == [o - offs[0] for o in offs]


def test_apply_workload_on_pick_gemm_size():
    offs = [-256, -8, 8, 256]
    for free in (0, 10**9, 10**12):
        ns = apply_workload(pick_gemm_size(free), offs)
        assert all(2048 <= n <= 16384 for n in ns)
        assert [n - ns[0] for n in ns] == [o - offs[0] for o in offs]


def test_build_result_without_challenge_is_v1_identical():
    from gpucheck.resultjson import build_result, canonical_dumps, run_id
    from gpucheck.diagnose import diagnose
    from gpucheck.metrics import derive
    from gpucheck.run import execute_run, RunConfig, NullLoad
    from gpucheck.specs import lookup

    a = make_fake("healthy")
    spec = lookup(a.inventory().model_name)
    t = [0.0]
    run = execute_run(a, NullLoad(), RunConfig(30, 60, 1.0), spec,
                      clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    m = derive(run, spec)
    f = diagnose(m, run.preflight, run.inventory, spec)
    base = build_result(run, m, f)
    assert "challenge" not in base and "challenge_workload" not in base["config"]
    assert canonical_dumps(build_result(run, m, f, None, None)) == canonical_dumps(base)
    with_c = build_result(run, m, f, CHALLENGE, {"gemm_n": [1, 2], "order_seed": 3})
    assert with_c["challenge"] == {"challenge_id": "ch1", "nonce": "n0nce"}
    assert with_c["config"]["challenge_workload"] == {"gemm_n": [1, 2], "order_seed": 3}
    assert run_id(with_c) != run_id(base)


def test_certify_fake_posts_challenge_and_workload(server, tmp_path, capsys):
    rc = main(["--fake", "healthy", "--quick", "--certify", "--yes-upload",
               "--server", server["url"], "--json", str(tmp_path / "r.json")])
    assert rc == 0
    posts = [b for p, b in server["requests"] if p == "/v1/runs"]
    assert len(posts) == 1
    body = json.loads(posts[0])
    assert body["challenge"] == {"challenge_id": "ch1", "nonce": "n0nce"}
    assert body["config"]["challenge_workload"] == {
        "gemm_n": [8192 - 64, 8192, 8192 + 16, 8192 + 128],
        "order_seed": 77,
    }


def test_certify_offline_is_argparse_error(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--fake", "healthy", "--certify", "--offline"])
    assert ei.value.code == 2


def test_certify_challenge_failure_exit_1(server, tmp_path, capsys):
    server["status"] = 404
    rc = main(["--fake", "healthy", "--certify", "--server", server["url"],
               "--json", str(tmp_path / "r.json")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "certification disabled" in err
    assert len(err.strip().splitlines()) == 1


def test_certify_real_adapter_without_torch_exits_1(server, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_make_adapter", lambda args: make_fake("healthy"))

    def boom(*a, **k):
        raise LoadUnavailable("torch is not installed")

    monkeypatch.setattr(cli, "TorchGemmLoad", boom)
    rc = main(["--certify", "--server", server["url"], "--json", str(tmp_path / "r.json")])
    err = capsys.readouterr().err
    assert rc == 1
    assert len(err.strip().splitlines()) == 1
    assert "load generator" in err
