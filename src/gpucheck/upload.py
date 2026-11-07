"""Opt-in upload client (stdlib only). Never raises: failures become UploadOutcome(ok=False)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .resultjson import canonical_dumps, run_id


@dataclass(frozen=True)
class UploadOutcome:
    ok: bool
    url: str | None
    grade: dict | None
    delete_token: str | None
    error: str | None
    n: int | None = None
    reason: str | None = None  # server's explanation when grade is None


def _fail(e: BaseException) -> UploadOutcome:
    msg = " ".join(f"{type(e).__name__}: {e}".split())
    return UploadOutcome(False, None, None, None, msg)


class ChallengeUnavailable(Exception):
    """The server could not issue a certification challenge."""


def fetch_challenge(server: str, timeout_s: float = 20) -> dict:
    """POST /v1/challenges; raises ChallengeUnavailable with a one-line message."""
    try:
        req = urllib.request.Request(
            f"{server.rstrip('/')}/v1/challenges",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())
        if not isinstance(body, dict) or not isinstance(body.get("workload"), dict):
            raise ValueError("unexpected challenge response shape")
        for k in ("challenge_id", "nonce"):
            if not isinstance(body.get(k), str):
                raise ValueError(f"challenge missing {k}")
        w = body["workload"]
        offs = w.get("gemm_n_offsets")
        if (
            not isinstance(offs, list)
            or not offs
            or not all(type(o) is int for o in offs)  # type() rejects bools
            or type(w.get("order_seed")) is not int
        ):
            raise ValueError("challenge workload malformed")
        return body
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ChallengeUnavailable("server has certification disabled") from e
        raise ChallengeUnavailable(f"challenge request failed (HTTP {e.code})") from e
    except Exception as e:  # noqa: BLE001
        msg = " ".join(f"{type(e).__name__}: {e}".split())
        raise ChallengeUnavailable(f"could not fetch challenge ({msg})") from e


def upload(result: dict, server: str, timeout_s: float = 20) -> UploadOutcome:
    try:
        req = urllib.request.Request(
            f"{server.rstrip('/')}/v1/runs",
            data=canonical_dumps(result).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Run-Id": run_id(result)},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())
        if not isinstance(body, dict):
            raise ValueError("unexpected response shape")
        grade = body.get("grade")
        return UploadOutcome(
            True,
            body.get("url"),
            grade if isinstance(grade, dict) else None,
            body.get("delete_token"),
            None,
            body.get("n") if isinstance(body.get("n"), int) else None,
            body.get("reason") if isinstance(body.get("reason"), str) else None,
        )
    except Exception as e:  # noqa: BLE001 - upload must never break the local run
        return _fail(e)
