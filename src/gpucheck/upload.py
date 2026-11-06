"""Opt-in upload client (stdlib only). Never raises: failures become UploadOutcome(ok=False)."""

from __future__ import annotations

import json
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


def _fail(e: BaseException) -> UploadOutcome:
    msg = " ".join(f"{type(e).__name__}: {e}".split())
    return UploadOutcome(False, None, None, None, msg)


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
        )
    except Exception as e:  # noqa: BLE001 - upload must never break the local run
        return _fail(e)
