"""Opt-in upload client (stdlib only). Never raises: failures become UploadOutcome(ok=False)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .lot import lot_request
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
    certifiable: bool | None = None  # server's verdict on a --certify upload (None: not said)
    certify_note: str | None = None  # why the server refused to certify
    card_url: str | None = None  # the card's history page


KEY_REJECTED = "API key was rejected by the server"


class _Redirected(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"server redirected the upload (HTTP {code}); check the --server URL")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the API key to whatever host the server names."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise _Redirected(code)


_opener = urllib.request.build_opener(_NoRedirect)


def _open(req, timeout):
    return _opener.open(req, timeout=timeout)


def _error_text(e: BaseException, api_key: str | None = None) -> str:
    if isinstance(e, urllib.error.HTTPError) and e.code == 401:
        return KEY_REJECTED
    if isinstance(e, _Redirected):
        return str(e)
    text = " ".join(f"{type(e).__name__}: {e}".split())
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text


def _fail(e: BaseException, api_key: str | None = None) -> UploadOutcome:
    return UploadOutcome(False, None, None, None, _error_text(e, api_key))


def _headers(api_key: str | None, extra: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json", **(extra or {})}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
KEY_INVALID = "the API key contains invalid characters"


def key_error(server: str, api_key: str | None) -> str | None:
    """Why `api_key` must not be sent to `server`, or None. The text never contains the key."""
    if not api_key:
        return None
    if any(ch.isspace() or not ch.isprintable() for ch in api_key):
        return KEY_INVALID
    parts = urllib.parse.urlsplit(server)
    if parts.scheme != "https":
        host = parts.hostname or server
        if host not in LOCAL_HOSTS:
            return f"refusing to send the API key over plain http to {host}; use an https server URL"
    return None


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


def upload(result: dict, server: str, timeout_s: float = 20, api_key: str | None = None) -> UploadOutcome:
    if (refused := key_error(server, api_key)) is not None:
        return UploadOutcome(False, None, None, None, refused)
    try:
        req = urllib.request.Request(
            f"{server.rstrip('/')}/v1/runs",
            data=canonical_dumps(result).encode(),
            method="POST",
            headers=_headers(api_key, {"X-Run-Id": run_id(result)}),
        )
        with _open(req, timeout_s) as resp:
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
            body.get("certifiable") if isinstance(body.get("certifiable"), bool) else None,
            body.get("certify_note") if isinstance(body.get("certify_note"), str) else None,
            body.get("card_url") if isinstance(body.get("card_url"), str) else None,
        )
    except Exception as e:  # noqa: BLE001 - upload must never break the local run
        return _fail(e, api_key)


@dataclass(frozen=True)
class LotUploadOutcome:
    ok: bool
    url: str | None
    runs: list[dict]
    error: str | None


def upload_lot(
    summary: dict,
    results: list[dict],
    server: str,
    api_key: str | None = None,
    timeout_s: float = 120,
) -> LotUploadOutcome:
    """POST the whole lot in one request. Never raises."""
    if (refused := key_error(server, api_key)) is not None:
        return LotUploadOutcome(False, None, [], refused)
    try:
        data = canonical_dumps(lot_request(summary, results)).encode()
        req = urllib.request.Request(
            f"{server.rstrip('/')}/v1/lots", data=data, method="POST", headers=_headers(api_key)
        )
        with _open(req, timeout_s) as resp:
            body = json.loads(resp.read().decode())
        if not isinstance(body, dict):
            raise ValueError("unexpected response shape")
        runs = body.get("runs")  # the server accepted the lot even if it said nothing usable about the runs
        runs = [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []
        url = body.get("url")
        return LotUploadOutcome(True, url if isinstance(url, str) else None, runs, None)
    except Exception as e:  # noqa: BLE001 - upload must never break the local run
        return LotUploadOutcome(False, None, [], _error_text(e, api_key))
