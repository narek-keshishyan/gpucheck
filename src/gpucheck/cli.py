"""Command-line entry point: run, analyse, write result.json, print report."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .adapter import AdapterUnavailable, NvmlAdapter, gpu_count
from .diagnose import diagnose
from .fakes import make_fakes
from .lot import (
    MAX_LABEL_LEN,
    ABORTED,
    NOT_TESTED,
    TESTED,
    CardOutcome,
    lot_exit_code,
    lot_request,
    lot_summary,
    mark_uploaded,
    new_lot_id,
    parse_gpus,
    previous_upload_with_tokens,
    read_lot,
    render_lot_table,
    lot_finished,
    settle,
    write_card,
    write_summary,
)
from .load import LoadUnavailable, TorchGemmLoad
from .metrics import derive
from .report import NOT_COVERED, render_report
from .resultjson import build_result, canonical_dumps
from .run import NullLoad, RunConfig, execute_run
from .specs import lookup
from .upload import (
    KEY_INVALID,
    ChallengeUnavailable,
    UploadOutcome,
    fetch_challenge,
    key_error,
    upload,
    upload_lot,
)

DEFAULT_SERVER = "https://gpucheck-api.fly.dev"

EXIT_OK, EXIT_REFUSED, EXIT_NO_GPU, EXIT_ABORTED = 0, 1, 2, 3
EXIT_INTERRUPTED = 130

INTERRUPTED = "interrupted"
NOT_REACHED = "not reached: the run stopped before this card"
ACCOUNT_NOTE = "Account uploads cannot be self-deleted."
NO_TOKENS_NOTE = "The server returned no delete tokens: this was already uploaded."
UPLOAD_INTERRUPTED = "upload interrupted; nothing was changed on disk"
UNMATCHED_TOKENS = (
    "warning: the server's response had entries gpucheck could not match to a card; save these tokens yourself:"
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpucheck", description="Stress-test an NVIDIA GPU and report what was observed")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="shorter run (30s baseline, 150s load)")
    g.add_argument("--long", action="store_true", help="longer run (60s baseline, 1800s load)")
    p.add_argument("--offline", action="store_true", help="never upload; report locally only")
    p.add_argument("--certify", action="store_true", help="request a one-time server challenge so the result can earn a certificate (requires a server; excludes --offline/--dry-run/--quick)")
    p.add_argument("--yes-upload", action="store_true", help="upload without prompting")
    p.add_argument("--dry-run", action="store_true", help="show what would be uploaded")
    p.add_argument("--json", metavar="PATH", default="./gpucheck-result.json", help="where to write result.json")
    p.add_argument("--server", metavar="URL", default=None, help="upload server URL")
    p.add_argument("--api-key", metavar="KEY", default=None, help="send this API key with the upload; prefer the GPUCHECK_API_KEY environment variable, because command-line arguments are visible to other users of the machine. The key is never saved or printed")
    p.add_argument("--gpus", metavar="SPEC", default=None, help='which GPUs to test: "all" or indices such as "0,2" (default: 0)')
    p.add_argument("--include-serial", action="store_true", help="also record the card's serial number in the result")
    p.add_argument("--label", metavar="TEXT", default=None, help=f"a name for a multi-GPU lot, such as a rack or chassis (at most {MAX_LABEL_LEN} characters)")
    p.add_argument("--lot-dir", metavar="DIR", default=None, help="where a multi-GPU run writes its files (default: ./gpucheck-lot-<id>)")
    p.add_argument("--upload-saved", metavar="PATH", default=None, help="upload a saved result file or lot directory without testing any GPU")
    p.add_argument("--fake", metavar="PROFILE", help=argparse.SUPPRESS)
    return p


def _parse(argv):
    p = _parser()
    args = p.parse_args(argv)
    if args.certify and args.offline:
        p.error("--certify cannot be combined with --offline")
    if args.certify and args.dry_run:
        p.error("--certify cannot be combined with --dry-run")
    if args.certify and args.quick:
        p.error("--certify cannot be combined with --quick")
    if args.upload_saved is not None:
        for flag, used in (
            ("--offline", args.offline), ("--dry-run", args.dry_run), ("--certify", args.certify),
            ("--fake", args.fake), ("--gpus", args.gpus is not None), ("--quick", args.quick),
            ("--long", args.long), ("--label", args.label is not None),
            ("--lot-dir", args.lot_dir is not None), ("--include-serial", args.include_serial),
        ):
            if used:
                p.error(f"--upload-saved cannot be combined with {flag}")
    if args.label is not None and len(args.label) > MAX_LABEL_LEN:
        p.error(f"--label must be at most {MAX_LABEL_LEN} characters")
    return args


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def challenge_expired(challenge: dict, now: datetime | None = None) -> bool:
    """True when the challenge's expires_at (ISO 8601) is in the past.

    An unparseable or missing expiry is not treated as expired: the server decides."""
    raw = challenge.get("expires_at")
    if not isinstance(raw, str):
        return False
    try:
        exp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return (now or _now()) > exp


def format_certify(out: UploadOutcome) -> str | None:
    """The server's certification verdict for a --certify upload, or None if it said nothing."""
    if out.certifiable is True:
        return "Certifiable: yes"
    if out.certifiable is False:
        return f"Not certifiable: {out.certify_note or 'no reason given'}"
    return None


def _make_adapter(args):
    if args.fake:
        cards = getattr(args, "fake_cards", None) or make_fakes(args.fake)
        return cards[getattr(args, "gpu_index", 0)]
    return NvmlAdapter(index=getattr(args, "gpu_index", 0), include_serial=args.include_serial)


def _make_load(args, challenge=None, device_index: int = 0):
    if args.fake:  # fake adapters supply their own achieved_tflops
        return NullLoad()
    if challenge is not None:
        # Certification requires the real load generator; LoadUnavailable propagates.
        return TorchGemmLoad(challenge["workload"], device_index=device_index)
    try:
        return TorchGemmLoad(device_index=device_index)
    except LoadUnavailable:
        hint = ""
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            hint = " (CUDA_VISIBLE_DEVICES is set, which may be hiding the GPU)"
        print(
            f"no torch/CUDA — running without load; performance metrics will be gaps{hint}",
            file=sys.stderr,
        )
        return NullLoad()


def _load_available() -> bool:
    """Whether the load generator can be built. Constructing it only imports torch and asks
    whether CUDA is available; it creates no CUDA context (that happens in start())."""
    try:
        TorchGemmLoad()
    except LoadUnavailable:
        return False
    return True


def _config(args) -> RunConfig:
    if args.quick:
        return RunConfig(baseline_s=30, load_s=150, hz=1.0)
    if args.long:
        return RunConfig(baseline_s=60, load_s=1800, hz=1.0)
    return RunConfig(baseline_s=60, load_s=600, hz=1.0)


def _timer(args) -> dict:
    """clock/sleep overrides for execute_run: a virtual clock for simulated hardware, else none."""
    if not args.fake:
        return {}
    t = [0.0]

    def clock() -> float:
        return t[0]

    def sleep(s: float) -> None:
        t[0] += s

    return {"clock": clock, "sleep": sleep}


def _execute(adapter, args, challenge=None, device_index: int = 0):
    spec = lookup(adapter.inventory().model_name)
    load = _make_load(args, challenge, device_index=device_index)
    return spec, load, execute_run(adapter, load, _config(args), spec, **_timer(args))


MAX_REASON_LEN = 300


def _one_line(text: str) -> str:
    """An exception's text as a card reason: whitespace runs collapsed, at most 300 characters."""
    return " ".join(text.split())[:MAX_REASON_LEN]


class _Unreachable:
    """Stands in for a card whose adapter could not be created, so the lot records it and moves on."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def inventory(self):
        raise self._error


class _CardInterrupted(BaseException):
    """Ctrl-C while a lot card was settling or running; carries that card's outcome."""

    def __init__(self, outcome: CardOutcome) -> None:
        super().__init__(outcome)
        self.outcome = outcome


def _lot_adapters(args, selected: list[int]) -> list:
    if args.fake:
        return [args.fake_cards[i] for i in selected]
    adapters = []
    for i in selected:
        try:
            adapters.append(NvmlAdapter(index=i, include_serial=args.include_serial))
        except Exception as e:  # AdapterUnavailable or anything else: this card only
            adapters.append(_Unreachable(e))
    return adapters


def _run_card(
    adapter, args, *, position: int, gpu_index: int | None, lot: dict, settle_first: bool
) -> CardOutcome:
    """Test one card of a lot. A card that cannot be read, refuses preflight or fails is recorded, not raised.

    Ctrl-C while settling or running raises _CardInterrupted carrying this card's outcome."""

    def outcome(inventory, kind, reason, result=None) -> CardOutcome:
        return CardOutcome(position, gpu_index, inventory, kind, reason, result)

    try:
        inventory = adapter.inventory()
    except Exception as e:
        return outcome(None, NOT_TESTED, _one_line(f"could not read the card: {e}"))
    timer = _timer(args)
    started = _now()
    try:
        settle_s = settle(adapter, sleep=timer.get("sleep", time.sleep)) if settle_first else None
        spec = lookup(inventory.model_name)
        load = _make_load(args, None, device_index=gpu_index)
        run = execute_run(adapter, load, _config(args), spec, **timer)
    except KeyboardInterrupt:
        raise _CardInterrupted(outcome(inventory, NOT_TESTED, INTERRUPTED)) from None
    except Exception as e:  # one card's failure must not lose the rest of the lot
        return outcome(inventory, NOT_TESTED, _one_line(f"run failed: {type(e).__name__}: {e}"))
    ended = _now()
    if run.aborted == "preflight_refused":
        return outcome(run.inventory, NOT_TESTED, "; ".join(run.preflight.reasons))
    metrics = derive(run, spec)
    findings = diagnose(metrics, run.preflight, run.inventory, spec)
    result = build_result(
        run, metrics, findings,
        timing={"started_utc": _iso(started), "ended_utc": _iso(ended), "settle_s": settle_s},
        lot=lot,
    )
    if run.aborted:
        return outcome(run.inventory, ABORTED, run.aborted, result)
    return outcome(run.inventory, TESTED, None, result)


def _main_lot(adapters, args, gpu_indices: list[int], api_key: str | None = None) -> int:
    """Test every card in turn and write the lot; gpu_indices[i] is the index adapters[i] was selected by."""
    if args.certify:
        print("gpucheck: --certify cannot be combined with a multi-GPU run", file=sys.stderr)
        return EXIT_REFUSED
    lot_id = new_lot_id()
    directory = Path(args.lot_dir or f"./gpucheck-lot-{lot_id[:8]}")
    if (directory / "lot.json").exists():
        print(f"gpucheck: {directory} already contains a lot; choose another --lot-dir", file=sys.stderr)
        return EXIT_REFUSED
    if directory.exists() and not directory.is_dir():
        print(f"gpucheck: {directory} is not a directory", file=sys.stderr)
        return EXIT_REFUSED
    try:  # find out now, not after every card has run, that the lot cannot be saved
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass
    except OSError as e:
        print(f"gpucheck: cannot write to {directory}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    created = _iso(_now())
    lot_ref = {"lot_id": lot_id, "size": len(adapters), "label": args.label}
    cards: list[CardOutcome] = []
    saved: set[int] = set()  # positions whose gpu<position>.json is on disk

    def save(finished: bool, rest_reason: str) -> None:
        """Write the card files not yet on disk, then lot.json; positions not reached get rest_reason."""
        rest = [CardOutcome(p, gpu_indices[p], None, NOT_TESTED, rest_reason, None)
                for p in range(len(cards), len(adapters))]
        for card in cards:
            if card.result is not None and card.position not in saved:
                write_card(directory, card)
                saved.add(card.position)
        write_summary(directory, lot_summary(lot_id, args.label, created, cards + rest, finished=finished))

    interrupted = False
    for pos, adapter in enumerate(adapters):
        try:
            cards.append(_run_card(
                adapter, args, position=pos, gpu_index=gpu_indices[pos],
                lot={**lot_ref, "position": pos}, settle_first=pos > 0,
            ))
            try:  # so a dropped session or a kill later in the lot keeps this card
                save(False, NOT_REACHED)
            except OSError as e:  # retried at the end; the result stays in memory
                print(f"warning: could not save GPU {pos} to {directory}: {e}", file=sys.stderr)
        except _CardInterrupted as stop:
            cards.append(stop.outcome)
            interrupted = True
        except KeyboardInterrupt:  # between cards, while reading the inventory, or while saving
            interrupted = True
        if interrupted:
            break
    for pos in range(len(cards), len(adapters)):
        cards.append(CardOutcome(pos, gpu_indices[pos], None, NOT_TESTED, INTERRUPTED, None))
    summary = lot_summary(lot_id, args.label, created, cards)
    try:
        save(True, INTERRUPTED)
        write_error = None
    except OSError as e:
        write_error = e
    for card in cards:
        if card.result is not None:
            print(render_report(card.result))
    print(render_lot_table(summary))
    print(NOT_COVERED)
    if write_error is not None:  # nothing saved to upload from
        print(f"gpucheck: could not write the lot to {directory}: {write_error}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"Lot written to {directory}")
    if interrupted:  # an interrupted lot is never uploaded
        return EXIT_INTERRUPTED
    results = [c.result for c in cards if c.result is not None]
    if results and not args.offline:
        _maybe_upload_lot(summary, results, directory, args, api_key)
    return lot_exit_code(cards)


def _maybe_upload_lot(summary: dict, results: list[dict], directory: Path, args, api_key: str | None) -> None:
    if args.dry_run:
        print(canonical_dumps(lot_request(summary, results)))  # the key is a header, never in the body
        return
    server = args.server or DEFAULT_SERVER
    later = f"Not uploaded. The lot is saved in {directory}; upload it later with: gpucheck --upload-saved {directory}"
    if (refused := key_error(server, api_key)) is not None:
        print(f"gpucheck: {refused}", file=sys.stderr)
        print(later)
        return
    if not args.yes_upload:
        if not sys.stdin.isatty():
            print(later)
            return
        if not _confirm(server, "lot"):
            return
    try:  # a lot upload can take minutes; Ctrl-C must not end in a traceback
        out = upload_lot(summary, results, server, api_key)
    except KeyboardInterrupt:
        print(f"upload interrupted; the lot is saved in {directory}. Retry with: gpucheck --upload-saved {directory}")
        return
    if not out.ok:
        print(
            f"warning: upload failed ({out.error}); the lot is saved in {directory}. "
            f"Retry with: gpucheck --upload-saved {directory}"
        )
        return
    _report_lot_upload(out, directory, server, api_key)


def _report_lot_upload(out, directory: Path, server: str, api_key: str | None) -> None:
    """Print a successful lot upload and record it in lot.json."""
    print(f"Lot page: {out.url}")
    for r in out.runs:
        print(f"GPU {r.get('position')}: {r.get('url')}")
        if r.get("card_url"):
            print(f"  Card history: {r['card_url']}")
    tokens = [(r.get("position"), r.get("delete_token")) for r in out.runs if r.get("delete_token")]
    try:
        written = mark_uploaded(directory, server, out)
    except Exception as e:  # the tokens exist nowhere else, so show them
        print(f"warning: could not update {directory}/lot.json ({e}); save these delete tokens now")
        for position, token in tokens:
            print(f"GPU {position} delete token: {token}")
        return
    kept = [r.get("delete_token") for r in written["uploaded"]["runs"]]
    unsaved = [(position, token) for position, token in tokens if token not in kept]
    if len(unsaved) < len(tokens):
        print(f"Delete tokens were saved in {directory}/lot.json")
    if unsaved:  # a response entry that names no card: its token is not in lot.json
        print(UNMATCHED_TOKENS)
        for position, token in unsaved:
            print(f"Unsaved delete token (position {position!r}): {token}")
    if not tokens:
        print(ACCOUNT_NOTE if api_key else NO_TOKENS_NOTE)


def format_grade(out: UploadOutcome, model: str | None) -> str | None:
    """One-line grade summary from the server's response, or None if it said nothing."""
    g = out.grade
    if g:
        letter = g.get("letter", "?")
        ratio = g.get("ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            return f"Grade {letter}"
        n = g.get("n", out.n)
        extra = f"n={n}" + (", low confidence" if g.get("low_confidence") else "")
        return f"Grade {letter} — {ratio * 100:.1f}% of the median clean reference run for {model or 'this model'} ({extra})"
    if out.reason:
        return f"Grade: {out.reason}"
    if out.n is not None:
        return f"Grade: insufficient reference (n={out.n})"
    return None


def _confirm(server: str, what: str = "result") -> bool:
    try:
        return input(f"Upload this {what} to {server}? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):  # closed stdin or Ctrl-C at the prompt means "no"
        print()
        return False


def _maybe_upload(result: dict, args, challenge: dict | None = None, api_key: str | None = None) -> None:
    if args.dry_run:
        print(canonical_dumps(result))
        return
    server = args.server or DEFAULT_SERVER
    later = f"gpucheck --upload-saved {args.json}"
    if (refused := key_error(server, api_key)) is not None:
        print(f"gpucheck: {refused}", file=sys.stderr)
        print(f"Not uploaded. The result is saved in {args.json}; upload it later with: {later}")
        return
    if not args.yes_upload:
        if not sys.stdin.isatty():
            print(f"Not uploaded. To upload this result, re-run with --yes-upload, or upload the saved file later with: {later}")
            return
        if not _confirm(server):
            return
    out = upload(result, server, api_key=api_key)
    if not out.ok:
        print(f"warning: upload failed ({out.error}); result.json is saved locally. Upload it later with: {later}")
        return
    _report_upload(out, result, challenge, api_key)


def _report_upload(out: UploadOutcome, result: dict, challenge: dict | None, api_key: str | None) -> None:
    """Print a successful single-result upload."""
    if out.url:
        print(f"Result page: {out.url}")
    if out.card_url:
        print(f"Card history: {out.card_url}")
    line = format_grade(out, (result.get("inventory") or {}).get("model_name"))
    if line:
        print(line)
    if challenge is not None:
        cline = format_certify(out)
        if cline:
            print(cline)
    if out.delete_token:
        print(f"Delete token: {out.delete_token}")
        print("(save this token to delete the page later)")
    elif api_key:
        print(ACCOUNT_NOTE)
    else:
        print(NO_TOKENS_NOTE)


def _main_upload_saved(path_arg: str, args, api_key: str | None) -> int:
    """Upload a saved result file or lot directory. No GPU, no prompt; 0 on success, 1 on any failure."""
    path = Path(path_arg)
    server = args.server or DEFAULT_SERVER
    if (refused := key_error(server, api_key)) is not None:
        print(f"gpucheck: {refused}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        if path.is_dir():
            summary, results = read_lot(path)
            if not results:
                print(f"gpucheck: {path} has no tested card to upload", file=sys.stderr)
                return EXIT_REFUSED
            if not lot_finished(summary):
                print("note: this lot did not finish; cards that were not reached are recorded as not tested")
            before = previous_upload_with_tokens(summary, server)
            if before:
                print(
                    f"note: this lot was uploaded to {before} before; its delete tokens for that server "
                    'will be kept in lot.json under "previous_uploads".'
                )
            try:
                out = upload_lot(summary, results, server, api_key)
            except KeyboardInterrupt:
                print(UPLOAD_INTERRUPTED)
                return EXIT_INTERRUPTED
            if not out.ok:
                print(f"gpucheck: upload failed ({out.error}); {path} is unchanged", file=sys.stderr)
                return EXIT_REFUSED
            _report_lot_upload(out, path, server, api_key)
            return EXIT_OK
        result = _read_result(path)
    except (OSError, ValueError) as e:
        print(f"gpucheck: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if result.get("lot") is not None:  # the server takes a lot's cards only as a lot
        print(
            f"gpucheck: {path} is part of a lot; upload its directory instead: "
            f"gpucheck --upload-saved {path.parent}",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    try:
        out = upload(result, server, api_key=api_key)
    except KeyboardInterrupt:
        print(UPLOAD_INTERRUPTED)
        return EXIT_INTERRUPTED
    if not out.ok:
        print(f"gpucheck: upload failed ({out.error}); {path} is unchanged", file=sys.stderr)
        return EXIT_REFUSED
    _report_upload(out, result, None, api_key)
    return EXIT_OK


def _read_result(path: Path) -> dict:
    try:
        result = json.loads(path.read_text())
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e.strerror or e}") from None
    except ValueError:
        raise ValueError(f"{path} is not valid JSON") from None
    if not isinstance(result, dict) or "schema_version" not in result:
        raise ValueError(f"{path} is not a gpucheck result")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    api_key = next((k for k in (c.strip() for c in (args.api_key, os.environ.get("GPUCHECK_API_KEY")) if c) if k), None)
    args.api_key = None  # held only as an argument from here on
    if api_key and any(ch.isspace() or not ch.isprintable() for ch in api_key):
        print(f"gpucheck: {KEY_INVALID}", file=sys.stderr)
        return EXIT_REFUSED
    if args.upload_saved is not None:
        return _main_upload_saved(args.upload_saved, args, api_key)
    index = 0
    args.gpu_index = 0
    lot_adapters = None
    try:
        if not args.fake:
            count = gpu_count()
            if count > 1 and "CUDA_VISIBLE_DEVICES" in os.environ:
                # torch would number only the visible cards, NVML numbers all of them:
                # the load would run on a different card from the one measured.
                # An empty value hides every card from CUDA, so it is refused too.
                print(
                    "gpucheck: CUDA_VISIBLE_DEVICES is set, so PyTorch and NVML would number the GPUs "
                    "differently; unset it and choose cards with --gpus",
                    file=sys.stderr,
                )
                return EXIT_REFUSED
            gpus_spec = args.gpus
        else:
            args.fake_cards = make_fakes(args.fake)  # unknown profile: handled below
            count = len(args.fake_cards)
            gpus_spec = "all" if args.gpus is None else args.gpus
        try:
            selected = parse_gpus(gpus_spec, count)
        except ValueError as e:
            print(f"gpucheck: --gpus: {e}", file=sys.stderr)
            return EXIT_REFUSED
        if len(selected) > 1:
            if not args.fake and not _load_available():
                print(
                    'gpucheck: a multi-GPU run needs the load generator (PyTorch with CUDA); '
                    'install it with: pip install "gpucheck-cli[load]"',
                    file=sys.stderr,
                )
                return EXIT_REFUSED
            lot_adapters = _lot_adapters(args, selected)
        else:
            args.gpu_index = selected[0]
            if not args.fake:
                index = selected[0]
                if args.gpus is None and count > 1:
                    print(f"{count} GPUs detected; testing GPU 0 only. Use --gpus all to test every card.")
            adapter = _make_adapter(args)
    except AdapterUnavailable as e:
        print(f"gpucheck: no NVIDIA GPU or driver available ({e}).", file=sys.stderr)
        return EXIT_NO_GPU
    except ValueError as e:  # a bad --fake profile name (make_fakes) or an unusable adapter
        print(f"gpucheck: {e}", file=sys.stderr)
        return EXIT_NO_GPU
    if lot_adapters is not None:
        return _main_lot(lot_adapters, args, selected, api_key)
    challenge = None
    if args.certify:
        try:
            challenge = fetch_challenge(args.server or DEFAULT_SERVER)
        except ChallengeUnavailable as e:
            print(f"gpucheck: certification unavailable: {e}", file=sys.stderr)
            return EXIT_REFUSED
    started = _now()
    try:
        spec, load, run = _execute(adapter, args, challenge, device_index=index)
    except LoadUnavailable as e:
        if args.certify:
            print(f"gpucheck: certification requires the GPU load generator ({e}).", file=sys.stderr)
        else:
            print(f"gpucheck: the GPU load generator failed to start ({e}).", file=sys.stderr)
        return EXIT_REFUSED
    except AdapterUnavailable as e:
        print(f"gpucheck: no NVIDIA GPU or driver available ({e}).", file=sys.stderr)
        return EXIT_NO_GPU

    ended = _now()
    if run.aborted == "preflight_refused":
        print("gpucheck: preflight refused to run:")
        for reason in run.preflight.reasons:
            print(f"  - {reason}")
        return EXIT_REFUSED

    metrics = derive(run, spec)
    findings = diagnose(metrics, run.preflight, run.inventory, spec)
    challenge_workload = None
    if challenge is not None:
        challenge_workload = getattr(load, "resolved_workload", None)
        if challenge_workload is None:
            # Test/demo-only path (--fake): no real load ran, so synthesize the
            # echo straight from the prescription. Real runs always echo what
            # TorchGemmLoad actually resolved.
            w = challenge["workload"]
            challenge_workload = {
                "gemm_n": [8192 + o for o in w["gemm_n_offsets"]],
                "order_seed": w["order_seed"],
            }
    result = build_result(
        run, metrics, findings, challenge, challenge_workload,
        timing={"started_utc": _iso(started), "ended_utc": _iso(ended), "settle_s": None},
    )
    Path(args.json).write_text(canonical_dumps(result))  # before any upload consideration
    print(render_report(result))
    print(f"Result written to {args.json}")
    if challenge is not None and challenge_expired(challenge):
        print("warning: challenge expired before upload; run again to certify — result saved locally")
    elif not args.offline:
        _maybe_upload(result, args, challenge, api_key)
    return EXIT_ABORTED if run.aborted else EXIT_OK
