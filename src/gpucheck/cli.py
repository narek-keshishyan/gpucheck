"""Command-line entry point: run, analyse, write result.json, print report."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .adapter import AdapterUnavailable, NvmlAdapter
from .diagnose import diagnose
from .fakes import make_fake
from .load import LoadUnavailable, TorchGemmLoad
from .metrics import derive
from .report import render_report
from .resultjson import build_result, canonical_dumps
from .run import NullLoad, RunConfig, execute_run
from .specs import lookup
from .upload import ChallengeUnavailable, UploadOutcome, fetch_challenge, upload

DEFAULT_SERVER = "https://gpucheck-api.fly.dev"

EXIT_OK, EXIT_REFUSED, EXIT_NO_GPU, EXIT_ABORTED = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpucheck", description="GPU health report card")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="shorter run (30s baseline, 150s load)")
    g.add_argument("--long", action="store_true", help="longer run (60s baseline, 1800s load)")
    p.add_argument("--offline", action="store_true", help="never upload; report locally only")
    p.add_argument("--certify", action="store_true", help="request a server challenge and run its prescribed workload (requires a server; excludes --offline/--dry-run)")
    p.add_argument("--yes-upload", action="store_true", help="upload without prompting")
    p.add_argument("--dry-run", action="store_true", help="show what would be uploaded")
    p.add_argument("--json", metavar="PATH", default="./gpucheck-result.json", help="where to write result.json")
    p.add_argument("--server", metavar="URL", default=None, help="upload server URL")
    p.add_argument("--fake", metavar="PROFILE", help=argparse.SUPPRESS)
    return p


def _parse(argv):
    p = _parser()
    args = p.parse_args(argv)
    if args.certify and args.offline:
        p.error("--certify cannot be combined with --offline")
    if args.certify and args.dry_run:
        p.error("--certify cannot be combined with --dry-run")
    return args


def _make_adapter(args):
    return make_fake(args.fake) if args.fake else NvmlAdapter()


def _make_load(args, challenge=None):
    if args.fake:  # fake adapters supply their own achieved_tflops
        return NullLoad()
    if challenge is not None:
        # Certification requires the real load generator; LoadUnavailable propagates.
        return TorchGemmLoad(challenge["workload"])
    try:
        return TorchGemmLoad()
    except LoadUnavailable:
        print(
            "no torch/CUDA — running without load; performance metrics will be gaps",
            file=sys.stderr,
        )
        return NullLoad()


def _config(args) -> RunConfig:
    if args.quick:
        return RunConfig(baseline_s=30, load_s=150, hz=1.0)
    if args.long:
        return RunConfig(baseline_s=60, load_s=1800, hz=1.0)
    return RunConfig(baseline_s=60, load_s=600, hz=1.0)


def _execute(adapter, args, challenge=None):
    spec = lookup(adapter.inventory().model_name)
    load = _make_load(args, challenge)
    kwargs = {}
    if args.fake:  # simulated hardware: use a virtual clock instead of real waiting
        t = [0.0]

        def clock() -> float:
            return t[0]

        def sleep(s: float) -> None:
            t[0] += s

        kwargs = {"clock": clock, "sleep": sleep}
    return spec, load, execute_run(adapter, load, _config(args), spec, **kwargs)


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
        return f"Grade {letter} — delivers {ratio * 100:.1f}% of a healthy {model or 'card of this model'} ({extra})"
    if out.reason:
        return f"Grade: {out.reason}"
    if out.n is not None:
        return f"Grade: insufficient reference (n={out.n})"
    return None


def _confirm(server: str) -> bool:
    try:
        return input(f"Upload this result to {server}? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):  # closed stdin or Ctrl-C at the prompt means "no"
        print()
        return False


def _maybe_upload(result: dict, args) -> None:
    if args.dry_run:
        print(canonical_dumps(result))
        return
    server = args.server or DEFAULT_SERVER
    if not args.yes_upload:
        if not sys.stdin.isatty():
            print("Not uploaded. To upload this result, re-run with --yes-upload.")
            return
        if not _confirm(server):
            return
    out = upload(result, server)
    if not out.ok:
        print(f"warning: upload failed ({out.error}); result.json is saved locally")
        return
    if out.url:
        print(f"Result page: {out.url}")
    line = format_grade(out, (result.get("inventory") or {}).get("model_name"))
    if line:
        print(line)
    if out.delete_token:
        print(f"Delete token: {out.delete_token}")
        print("(save this token to delete the page later)")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        adapter = _make_adapter(args)
    except AdapterUnavailable as e:
        print(f"gpucheck: no NVIDIA GPU or driver available ({e}).", file=sys.stderr)
        return EXIT_NO_GPU
    except ValueError as e:  # only _make_adapter resolves --fake, so this is a bad profile name
        print(f"gpucheck: {e}", file=sys.stderr)
        return EXIT_NO_GPU
    challenge = None
    if args.certify:
        try:
            challenge = fetch_challenge(args.server or DEFAULT_SERVER)
        except ChallengeUnavailable as e:
            print(f"gpucheck: certification unavailable: {e}", file=sys.stderr)
            return EXIT_REFUSED
    try:
        spec, load, run = _execute(adapter, args, challenge)
    except LoadUnavailable as e:
        print(f"gpucheck: certification requires the GPU load generator ({e}).", file=sys.stderr)
        return EXIT_REFUSED
    except AdapterUnavailable as e:
        print(f"gpucheck: no NVIDIA GPU or driver available ({e}).", file=sys.stderr)
        return EXIT_NO_GPU

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
    result = build_result(run, metrics, findings, challenge, challenge_workload)
    Path(args.json).write_text(canonical_dumps(result))  # before any upload consideration
    print(render_report(result))
    print(f"Result written to {args.json}")
    if not args.offline:
        _maybe_upload(result, args)
    return EXIT_ABORTED if run.aborted else EXIT_OK
