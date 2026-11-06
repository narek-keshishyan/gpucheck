"""Command-line entry point: run, analyse, write result.json, print report."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .adapter import AdapterUnavailable, NvmlAdapter
from .diagnose import diagnose
from .fakes import make_fake
from .metrics import derive
from .report import render_report
from .resultjson import build_result, canonical_dumps
from .run import NullLoad, RunConfig, execute_run
from .specs import lookup

DEFAULT_SERVER = "https://gpucheck.example.invalid"  # placeholder until Task 10

EXIT_OK, EXIT_REFUSED, EXIT_NO_GPU, EXIT_ABORTED = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpucheck", description="GPU health report card")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="shorter run (30s baseline, 150s load)")
    g.add_argument("--long", action="store_true", help="longer run (60s baseline, 1800s load)")
    p.add_argument("--offline", action="store_true", help="never upload; report locally only")
    p.add_argument("--yes-upload", action="store_true", help="upload without prompting")
    p.add_argument("--dry-run", action="store_true", help="show what would be uploaded")
    p.add_argument("--json", metavar="PATH", default="./gpucheck-result.json", help="where to write result.json")
    p.add_argument("--server", metavar="URL", default=DEFAULT_SERVER, help="upload server URL")
    p.add_argument("--fake", metavar="PROFILE", help=argparse.SUPPRESS)
    return p


def _make_adapter(args):
    return make_fake(args.fake) if args.fake else NvmlAdapter()


def _make_load():
    # TODO(Task 9): return TorchGemmLoad() when torch is available.
    return NullLoad()


def _config(args) -> RunConfig:
    if args.quick:
        return RunConfig(baseline_s=30, load_s=150, hz=1.0)
    if args.long:
        return RunConfig(baseline_s=60, load_s=1800, hz=1.0)
    return RunConfig(baseline_s=60, load_s=600, hz=1.0)


def _execute(adapter, args):
    spec = lookup(adapter.inventory().model_name)
    kwargs = {}
    if args.fake:  # simulated hardware: use a virtual clock instead of real waiting
        t = [0.0]

        def clock() -> float:
            return t[0]

        def sleep(s: float) -> None:
            t[0] += s

        kwargs = {"clock": clock, "sleep": sleep}
    return spec, execute_run(adapter, _make_load(), _config(args), spec, **kwargs)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        adapter = _make_adapter(args)
    except AdapterUnavailable as e:
        print(f"gpucheck: no NVIDIA GPU or driver available ({e}).", file=sys.stderr)
        return EXIT_NO_GPU
    except ValueError as e:  # only _make_adapter resolves --fake, so this is a bad profile name
        print(f"gpucheck: {e}", file=sys.stderr)
        return EXIT_NO_GPU
    try:
        spec, run = _execute(adapter, args)
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
    result = build_result(run, metrics, findings)
    Path(args.json).write_text(canonical_dumps(result))  # before any upload consideration
    print(render_report(result))
    print(f"Result written to {args.json}")
    # Upload (--yes-upload / --dry-run / --server) is wired in Task 10; --offline skips it.
    return EXIT_ABORTED if run.aborted else EXIT_OK
