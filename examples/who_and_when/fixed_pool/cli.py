"""Explicit prepare/run/report commands. No key is read outside the paid run command."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from pathlib import Path

import httpx

from .protocol import DEFAULT_CONFIG, ROOT, load_run, prepare
from .report import save_report
from .runner import HaltRun, redact, resolve_lost_call, run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Controlled adaptive and fixed six-judge comparison"
    )
    subs = result.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="Freeze data/prompts/settings; no LLM calls or key needed")
    p.add_argument("--phase", choices=("smoke", "pilot", "main"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / ".cache/ww-fixed-pool")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = subs.add_parser("run", help="Execute or resume frozen run (requires key unless --dry-run)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--max-cost-usd",
        type=float,
        help="Cumulative limit for this directory, including past attempts",
    )
    p.add_argument(
        "--key-file", type=Path, help="Read a key from this file instead of OPENROUTER_API_KEY"
    )
    p.add_argument(
        "--pilot-dir", type=Path, help="Completed pilot with identical settings (required for main)"
    )
    p.add_argument("--dry-run", action="store_true", help="Validate frozen inputs; no API requests")
    p = subs.add_parser("report", help="Recompute metrics, paired CI and expenses offline")
    p.add_argument("--output-dir", type=Path, required=True)
    p = subs.add_parser(
        "archive", help="Audit local artifacts and create an experiment archive; offline"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--archive-path", type=Path, required=True)
    p = subs.add_parser("resolve-lost-call", help="Record a verified charge after a lost response")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--call-id", required=True)
    p.add_argument("--charged-usd", type=float, required=True)
    p.add_argument("--note", required=True, help="Evidence used to verify the charge")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    key = ""
    try:
        if args.command == "prepare":
            manifest = prepare(args.output_dir, args.phase, args.data_dir, args.config)
            print(
                f"Prepared {len(manifest['entries'])} traces, two arms, "
                f"{len(manifest['entries']) * 16} primary calls. No inference performed."
            )
        elif args.command == "run":
            manifest, _, _ = load_run(args.output_dir)
            if args.dry_run:
                print(
                    f"Validated {manifest['phase']}: {len(manifest['entries'])} frozen traces, "
                    f"{len(manifest['entries']) * 16} primary calls. No API requests."
                )
                return 0
            if (
                args.max_cost_usd is None
                or not math.isfinite(args.max_cost_usd)
                or args.max_cost_usd <= 0
            ):
                arguments.error(
                    "run requires a positive --max-cost-usd (cumulative, not an extra allowance)"
                )
            key = (
                args.key_file.read_text().strip()
                if args.key_file
                else os.environ.get("OPENROUTER_API_KEY", "").strip()
            )
            if not key:
                arguments.error(
                    "Set OPENROUTER_API_KEY or supply --key-file; keys are never command-line values"
                )
            try:
                asyncio.run(run(args.output_dir, key, args.max_cost_usd, pilot_path=args.pilot_dir))
            finally:
                save_report(args.output_dir)
        elif args.command == "report":
            report = save_report(args.output_dir)
            print(
                f"Report complete={report['complete']}; "
                f"known expense=${report['accounting_all_attempts']['known_cost_usd']:.6f}; "
                f"unknown costs={report['accounting_all_attempts']['unknown_cost_calls']}"
            )
        elif args.command == "archive":
            from .archive import export_archive

            audit = export_archive(args.output_dir, args.archive_path)
            print(f"Archive: {args.archive_path}; storage audit passed={audit['storage_ok']}")
        elif args.command == "resolve-lost-call":
            resolve_lost_call(args.output_dir, args.call_id, args.charged_usd, args.note)
            print("Recorded manual reconciliation; this does not invoke a model.")
    except KeyboardInterrupt:
        print(
            "Interrupted. In-flight records are retained; unknown requests are not automatically replayed."
        )
        return 130
    except (HaltRun, ValueError, OSError, httpx.HTTPError) as exc:
        print(redact(f"Stopped: {exc}", key))
        return 2
    return 0
