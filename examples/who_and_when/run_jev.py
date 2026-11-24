"""Evaluate the same labeled Who&When traces with chat judges, a Jev pool, or direct Jev."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from jev_protocol import REVISION, select_cases, summarize, verify_snapshot

import autojudge
from autojudge.meta_agents import PoolGenerator
from autojudge.meta_agents.decision_pool import DecisionPoolGenerator, direct_decision_pool
from autojudge.meta_agents.graph_gen import get_parallel_graph
from autojudge.meta_agents.prompts import examples_no_tools
from autojudge.meta_agents.prompts.output_schema_prompts import ww_bench
from autojudge.model_connection import chat_api_key, decision_connection, model_http_client
from autojudge.pipeline import PipelineBuilder
from autojudge.pipeline.decisions import (
    DECISION_CONCURRENCY,
    DECISION_MAX_ATTEMPTS,
    FINAL_QUESTION,
    JEV_MODEL,
    DecisionNode,
    DecisionTrace,
)
from autojudge.pipeline.node import AGENT_NODE_TEMPERATURE

MODES = ("chat", "pool", "direct")


def load_rows(path: Path, limit: int | None) -> list[dict]:
    if path.suffix == ".parquet":
        from pyarrow import parquet

        source = parquet.ParquetFile(path)
        if limit is None:
            rows = source.read().to_pylist()
        else:
            batch = next(source.iter_batches(batch_size=limit), None)
            rows = batch.to_pylist() if batch is not None else []
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = rows[:limit]
    if not rows:
        raise ValueError("No labeled traces selected")
    return rows


def usage_record(usage, rates: list[float] | None, *, provider_cost=None) -> dict:
    if usage is None:
        return {"cost_usd": None, "source": "unavailable"}
    cost = provider_cost
    source = "provider"
    if cost is None and rates is not None:
        cost = (usage.input_tokens * rates[0] + usage.output_tokens * rates[1]) / 1_000_000
        source = "token_rate_estimate"
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "requests": usage.requests,
        "cost_usd": cost,
        "source": source if cost is not None else "unavailable",
    }


async def run(args: argparse.Namespace) -> dict:
    controlled = args.phase != "quick"
    if controlled:
        verify_snapshot(args.input, "hand" if args.phase == "pilot" else "algo")
        if args.phase == "main":
            verify_snapshot(args.hand, "hand")
        hand = load_rows(args.hand, None) if args.phase == "main" else []
        selected = select_cases(load_rows(args.input, None), hand, args.phase)
    else:
        selected = [(row, "quick") for row in load_rows(args.input, args.limit)]
    cases = []
    for row, cohort in selected:
        trace = DecisionTrace(query=row["question"], history_for_evaluating=row["history"])
        trace.candidates()
        # Who&When labels are 0-based. Both AutoJudge arms predict 1-based positions.
        reference = {"agent": str(row["mistake_agent"]), "step": int(row["mistake_step"]) + 1}
        cases.append((str(row["question_ID"]), trace, reference, cohort))
    if len({case[0] for case in cases}) != len(cases):
        raise ValueError("Duplicate question_ID in selected traces")
    if {"pool", "direct"}.intersection(args.mode):
        decision_connection()
    if {"chat", "pool"}.intersection(args.mode):
        chat_api_key()

    count = 6 if controlled else None

    def make_generators():
        generators = {}
        if "pool" in args.mode:
            generators["pool"] = DecisionPoolGenerator(
                model=args.meta_model, temperature=0.3, specialist_count=count
            )
        if "chat" in args.mode:
            generators["chat"] = PoolGenerator(
                model=args.meta_model,
                temperature=0.3,
                taxonomy=ww_bench.taxonomy,
                output_schema=ww_bench.output_schema,
                examples=examples_no_tools,
                specialist_count=count,
            )
        return generators

    generators = make_generators()

    manifest = {
        "phase": args.phase,
        "seed": args.seed,
        "modes": args.mode,
        "trace_workers": args.workers,
        "packages": {name: version(name) for name in ("httpx", "pydantic", "pyarrow")}
        if controlled
        else {},
        "source_sha256": {
            (
                file.relative_to(Path(autojudge.__file__).parent).as_posix()
                if file.is_relative_to(Path(autojudge.__file__).parent)
                else "runner/" + file.name
            ): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in [
                *sorted(Path(autojudge.__file__).parent.rglob("*.py")),
                Path(__file__).resolve(),
                Path(__file__).with_name("jev_protocol.py").resolve(),
            ]
        },
        "dataset_revision": REVISION if controlled else None,
        "meta_model": args.meta_model,
        "chat_model": args.chat_model,
        "decision_model": JEV_MODEL,
        "decision_concurrency": DECISION_CONCURRENCY,
        "decision_max_attempts": DECISION_MAX_ATTEMPTS,
        "meta_temperature": 0.3,
        "chat_temperature": AGENT_NODE_TEMPERATURE,
        "specialist_count": count,
        "meta_rates_usd_per_million": args.meta_rates,
        "chat_rates_usd_per_million": args.chat_rates,
        "meta_prompts": {m: g._get_system_prompt() for m, g in generators.items()},
        "cases": [
            {
                "trace_id": cid,
                "reference": ref,
                "cohort": group,
                "trace_sha256": hashlib.sha256(trace.model_dump_json().encode()).hexdigest(),
            }
            for cid, trace, ref, group in cases
        ],
    }
    report = {"runs": []}
    manifest_path = args.output.with_suffix(".manifest.json")
    recorded = set()
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Cannot resume: input manifest or run settings have changed")
        report["runs"] = [
            json.loads(line) for line in args.output.read_text().splitlines() if line.strip()
        ]
        summarize(manifest, report["runs"])
        recorded = {(row["trace_id"], row["mode"]) for row in report["runs"]}
        if args.output.stat().st_size and not args.output.read_bytes().endswith(b"\n"):
            raise ValueError("Cannot resume: results must end at a complete JSONL line")
    with args.output.open("a" if args.resume else "x", encoding="utf-8") as output_file:
        if not args.resume:
            with manifest_path.open("x", encoding="utf-8") as manifest_file:
                json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        async with model_http_client() as client:
            case_iterator = iter(cases)

            async def worker(local_generators) -> None:
                for case_id, trace, reference, cohort in case_iterator:
                    # Rotate arm order deterministically to reduce first-request bias.
                    rotation = int(
                        hashlib.sha256(f"{args.seed}:{case_id}".encode()).hexdigest(), 16
                    )
                    rotation %= len(args.mode) if controlled else 1
                    for mode in args.mode[rotation:] + args.mode[:rotation]:
                        if (case_id, mode) in recorded:
                            continue
                        start = perf_counter()
                        row = {
                            "trace_id": case_id,
                            "mode": mode,
                            "reference": reference,
                            "cohort": cohort,
                            "meta_model": args.meta_model if mode != "direct" else None,
                        }
                        pipeline = None
                        meta_seconds = None
                        generator = local_generators.get(mode)
                        if generator is not None:
                            generator._usage = None
                        try:
                            if mode == "direct":
                                pool = direct_decision_pool()
                            elif mode == "pool":
                                pool = await local_generators[mode].create_pool(trace)
                            else:
                                pool = await local_generators[mode].create_pool(trace.model_dump())
                                for node in pool:
                                    node.model = args.chat_model
                            meta_seconds = perf_counter() - start if generator is not None else 0.0
                            if len({node.name for node in pool}) != len(pool):
                                raise ValueError("Expected unique judge names")
                            if controlled and mode != "direct" and len(pool) != count + 1:
                                raise ValueError(f"Expected exactly {count} specialists")
                            if controlled and mode == "chat" and pool.has_agent("FINAL_AGGREGATOR"):
                                pool.get_agent("FINAL_AGGREGATOR").instructions = (
                                    FINAL_QUESTION
                                    + " Return only a JSON object with agent (the exact "
                                    "author name from the trace), step (1-based integer), and reason."
                                )
                            if sum(node.name == "FINAL_AGGREGATOR" for node in pool) != 1:
                                raise ValueError("Expected exactly one FINAL_AGGREGATOR")
                            row["pool"] = pool.full_agents_data
                            pipeline = (
                                PipelineBuilder()
                                .create_from_pool(pool, get_parallel_graph(pool))
                                .build()
                            )
                            result = await pipeline.ainvoke(
                                trace.model_dump(), decision_client=client
                            )
                            if mode == "chat":
                                result = json.loads(
                                    result.strip()
                                    .removeprefix("```json")
                                    .removeprefix("```")
                                    .removesuffix("```")
                                    .strip()
                                )
                            if type(result["step"]) not in (int, str):
                                raise ValueError(
                                    "Predicted step is not an integer or integer string"
                                )
                            step = int(result["step"])
                            if not isinstance(result["agent"], str) or not result["agent"].strip():
                                raise ValueError("Invalid predicted agent")
                            if not 1 <= step <= len(trace.history_for_evaluating):
                                raise ValueError("Predicted step is out of range")
                            row["prediction"] = {**result, "step": step}
                        except Exception as exc:  # noqa: BLE001 - score per-trace failures in the denominator.
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        row["elapsed_seconds"] = perf_counter() - start
                        row["meta_seconds"] = meta_seconds
                        row["meta_usage"] = (
                            usage_record(generator._usage, args.meta_rates)
                            if generator
                            else {"cost_usd": 0.0, "source": "no_call"}
                        )
                        row["judge_usage"] = []
                        if pipeline and getattr(pipeline, "_trace", None) is not None:
                            for node in pipeline.trace.node_traces:
                                record = usage_record(
                                    node.usage,
                                    args.chat_rates if mode == "chat" else None,
                                    provider_cost=(
                                        node.decision["usage"]["cost"] if node.decision else None
                                    ),
                                )
                                row["judge_usage"].append(
                                    {
                                        **record,
                                        "name": node.node_name,
                                        "model": node.model,
                                        "stage": (
                                            "final"
                                            if node.node_name == "FINAL_AGGREGATOR"
                                            else "specialists"
                                        ),
                                    }
                                )
                        row["decision_attempts"] = (
                            {
                                node.name: node.attempts
                                for node in pipeline.execution_order
                                if isinstance(node, DecisionNode) and node.attempts
                            }
                            if pipeline
                            else {}
                        )
                        row["unpriced_decision_attempts"] = sum(
                            "usage" not in attempt
                            for attempts in row["decision_attempts"].values()
                            for attempt in attempts
                        )
                        costs = [
                            rec["cost_usd"] for rec in [row["meta_usage"], *row["judge_usage"]]
                        ]
                        row["total_cost_usd"] = (
                            sum(costs)
                            if "error" not in row
                            and not row["unpriced_decision_attempts"]
                            and all(c is not None for c in costs)
                            else None
                        )
                        if pipeline and getattr(pipeline, "_trace", None) is not None:
                            row["nodes"] = [
                                {
                                    "name": node.node_name,
                                    "model": node.model,
                                    "decision": node.decision,
                                    "output": pipeline.node_session.node_executions[
                                        node.node_id
                                    ].output,
                                }
                                for node in pipeline.trace.node_traces
                            ]
                        report["runs"].append(row)
                        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output_file.flush()

            await asyncio.gather(
                worker(generators), *(worker(make_generators()) for _ in range(args.workers - 1))
            )
    report["metrics"] = summarize(manifest, report["runs"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Who&When parquet or JSONL with original columns")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL results file")
    parser.add_argument("--mode", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument(
        "--meta-model", default=os.getenv("POOL_GEN_MODEL", "google/gemini-2.5-flash")
    )
    parser.add_argument(
        "--chat-model", default=os.getenv("AGENT_NODE_MODEL", "google/gemini-2.5-flash")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--phase", choices=("quick", "pilot", "main"), default="quick")
    parser.add_argument("--hand", type=Path, help="Frozen Hand parquet for overlap checks in main")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent traces")
    parser.add_argument(
        "--resume", action="store_true", help="Continue an interrupted matching run"
    )
    parser.add_argument(
        "--meta-rates",
        type=float,
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="USD per million MetaAgent tokens",
    )
    parser.add_argument(
        "--chat-rates",
        type=float,
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="USD per million chat judge tokens",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if len(set(args.mode)) != len(args.mode):
        parser.error("--mode entries must be unique")
    if args.phase != "quick" and args.limit is not None:
        parser.error("--limit is only for quick runs")
    if args.phase == "main" and args.hand is None:
        parser.error("--phase main requires --hand")
    for rates in (args.meta_rates, args.chat_rates):
        if rates is not None and any(not math.isfinite(r) or r < 0 for r in rates):
            parser.error("Token rates must be finite and nonnegative")
    report = asyncio.run(run(args))
    print(json.dumps(report["metrics"], indent=2))
    if any(arm["failed"] for arm in report["metrics"]["all"]["arms"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
