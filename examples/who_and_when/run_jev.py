"""Evaluate the same labeled Who&When traces with chat judges, a Jev pool, or direct Jev."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

from autojudge.meta_agents import PoolGenerator
from autojudge.meta_agents.decision_pool import DecisionPoolGenerator, direct_decision_pool
from autojudge.meta_agents.graph_gen import get_parallel_graph
from autojudge.meta_agents.prompts import examples_no_tools
from autojudge.meta_agents.prompts.output_schema_prompts import ww_bench
from autojudge.model_connection import chat_api_key, decision_connection, model_http_client
from autojudge.pipeline import PipelineBuilder
from autojudge.pipeline.decisions import DecisionTrace

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


async def run(args: argparse.Namespace) -> dict:
    cases = []
    for row in load_rows(args.input, args.limit):
        trace = DecisionTrace(query=row["question"], history_for_evaluating=row["history"])
        trace.candidates()
        # Who&When labels are 0-based. Both AutoJudge arms predict 1-based positions.
        reference = {"agent": str(row["mistake_agent"]), "step": int(row["mistake_step"]) + 1}
        cases.append((str(row["question_ID"]), trace, reference))
    if len({case[0] for case in cases}) != len(cases):
        raise ValueError("Duplicate question_ID in selected traces")
    if {"pool", "direct"}.intersection(args.mode):
        decision_connection()
    if {"chat", "pool"}.intersection(args.mode):
        chat_api_key()

    generators = {}
    if "pool" in args.mode:
        generators["pool"] = DecisionPoolGenerator(model=args.meta_model, temperature=0.3)
    if "chat" in args.mode:
        generators["chat"] = PoolGenerator(
            model=args.meta_model,
            temperature=0.3,
            taxonomy=ww_bench.taxonomy,
            output_schema=ww_bench.output_schema,
            examples=examples_no_tools,
        )

    report = {"runs": []}
    with args.output.open("x", encoding="utf-8") as output_file:
        async with model_http_client() as client:
            for case_id, trace, reference in cases:
                for mode in args.mode:
                    start = perf_counter()
                    row = {
                        "trace_id": case_id,
                        "mode": mode,
                        "reference": reference,
                        "meta_model": args.meta_model if mode != "direct" else None,
                    }
                    pipeline = None
                    generator = generators.get(mode)
                    if generator is not None:
                        generator._usage = None
                    try:
                        if mode == "direct":
                            pool = direct_decision_pool()
                        elif mode == "pool":
                            pool = await generators[mode].create_pool(trace)
                        else:
                            pool = await generators[mode].create_pool(trace.model_dump())
                            for node in pool:
                                node.model = args.chat_model
                        if sum(node.name == "FINAL_AGGREGATOR" for node in pool) != 1:
                            raise ValueError("Expected exactly one FINAL_AGGREGATOR")
                        row["pool"] = pool.full_agents_data
                        pipeline = (
                            PipelineBuilder()
                            .create_from_pool(pool, get_parallel_graph(pool))
                            .build()
                        )
                        result = await pipeline.ainvoke(trace.model_dump(), decision_client=client)
                        if mode == "chat":
                            result = json.loads(
                                result.strip()
                                .removeprefix("```json")
                                .removeprefix("```")
                                .removesuffix("```")
                                .strip()
                            )
                        row["prediction"] = {**result, "step": int(result["step"])}
                    except Exception as exc:  # noqa: BLE001 - score per-trace failures in the denominator.
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    row["elapsed_seconds"] = perf_counter() - start
                    row["meta_estimated_cost_usd"] = generator.cost.total_price if generator else 0
                    row["judge_cost_usd"] = pipeline.cost.total_price if pipeline else 0
                    row["judge_cost_source"] = "estimate" if mode == "chat" else "Jev receipts"
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
    grade(report, args.mode)
    return report


def grade(report: dict, modes: list[str]) -> None:
    report["metrics"] = {}
    for mode in modes:
        rows = [row for row in report["runs"] if row["mode"] == mode]
        report["metrics"][mode] = {
            "traces": len(rows),
            "failed": sum("error" in row for row in rows),
            **{
                f"{field}_accuracy": sum(
                    row.get("prediction", {}).get(field) == row["reference"][field] for row in rows
                )
                / len(rows)
                for field in ("agent", "step")
            },
            "total_recorded_cost_usd": sum(
                row["meta_estimated_cost_usd"] + row["judge_cost_usd"] for row in rows
            ),
            "mean_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows) / len(rows),
        }


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
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if len(set(args.mode)) != len(args.mode):
        parser.error("--mode entries must be unique")
    report = asyncio.run(run(args))
    print(json.dumps(report["metrics"], indent=2))
    if any(arm["failed"] for arm in report["metrics"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
