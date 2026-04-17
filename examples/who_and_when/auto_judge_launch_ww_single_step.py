"""Single-step sequential evaluation with early stopping.

One judge pool is generated per trace (same as the standard approach). Then
each summarized step is fed to the pipeline **individually** — judges see
exactly one step, with no surrounding context.

Evaluation stops as soon as FINAL_AGGREGATOR returns a "poor" verdict.
If no step is flagged, the trace is recorded as having no mistake found.

Cost vs. standard approach: same pool/graph generation cost, but N pipeline
runs per trace instead of 1. In practice most mistakes appear early, so the
average pipeline-run count is well below N.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import asyncio
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from autojudge.meta_agents import PoolGenerator
from autojudge.pipeline import PipelineBuilder
from autojudge.agent_pool import AgentPool
from autojudge.pipeline.types import GraphDict
from autojudge.utils.langfuse_utils import ainvoke_with_lf
from autojudge.utils import get_logger
from maseval import get_langfuse_judge_client
from pydantic_ai.messages import ModelMessagesTypeAdapter
from autojudge.meta_agents.prompts import examples_no_tools as examples
from autojudge.meta_agents.prompts import ww_output_schema, ww_taxonomy

logger = get_logger(__name__)



def _serialize_pipeline_trace(pipeline) -> list:
    """Serialize per-node message histories from a completed pipeline."""
    if pipeline is None or pipeline.trace is None:
        return []
    result = []
    for node_trace in pipeline.trace.node_traces:
        try:
            messages = ModelMessagesTypeAdapter.dump_python(
                node_trace.message_history, mode="json"
            )
        except Exception as e:
            messages = [{"_serialization_error": str(e), "_repr": repr(node_trace.message_history)}]
        result.append({
            "node_name": node_trace.node_name,
            "model": node_trace.model,
            "messages": messages,
        })
    return result


def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    graph_dict: GraphDict = {}
    for agent in agent_pool.full_agents_data:
        if agent["name"] != "FINAL_AGGREGATOR":
            graph_dict[agent["name"]] = ["FINAL_AGGREGATOR"]
    graph_dict["FINAL_AGGREGATOR"] = []
    return graph_dict


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        t = "\n".join(lines[1:-1]).strip()
    return t

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(save_folder: str, df, df_summary=None, split: str = "algo", test_mode: bool = False):
    logger.info("===Starting Who&When evaluation (single-step sequential with early stopping)===")

    pool_gen = PoolGenerator(
        output_schema=ww_output_schema, taxonomy=ww_taxonomy, examples=examples
    )
    judge_client = get_langfuse_judge_client()

    local_results_dir = Path(__file__).resolve().parent / "results" / save_folder

    done_traces: list[str] = []
    if local_results_dir.exists():
        done_traces = [f.split(".")[0] for f in os.listdir(local_results_dir) if f.endswith(".json")]

    failed_traces_ids: list[str] = []
    if local_results_dir.exists() and "failed_traces.txt" in os.listdir(local_results_dir):
        with open(local_results_dir / "failed_traces.txt") as fh:
            for line in fh:
                if line.startswith("Task ID:"):
                    failed_traces_ids.append(line.split(":")[1].strip())

    failed_traces: list[dict] = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        task_id = row["question_ID"]

        if task_id in done_traces:
            logger.info(f"Task {task_id} already processed, skipping...")
            continue
        if task_id in failed_traces_ids:
            logger.info(f"Task {task_id} previously failed, skipping...")
            continue

        logger.info(f"Processing task {idx + 1}/{len(df)}: {task_id}")

        try:
            # Use raw history directly. Summarized-step variant is commented out above.
            history_for_judges = [json.dumps(s, ensure_ascii=False) for s in row["history"]]
            input_type = "raw"
            logger.info(f"[INPUT] {len(history_for_judges)} raw steps for task {task_id}")

            query = json.dumps(row["question"], ensure_ascii=False)

            if "groundtruth" in df.columns:
                ground_truth = row["groundtruth"]
                correct_answer = str(row["is_corrected"])
            else:
                ground_truth = row.get("ground_truth", "")
                correct_answer = str(row.get("is_correct", ""))

            # ------------------------------------------------------------------
            # Generate judge pool ONCE for this trace (uses first step as sample)
            # ------------------------------------------------------------------
            sample_input = {
                "query": query,
                "step_index": 1,
                "step_to_evaluate": json.loads(history_for_judges[0]) if history_for_judges else {},
            }

            pool = None
            for attempt in range(3):
                try:
                    pool = await pool_gen.create_pool(sample_input)
                    if any(a.get("name") == "FINAL_AGGREGATOR" for a in pool.full_agents_data):
                        break
                    logger.warning(f"Attempt {attempt + 1}: FINAL_AGGREGATOR missing, retrying...")
                    pool = None
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Pool gen attempt {attempt + 1} failed: {e}")

            if pool is None:
                raise RuntimeError("Could not create a valid pool with FINAL_AGGREGATOR after 3 attempts")

            logger.info(f"Pool ready: {len(pool)} judges")

            graph = get_parallel_graph(pool)
            builder = PipelineBuilder()

            trace_metadata = {
                "task_id": task_id,
                "ground_truth": ground_truth,
                "correct_answer": correct_answer,
            }

            _tags = [
                split,
                "test" if test_mode else "full",
                "single_step",
                f"task_id:{task_id}",
                f"folder:{save_folder}",
            ]

            # ------------------------------------------------------------------
            # Outer Langfuse span for this trace
            # ------------------------------------------------------------------
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{task_id}",
                input={"task_id": task_id, "n_steps": len(history_for_judges)},
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(tags=_tags)

                guilty_result: dict | None = None
                guilty_step: int | None = None
                step_results: list[dict] = []
                last_pipeline = None

                # --------------------------------------------------------------
                # Step-by-step loop with early stop
                # --------------------------------------------------------------
                for step_index, step_str in enumerate(history_for_judges, start=1):
                    logger.info(f"  Evaluating step {step_index}/{len(history_for_judges)}...")

                    step_input = {
                        "query": query,
                        "step_index": step_index,
                        "step_to_evaluate": json.loads(step_str),
                    }

                    raw_result = None
                    current_pipeline = None
                    for attempt in range(1, 4):
                        pipeline = builder.create_from_pool(pool, graph).build()
                        raw, _tid = await ainvoke_with_lf(pool, pipeline, step_input, graph)
                        if raw and raw.strip():
                            raw_result = raw
                            current_pipeline = pipeline
                            break
                        logger.warning(
                            f"  [step {step_index}] Empty result on attempt {attempt}, retrying..."
                        )

                    if raw_result is None:
                        logger.warning(f"  Step {step_index}: pipeline returned empty result, recording error.")
                        step_results.append({"step": step_index, "error": "pipeline_empty"})
                        continue

                    last_pipeline = current_pipeline
                    clean = _strip_fences(raw_result)
                    try:
                        result_dict = json.loads(clean)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"  [step {step_index}] Could not parse JSON: {repr(clean)}"
                        )
                        result_dict = {"raw": clean}

                    verdict = result_dict.get("verdict", "").lower() if isinstance(result_dict, dict) else ""
                    step_results.append({"step": step_index, "result": result_dict})
                    logger.info(
                        f"  Step {step_index} → verdict={verdict!r}  agent={result_dict.get('agent', '?')}"
                    )

                    if verdict == "poor":
                        guilty_result = result_dict
                        guilty_step = step_index
                        logger.info(f"  [EARLY STOP] Mistake found at step {step_index}.")
                        break

                span.update(
                    output={
                        "guilty_step": guilty_step,
                        "guilty_result": guilty_result,
                        "steps_evaluated": len(step_results),
                    }
                )
                span.end()

            # ------------------------------------------------------------------
            # Build final answer in same {agent, step, reason} shape as other scripts
            # ------------------------------------------------------------------
            if guilty_result is not None:
                final_score = {
                    "agent": guilty_result.get("agent", ""),
                    "step": guilty_step,
                    "reason": guilty_result.get("reason") or guilty_result.get("justification", ""),
                }
            else:
                final_score = {
                    "agent": None,
                    "step": None,
                    "reason": "No step was flagged as poor by any judge.",
                }

            # Coerce step to int just in case
            if isinstance(final_score.get("step"), str):
                try:
                    final_score["step"] = int(str(final_score["step"]).strip().strip(","))
                except (ValueError, TypeError):
                    pass

            serializable_results = {
                "summarizer_score": {
                    "metric_name": "summarizer_score",
                    "approach": "single_step_sequential_early_stop",
                    "scores": [
                        {
                            "item_id": "overall_score",
                            "score": final_score,
                            "idx": idx,
                            "task_id": task_id,
                            "ground_truth": ground_truth,
                            "correct_answer": correct_answer,
                            "gt_agent": row["mistake_agent"],
                            "gt_step": row["mistake_step"],
                            "gt_mistake_reason": row["mistake_reason"],
                        }
                    ],
                },
                "debug": {
                    "input_type": input_type,
                    "n_steps_sent_to_judges": len(history_for_judges),
                    "n_raw_steps_in_dataset": len(row["history"]),
                    "n_steps_evaluated": len(step_results),
                    "early_stop": guilty_step is not None,
                    "step_results": step_results,
                    "judge_node_traces": _serialize_pipeline_trace(last_pipeline),
                    "langfuse_trace_id": span.trace_id,
                },
            }

            local_results_dir.mkdir(parents=True, exist_ok=True)
            output_file = local_results_dir / f"{task_id}.json"
            with open(output_file, "w") as fh:
                json.dump(serializable_results, fh, indent=2)
            logger.info(f"Saved: {output_file}")

        except Exception as e:
            error_msg = str(e)
            safe_error = error_msg.replace("{", "{{").replace("}", "}}")
            logger.error(f"Error processing task {task_id}: {safe_error}", exc_info=True)
            failed_traces.append(
                {"task_id": task_id, "task_index": idx + 1, "error": error_msg, "error_type": type(e).__name__}
            )
            print(f"\n!  Failed task {idx + 1}/{len(df)}: {task_id}\n   Error: {error_msg}\n")

    # --------------------------------------------------------------------------
    # Write failed_traces.txt
    # --------------------------------------------------------------------------
    if failed_traces:
        local_results_dir.mkdir(parents=True, exist_ok=True)
        failed_file = local_results_dir / "failed_traces.txt"
        with open(failed_file, "w") as fh:
            fh.write(f"Failed traces: {len(failed_traces)} out of {len(df)}\n{'=' * 80}\n\n")
            for ft in failed_traces:
                fh.write(
                    f"Task ID: {ft['task_id']}\n"
                    f"Index: {ft['task_index']}/{len(df)}\n"
                    f"Error Type: {ft['error_type']}\n"
                    f"Error Message: {ft['error']}\n"
                    f"{'-' * 80}\n\n"
                )
        logger.warning(f"{len(failed_traces)} traces failed. Details: {failed_file}")

    total_failed = len(failed_traces) + len(failed_traces_ids)
    logger.info(
        f"Done: {len(df) - total_failed}/{len(df)} successful, {total_failed} failed"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential single-step evaluation with early stopping."
    )
    parser.add_argument(
        "--split",
        choices=["hand", "algo"],
        default="hand",
        help="Dataset split: 'hand' (Hand-Crafted) or 'algo' (Algorithm-Generated). Default: algo.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run on 1 trace only.",
    )
    args = parser.parse_args()

    df_handcrafted = pd.read_parquet(
        "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet"
    )
    df_algorithm = pd.read_parquet(
        "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet"
    )

    if args.split == "hand":
        df = df_handcrafted
        folder = "hand_single_test" if args.test else "hand_single"
    else:
        df = df_algorithm
        folder = "algo_single_test" if args.test else "algo_single"

    if args.test:
        print(f"[TEST MODE] Running on 1 trace only (split={args.split}).")
        df = df[:1]

    asyncio.run(
        main(
            save_folder=folder,
            df=df,
            split=args.split,
            test_mode=args.test,
        )
    )
