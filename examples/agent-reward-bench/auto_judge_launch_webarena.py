"""AutoJudge evaluation for WebArena agent trajectories (full trace, no DB tool)."""
import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))  # for autojudge
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # for maseval

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from maseval import get_langfuse_judge_client

from autojudge.meta_agents import PoolGenerator
from autojudge.meta_agents.graph_gen import get_parallel_graph
from autojudge.pipeline import PipelineBuilder
from autojudge.utils import get_logger

from autojudge.meta_agents.prompts import (
    examples_no_tools as examples,
    webarena_output_schema as output_schema,
    webarena_taxonomy as taxonomy,
)

logger = get_logger(__name__)


async def create_pool_with_retries(pool_gen, judge_input: dict, max_attempts: int = 3):
    context_feedback = None
    attempt_errors = []

    for attempt in range(1, max_attempts + 1):
        try:
            pool = await pool_gen.create_pool(str(judge_input), context=context_feedback)
            if any(
                agent.get("name") == "FINAL_AGGREGATOR"
                if isinstance(agent, dict)
                else agent.name == "FINAL_AGGREGATOR"
                for agent in pool.full_agents_data
            ):
                return pool

            context_feedback = (
                "Previous attempt was invalid: include FINAL_AGGREGATOR exactly named "
                "'FINAL_AGGREGATOR' and return a non-empty list of valid agents."
            )
            attempt_errors.append(f"attempt {attempt}/{max_attempts}: missing FINAL_AGGREGATOR")
        except Exception as exc:
            if "No valid agents generated" in str(exc):
                context_feedback = (
                    "Previous attempt returned no valid agents. Return a non-empty list of "
                    "valid agent schemas and include FINAL_AGGREGATOR exactly once."
                )
            elif "output validation" in str(exc).lower() or "validation" in str(exc).lower():
                context_feedback = (
                    "Your previous response was NOT valid JSON matching the exact schema requested. "
                    "Ensure your response is a valid strict JSON list of objects. DO NOT use unescaped newlines in strings. "
                    "Include FINAL_AGGREGATOR exactly once."
                )
            else:
                context_feedback = f"Previous attempt failed with error: {str(exc)[:200]}. Fix the schema."

            attempt_errors.append(f"attempt {attempt}/{max_attempts}: {type(exc).__name__}: {str(exc)[:300]}")
            logger.warning(f"Pool creation attempt {attempt} failed:", exc_info=True)

    error_block = "\n".join(f"- {e}" for e in attempt_errors)
    raise RuntimeError(f"Failed to create pool after {max_attempts} attempts. Errors:\n{error_block}")


def format_trace_for_judge(trace_data: dict) -> str:
    """Format WebArena trace exactly like ARB (excluding images)."""
    steps = trace_data.get("steps", [])

    STEP_TEMPLATE = (
        "-----\n"
        "Step: {step_number}\n"
        "URL: {url}\n"
        "Action: {action}\n"
        "Reasoning: {reasoning}\n"
    )
    ACTION_TEMPLATE = (
        "The agent performed the following actions:\n"
        "{steps}\n"
        "-----\n"
    )
    AXTREE_TEMPLATE = (
        "The last accessibility tree is:\n"
        "{axtree}\n"
    )

    steps_str = ""
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        steps_str += STEP_TEMPLATE.format(
            step_number=i + 1,
            url=step.get("url", "unknown"),
            action=step.get("action", "unknown"),
            reasoning=step.get("reasoning", "")
        )

    action_msg = ACTION_TEMPLATE.format(steps=steps_str)

    last_step = steps[-1] if steps else {}
    axtree_content = ""
    if "axtree_pruned" in last_step and last_step["axtree_pruned"]:
        axtree_content = last_step["axtree_pruned"]
    elif "axtree" in last_step and last_step["axtree"]:
        axtree_content = last_step["axtree"]

    if axtree_content:
        axtree_msg = AXTREE_TEMPLATE.format(axtree=axtree_content)
    else:
        axtree_msg = ""

    return action_msg + "\n" + axtree_msg


async def evaluate_trace(trace_id: str, trace_data: dict, pool_gen, judge_client) -> dict:
    # Prepare judge input
    history_for_evaluating = format_trace_for_judge(trace_data)
    judge_input_dict = {
        "goal": trace_data.get("goal", ""),
        "trace_id": trace_id,
        "history_for_evaluating": history_for_evaluating,
    }
    judge_input = str(judge_input_dict)

    # Don't send the entire massive trace to the generator LLM
    pool_gen_input_dict = {
        "goal": trace_data.get("goal", ""),
        "trace_id": trace_id,
    }
    pool = await create_pool_with_retries(pool_gen, pool_gen_input_dict)

    # Build pipeline
    graph = get_parallel_graph(pool)
    builder = PipelineBuilder()
    pipeline = builder.create_from_pool(pool, graph).build()

    ctx_mgr = judge_client.start_as_current_span(
        name=f"evaluate_webarena_{trace_id}",
        input={"task_id": trace_id},
        metadata={"task_id": trace_id, "benchmark": "webarena"}
    ) if judge_client else contextlib.nullcontext()

    with ctx_mgr as span:
        if judge_client:
            judge_client.update_current_trace(tags=["webarena", "arb", "autojudge", f"task_id:{trace_id}"])

        raw_result = await pipeline.ainvoke(judge_input)

        result = raw_result.get("response") if isinstance(raw_result, dict) else raw_result

        # Clean JSON output
        clean = result.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            result_dict = json.loads(clean)
        except json.JSONDecodeError:
            result_dict = {"raw": result}

        if span and hasattr(span, 'update'):
            span.update(output={"result": result_dict})

        if judge_client:
            judge_client.flush()

    # Parse judge predictions
    judge_predictions = {}
    success_val = result_dict.get("success", "").lower().strip() if isinstance(result_dict.get("success"), str) else None
    side_val = result_dict.get("side", "").lower().strip() if isinstance(result_dict.get("side"), str) else None
    loop_val = result_dict.get("loop", "").lower().strip() if isinstance(result_dict.get("loop"), str) else None
    optimal_val = result_dict.get("optimal", "").lower().strip() if isinstance(result_dict.get("optimal"), str) else None

    judge_predictions["trajectory_success"] = success_val in ["yes", "successful"] if success_val else None
    judge_predictions["trajectory_side_effect"] = side_val == "yes" if side_val else None
    judge_predictions["trajectory_looping"] = loop_val == "yes" if loop_val else None
    judge_predictions["trajectory_optimality"] = optimal_val if optimal_val else None

    if not all([success_val, side_val, loop_val, optimal_val]):
        missing = [k for k, v in {"success": success_val, "side": side_val, "loop": loop_val, "optimal": optimal_val}.items() if not v]
        logger.warning(f"Missing fields {missing} in judge output for {trace_id}")

    return {
        "task_id": trace_id,
        "benchmark": trace_data.get("benchmark", "webarena"),
        "model": trace_data.get("model", ""),
        "ground_truth": {
            "task_success": trace_data.get("summary_info", {}).get("task_success"),
        },
        "judge_predictions": judge_predictions,
        "judge_output": result_dict,
    }


async def main(data_dir: str, save_folder: str, max_traces: int | None = None):
    logger.info("===Starting WebArena evaluation===")

    pool_gen = PoolGenerator(output_schema=output_schema, taxonomy=taxonomy, examples=examples)
    judge_client = get_langfuse_judge_client()

    data_dir = Path(data_dir)
    cleaned_dir = data_dir / "cleaned"
    pruned_dir = data_dir / "pruned"

    if not cleaned_dir.exists():
        logger.error(f"Cleaned directory {cleaned_dir} not found.")
        return

    trace_files = sorted(cleaned_dir.rglob("*.json"))

    if not trace_files:
        logger.warning(f"No JSON files found in {cleaned_dir}")
        return

    if max_traces is not None:
        trace_files = trace_files[:max_traces]

    logger.info(f"Found {len(trace_files)} traces to evaluate")

    local_results_dir = Path(__file__).resolve().parent / "results" / save_folder
    local_results_dir.mkdir(parents=True, exist_ok=True)

    done_traces = set()
    if local_results_dir.exists():
        for res in os.listdir(local_results_dir):
            if res.endswith(".json"):
                done_traces.add(Path(res).stem)

    failed_traces = []
    failed_file = local_results_dir / "failed_traces.txt"
    failed_traces_ids = []

    if failed_file.exists():
        with open(failed_file) as fh:
            for line in fh:
                if line.startswith("Task ID:"):
                    failed_traces_ids.append(line.split(":")[1].strip())

    for idx, trace_file in enumerate(trace_files):
        trace_id = trace_file.stem

        if trace_id in done_traces:
            logger.info(f"Task {trace_id} already processed, skipping...")
            continue
        if trace_id in failed_traces_ids:
            logger.info(f"Task {trace_id} previously failed, skipping...")
            continue

        logger.info(f"Processing task {idx + 1}/{len(trace_files)}: {trace_id}")

        try:
            with open(trace_file) as f:
                trace_data = json.load(f)

            output_data = None
            try:
                output_data = await evaluate_trace(trace_id, trace_data, pool_gen, judge_client)
            except Exception as e:
                logger.warning(f"Error evaluating cleaned trace {trace_id}: {e}. Retrying with pruned trace.")
                pruned_file = pruned_dir / f"{trace_id}.json"
                if pruned_file.exists():
                    with open(pruned_file) as pf:
                        pruned_data = json.load(pf)
                    output_data = await evaluate_trace(trace_id, pruned_data, pool_gen, judge_client)
                else:
                    raise RuntimeError(f"Cleaned trace failed and pruned trace {pruned_file} not found.") from e

            if output_data:
                out_file = local_results_dir / f"{trace_id}.json"
                with open(out_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Saved: {out_file}")

        except Exception as e:
            error_msg = str(e)
            logger.error("Error processing task {}: {}", trace_id, error_msg, exc_info=True)
            failed_traces.append({"task_id": trace_id, "error": error_msg})

    if failed_traces:
        with open(failed_file, "w") as fh:
            fh.write(f"Failed traces: {len(failed_traces)}\n{'=' * 80}\n\n")
            for ft in failed_traces:
                fh.write(f"Task ID: {ft['task_id']}\nError Message: {ft['error']}\n{'-' * 80}\n\n")

    logger.info(f"Completed: {len(trace_files) - len(failed_traces)}/{len(trace_files)} successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WebArena with AutoJudge")
    parser.add_argument("--data-dir", default="examples/agent-reward-bench/data", help="Path to base data directory containing cleaned/ and pruned/ folders")
    parser.add_argument("--save-folder", default="webarena_results_fixed", help="Output folder name")
    parser.add_argument("--max-traces", type=int, default=None, help="Max traces to evaluate")
    parser.add_argument("--test", action="store_true", help="Test mode: run on 5 traces")

    args = parser.parse_args()
    max_traces = 5 if args.test else args.max_traces
    asyncio.run(main(data_dir=args.data_dir, save_folder=args.save_folder, max_traces=max_traces))
