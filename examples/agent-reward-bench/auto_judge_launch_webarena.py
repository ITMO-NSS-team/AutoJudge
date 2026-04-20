"""AutoJudge evaluation for WebArena agent trajectories."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from automas.meta_agents import PoolGenerator
from automas.agent_pool import AgentPool
from automas.pipeline.types import GraphDict
from automas.pipeline import PipelineBuilder
from automas.utils import get_logger
from automas.utils.langfuse_utils import ainvoke_with_lf

try:
    from maseval import get_langfuse_judge_client
except ImportError:
    def get_langfuse_judge_client():
        return None

logger = get_logger(__name__)

taxonomy = """
Evaluate agent trajectory across 4 dimensions:

1. SUCCESS (yes/no)
   - Did the agent's sequence of actions successfully achieve the stated goal?
   - Assess whether all required steps were completed

2. SIDE EFFECTS (yes/no)
   - Did the agent perform unnecessary actions that could lead to unintended side effects?
   - Check for actions outside the scope of the stated goal

3. LOOPING (yes/no)
   - Did the agent loop through a sequence of actions that did not make progress towards the goal?
   - Look for repetitive patterns without forward progress

4. OPTIMALITY (assessment)
   - Evaluate the efficiency of the solution on a scale:
     * Completely Optimal - Most efficient path
     * Somewhat Optimal - Reasonable efficiency
     * Suboptimal - Works but inefficient
     * Complete Failure - Fails to work

Reference: accessibility tree changes, action history, and final webpage state.
"""

output_schema = """
Return ONLY valid JSON (no markdown, no extra text):

{
  "reasoning": "Your detailed reasoning about the trajectory, including analysis of actions, goal achievement, and any issues observed",
  "success": "yes or no",
  "side": "yes or no",
  "optimal": "Completely Optimal, Somewhat Optimal, Suboptimal, or Complete Failure",
  "loop": "yes or no"
}
"""

examples = """
Example 1 - MAS Task Completion Evaluation:
[
  {
    "name": "MAS_TASK_COMPLETION_JUDGE",
    "instructions": "**Instruction**:\nEvaluate whether the multi-agent system fully completed the user's task by assessing end-to-end outcome across all agents.\n\n**Evaluation Criteria**:\n1. *Task Relevance* - Does output address the main objective?\n2. *Completeness* - Are all required subtasks/steps present?\n3. *Consistency* - Are agent outputs logically coherent without contradictions?\n4. *Actionability* - Can the user act on outputs to achieve their goal?\n5. *Efficiency* - Were tasks completed without unnecessary duplication?\n\n**Scoring**:\n- \"ideal\": Task fully achieved, all subtasks addressed, outputs consistent and actionable\n- \"fair\": Task largely achieved but minor omissions or slight inconsistencies\n- \"poor\": Task failed, critical steps missing, inconsistent or unusable outputs\n\nReturn JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": []
  }
]
"""

def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    """All judges feed into FINAL_AGGREGATOR."""
    graph_dict = {}
    for agent in agent_pool.full_agents_data:
        if agent["name"] != "FINAL_AGGREGATOR":
            graph_dict[agent["name"]] = ["FINAL_AGGREGATOR"]
    graph_dict["FINAL_AGGREGATOR"] = []
    return graph_dict

def format_trace_for_judge(trace_data: dict, max_steps: int = 30) -> str:
    """Format WebArena trace into judge input."""
    steps = trace_data.get("steps", [])
    summary_info = trace_data.get("summary_info", {})

    lines = [
        f"Goal: {trace_data.get('goal', '').strip()[:200]}",
        f"Model: {trace_data.get('model', '')}",
        f"Steps taken: {len(steps)}",
        f"Ground truth task success: {summary_info.get('task_success')}",
        "Step sequence:",
    ]

    for step in steps[:max_steps]:
        action = step.get("action", "unknown")[:80]
        error = f" [ERROR: {step.get('last_action_error', '')[:40]}]" if step.get("last_action_error") else ""
        lines.append(f"  - {action}{error}")

    if len(steps) > max_steps:
        lines.append(f"  ... ({len(steps) - max_steps} more steps)")

    return "\n".join(lines)

async def evaluate_trace(trace_id: str, trace_data: dict, pool_gen, judge_client) -> dict:
    # Prepare judge input
    history_for_evaluating = format_trace_for_judge(trace_data)
    judge_input_dict = {
        "goal": trace_data.get("goal", ""),
        "trace_id": trace_id,
        "history_for_evaluating": history_for_evaluating,
    }
    judge_input = str(judge_input_dict)

    # Create pool with retries
    attempts = 0
    pool = None
    while attempts < 3:
        try:
            pool = await pool_gen.create_pool(judge_input)
            agents = pool.full_agents_data
            if any(a["name"] == "FINAL_AGGREGATOR" for a in agents):
                break
        except Exception as e:
            logger.warning(f"Pool creation attempt {attempts + 1} failed: {e}")
        attempts += 1

    if pool is None:
        raise RuntimeError("Failed to create pool after 3 attempts")

    # Build pipeline
    graph = get_parallel_graph(pool)
    builder = PipelineBuilder()
    pipeline = builder.create_from_pool(pool, graph).build()

    # Execute with Langfuse tracing (if available)
    span = None
    if judge_client:
        span = judge_client.start_as_current_span(
            name=f"evaluate_webarena_{trace_id}",
            input={"task_id": trace_id},
            metadata={"task_id": trace_id, "benchmark": "webarena"},
        )
        span.__enter__()
        judge_client.update_current_trace(tags=["webarena", f"task_id:{trace_id}"])

    result, trace_id_lf = await ainvoke_with_lf(pool, pipeline, judge_input, graph)

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

    if span:
        span.update(output={"result": result_dict})
        span.__exit__(None, None, None)

    # Parse judge predictions
    judge_predictions = {}
    success_val = result_dict.get("success", "").lower().strip() if isinstance(result_dict.get("success"), str) else None
    side_val = result_dict.get("side", "").lower().strip() if isinstance(result_dict.get("side"), str) else None
    loop_val = result_dict.get("loop", "").lower().strip() if isinstance(result_dict.get("loop"), str) else None
    optimal_val = result_dict.get("optimal", "").lower().strip() if isinstance(result_dict.get("optimal"), str) else None

    judge_predictions["trajectory_success"] = success_val == "yes" if success_val else None
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
                done_traces.add(res.split(".")[0])

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
            logger.error(f"Error processing task {trace_id}: {error_msg}", exc_info=True)
            failed_traces.append({"task_id": trace_id, "error": error_msg})

    if failed_traces:
        with open(failed_file, "w") as fh:
            fh.write(f"Failed traces: {len(failed_traces)}\n{'=' * 80}\n\n")
            for ft in failed_traces:
                fh.write(f"Task ID: {ft['task_id']}\nError Message: {ft['error']}\n{'-' * 80}\n\n")

    logger.info(f"Completed: {len(trace_files) - len(failed_traces)}/{len(trace_files)} successful")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WebArena with AutoJudge")
    parser.add_argument("--data-dir", default="data/test", help="Path to base data directory containing cleaned/ and pruned/ folders")
    parser.add_argument("--save-folder", required=True, help="Output folder name")
    parser.add_argument("--max-traces", type=int, default=None, help="Max traces to evaluate")
    parser.add_argument("--test", action="store_true", help="Test mode: run on 5 traces")

    args = parser.parse_args()
    max_traces = 5 if args.test else args.max_traces
    asyncio.run(main(data_dir=args.data_dir, save_folder=args.save_folder, max_traces=max_traces))
