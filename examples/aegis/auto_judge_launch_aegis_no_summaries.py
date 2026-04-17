import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(".env")

from autojudge.meta_agents import PoolGenerator
from autojudge.meta_agents.prompts import DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool
from autojudge.agent_pool import AgentPool
from autojudge.pipeline.types import GraphDict
from autojudge.pipeline import PipelineBuilder
from autojudge.pipeline.node_session import NodeExecution, NodeSessionError
from autojudge.utils import get_logger
from autojudge.utils.langfuse_utils import setup_langfuse_instrumentation
from maseval import get_langfuse_judge_client
from autojudge.db.db_tools import get_content_tool
from autojudge.meta_agents.prompts import examples_no_tools as examples
from autojudge.meta_agents.prompts import aegis_output_schema, aegis_taxonomy
from datasets import load_dataset

logger = get_logger(__name__)

POOL_GENERATION_ATTEMPTS = 3
FINAL_AGENT_ONLY_ATTEMPTS = 8
MISSING_DEPENDENCY_RECOVERY_ATTEMPTS = 3
DEPENDENCY_NODE_ATTEMPTS = 5
TRANSIENT_ERROR_RETRY_DELAY_SECONDS = 1.5

def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    graph_dict: GraphDict = {}
    _agents_info = agent_pool.full_agents_data

    for agent in _agents_info:
        if agent.get("name") != "FINAL_AGGREGATOR":
            graph_dict[agent.get("name")] = ["FINAL_AGGREGATOR"]

    graph_dict["FINAL_AGGREGATOR"] = []

    return graph_dict


def is_transient_chat_completion_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "Invalid response from openrouter chat completions endpoint" in message
        and "validation errors for ChatCompletion" in message
        and "input_value=None" in message
    )


def has_final_aggregator(pool: AgentPool) -> bool:
    return any(agent.get("name") == "FINAL_AGGREGATOR" for agent in pool.full_agents_data)


async def create_pool_with_retries(pool_gen: PoolGenerator, judge_input: dict) -> AgentPool:
    context_feedback = None
    attempt_errors: list[str] = []

    for attempt in range(1, POOL_GENERATION_ATTEMPTS + 1):
        try:
            logger.info("Pool generation attempt %s/%s", attempt, POOL_GENERATION_ATTEMPTS)
            pool = await pool_gen.create_pool(judge_input, context=context_feedback)

            if has_final_aggregator(pool):
                return pool

            context_feedback = (
                "Previous attempt was invalid: include FINAL_AGGREGATOR exactly named "
                "'FINAL_AGGREGATOR' and return a non-empty list of valid agents."
            )
            attempt_errors.append(
                f"attempt {attempt}/{POOL_GENERATION_ATTEMPTS}: missing FINAL_AGGREGATOR"
            )

        except Exception as exc:
            if "No valid agents generated" in str(exc):
                context_feedback = (
                    "Previous attempt returned no valid agents. Return a non-empty list of "
                    "valid agent schemas and include FINAL_AGGREGATOR exactly once."
                )

            attempt_errors.append(
                f"attempt {attempt}/{POOL_GENERATION_ATTEMPTS}: {type(exc).__name__}: {str(exc)[:300]}"
            )

    error_block = "\n".join(f"- {err}" for err in attempt_errors)
    raise RuntimeError(
        "Pool generation failed after retries. Attempts:\n"
        f"{error_block}"
    )


async def execute_node_with_retries(
    pipeline,
    node,
    max_attempts: int,
    node_input: str,
) -> str:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            agent = node.build_agent()
            result = await agent.run(node_input)
            node._usage = result.usage()
            pipeline.node_session.add_node_execution(
                NodeExecution(
                    node_id=node.id,
                    node_name=node.name,
                    output=result.output,
                )
            )
            return result.output
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Node %s attempt %s/%s failed: %s",
                node.name,
                attempt,
                max_attempts,
                exc,
            )

            if is_transient_chat_completion_error(exc) and attempt < max_attempts:
                await asyncio.sleep(TRANSIENT_ERROR_RETRY_DELAY_SECONDS * attempt)
                continue

            if attempt < max_attempts:
                await asyncio.sleep(0.5)

    raise RuntimeError(
        f"Node {node.name} failed after {max_attempts} attempts: {type(last_error).__name__}: {last_error}"
    ) from last_error


async def recover_missing_dependencies_for_final(
    pipeline,
    max_rounds: int = MISSING_DEPENDENCY_RECOVERY_ATTEMPTS,
) -> None:
    final_node = pipeline.execution_order[-1]

    for round_index in range(1, max_rounds + 1):
        missing_parents = [
            parent for parent in final_node.parents
            if parent.id not in pipeline.node_session.node_executions
        ]

        if not missing_parents:
            return

        logger.warning(
            "Dependency recovery round %s/%s for FINAL_AGGREGATOR. Missing parents: %s",
            round_index,
            max_rounds,
            [parent.name for parent in missing_parents],
        )

        progressed = False
        for parent_node in missing_parents:
            try:
                pipeline.node_session.validate_dependencies(parent_node)
                parent_input = pipeline.node_session.get_input_for_node(parent_node)
                await execute_node_with_retries(
                    pipeline,
                    parent_node,
                    max_attempts=DEPENDENCY_NODE_ATTEMPTS,
                    node_input=parent_input,
                )
                progressed = True
            except NodeSessionError as dep_exc:
                logger.warning(
                    "Cannot execute missing parent %s yet due to unmet dependencies: %s",
                    parent_node.name,
                    dep_exc,
                )
            except Exception as parent_exc:
                logger.warning(
                    "Failed to recover missing parent %s: %s",
                    parent_node.name,
                    parent_exc,
                )

        if not progressed:
            break

    unresolved = [
        parent.name for parent in final_node.parents
        if parent.id not in pipeline.node_session.node_executions
    ]
    if unresolved:
        raise RuntimeError(
            "Could not recover FINAL_AGGREGATOR dependencies. "
            f"Still missing: {unresolved}"
        )


async def run_final_agent_only_with_retries(
    pipeline,
    max_attempts: int = FINAL_AGENT_ONLY_ATTEMPTS,
):
    final_node = pipeline.execution_order[-1]
    if final_node.name != "FINAL_AGGREGATOR":
        raise RuntimeError(
            f"Final node is '{final_node.name}', expected 'FINAL_AGGREGATOR'"
        )

    pipeline.node_session.validate_dependencies(final_node)
    final_input = pipeline.node_session.get_input_for_node(final_node)
    parent_names = [parent.name for parent in final_node.parents]
    parent_status = [
        f"{parent.name}:{'ready' if parent.id in pipeline.node_session.node_executions else 'missing'}"
        for parent in final_node.parents
    ]
    input_preview = " ".join(final_input.split())[:300]

    last_error = None
    attempt_failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            logger.warning(
                "Running FINAL_AGGREGATOR only (attempt %s/%s)",
                attempt,
                max_attempts,
            )
            result_output = await execute_node_with_retries(
                pipeline,
                final_node,
                max_attempts=1,
                node_input=final_input,
            )

            logger.info(
                "FINAL_AGGREGATOR succeeded on attempt %s/%s",
                attempt,
                max_attempts,
            )
            return result_output

        except Exception as exc:
            last_error = exc
            failure_summary = (
                f"attempt {attempt}/{max_attempts}: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            attempt_failures.append(failure_summary)
            logger.warning(
                "FINAL_AGGREGATOR attempt %s/%s failed: %s",
                attempt,
                max_attempts,
                exc,
            )

            if is_transient_chat_completion_error(exc) and attempt < max_attempts:
                await asyncio.sleep(TRANSIENT_ERROR_RETRY_DELAY_SECONDS * attempt)


async def main(save_folder: str, split: str = "test", max_traces: int | None = None):
    logger.info("===Starting AEGIS evaluation WITHOUT SUMMARIES===")

    pool_gen = PoolGenerator(
        output_schema=aegis_output_schema,
        taxonomy=aegis_taxonomy,
        examples=examples,
        use_tools=False,
        prompt_template=DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool,
    )
    judge_client = get_langfuse_judge_client()
    setup_langfuse_instrumentation()
    logger.info("Initialized generators and Langfuse client")

    # No summaries loaded - using full raw traces instead

    ds = load_dataset("Fancylalala/AEGIS", split=split)
    logger.info(f"Loaded {len(ds)} traces from AEGIS {split} split")

    if max_traces is not None:
        ds = ds.select(range(min(max_traces, len(ds))))
        logger.info(f"Limited to {len(ds)} traces")

    local_results_dir = Path(__file__).resolve().parent / "results" / save_folder
    local_results_dir.mkdir(parents=True, exist_ok=True)

    done_traces_with_occurrence = set()  # Track (trace_id, occurrence) pairs
    if local_results_dir.exists():
        for res in os.listdir(local_results_dir):
            if res.endswith(".json"):
                filepath = local_results_dir / res
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("input_type") == "summary_context":
                            res_cropped = res.split(".")[0]
                            # Parse new format: <trace_id>_occ<N>.json
                            match = re.match(r"^(.+)_occ(\d+)$", res_cropped)
                            if match:
                                base_id = match.group(1)
                                occurrence = int(match.group(2))
                                done_traces_with_occurrence.add((base_id, occurrence))
                            else:
                                # Backward compatibility with legacy results saved as <trace_id>.json.
                                # These correspond to the first occurrence (occurrence 0).
                                base_id = data.get("task_id") or data.get("summary_file_used") or res_cropped
                                done_traces_with_occurrence.add((str(base_id), 0))
                except Exception:
                    pass

    failed_traces = []
    failed_file = local_results_dir / "failed_traces.txt"
    failed_traces_ids = []
    
    if failed_file.exists():
        with open(failed_file) as fh:
            for line in fh:
                if line.startswith("Task ID:"):
                    failed_traces_ids.append(line.split(":")[1].strip())

    # Track occurrence count per unique trace_id
    occurrence_counter = {}
    
    for idx, trace in enumerate(ds):
        trace_id = str(trace["id"]).replace("/", "_")  # Ensure valid filename
        
        # Count this occurrence of the trace_id
        if trace_id not in occurrence_counter:
            occurrence_counter[trace_id] = 0
        else:
            occurrence_counter[trace_id] += 1
        
        occurrence = occurrence_counter[trace_id]
        
        # Check if this specific occurrence was already processed
        if (trace_id, occurrence) in done_traces_with_occurrence:
            logger.info(f"Task {trace_id} occurrence {occurrence} already processed, skipping...")
            continue
        if trace_id in failed_traces_ids:
            logger.info(f"Task {trace_id} previously failed, skipping...")
            continue

        trace_input = trace.get("input", {})
        question = trace_input.get("query", "")

        # Use full raw conversation from trace input
        trace_input = trace.get("input", {})
        conversation_history = trace_input.get("conversation_history", [])
        if not conversation_history:
            logger.warning(f"No conversation_history found for {trace_id}, skipping trace")
            continue

        history_for_evaluating = json.dumps(conversation_history, indent=2)
        input_type = "raw_trace"

        logger.info(f"Processing task {idx + 1}/{len(ds)}: {trace_id} (using {input_type})")

        judge_input_dict = {
            "query": question,
            "trace_id": trace_id,
            "history_for_evaluating": history_for_evaluating,
        }
        judge_input = str(judge_input_dict)

        try:
            pool = await create_pool_with_retries(pool_gen, dict(query=question, history_for_evaluating=history_for_evaluating))
            for agent in pool.full_agents_data:
                agent["mcp_tools"] = [get_content_tool]

            graph = get_parallel_graph(pool)
            builder = PipelineBuilder()
            pipeline = builder.create_from_pool(pool, graph).build()
            pipeline.node_session.context_deps = trace_id
            
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{trace_id}",
                input={"task_id": trace_id, "input_type": input_type},
                metadata={
                    "task_id": trace_id,
                    "input_type": input_type
                },
            ) as span:
                judge_client.update_current_trace(tags=["aegis" ,"test" if max_traces else "full", f"task_id:{trace_id}", split, "raw_trace"])

                try:
                    raw_result = await pipeline.ainvoke(judge_input)
                except Exception as pipeline_error:
                    final_node = pipeline.execution_order[-1]
                    final_missing_output = (
                        final_node.id not in pipeline.node_session.node_executions
                    )

                    if final_node.name == "FINAL_AGGREGATOR" and final_missing_output:
                        logger.warning(
                            "Pipeline failed before producing FINAL_AGGREGATOR output. "
                            "Retrying FINAL_AGGREGATOR only without recreating pool. "
                            "Original error: %s",
                            pipeline_error,
                        )

                        await recover_missing_dependencies_for_final(pipeline)

                        raw_result = await run_final_agent_only_with_retries(
                            pipeline,
                            max_attempts=FINAL_AGENT_ONLY_ATTEMPTS,
                        )
                    else:
                        raise

                clean = raw_result.strip()
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
                    result_dict = {"raw": raw_result}

                span.update(output={"result": result_dict})
                span.end()

            # Save result with occurrence number to track duplicates
            result_filename = f"{trace_id}_occ{occurrence}.json"
            out_file = local_results_dir / result_filename
            output_data = {
                "task_id": trace_id,
                "occurrence": occurrence,
                "input_type": input_type,
                "dataset_split": split,
                "ground_truth": {
                    "faulty_agents": trace.get("ground_truth", {}).get("injected_agents", []),
                },
                "model_detection": {
                    "faulty_agents": result_dict.get("faulty_agents", []),
                },
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {out_file}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing task {trace_id}: {error_msg}", exc_info=True)
            failed_traces.append({"task_id": trace_id, "error": error_msg})

    if failed_traces:
        with open(failed_file, "w") as fh:
            fh.write(f"Failed traces: {len(failed_traces)}\n{'=' * 80}\n\n")
            for ft in failed_traces:
                fh.write(
                    f"Task ID: {ft['task_id']}\n"
                    f"Error Message: {ft['error']}\n"
                    f"{'-' * 80}\n\n"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS Evaluation over Raw Traces (No Summaries, No Tools).")
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run on 5 traces only.",
    )
    args = parser.parse_args()

    # folder for output
    folder = "aegis_eval_summaries_test" if args.test else "aegis_eval_summaries_fixed"
    
    max_traces = 5 if args.test else None

    asyncio.run(
        main(
            save_folder=folder,
            split=args.split,
            max_traces=max_traces,
        )
    )
