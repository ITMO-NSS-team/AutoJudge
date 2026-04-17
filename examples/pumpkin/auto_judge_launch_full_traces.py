import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

from autojudge.meta_agents import PoolGenerator
from autojudge.agent_pool import AgentPool
from autojudge.pipeline.types import GraphDict
from autojudge.pipeline import PipelineBuilder
from autojudge.pipeline.node_session import NodeExecution, NodeSessionError
from autojudge.utils import get_logger
from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task
from autojudge.meta_agents.prompts import examples_no_tools as examples
from autojudge.meta_agents.prompts import pumpkin_output_schema, pumpkin_taxonomy
from dotenv import load_dotenv
from autojudge.meta_agents.graph_gen import get_parallel_graph

import json
import os

load_dotenv(".env")
logger = get_logger(__name__)

POOL_GENERATION_ATTEMPTS = 3
FINAL_AGENT_ONLY_ATTEMPTS = 8
MISSING_DEPENDENCY_RECOVERY_ATTEMPTS = 3
DEPENDENCY_NODE_ATTEMPTS = 5
TRANSIENT_ERROR_RETRY_DELAY_SECONDS = 1.5


def build_history_for_evaluating(query) -> str:
    if query.agent_states:
        return json.dumps(
            [state.model_dump(mode="json") for state in query.agent_states],
            ensure_ascii=False,
        )

    if query.dialogue_history:
        return json.dumps(
            [message.model_dump(mode="json") for message in query.dialogue_history],
            ensure_ascii=False,
        )

    if query.agent_responses:
        return json.dumps(
            [response.model_dump(mode="json") for response in query.agent_responses],
            ensure_ascii=False,
        )

    return "[]"


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

    attempts_block = "\n".join(f"- {item}" for item in attempt_failures)
    raise RuntimeError(
        "FINAL_AGGREGATOR failed after "
        f"{max_attempts} attempts. "
        f"final_node_id={final_node.id}; "
        f"parent_count={len(parent_names)}; "
        f"parents={parent_names}; "
        f"parent_status={parent_status}; "
        f"input_preview={input_preview}. "
        f"Attempt failures:\n{attempts_block}"
    ) from last_error


async def main(
    name: str,
    save_folder: str,
    num_traces: int | None = None,
):
    logger.info(f"Starting autojudge evaluation for task name: {name}")

    pool_gen = PoolGenerator(
        output_schema=pumpkin_output_schema, taxonomy=pumpkin_taxonomy, examples=examples
    )
    lf = get_langfuse_download_client()
    judge_client = get_langfuse_judge_client()
    logger.info("Initialized generators and Langfuse clients")

    traces_page1 = lf.api.trace.list(name=name, limit=50, page=1)
    traces_page2 = lf.api.trace.list(name=name, limit=50, page=2)
    traces_page3 = lf.api.trace.list(name=name, limit=50, page=3)
    traces_page4 = lf.api.trace.list(name=name, limit=50, page=4)

    all_traces = (
        traces_page1.data + traces_page2.data + traces_page3.data + traces_page4.data
    )
    task_ids = [item.id for item in all_traces]

    if not task_ids:
        raise ValueError(f"No tasks found in trace {name}")

    print(f"Found {len(task_ids)} tasks in trace {name}")

    # continue processing that was already started
    done_traces = []
    local_results_dir = Path(__file__).resolve().parent / "results" / save_folder
    results_dir = local_results_dir

    if results_dir.exists():
        for res in os.listdir(results_dir):
            if res.endswith(".json"):
                res_cropped = res.split(".")[0]
                done_traces.append(res_cropped)

    # skip failed traces
    failed_traces_ids = []
    failed_traces = []

    if local_results_dir.exists():
        if "failed_traces.txt" in os.listdir(local_results_dir):
            with open(local_results_dir / "failed_traces.txt", "r") as f:
                for line in f:
                    if line.startswith("Task ID:"):
                        failed_traces_ids.append(line.split(":")[1].strip())

    if num_traces is not None:
        task_ids = task_ids[:num_traces]

    for idx, task_id in enumerate(task_ids):
        if task_id in done_traces:
            logger.info(f"Task {task_id} already processed, skipping...")
            continue

        if task_id in failed_traces_ids:
            logger.info(f"Task {task_id} already failed, skipping...")
            continue

        logger.info(f"Processing task {idx + 1}/{len(task_ids)}: {task_id}")
        serializable_results = {}

        try:
            trace_data = lf.api.trace.get(task_id)
            query = parse_langfuse_task(trace_data)
            parsed_query_preview = (query.user_query or "")[:100]
            logger.debug(f"Parsed task query: {parsed_query_preview}...")

            trace_metadata = {"task_id": task_id}
            if hasattr(trace_data, "output") and trace_data.output:
                if "ground_truth" in trace_data.output:
                    trace_metadata["ground_truth"] = trace_data.output["ground_truth"]
                if "response" in trace_data.output:
                    trace_metadata["mas_response"] = trace_data.output["response"]
                if (
                    "ground_truth" in trace_data.output
                    and "response" in trace_data.output
                ):
                    trace_metadata["correct_answer"] = (
                        trace_data.output["response"]
                        == trace_data.output["ground_truth"]
                    )
                    logger.debug(f"Correct answer: {trace_metadata['correct_answer']}")

            history_for_evaluating = build_history_for_evaluating(query)
            logger.debug(
                "Prepared trace history for evaluating: %s chars",
                len(history_for_evaluating),
            )

            judge_input = {
                "query": query.user_query,
                "history_for_evaluating": history_for_evaluating,
            }

            logger.info("Generating judge pool...")

            try:
                pool = await create_pool_with_retries(pool_gen, judge_input)

            except Exception as e:
                error_msg = str(e)
                safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
                logger.error(
                    f"Error processing task {task_id}: {safe_error_msg}", exc_info=True
                )

                failed_traces.append(
                    {
                        "task_id": task_id,
                        "task_index": idx + 1,
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    }
                )

                print(f"\n!  Failed task {idx + 1}/{len(task_ids)}: {task_id}")
                print(f"   Error: {error_msg}\n")
                continue

            logger.info(f"Created pool with {len(pool)} judges")

            logger.info("Generating evaluation graph...")
            graph = get_parallel_graph(pool)
            logger.debug(f"Graph structure: {graph}")

            builder = PipelineBuilder()
            pipeline = builder.create_from_pool(pool, graph).build()
            pipeline.to_mermaid_lr(visualize=True)
            logger.info(f"Built pipeline with {len(pipeline.execution_order)} nodes")

            with judge_client.start_as_current_span(
                name=f"evaluate_task_{task_id}",
                input={"task_id": task_id, "trace_id": task_id},
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(
                    tags = [
                        "big_mas",
                        "full",
                        "gaia_without_summary",
                        f"task_id:{task_id}",
                        f"folder:{save_folder}",
                    ]
                )

                logger.info("Executing evaluation pipeline...")
                try:
                    result = await pipeline.ainvoke(judge_input)
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

                        result = await run_final_agent_only_with_retries(
                            pipeline,
                            max_attempts=FINAL_AGENT_ONLY_ATTEMPTS,
                        )
                    else:
                        raise

                trace_id = None
                logger.info(f"Pipeline execution completed. Trace ID: {trace_id}")

                span.update(output={"result": result, "trace_id": trace_id})
                span.end()

            result_clean = result.strip()
            if result_clean.startswith("```"):
                lines = result_clean.splitlines()
                # drop first line (```json or ```) and last line (```)
                result_clean = "\n".join(lines[1:-1]).strip()

            result_dict = json.loads(result_clean)

            serializable_results["summarizer_score"] = {
                "metric_name": "summarizer_score",
                "scores": [
                    {
                        "item_id": "overall_score",
                        "score": result_dict["score"],
                        "justification": result_dict["justification"],
                    }
                ],
            }

            output_dir = local_results_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / Path(f"{task_id}.json")

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}\n\n")

            print(f"Result: {result}")
            print(f"Langfuse trace ID: {trace_id}")
            print(f"Launch № {idx + 1} from {len(task_ids)}")

        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            logger.error(
                f"Error processing task {task_id}: {safe_error_msg}", exc_info=True
            )

            failed_traces.append(
                {
                    "task_id": task_id,
                    "task_index": idx + 1,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                }
            )

            print(f"\n!  Failed task {idx + 1}/{len(task_ids)}: {task_id}")
            print(f"   Error: {error_msg}\n")
            continue

    output_dir = local_results_dir
    output_dir.mkdir(exist_ok=True)
    failed_file = output_dir / "failed_traces.txt"

    if failed_traces:
        first_run = not failed_file.exists()
        with open(failed_file, "a") as f:
            if first_run:
                f.write(f"Failed traces: {len(failed_traces)} out of {len(task_ids)}\n")
                f.write("=" * 80 + "\n\n")

            for failed in failed_traces:
                f.write(f"Task ID: {failed['task_id']}\n")
                f.write(f"Index: {failed['task_index']}/{len(task_ids)}\n")
                f.write(f"Error Type: {failed['error_type']}\n")
                f.write(f"Error Message: {failed['error']}\n")
                f.write("-" * 80 + "\n\n")

    if len(failed_traces) > 0:
        logger.warning(
            f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n"
        )

    if failed_traces_ids:
        logger.info(
            f"Completed evaluation: {len(task_ids) - (len(failed_traces) + len(failed_traces_ids))}/{len(task_ids)} successful, {len(failed_traces) + len(failed_traces_ids)} failed"
        )
    else:
        logger.info(
            f"Completed evaluation: {len(task_ids) - len(failed_traces)}/{len(task_ids)} successful, {len(failed_traces)} failed"
        )


if __name__ == "__main__":
    asyncio.run(
        main(
            name="gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb", # big mas
            # name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097",  # small mas
            save_folder="big_mas_no_sum",
            num_traces=165,
        )
    )
