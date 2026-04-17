import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

from autojudge.meta_agents import PoolGenerator
from autojudge.agent_pool import AgentPool
from autojudge.pipeline.types import GraphDict
from autojudge.pipeline import PipelineBuilder
from autojudge.utils.langfuse_utils import ainvoke_with_lf
from autojudge.utils import get_logger
from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task
from autojudge.meta_agents.prompts import examples_tools as examples
from autojudge.meta_agents.prompts import pumpkin_output_schema, pumpkin_taxonomy
from dotenv import load_dotenv
import json
import os
import pandas as pd

load_dotenv(".env")
logger = get_logger(__name__)


def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    graph_dict = {}
    _agents_info = agent_pool.full_agents_data

    for agent in _agents_info:
        if agent["name"] != "FINAL_AGGREGATOR":
            graph_dict[agent["name"]] = ["FINAL_AGGREGATOR"]

    graph_dict["FINAL_AGGREGATOR"] = []

    return graph_dict


async def main(
    name: str,
    save_folder: str,
    df_summary,
    table_name: str,
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
            logger.debug(f"Parsed task query: {query.user_query[:100]}...")

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

            q = df_summary[df_summary["task_id"] == task_id]["summary"].values[0]
            if q is None:
                logger.error(f"Task {task_id} not found in summary dataframe")
                continue

            logger.debug(f"Parsed task query: {q}...")

            # agent_states = [
            #     state.model_dump(mode="json") for state in query.agent_states
            # ]

            judge_input = {
                "query": query.user_query,
                "history_for_evaluating": str(q),
                "table_name": table_name,
            }

            logger.info("Generating judge pool...")

            try:
                attempts = 0
                while attempts < 3:
                    pool = await pool_gen.create_pool(judge_input)
                    agents_info = pool.full_agents_data
                    final_agent = any(
                        agent.get("name") == "FINAL_AGGREGATOR" for agent in agents_info
                    )
                    if final_agent:
                        break
                    attempts += 1

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
                    tags=[
                        "test",
                        f"task_id:{task_id}",
                    ]
                )

                logger.info("Executing evaluation pipeline...")
                result, trace_id = await ainvoke_with_lf(
                    pool, pipeline, judge_input, graph
                )
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
    summaries_directory = Path("path to our_mas (GHOST) summaries")

    summary = []
    for dir in summaries_directory.iterdir():
        if dir.is_file() and dir.suffix == ".json":
            with open(dir, "r") as f:
                data = json.load(f)
                summary.append([data, dir.name.split(".")[0]])
    df_summary = pd.DataFrame(summary, columns=["summary", "task_id"])

    asyncio.run(
        main(
            # name="gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb", # big mas
            name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097",  # small mas
            save_folder="test",
            df_summary=df_summary,
            table_name="our_mas",
            # num_traces=30,
        )
    )
