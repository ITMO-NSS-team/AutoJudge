import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

from autojudge.meta_agents import PoolGenerator
from autojudge.pipeline import PipelineBuilder
from autojudge.agent_pool import AgentPool
from autojudge.pipeline.types import GraphDict
from autojudge.utils.langfuse_utils import ainvoke_with_lf
from autojudge.utils import get_logger
from autojudge.meta_agents.prompts import examples_no_tools as examples
from autojudge.meta_agents.prompts import ae_output_schema, ae_taxonomy
from maseval import get_langfuse_judge_client
from autojudge.meta_agents.graph_gen import get_parallel_graph

import json
import pandas as pd

logger = get_logger(__name__)


async def main(
    save_folder: str, df, df_full, table_name: str, num_traces: int | None = None, tag: str = "ae_afworld_full_trace"
):
    logger.info("===Starting evaluation===")

    pool_gen = PoolGenerator(
        output_schema=ae_output_schema, taxonomy=ae_taxonomy, examples=examples, use_summary=False
    )
    judge_client = get_langfuse_judge_client()
    logger.info("Initialized generators and Langfuse client")

    # continue processing that was already started
    done_traces = []
    local_results_dir = Path(__file__).resolve().parent / "results" / save_folder
    results_dir = local_results_dir

    if results_dir.exists():
        for res in os.listdir(results_dir):
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
        df = df[:num_traces]

    for idx in range(len(df)):
        id = df.iloc[idx]["trajectory_id"]
        if id in done_traces:
            trajectory_id = id
            logger.info(f"Task {trajectory_id} already processed, skipping...")
            continue

        if id in failed_traces_ids:
            logger.info(f"Task {id} already failed, skipping...")
            continue

        task = id
        logger.info(f"Processing task {idx + 1}/{len(df)}: {task}")
        serializable_results = {}

        try:
            trace_data = {
                "step_annotations": df.iloc[idx]["step_annotations"],
                "critical_failure_step": df.iloc[idx]["critical_failure_step"],
                "critical_failure_module": df.iloc[idx]["critical_failure_module"],
                "task_id": id,
                "trace_id": id,
            }
            q = df_full[df_full["question_ID"] == id]["messages"].values[0]
            logger.debug(f"Parsed task query: {q}...")

            trace_metadata = {
                "task_id": trace_data["task_id"],
                "trace_id": trace_data["trace_id"],
            }


            judge_input = {
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
                    f"Error processing task {id}: {safe_error_msg}",
                    exc_info=True,
                )

                failed_traces.append(
                    {
                        "task_id": id,
                        "task_index": idx + 1,
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    }
                )

                print(f"\n!  Failed task {idx + 1}/{len(df)}: {id}")
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
                name=f"evaluate_task_{task}",
                input={
                    "task_id": id,
                    "trace_id": id,
                },
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(tags=[tag, f"task_id:{id}"])

                logger.info("Executing evaluation pipeline...")
                result, trace_id = await ainvoke_with_lf(
                    pool, pipeline, judge_input, graph
                )
                logger.info(f"Pipeline execution completed. Trace ID: {trace_id}")

                span.update(output={"result": result, "trace_id": trace_id})
                span.end()

            result_dict = json.loads(
                result.replace("```json", "").replace("```", "").strip()
            )

            serializable_results["summarizer_score"] = {
                "metric_name": "summarizer_score",
                "scores": [
                    {
                        "item_id": "overall_score",
                        "score": result_dict,
                        "idx": str(idx),
                        "task_id": str(id),
                        "step_annotations": df.iloc[idx]["step_annotations"],
                        "critical_failure_step": str(df.iloc[idx]["critical_failure_step"]),
                        "critical_failure_module": df.iloc[idx]["critical_failure_module"],
                    }
                ],
            }

            output_dir = local_results_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / Path(f"{df.iloc[idx]['trajectory_id']}.json")

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}\n\n")

            print(f"Result: {result}")
            print(f"Langfuse trace ID: {trace_id}")

        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            task_id = id
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

            print(f"\n!  Failed task {idx + 1}/{len(df)}: {task_id}")
            print(f"   Error: {error_msg}\n")
            continue

    output_dir = local_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_file = output_dir / "failed_traces.txt"

    if failed_traces:
        first_run = not failed_file.exists()
        with open(failed_file, "a") as f:
            if first_run:
                f.write(f"Failed traces: {len(failed_traces)} out of {len(df)}\n")
                f.write("=" * 80 + "\n\n")

            for failed in failed_traces:
                f.write(f"Task ID: {failed['task_id']}\n")
                f.write(f"Index: {failed['task_index']}/{len(df)}\n")
                f.write(f"Error Type: {failed['error_type']}\n")
                f.write(f"Error Message: {failed['error']}\n")
                f.write("-" * 80 + "\n\n")

    if len(failed_traces) > 0:
        logger.warning(
            f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n"
        )

    if failed_traces_ids:
        logger.info(
            f"Completed evaluation: {len(df) - (len(failed_traces) + len(failed_traces_ids))}/{len(df)} successful, {len(failed_traces) + len(failed_traces_ids)} failed"
        )
    else:
        logger.info(
            f"Completed evaluation: {len(df) - (len(failed_traces))}/{len(df)} successful, {len(failed_traces)} failed"
        )


if __name__ == "__main__":
    df_labels = pd.read_json("/home/alina/Desktop/AutoJudge/examples/agent_error_bench/AgentErrorBench/Label/gaia_labels.json")
    raw_data_directory = Path("/home/alina/Desktop/AutoJudge/examples/agent_error_bench/AgentErrorBench/Original_Failure_Trajectory/GAIA")
    summary = []
    for file_path in raw_data_directory.iterdir():
        if file_path.is_file() and file_path.suffix == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
                summary.append([data["messages"], file_path.stem]) 
    messages = pd.DataFrame(summary, columns=["messages", "question_ID"])

    asyncio.run(
        main(
            save_folder="test_gaia_full_trace",
            df=df_labels,
            df_full=messages,
            table_name="agent_error",
            tag="FIXED_gaia_full_trace"
        )
    )
