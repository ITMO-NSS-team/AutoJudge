import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from maseval import get_langfuse_judge_client

from autojudge.agent_pool import AgentPool
from autojudge.meta_agents import PoolGenerator
from autojudge.meta_agents.graph_gen import get_parallel_graph
from autojudge.meta_agents.prompts import examples_tools as examples
from autojudge.meta_agents.prompts import trail_output_schema, trail_taxonomy
from autojudge.pipeline import PipelineBuilder
from autojudge.pipeline.types import GraphDict
from autojudge.utils import get_logger
from autojudge.utils.langfuse_utils import ainvoke_with_lf

logger = get_logger(__name__)

async def main(save_folder: str, df):
    logger.info("===Starting TRAIL evaluation===")

    pool_gen = PoolGenerator(
        output_schema=trail_output_schema, taxonomy=trail_taxonomy, examples=examples
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

    for idx in range(len(df)):
        if df.iloc[idx]["trace_id"] in done_traces:
            trace_id = df.iloc[idx]["trace_id"]
            logger.info(f"Task {trace_id} already processed, skipping...")
            continue

        if df.iloc[idx]["trace_id"] in failed_traces_ids:
            logger.info(
                f"Task {df.iloc[idx]["trace_id"]} already failed, skipping..."
            )
            continue

        task = df.iloc[idx]["trace_id"]
        logger.info(f"Processing task {idx + 1}/{len(df)}: {task}")
        serializable_results = {}

        try:
            trace_data = {
                "history": df.iloc[idx]["history"],
                "question": "You should evaluate the trace.",
                "task_id": df.iloc[idx]["task_id"],
                "trace_id": df.iloc[idx]["trace_id"],
            }
            q = trace_data["question"][:100]

            logger.debug(f"Parsed task query: {q}...")

            trace_metadata = {
                "task_id": trace_data["task_id"],
                "trace_id": trace_data["trace_id"],
            }

            judge_input = str(
                {
                    "query": trace_data["question"],
                    "history_for_evaluating": trace_data["history"],
                }
            )

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
                    f"Error processing task {df.iloc[idx]["trace_id"]}: {safe_error_msg}",
                    exc_info=True,
                )

                failed_traces.append(
                    {
                        "task_id": df.iloc[idx]["trace_id"],
                        "task_index": idx + 1,
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    }
                )

                print(
                    f"\n!  Failed task {idx + 1}/{len(df)}: {df.iloc[idx]["trace_id"]}"
                )
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
                    "task_id": df.iloc[idx]["trace_id"],
                    "trace_id": df.iloc[idx]["trace_id"],
                },
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(tags=["test", f"task_id:{task}"])

                logger.info("Executing evaluation pipeline...")
                result, trace_id = await ainvoke_with_lf(
                    pool, pipeline, judge_input, graph
                )
                logger.info(f"Pipeline execution completed. Trace ID: {trace_id}")

                span.update(output={"result": result, "trace_id": trace_id})
                span.end()

            if result.startswith("```json"):
                result = result.strip("```json").strip("```")
            else:
                result = result.strip()

            result_dict = json.loads(result)

            serializable_results["summarizer_score"] = {
                "metric_name": "summarizer_score",
                "scores": [
                    {
                        "item_id": "overall_score",
                        "score": result_dict,
                        "idx": idx,
                        "task_id": df.iloc[idx]["trace_id"],
                        "filename": df.iloc[idx]["filename"],
                    }
                ],
            }

            output_dir = local_results_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / Path(f"{df.iloc[idx]['trace_id']}.json")

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}\n\n")

            print(f"Result: {result}")
            print(f"Langfuse trace ID: {trace_id}")

        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            task_id = df.iloc[idx]["trace_id"]
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


def create_gaia_dataframe():
    gaia_dir = Path("path/to/your/directory")

    json_files = list(gaia_dir.glob("*.json"))

    if not json_files:
        return None

    files_to_process = json_files

    data_list = []

    for i, file_path in enumerate(files_to_process, 1):
        try:
            print(f"[{i:2d}/{len(files_to_process)}] {file_path.name}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["filename"] = file_path.name
            data["file_path"] = str(file_path)

            data_list.append(data)

        except Exception as e:
            print(f"Error during processing of {file_path.name}: {e}")
            continue

    df = pd.json_normalize(data_list)
    print(df.head(3))
    print(df.info())
    return df


if __name__ == "__main__":
    df = create_gaia_dataframe()
    asyncio.run(main(save_folder="test", df=df))
