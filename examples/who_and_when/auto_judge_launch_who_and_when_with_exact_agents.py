"""Launches auto-judge on who_and_when dataset with a pool that has 3 fixed agents: GUILTY_AGENT_FINDER and STEP_OF_ERROR_FINDER."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

from autojudge.meta_agents import PoolGenerator_WW
from autojudge.pipeline import PipelineBuilder
from autojudge.meta_agents import GraphGenerator
from autojudge.utils.langfuse_utils import ainvoke_with_lf
from autojudge.utils import get_logger
from maseval import get_langfuse_judge_client
from autojudge.meta_agents.prompts import examples_no_tools as examples
from autojudge.meta_agents.prompts import ww_output_schema, ww_taxonomy
import json
import pandas as pd
from toon_format import encode

logger = get_logger(__name__)


async def main(
    save_folder: str, df, df_summary, table_name: str, num_traces: int | None = None
):
    logger.info("===Starting Who&When evaluation===")

    pool_gen = PoolGenerator_WW(
        output_schema=ww_output_schema, taxonomy=ww_taxonomy, examples=examples
    )
    graph_gen = GraphGenerator()

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
        id = df.iloc[idx]["question_ID"]
        if id in done_traces:
            question_id = id
            logger.info(f"Task {question_id} already processed, skipping...")
            continue

        if id in failed_traces_ids:
            logger.info(f"Task {id} already failed, skipping...")
            continue

        task = id
        logger.info(f"Processing task {idx + 1}/{len(df)}: {task}")
        serializable_results = {}

        try:
            trace_data = {
                "history": df.iloc[idx]["history"],
                "question": df.iloc[idx]["question"],
                "task_id": id,
                "trace_id": id,
            }
            q = df_summary[df_summary["question_ID"] == id]["summary"].values[0]
            logger.debug(f"Parsed task query: {q}...")

            trace_metadata = {
                "task_id": trace_data["task_id"],
                "trace_id": trace_data["trace_id"],
            }

            if "groundtruth" in df.iloc[idx].keys():
                trace_metadata["ground_truth"] = df.iloc[idx]["groundtruth"]
                trace_metadata["correct_answer"] = df.iloc[idx]["is_corrected"]
            else:
                trace_metadata["ground_truth"] = df.iloc[idx]["ground_truth"]
                trace_metadata["correct_answer"] = df.iloc[idx]["is_correct"]

            judge_input = {
                "query": encode(trace_data["question"]),
                "history_for_evaluating": str(q),
                "table_name": table_name,
            }

            logger.info("Generating judge pool...")

            try:
                attempts = 0
                while attempts < 3:
                    missing_agents = []
                    pool = await pool_gen.create_pool(judge_input)
                    agents_info = pool.full_agents_data
                    final_agent = any(
                        agent.get("name") == "FINAL_AGGREGATOR" for agent in agents_info
                    )
                    guilty_agent_finder = any(
                        agent.get("name") == "GUILTY_AGENT_FINDER"
                        for agent in agents_info
                    )
                    step_of_error_finder = any(
                        agent.get("name") == "STEP_OF_ERROR_FINDER"
                        for agent in agents_info
                    )

                    if final_agent and guilty_agent_finder and step_of_error_finder:
                        logger.info(f"Generated correct pool on {attempts + 1} attempt")
                        break
                    else:
                        missing_agents.append(
                            "FINAL_AGGREGATOR" if not final_agent else ""
                        )
                        missing_agents.append(
                            "GUILTY_AGENT_FINDER" if not guilty_agent_finder else ""
                        )
                        missing_agents.append(
                            "STEP_OF_ERROR_FINDER" if not step_of_error_finder else ""
                        )

                    attempts += 1

            except Exception as e:
                logger.error(
                    f"Error processing task {id}: There are missing {missing_agents} agents in the pool",
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
            graph = await graph_gen.create_graph(pool, str(judge_input))
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
                judge_client.update_current_trace(tags=["test", f"task_id:{id}"])

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
                        "idx": idx,
                        "task_id": id,
                        "ground_truth": trace_metadata["ground_truth"],
                        "correct_answer": str(trace_metadata["correct_answer"]),
                        "gt_agent": df.iloc[idx]["mistake_agent"],
                        "gt_step": df.iloc[idx]["mistake_step"],
                        "gt_mistake_reason": df.iloc[idx]["mistake_reason"],
                    }
                ],
            }

            output_dir = local_results_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / Path(f"{df.iloc[idx]['question_ID']}.json")

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

    if failed_traces_ids:
        with open(failed_file, "w") as f:
            f.write(f"Failed traces: {len(failed_traces)} out of {len(df)}\n")
            f.write("=" * 80 + "\n\n")

            for failed in failed_traces:
                f.write(f"Task ID: {failed['task_id']}\n")
                f.write(f"Index: {failed['task_index']}/{len(df)}\n")
                f.write(f"Error Type: {failed['error_type']}\n")
                f.write(f"Error Message: {failed['error']}\n")
                f.write("-" * 80 + "\n\n")
    else:
        if failed_traces:
            with open(failed_file, "w") as f:
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
    # handcrafted dataset
    df_handcrafted = pd.read_parquet(
        "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet"
    )
    # or llm-generated dataset
    # df_algorithm = pd.read_parquet(
    #     "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet"
    # )

    summaries_directory = Path("path to who_and_when summaries")

    summary = []
    for dir in summaries_directory.iterdir():
        if dir.is_file() and dir.suffix == ".json":
            with open(dir, "r") as f:
                data = json.load(f)
                summary.append([data, dir.name.split(".")[0]])
    df_summary = pd.DataFrame(summary, columns=["summary", "question_ID"])

    asyncio.run(
        main(
            save_folder="test",
            df=df_handcrafted[:],
            df_summary=df_summary,
            table_name="who_when",
            # num_traces=30,
        )
    )
