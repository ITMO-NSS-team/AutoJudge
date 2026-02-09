import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

from automas.meta_agents import GraphGenerator, PoolGenerator
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from automas.utils import get_logger
from maseval import get_langfuse_judge_client
from dotenv import load_dotenv
import json
import os
import pandas as pd

load_dotenv(".env")
logger = get_logger(__name__)

async def main(save_folder: str, df):
    logger.info(f"===Starting Who&When evaluation===")
    
    pool_gen = PoolGenerator()
    graph_gen = GraphGenerator()
    judge_client = get_langfuse_judge_client()
    logger.info("Initialized generators and Langfuse client")
    
    # continue processing that was already started
    # ======================
    done_traces = []
    local_results_dir = (Path(__file__).resolve().parent / "results" / save_folder)
    results_dir = local_results_dir

    if results_dir.exists():
        for res in os.listdir(results_dir):
            res_cropped = res.split(".")[0]
            done_traces.append(res_cropped)
    # ======================

    failed_traces = [] 
    
    for idx in range(len(df)):
        if df.iloc[idx]["question_ID"] in done_traces:
            logger.info(f"Task {df.iloc[idx]["question_ID"]} already processed, skipping...")
            continue
        
        logger.info(f"Processing task {idx + 1}/{len(df)}: {df.iloc[idx]["question_ID"]}")
        serializable_results = {}
        
        try:
            trace_data = {
            "history": df.iloc[idx]["history"],
            "question": df.iloc[idx]["question"],
            "task_id": df.iloc[idx]["question_ID"],
            "trace_id": df.iloc[idx]["question_ID"],
            }

            logger.debug(f"Parsed task query: {trace_data["question"][:100]}...")
            
            trace_metadata = {
                "task_id": trace_data["task_id"],
                "trace_id": trace_data["trace_id"]
                }

            if 'groundtruth' in df.iloc[idx].keys():
                trace_metadata["ground_truth"] = df.iloc[idx]["groundtruth"]
                trace_metadata["correct_answer"] = df.iloc[idx]["is_corrected"]
            else:
                trace_metadata["ground_truth"] = df.iloc[idx]["ground_truth"]
                trace_metadata["correct_answer"] = df.iloc[idx]["is_correct"]
            
            judge_input = str({"query": trace_data["question"], "history_for_evaluating": trace_data["history"]})
            
            logger.info("Generating judge pool...")
            pool = await pool_gen.create_pool(judge_input)
            logger.info(f"Created pool with {len(pool)} judges")
            
            logger.info("Generating evaluation graph...")
            graph = await graph_gen.create_graph(pool, judge_input)
            logger.debug(f"Graph structure: {graph}")

            builder = PipelineBuilder()
            pipeline = builder.create_from_pool(pool, graph).build()
            pipeline.to_mermaid_lr(visualize=True)
            logger.info(f"Built pipeline with {len(pipeline.execution_order)} nodes")

            with judge_client.start_as_current_span(
                name=f"evaluate_task_{df.iloc[idx]["question_ID"]}",
                input={"task_id": df.iloc[idx]["question_ID"], "trace_id": df.iloc[idx]["question_ID"]},
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(
                    tags=["who_and_when_eval_30_traces", f"task_id:{df.iloc[idx]["question_ID"]}"]
                )
                
                logger.info("Executing evaluation pipeline...")
                result, trace_id = await ainvoke_with_lf(pool, pipeline, judge_input, graph)
                logger.info(f"Pipeline execution completed. Trace ID: {trace_id}")
                
                span.update(output={"result": result, "trace_id": trace_id})
                span.end()

            result_dict = json.loads(result)

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
            output_file = output_dir / Path(f"{df.iloc[idx]['question_ID']}.json")

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}\n\n")

            print(f"Result: {result}")
            print(f"Langfuse trace ID: {trace_id}")
            
        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            task_id = df.iloc[idx]["question_ID"]
            logger.error(f"Error processing task {task_id}: {safe_error_msg}", exc_info=True)
            
            failed_traces.append({
                "task_id": task_id,
                "task_index": idx + 1,
                "error": error_msg,
                "error_type": type(e).__name__
            })
            
            print(f"\n!  Failed task {idx + 1}/{len(df)}: {task_id}")
            print(f"   Error: {error_msg}\n")
            continue 
    
    if failed_traces:
        output_dir = local_results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_file = output_dir / "failed_traces.txt"
        
        with open(failed_file, "w") as f:
            f.write(f"Failed traces: {len(failed_traces)} out of {len(df)}\n")
            f.write("=" * 80 + "\n\n")
            
            for failed in failed_traces:
                f.write(f"Task ID: {failed['task_id']}\n")
                f.write(f"Index: {failed['task_index']}/{len(df)}\n")
                f.write(f"Error Type: {failed['error_type']}\n")
                f.write(f"Error Message: {failed['error']}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.warning(f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n")
        print(f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n")
    
    logger.info(f"Completed evaluation: {len(df) - len(failed_traces)}/{len(df)} successful, {len(failed_traces)} failed")


if __name__ == "__main__":
    df_handcrafted = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet")
    # df_algorithm = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet")

    asyncio.run(main(
        save_folder="who_and_when_res_30_traces",
        df=df_handcrafted[:30]
    ))
