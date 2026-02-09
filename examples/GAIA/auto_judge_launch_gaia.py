import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

from automas.meta_agents import GraphGenerator, PoolGenerator
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from automas.utils import get_logger
from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task
from dotenv import load_dotenv
import json
import os

load_dotenv(".env")
logger = get_logger(__name__)

async def main(name: str, save_folder: str):
    logger.info(f"Starting GAIA evaluation for task name: {name}")
    
    pool_gen = PoolGenerator()
    graph_gen = GraphGenerator()
    lf = get_langfuse_download_client()
    judge_client = get_langfuse_judge_client()
    logger.info("Initialized generators and Langfuse clients")

    traces_page1 = lf.api.trace.list(name=name, limit=30, page=1)
    logger.info(f"Retrieved {len(traces_page1.data)} traces from Langfuse")
    
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
    
    for idx, task in enumerate(traces_page1.data):
        if task.id in done_traces:
            logger.info(f"Task {task.id} already processed, skipping...")
            continue
        
        logger.info(f"Processing task {idx + 1}/{len(traces_page1.data)}: {task.id}")
        serializable_results = {}
        
        try:
            trace_data = lf.api.trace.get(task.id)
            query = parse_langfuse_task(trace_data)
            logger.debug(f"Parsed task query: {query.user_query[:100]}...")
            
            trace_metadata = {"task_id": task.id}
            if hasattr(trace_data, "output") and trace_data.output:
                if "ground_truth" in trace_data.output:
                    trace_metadata["ground_truth"] = trace_data.output["ground_truth"]
                if "response" in trace_data.output:
                    trace_metadata["mas_response"] = trace_data.output["response"]
                if "ground_truth" in trace_data.output and "response" in trace_data.output:
                    trace_metadata["correct_answer"] = (
                        trace_data.output["response"] == trace_data.output["ground_truth"]
                    )
                    logger.debug(f"Correct answer: {trace_metadata['correct_answer']}")
            
            judge_input = str({"query": query.user_query, "history_for_evaluating": query.agent_states})
            
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
                name=f"evaluate_task_{task.id}",
                input={"task_id": task.id, "trace_id": task.id},
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(
                    tags=["gaia_eval_30_traces", f"task_id:{task.id}"]
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
            output_file = output_dir / Path(f"{task.id}.json")

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}\n\n")

            print(f"Result: {result}")
            print(f"Langfuse trace ID: {trace_id}")
            print(f"Launch № {idx + 1} from {len(traces_page1.data)}")
            
        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            logger.error(f"Error processing task {task.id}: {safe_error_msg}", exc_info=True)
            
            failed_traces.append({
                "task_id": task.id,
                "task_index": idx + 1,
                "error": error_msg,
                "error_type": type(e).__name__
            })
            
            print(f"\n!  Failed task {idx + 1}/{len(traces_page1.data)}: {task.id}")
            print(f"   Error: {error_msg}\n")
            continue 
    
    if failed_traces:
        output_dir = Path(__file__).parent / save_folder
        output_dir.mkdir(exist_ok=True)
        failed_file = output_dir / "failed_traces.txt"
        
        with open(failed_file, "w") as f:
            f.write(f"Failed traces: {len(failed_traces)} out of {len(traces_page1.data)}\n")
            f.write("=" * 80 + "\n\n")
            
            for failed in failed_traces:
                f.write(f"Task ID: {failed['task_id']}\n")
                f.write(f"Index: {failed['task_index']}/{len(traces_page1.data)}\n")
                f.write(f"Error Type: {failed['error_type']}\n")
                f.write(f"Error Message: {failed['error']}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.warning(f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n")
        print(f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n")
    
    logger.info(f"Completed evaluation: {len(traces_page1.data) - len(failed_traces)}/{len(traces_page1.data)} successful, {len(failed_traces)} failed")


if __name__ == "__main__":
    asyncio.run(main(
        name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097",
        save_folder="gaia_res_30_traces"
    ))
