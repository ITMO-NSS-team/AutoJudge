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
from maseval import get_langfuse_judge_client
import json
import pandas as pd
from toon_format import encode

logger = get_logger(__name__)

from .prompts import taxonomy, output_schema

examples = """
Example 1 - MAS Task Completion Evaluation:
[
  {
    "name": "MAS_TASK_COMPLETION_JUDGE",
    "instructions": "**Instruction**:
Evaluate whether the multi-agent system fully completed the user's task by assessing end-to-end outcome across all agents.

**Evaluation Criteria**:
1. *Task Relevance* - Does output address the main objective?
2. *Completeness* - Are all required subtasks/steps present?
3. *Consistency* - Are agent outputs logically coherent without contradictions?
4. *Actionability* - Can the user act on outputs to achieve their goal?
5. *Efficiency* - Were tasks completed without unnecessary duplication?

You must use the available tools at least once!

**Scoring**:
- \"ideal\": Task fully achieved, all subtasks addressed, outputs consistent and actionable
- \"fair\": Task largely achieved but minor omissions or slight inconsistencies
- \"poor\": Task failed, critical steps missing, inconsistent or unusable outputs

Return JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": [get_content_tool]
  }
]

Example 2 - MAS Complexity Assessment:
[
  {
    "name": "MAS_COMPLEXITY_JUDGE",
    "instructions": "**Instruction**:
Evaluate complexity and interconnectedness of the multi-agent system.

**Evaluation Criteria**:
1. *Agent Density* - Is number of agents appropriate for system scope?
2. *Interconnection Quality* - Are agent connections well-designed and efficient?
3. *System Scalability* - Can architecture accommodate growth and maintainability?

You must use the available tools at least once!

**Scoring**:
- \"ideal\": Complexity perfectly balanced with optimal density and connections
- \"fair\": Complexity manageable but has scalability or efficiency issues
- \"poor\": Complexity poorly managed with density or connection problems

Return single JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": [get_content_tool]
  }
]

Example 3 - Tool Performance Evaluation:
[
  {
    "name": "TOOL_PERFORMANCE_JUDGE",
    "instructions": "**Instruction**:
Assess whether tools successfully fulfilled user requests by evaluating execution outcome quality.

**Evaluation Criteria**:
1. *Task Completion* - Did tool fully accomplish the request?
2. *Accuracy* - Is output accurate, relevant, and logically consistent?
3. *Clarity* - Is output clear, structured, and in expected format?
4. *Failure Handling* - Any errors or unrelated information returned?

You must use the available tools at least once!

**Scoring** (strict - zero tolerance for errors):
- \"ideal\": Output perfectly solves task, all parts correct and complete
- \"fair\": Output mostly correct but minor issues or omissions
- \"poor\": Output fails task, incorrect, incomplete, or misleading

Return JSON list: [{\"state_id\": \"...\", \"justification\": \"...\", \"score\": \"ideal|fair|poor\"}]",
    "mcp_tools": [get_content_tool]
  }
]

Example 4 - Environment Setup Error Detection:
[
  {
    "name": "MAS_ENVIRONMENT_SETUP_JUDGE",
    "instructions": "**Instruction**:
Analyze execution trace to identify environment setup and configuration errors that occurred BEFORE or DURING initialization.

**Scope**: Focus on initialization phase errors, NOT runtime API errors.

**Evaluation Criteria** - Look for trace entries showing:
1. *File System Issues* - Permission denied, access errors (PermissionError, errno 13)
2. *Credential Problems* - Missing API keys in config (KeyError: 'API_KEY')
3. *Environment Variables* - Missing or invalid env vars (os.environ KeyError)
4. *Config Files* - Missing or malformed configs (FileNotFoundError, JSONDecodeError)
5. *Dependencies* - Import errors or version conflicts (ModuleNotFoundError)
You must use the available tools at least once!

**Out of Scope**: HTTP status codes (401, 403, 429, 500), runtime API errors, network timeouts

**Scoring**:
- \"ideal\": No setup errors, clean initialization
- \"fair\": Minor warnings but system recovered with defaults
- \"poor\": Critical setup errors prevented system startup

Return JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": [get_content_tool]
  }
]

Example 5 - API Issues Detection:
[
  {
    "name": "MAS_API_ISSUES_JUDGE",
    "instructions": "**Instruction**:
Analyze execution trace to identify API-related errors during RUNTIME execution.

**Scope**: Focus on runtime API communication errors, NOT initialization/config errors.
You must use the available tools at least once!

**Evaluation Criteria** - Look for trace entries showing:
1. *Rate Limiting* - HTTP 429, "Rate limit exceeded" (RateLimitError)
2. *Auth Errors* - HTTP 401/403 during API calls, "Invalid token" (AuthenticationError)
3. *Server Errors* - HTTP 500/502/503/504, "Internal Server Error"
4. *Not Found* - HTTP 404, "Endpoint not found"
5. *Client Errors* - HTTP 400/422, "Bad Request", "Validation failed"
6. *Network Failures* - Connection timeout, "Connection refused" (ConnectionError)

**Out of Scope**: Environment variable errors, config file issues, local file permissions

**Scoring**:
- \"ideal\": No API errors, all external calls succeeded
- \"fair\": Minor/temporary API errors but system recovered
- \"poor\": Critical API errors prevented task completion or occurred repeatedly

Return JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": []
  }
]

Example 6 - Tool Selection Evaluation:
[
  {
    "name": "TOOL_SELECTION_JUDGE",
    "instructions": "**Instruction**:
Assess whether tool selections made by the agent are appropriate for the task.

**Evaluation Criteria**:
1. *Tool Relevance* - Does the selected tool directly address the node_role responsibility?
2. *Pipeline Position* - Is the tool suitable given the agent's position in the pipeline?
3. *Justification* - Is the tool selection clearly supported by the task requirements?

**Scoring**:
- \"ideal\": Tool selection perfectly matches node_role and is clearly justified
- \"fair\": Selection is relevant but potentially suboptimal for the task
- \"poor\": Selection is inappropriate or clearly mismatched to node_role

Return JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": []
  }
]"""


def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    graph_dict = {}
    _agents_info = agent_pool.full_agents_data

    for agent in _agents_info:
        if agent["name"] != "FINAL_AGGREGATOR":
            graph_dict[agent["name"]] = ["FINAL_AGGREGATOR"]

    graph_dict["FINAL_AGGREGATOR"] = []

    return graph_dict


async def main(
    save_folder: str, df, df_summary, table_name: str, num_traces: int | None = None, pool: AgentPool = None, n_set: int = 0
):
    logger.info(f"===Starting Who&When evaluation===")

    judge_client = get_langfuse_judge_client()
    logger.info("Initialized Langfuse client")

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

            logger.info("Checking judge pool...")

            pool = pool
            logger.info(f"Use pool with {len(pool)} judges")

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
                judge_client.update_current_trace(tags=[f"union_pool_gen_{n_set}", f"task_id:{id}"])

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

async def generate_pool_per_dataset(trace_reference: dict) -> AgentPool:
    pool_gen = PoolGenerator(
        output_schema=output_schema, taxonomy=taxonomy, examples=examples, pool_per_dataset=True
    )
    try:
        attempts = 0
        while attempts < 3:
            pool = await pool_gen.create_pool(trace_reference)
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
        print(f"\n!  Failed launching pool")
        raise e
    return pool 


async def run_multiple_pool_generators(df_summary: pd.DataFrame, batch_size: int = 3, how_many_batches: int = 3) -> list[AgentPool]:
    """
    Runs multiple pool generators in parallel for different subsets of the dataset.
    """
    batches = [
        df_summary.iloc[i:i + batch_size].to_dict(orient="records")
        for i in range(0, len(df_summary), batch_size)
    ]
    # Limit the number of batches
    batches = batches[:how_many_batches]
    tasks = [generate_pool_per_dataset(trace_reference=batch) for batch in batches]
    # parallel execution of pool generation for each batch
    all_pools = await asyncio.gather(*tasks)
    return all_pools

if __name__ == "__main__":
    from datasets import load_dataset

    df = load_dataset(
        "McGill-NLP/agent-reward-bench",
        "annotations",
        split="full"
    ).to_pandas()
    summaries_directory = Path("/Users/alina/Desktop/ITMO/AutoJudge/examples/who_and_when/step_summaries_gemini_2.5_flash/algo")

    summary = []
    for dir in summaries_directory.iterdir():
        if dir.is_file() and dir.suffix == ".json":
            with open(dir, "r") as f:
                data = json.load(f)
                summary.append([data, dir.name.split(".")[0]])
                
    df_summary = pd.DataFrame(summary, columns=["summary", "question_ID"])
    all_pools = asyncio.run(run_multiple_pool_generators(df_summary, batch_size=3))
    cnt = 0
    
    for pool in all_pools:
        cnt += 1
        logger.info(f"Use pool with {len(pool)} agents: {[agent['name'] for agent in pool.full_agents_data]}")
        asyncio.run(
            main(
                save_folder=f"algo_union_pool_results_{cnt}",
                df=df_algorithm,
                df_summary=df_summary,
                table_name="who_when",
                pool=pool,  # pass first pool for testing, can be modified to loop through all pools for different batches
                # num_traces=30,
                n_set=cnt
            )
        )
