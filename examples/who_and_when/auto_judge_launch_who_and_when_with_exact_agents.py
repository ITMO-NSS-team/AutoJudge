"""Launches auto-judge on who_and_when dataset with a pool that has 3 fixed agents: GUILTY_AGENT_FINDER and STEP_OF_ERROR_FINDER."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

from automas.meta_agents import PoolGenerator_WW
from automas.pipeline import PipelineBuilder
from automas.meta_agents import GraphGenerator
from automas.utils.langfuse_utils import ainvoke_with_lf
from automas.utils import get_logger
from maseval import get_langfuse_judge_client
import json
import pandas as pd
from toon_format import encode

logger = get_logger(__name__)

taxonomy = """
1) Guilty agent
2) Step of error
"""

output_schema = """
**OUTPUT FORMAT - STRICTLY REQUIRED:**
You must determine the most guilty agent in the evaluated 'history_for_evaluating', based on what the other judges wrote. You MUST return ONLY a valid JSON object with exactly these three fields:
{
  "agent": "Guilty Agent name from trace here",
  "step": "integer number (1, 2, 3...) of the message in trace sequence",
  "reason": "reason of your prediction"
}

**CRITICAL RULES:**
- Return ONLY the JSON object, nothing else
- NO markdown code fences (no ```json or ```)
- NO explanatory text before or after the JSON
- NO additional fields (no score, no confidence, no metadata)
- agent must be the exact agent name as it appears in the trace
- step must be a plain integer string: "1", "2", "3", etc.

**STEP RULE:** This is the sequential position of the message in 'history_for_evaluating' (1 = first message, 2 = second message, etc). Use ONLY integers. Do NOT use IDs, UUIDs, strings, or any other identifiers.

**VALID EXAMPLE:**
{
  "agent": "File_Surfer",
  "step": "1", 
  "reason": "The agent fails to collect price data for the daily tickets and season passes for California's Great America in 2024."
}


INVALID EXAMPLES (DO NOT USE THIS FORMAT):
```json
{
  "agent": "Orchestrator",
  "step": "21",
  "reason": "The Orchestrator is the most guilty agent. Despite the WebSurfer's repeated failures to find clear Vudu listings for 'The Tenant' and 'Nosferatu the Vampyre' (as noted in steps 13 and 17), and the subsequent 'ResponsibleAIPolicyViolation' error in step 21, the Orchestrator still allowed the final answer to be 'The Tenant' without any verified evidence of its availability on Vudu. This indicates a failure in the Orchestrator's decision-making process to ensure all constraints were met before providing a final answer. The Orchestrator also repeatedly asked the WebSurfer to check for Vudu availability without changing its strategy, indicating a lack of progress and looping, as highlighted by the Search Integrity Judge."
}
```

**INVALID STEP EXAMPLES:** "step1", "abc-123", "task_id_45", "first" — ONLY USE: "1", "2", "3", etc.
"""

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


async def main(
    save_folder: str, df, df_summary, table_name: str, num_traces: int | None = None
):
    logger.info("===Starting Who&When evaluation===")

    pool_gen = PoolGenerator_WW(
        output_schema=output_schema, taxonomy=taxonomy, examples=examples
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
