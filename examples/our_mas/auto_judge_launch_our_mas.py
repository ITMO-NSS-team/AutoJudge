import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

from automas.meta_agents import PoolGenerator
from automas.agent_pool import AgentPool
from automas.pipeline.types import GraphDict
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from automas.utils import get_logger
from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task
from dotenv import load_dotenv
import json
import os
import pandas as pd

load_dotenv(".env")
logger = get_logger(__name__)

taxonomy = """
**LLM Metrics (11 total)** - Input scores may be "ideal", "fair" or "poor":
- Overall score domain: {"ideal", "poor"}
- OBSERVATION_ALIGNMENT
- STATE_CONSISTENCY
- MAS_COMPLEXITY
- MAS_TASK_TRANSFER
- MAS_ROLES_DISTRIBUTION
- TASK_COMPLETENESS
- TOOL_SELECTION
- TOOL_PARAMETER_EXTRACTION
- MAS_TASK_COMPLETION
- MAS_PLANNING
- POLICY_ALIGNMENT
"""
output_schema = """**OUTPUT FORMAT - STRICTLY REQUIRED:**
You MUST return ONLY a valid JSON object with exactly these two fields:
{
  \"score\": \"ideal or poor\",
  \"justification\": \"string\"
}

**CRITICAL RULES:**
- Return ONLY the JSON object, nothing else
- NO markdown code fences (no ```json or ```)
- NO explanatory text before or after the JSON
- NO additional fields (no confidence, no metadata)
- score must be exactly \"ideal\" or \"poor\" (lowercase)
- justification must be a single string (concise, 1-3 sentences)

**VALID EXAMPLE:**
{\"score\": \"ideal\", \"justification\": \"System demonstrates strong performance across all metrics.\"}

**INVALID EXAMPLES:**
- ```json{\"score\": \"ideal\"}```  ← NO markdown fences
- Here is my assessment: {\"score\": \"ideal\"}  ← NO extra text
- {\"score\": \"IDEAL\"}  ← must be lowercase"""

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
    name: str,
    save_folder: str,
    df_summary,
    table_name: str,
    num_traces: int | None = None,
):
    logger.info(f"Starting AutoMAS evaluation for task name: {name}")

    pool_gen = PoolGenerator(
        output_schema=output_schema, taxonomy=taxonomy, examples=examples
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
