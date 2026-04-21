import json
import logging
import os
import sys
import re
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
# Add project root to path for maseval
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(".env")

# Import the extended instruction set to auto-enable tools
from automas.meta_agents.prompt_registry import DEFAULT_POOL_INSTRUCT_EXTENDED

from automas.meta_agents import PoolGenerator
from automas.agent_pool import AgentPool
from automas.pipeline.types import GraphDict
from automas.pipeline import PipelineBuilder
from automas.pipeline.node_session import NodeExecution, NodeSessionError
from automas.utils.langfuse_utils import setup_langfuse_instrumentation
from automas.utils import get_logger
from maseval import get_langfuse_judge_client
from automas.db.db_tools import get_content_tool
from datasets import load_dataset

logger = get_logger(__name__)

POOL_GENERATION_ATTEMPTS = 3
FINAL_AGENT_ONLY_ATTEMPTS = 8
MISSING_DEPENDENCY_RECOVERY_ATTEMPTS = 3
DEPENDENCY_NODE_ATTEMPTS = 5
TRANSIENT_ERROR_RETRY_DELAY_SECONDS = 1.5

taxonomy = """
### Functional Mistakes (FM-1.x - Task Execution Errors):
- FM-1.1: **Task specification deviation** - Agent deviates from specified task requirements (e.g., was asked to write code in Python, but used JavaScript).
- FM-1.2: **Role specification deviation** - Agent acts outside its designated role (e.g., a 'CodeWriter' agent starts criticizing other agents' work, which is the 'Critic's' role).
- FM-1.3: **Add redundant steps** - Agent adds unnecessary or duplicate steps (e.g., imports a library that was already imported in a previous step).
- FM-1.4: **Remove conversation history** - Agent ignores or removes important context from previous turns (e.g., ignores a user's correction from the previous message).
- FM-1.5: **Remove termination conditions** - Agent fails to define proper stopping criteria, leading to loops or unfinished tasks (e.g., writes a recursive function with no base case).

### Functional Mistakes (FM-2.x - Communication & Coordination Errors):
- FM-2.1: **Repeat handled tasks** - Agent redundantly handles already completed tasks (e.g., re-writes a piece of code that was already finalized and approved).
- FM-2.2: **Make request ambiguous** - Agent provides unclear or confusing instructions to other agents (e.g., asks another agent to "handle the data" without specifying how).
- FM-2.3: **Deviate from main goal** - Agent pursues objectives unrelated to the main task (e.g., starts discussing the history of programming languages in the middle of a coding task).
- FM-2.4: **Hide important information** - Agent withholds crucial information needed by other agents (e.g., knows a library has a bug but doesn't mention it).
- FM-2.5: **Ignore other agents** - Agent fails to consider input, corrections, or questions from other agents.
- FM-2.6: **Inconsistent reasoning** - Agent's logic contradicts its own previous statements (e.g., in step 2 agent says 'option A is best', but in step 4 says 'option A is a bad choice' without new information).

### Functional Mistakes (FM-3.x - Quality & Verification Errors):
- FM-3.1: **Premature termination** - Agent stops or declares the task complete before all requirements are met.
- FM-3.2: **Remove verification steps** - Agent skips necessary validation or testing steps (e.g., writes code but doesn't write any unit tests for it).
- FM-3.3: **Incorrect verification** - Agent performs flawed or wrong verification (e.g., writes a test that doesn't actually check for the correct condition).
"""

output_schema = """
**CRITICAL:** 
You are evaluating a multi-agent trace. 
The provided `history_for_evaluating` contains short summaries of each step rather than the full raw content. 
If you suspect an error in a step based on its summary, you MUST use the `get_content_tool` to fetch the full raw content from the database before making a final verdict.
Database state_id format is "{trace_id}_{step}" and table name is 'aegis'. 
You will receive the `trace_id` in the input.

**ANALYSIS WORKFLOW - CHAIN OF THOUGHT:**
1. **Agent Summary**: First, analyze and summarize what each agent has done throughout the conversation. List each agent and summarize their main actions, decisions, and contributions. Notice patterns.
2. **Error Analysis**: For each agent, carefully examine their actions against each error definition in the taxonomy. Look for violations of task requirements, role boundaries, communication issues, or quality problems. Note any potential errors with specific reasoning.
3. **Final Judgment**: Based on the context, determine which agents (if any) committed errors, and assign the appropriate error code. Ensure agent names match EXACTLY as they appear in the conversation log. Do not fabricate names.

**OUTPUT FORMAT - STRICTLY REQUIRED:**
Return EXACTLY AND ONLY a valid JSON object matching this structure. Do not output ANY markdown wrappers (no ```json or ```). Start directly with the `{{` character and end with `}}`.

{{
    "agent_summary": "[Analysis step 1: Summarize what each agent did]",
    "error_analysis": "[Analysis step 2: Your reasoning for identifying errors]",
    "faulty_agents": [
        {{
            "agent_name": "[AGENT NAME FROM TRACE]",
            "error_type": "FM-X.Y"
        }}
    ]
}}

**EXAMPLES:**
- Multiple Errors: 
{{
    "agent_summary": "Agent 1 gathered data, Agent 2 wrote code.",
    "error_analysis": "Agent 1 ignored user constraints causing FM-1.1. Agent 2 skipped tests causing FM-3.2.",
    "faulty_agents": [
        {{"agent_name": "Agent1", "error_type": "FM-1.1"}}, 
        {{"agent_name": "Agent2", "error_type": "FM-3.2"}}
    ]
}}

- No Errors: 
{{
    "agent_summary": "Agent A developed the plan and Agent B executed it.",
    "error_analysis": "Both agents followed instructions perfectly and verified their steps.",
    "faulty_agents": []
}}


- If no agents made errors, `faulty_agents` should be an empty list: [].
- `agent_name` MUST match names from the conversation history exactly.
- `error_type` MUST be one of the FM-X.Y codes exactly.
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

def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    graph_dict: GraphDict = {}
    _agents_info = agent_pool.full_agents_data

    for agent in _agents_info:
        if agent.get("name") != "FINAL_AGGREGATOR":
            graph_dict[agent.get("name")] = ["FINAL_AGGREGATOR"]

    graph_dict["FINAL_AGGREGATOR"] = []

    return graph_dict


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


async def main(save_folder: str, split: str = "test", max_traces: int | None = None):
    logger.info("===Starting AEGIS evaluation WITH SUMMARIES===")

    pool_gen = PoolGenerator(
        output_schema=output_schema, taxonomy=taxonomy, examples=examples
    )
    judge_client = get_langfuse_judge_client()
    setup_langfuse_instrumentation()
    logger.info("Initialized generators and Langfuse client")

    # Load Summaries
    summary_dir = Path(__file__).resolve().parent / "aegis_summaries"
    summaries_dict = {}
    if summary_dir.exists():
        for f in summary_dir.iterdir():
            if f.is_file() and f.suffix == ".json":
                with open(f) as fh:
                    data = json.load(fh)
                    trace_name = f.name.split(".")[0]
                    summaries_dict[trace_name] = data
        logger.info(f"Loaded {len(summaries_dict)} summaries from {summary_dir}")
    else:
        logger.warning(f"Summary directory {summary_dir} does not exist.")

    ds = load_dataset("Fancylalala/AEGIS", split=split)
    logger.info(f"Loaded {len(ds)} traces from AEGIS {split} split")

    if max_traces is not None:
        ds = ds.select(range(min(max_traces, len(ds))))
        logger.info(f"Limited to {len(ds)} traces")

    local_results_dir = Path(__file__).resolve().parent / "results" / save_folder
    local_results_dir.mkdir(parents=True, exist_ok=True)

    done_traces_with_occurrence = set()  # Track (trace_id, occurrence) pairs
    if local_results_dir.exists():
        for res in os.listdir(local_results_dir):
            if res.endswith(".json"):
                filepath = local_results_dir / res
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("input_type") == "summary_context":
                            res_cropped = res.split(".")[0]
                            # Parse new format: <trace_id>_occ<N>.json
                            match = re.match(r"^(.+)_occ(\d+)$", res_cropped)
                            if match:
                                base_id = match.group(1)
                                occurrence = int(match.group(2))
                                done_traces_with_occurrence.add((base_id, occurrence))
                            else:
                                # Backward compatibility with legacy results saved as <trace_id>.json.
                                # These correspond to the first occurrence (occurrence 0).
                                base_id = data.get("task_id") or data.get("summary_file_used") or res_cropped
                                done_traces_with_occurrence.add((str(base_id), 0))
                except Exception:
                    pass

    failed_traces = []
    failed_file = local_results_dir / "failed_traces.txt"
    failed_traces_ids = []
    
    if failed_file.exists():
        with open(failed_file) as fh:
            for line in fh:
                if line.startswith("Task ID:"):
                    failed_traces_ids.append(line.split(":")[1].strip())

    # Track occurrence count per unique trace_id
    occurrence_counter = {}
    
    for idx, trace in enumerate(ds):
        trace_id = str(trace["id"]).replace("/", "_")  # Ensure valid filename
        
        # Count this occurrence of the trace_id
        if trace_id not in occurrence_counter:
            occurrence_counter[trace_id] = 0
        else:
            occurrence_counter[trace_id] += 1
        
        occurrence = occurrence_counter[trace_id]
        
        # Check if this specific occurrence was already processed
        if (trace_id, occurrence) in done_traces_with_occurrence:
            logger.info(f"Task {trace_id} occurrence {occurrence} already processed, skipping...")
            continue
        if trace_id in failed_traces_ids:
            logger.info(f"Task {trace_id} previously failed, skipping...")
            continue

        q_input = trace.get("input", {})
        question = q_input.get("question", "")

        # Determine which summary file to load based on occurrence number
        # First occurrence uses base name, subsequent use base_2, base_3, etc.
        if occurrence == 0:
            summary_key = trace_id
        else:
            summary_key = f"{trace_id}_{occurrence + 1}"  # _2 for second, _3 for third, etc.
        
        # Use summaries instead of full trace where available
        if summary_key in summaries_dict:
            history_for_evaluating = json.dumps(summaries_dict[summary_key].get("step_by_step_summary", []), indent=2)
            input_type = "summary_context"
            actual_summary_key = summary_key
        else:
            logger.warning(f"Summary not found for {trace_id} (occurrence {occurrence}, tried {summary_key}), skipping trace")
            continue

        logger.info(f"Processing task {idx + 1}/{len(ds)}: {trace_id} (using {input_type})")

        judge_input_dict = {
            "query": question,
            "trace_id": trace_id,
            "history_for_evaluating": history_for_evaluating,
        }
        judge_input = str(judge_input_dict)

        try:
            pool = await create_pool_with_retries(pool_gen, dict(query=question, history_for_evaluating=history_for_evaluating))
            for agent in pool.full_agents_data:
                agent["mcp_tools"] = [get_content_tool]

            graph = get_parallel_graph(pool)
            builder = PipelineBuilder()
            pipeline = builder.create_from_pool(pool, graph).build()
            pipeline.node_session.context_deps = trace_id
            
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{trace_id}",
                input={"task_id": trace_id, "input_type": input_type},
                metadata={
                    "task_id": trace_id,
                    "input_type": input_type
                },
            ) as span:
                judge_client.update_current_trace(tags=["aegis" ,"test" if max_traces else "full", f"task_id:{trace_id}", split, "summary_context"])

                try:
                    raw_result = await pipeline.ainvoke(judge_input)
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

                        raw_result = await run_final_agent_only_with_retries(
                            pipeline,
                            max_attempts=FINAL_AGENT_ONLY_ATTEMPTS,
                        )
                    else:
                        raise
                
                clean = raw_result.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                if clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
                
                try:
                    result_dict = json.loads(clean)
                except json.JSONDecodeError:
                    result_dict = {"raw": raw_result}

                span.update(output={"result": result_dict})
                span.end()

            # Save result with occurrence number to track duplicates
            result_filename = f"{trace_id}_occ{occurrence}.json"
            out_file = local_results_dir / result_filename
            output_data = {
                "task_id": trace_id,
                "occurrence": occurrence,
                "summary_file_used": actual_summary_key,
                "input_type": input_type,
                "dataset_split": split,
                "ground_truth": {
                    "faulty_agents": trace.get("ground_truth", {}).get("injected_agents", []),
                },
                "model_detection": {
                    "faulty_agents": result_dict.get("faulty_agents", []),
                },
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {out_file}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing task {trace_id}: {error_msg}", exc_info=True)
            failed_traces.append({"task_id": trace_id, "error": error_msg})

    if failed_traces:
        with open(failed_file, "w") as fh:
            fh.write(f"Failed traces: {len(failed_traces)}\n{'=' * 80}\n\n")
            for ft in failed_traces:
                fh.write(
                    f"Task ID: {ft['task_id']}\n"
                    f"Error Message: {ft['error']}\n"
                    f"{'-' * 80}\n\n"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS Evaluation over Summaries.")
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run on 5 traces only.",
    )
    args = parser.parse_args()

    # folder for output
    folder = "aegis_eval_summaries_test" if args.test else "aegis_eval_summaries_fixed"
    
    max_traces = 5 if args.test else None

    asyncio.run(
        main(
            save_folder=folder,
            split=args.split,
            max_traces=max_traces,
        )
    )
