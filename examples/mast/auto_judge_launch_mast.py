import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

from automas.meta_agents import PoolGenerator
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from automas.utils import get_logger
from automas.agent_pool import AgentPool
from automas.pipeline.types import GraphDict
from maseval import get_langfuse_judge_client
import json
import pandas as pd
from toon_format import encode

logger = get_logger(__name__)

taxonomy = """
    "1.1 Disobey Task Specification: <yes or no>"
    "1.2 Disobey Role Specification: <yes or no>"
    "1.3 Step Repetition: <yes or no>"
    "1.4 Loss of Conversation History: <yes or no>"
    "1.5 Unaware of Termination Conditions: <yes or no>"
    "2.1 Conversation Reset: <yes or no>"
    "2.2 Fail to Ask for Clarification: <yes or no>"
    "2.3 Task Derailment: <yes or no>"
    "2.4 Information Withholding: <yes or no>"
    "2.5 Ignored Other Agent's Input: <yes or no>"
    "2.6 Action-Reasoning Mismatch: <yes or no>"
    "3.1 Premature Termination: <yes or no>"
    "3.2 No or Incorrect Verification: <yes or no>"
    "3.3 Weak Verification: <yes or no>"
"""

output_schema = """
**OUTPUT FORMAT - STRICTLY REQUIRED:**
You must determine all criteria for evaluated 'history_for_evaluating', based on what the other judges wrote. You MUST return ONLY a valid JSON object with exactly these two fields:
{"failure mode": "1.1 Disobey Task Specification\n\nFailure to adhere to the specified constraints or requirements of a given task, leading to suboptimal or incorrect outcomes.",
        "result": bool,
      },
      {
        "failure mode": "1.2 Disobey Role Specification \n\n\nFailure to adhere to the defined responsibilities and constraints of an assigned role, potentially leading to an agent behaving like another.",
        "result": bool,
      },
      {
        "failure mode": "1.3 Step Repetition\n\nUnnecessary reiteration of previously completed steps in a process, potentially causing delays or errors in task completion. ",
        "result": bool,
      },
      {
        "failure mode": "1.4 Loss of Conversation History\n\nUnexpected context truncation, disregarding recent interaction history and reverting to an antecedent conversational state.\n",
        "result": bool,
      },
      {
        "failure mode": "1.5 Unaware of Termination Conditions\n\nLack of recognition or understanding of the criteria that should trigger the termination of the agents\u2019 interaction, potentially leading to unnecessary continuation.",
        "result": bool,
      },
      {
        "failure mode": "2.1 Conversation reset \n\nUnexpected or unwarranted restarting of a dialogue, potentially losing context and progress made in the interaction.",
         "result": bool,
      },
      {
        "failure mode": "2.2 Fail to ask for clarification\n(between agents)\nInability to request additional information between agent when faced with unclear or incomplete data, potentially resulting in incorrect actions.",
        "result": bool,
      },
      {
        "failure mode": "2.3 Task derailment\n\nDeviation from the intended objective or focus of a given task, potentially resulting in irrelevant or unproductive actions.",
        "result": bool,
      },
      {
        "failure mode": "2.4 Information Witholding\n\nFailure to share or communicate important data that an agent has obtained before and did not share that information with other agents potentially leading to failures or inefficiencies. ",
        "result": bool,
      },
      {
        "failure mode": "2.5 Ignored Other Agents' Input\n\nDisregarding or failing to adequately consider input or recommendations provided by other agents in the system, potentially leading to suboptimal decisions or missed opportunities for collaboration.",
        "result": bool,

      },
      {
        "failure mode": "2.6 Reasoning-Action Mismatch\n\nInconsistency between reasoning and action\n\nDiscrepancy between the logical reasoning process and the actual actions taken by the agent, potentially resulting in unexpected or undesired behaviors.",
        "result": bool,
      },
      {
        "failure mode": "3.1 Premature Termination\n\nIll specified termination condition leading to premature termination\n\nEnding a dialogue, interaction or task before all necessary information has been exchanged or objectives have been met. Necessary information constitutes verification of outputs, key data (e.g. api tokens) etc. that are necessary for the success of the task, and agents could have obtained if they tried more or already obtained but failed to communicate to other agents before termination.",
        "result": bool,
      },
      {
        "failure mode": "3.2 No or Incomplete Verification\n\nLack of critical verification\n(system designed to but agents didn't)\n\n- \n\n- robustness (it covers also edge cases) \n- error handling etc.\n(broader)\n\nFailure to adequately validate or cross-check crucial information or decisions during the iterations, potentially leading to errors or vulnerabilities in the system.\n\n1. verifier exists in the system, but it is weak or badly design\n2. the system attempts to verify, but doesn't cover all aspects of the design to make a strong output\n\nThis is implicitly True if 3.3 Lack of result verification is true, so no longer need to explicitly specify as True.",
        "result": bool,
      },
      {
        "failure mode": "3.3 Incorrect Verification\n\nLack of result verification\n(no explicit design, only when the system benefits really ) \nOmission of proper checking or confirmation of task outcomes or system outputs, potentially allowing errors or inconsistencies to propagate undetected.",
        "result": bool,
      }

**VALID EXAMPLE:**
{"failure mode": "1.1 Disobey Task Specification\n\nFailure to adhere to the specified constraints or requirements of a given task, leading to suboptimal or incorrect outcomes.",
        "result": True,
      },
      {
        "failure mode": "1.2 Disobey Role Specification \n\n\nFailure to adhere to the defined responsibilities and constraints of an assigned role, potentially leading to an agent behaving like another.",
        "result": True,
      },
      {
        "failure mode": "1.3 Step Repetition\n\nUnnecessary reiteration of previously completed steps in a process, potentially causing delays or errors in task completion. ",
        "result": True,
      },
      {
        "failure mode": "1.4 Loss of Conversation History\n\nUnexpected context truncation, disregarding recent interaction history and reverting to an antecedent conversational state.\n",
        "result": True,
      },
      {
        "failure mode": "1.5 Unaware of Termination Conditions\n\nLack of recognition or understanding of the criteria that should trigger the termination of the agents\u2019 interaction, potentially leading to unnecessary continuation.",
        "result": True,
      },
      {
        "failure mode": "2.1 Conversation reset \n\nUnexpected or unwarranted restarting of a dialogue, potentially losing context and progress made in the interaction.",
         "result": False,
      },
      {
        "failure mode": "2.2 Fail to ask for clarification\n(between agents)\nInability to request additional information between agent when faced with unclear or incomplete data, potentially resulting in incorrect actions.",
        "result": False,
      },
      {
        "failure mode": "2.3 Task derailment\n\nDeviation from the intended objective or focus of a given task, potentially resulting in irrelevant or unproductive actions.",
        "result": False,
      },
      {
        "failure mode": "2.4 Information Witholding\n\nFailure to share or communicate important data that an agent has obtained before and did not share that information with other agents potentially leading to failures or inefficiencies. ",
        "result": False,
      },
      {
        "failure mode": "2.5 Ignored Other Agents' Input\n\nDisregarding or failing to adequately consider input or recommendations provided by other agents in the system, potentially leading to suboptimal decisions or missed opportunities for collaboration.",
        "result": False,

      },
      {
        "failure mode": "2.6 Reasoning-Action Mismatch\n\nInconsistency between reasoning and action\n\nDiscrepancy between the logical reasoning process and the actual actions taken by the agent, potentially resulting in unexpected or undesired behaviors.",
        "result": True,
      },
      {
        "failure mode": "3.1 Premature Termination\n\nIll specified termination condition leading to premature termination\n\nEnding a dialogue, interaction or task before all necessary information has been exchanged or objectives have been met. Necessary information constitutes verification of outputs, key data (e.g. api tokens) etc. that are necessary for the success of the task, and agents could have obtained if they tried more or already obtained but failed to communicate to other agents before termination.",
        "result": True,
      },
      {
        "failure mode": "3.2 No or Incomplete Verification\n\nLack of critical verification\n(system designed to but agents didn't)\n\n- \n\n- robustness (it covers also edge cases) \n- error handling etc.\n(broader)\n\nFailure to adequately validate or cross-check crucial information or decisions during the iterations, potentially leading to errors or vulnerabilities in the system.\n\n1. verifier exists in the system, but it is weak or badly design\n2. the system attempts to verify, but doesn't cover all aspects of the design to make a strong output\n\nThis is implicitly True if 3.3 Lack of result verification is true, so no longer need to explicitly specify as True.",
        "result": True,
      },
      {
        "failure mode": "3.3 Incorrect Verification\n\nLack of result verification\n(no explicit design, only when the system benefits really ) \nOmission of proper checking or confirmation of task outcomes or system outputs, potentially allowing errors or inconsistencies to propagate undetected.",
        "result": False,
      }

**INVALID EXAMPLES:**
- ```json{\"score\": \"ideal\"}``` 
- Here is my assessment: {\"score\": \"ideal\"} 
- {\"score\": \"IDEAL\"}"""

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

**Scoring**:
- \"ideal\": Task fully achieved, all subtasks addressed, outputs consistent and actionable
- \"fair\": Task largely achieved but minor omissions or slight inconsistencies
- \"poor\": Task failed, critical steps missing, inconsistent or unusable outputs

Return JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": []
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

**Scoring**:
- \"ideal\": Complexity perfectly balanced with optimal density and connections
- \"fair\": Complexity manageable but has scalability or efficiency issues
- \"poor\": Complexity poorly managed with density or connection problems

Return single JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": []
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

**Scoring** (strict - zero tolerance for errors):
- \"ideal\": Output perfectly solves task, all parts correct and complete
- \"fair\": Output mostly correct but minor issues or omissions
- \"poor\": Output fails task, incorrect, incomplete, or misleading

Return JSON list: [{\"state_id\": \"...\", \"justification\": \"...\", \"score\": \"ideal|fair|poor\"}]",
    "mcp_tools": []
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

**Out of Scope**: HTTP status codes (401, 403, 429, 500), runtime API errors, network timeouts

**Scoring**:
- \"ideal\": No setup errors, clean initialization
- \"fair\": Minor warnings but system recovered with defaults
- \"poor\": Critical setup errors prevented system startup

Return JSON: {\"score\": \"ideal|fair|poor\", \"justification\": \"...\"}",
    "mcp_tools": []
  }
]

Example 5 - API Issues Detection:
[
  {
    "name": "MAS_API_ISSUES_JUDGE",
    "instructions": "**Instruction**:
Analyze execution trace to identify API-related errors during RUNTIME execution.

**Scope**: Focus on runtime API communication errors, NOT initialization/config errors.

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
]"""


def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    graph_dict = {}
    _agents_info = agent_pool.full_agents_data

    for agent in _agents_info:
        if agent["name"] != "FINAL_AGGREGATOR":
            graph_dict[agent["name"]] = ["FINAL_AGGREGATOR"]

    graph_dict["FINAL_AGGREGATOR"] = []

    return graph_dict


async def main(save_folder: str, json_file_path: str):
    logger.info(f"===Starting Who&When evaluation from JSON file: {json_file_path}===")

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    logger.info(f"Loaded {len(df)} traces. Columns: {df.columns.tolist()}")

    required_cols = ["trace_id", "trace"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in JSON: {missing_cols}")

    pool_gen = PoolGenerator(
        output_schema=output_schema, taxonomy=taxonomy, examples=examples
    )
    judge_client = get_langfuse_judge_client()
    logger.info("Initialized generators and Langfuse client")

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
    new_failed_traces = []

    if local_results_dir.exists():
        if "failed_traces.txt" in os.listdir(local_results_dir):
            failed_traces = []
            with open(local_results_dir / "failed_traces.txt", "r") as f:
                for line in f:
                    if line.startswith("Task ID:"):
                        failed_traces_ids.append(line.split(":")[1].strip())

    for idx in range(len(df)):
        trace_id = str(df.iloc[idx]["trace_id"])

        if trace_id in done_traces:
            logger.info(f"Trace {trace_id} already processed, skipping...")
            continue

        if trace_id in failed_traces_ids:
            logger.info(f"Trace {trace_id} already failed, skipping...")
            continue

        logger.info(f"Processing trace {idx + 1}/{len(df)}: {trace_id}")
        serializable_results = {}

        try:
            history = df.iloc[idx]["trace"]
            mas_name = df.iloc[idx].get("mas_name", "unknown")
            benchmark_name = df.iloc[idx].get("benchmark_name", "unknown")
            round_num = df.iloc[idx].get("round", "unknown")

            if not history:
                raise ValueError(f"Empty 'trace' field in trace {trace_id}")

            question = f"Evaluate MAS trace from {mas_name} benchmark {benchmark_name}"

            trace_data = {
                "history": history,
                "question": question,
                "task_id": trace_id,
                "trace_id": trace_id,
            }

            logger.debug(f"Trace {trace_id}: {question[:100]}...")

            trace_metadata = {
                "task_id": trace_id,
                "trace_id": trace_id,
                "mas_name": mas_name,
                "benchmark_name": benchmark_name,
                "round": round_num,
            }

            trace_metadata.update({"annotations": df.iloc[idx]["annotations"]})

            judge_input = encode(
                {"question": question, "history_for_evaluating": trace_data["history"]}
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
                    f"Error processing task {trace_id}: {safe_error_msg}", exc_info=True
                )

                if failed_traces_ids:
                    new_failed_traces.append(
                        {
                            "task_id": trace_id,
                            "task_index": idx + 1,
                            "error": error_msg,
                            "error_type": type(e).__name__,
                        }
                    )
                else:
                    failed_traces.append(
                        {
                            "task_id": trace_id,
                            "task_index": idx + 1,
                            "error": error_msg,
                            "error_type": type(e).__name__,
                        }
                    )

                print(f"\n!  Failed task {idx + 1}/{len(df)}: {trace_id}")
                print(f"   Error: {error_msg}\n")
                continue

            logger.info(f"Created pool with {len(pool)} judges")

            logger.info("Generating evaluation graph...")
            graph = get_parallel_graph(pool)

            builder = PipelineBuilder()
            pipeline = builder.create_from_pool(pool, graph).build()
            pipeline.to_mermaid_lr(visualize=True)
            logger.info(f"Built pipeline with {len(pipeline.execution_order)} nodes")

            with judge_client.start_as_current_span(
                name=f"evaluate_trace_{trace_id}",
                input={"task_id": trace_id, "trace_id": trace_id},
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(
                    tags=[
                        "mast_dataset_eval",
                        f"trace_id:{trace_id}",
                        f"mas:{mas_name}",
                    ]
                )

                logger.info("Executing evaluation pipeline...")
                result, pipeline_trace_id = await ainvoke_with_lf(
                    pool, pipeline, judge_input, graph
                )
                logger.info(
                    f"Pipeline execution completed. Trace ID: {pipeline_trace_id}"
                )

                span.update(output={"result": result, "trace_id": pipeline_trace_id})
                span.end()

            result_dict = json.loads(result)

            serializable_results = {
                "trace_id": trace_id,
                "mas_name": mas_name,
                "benchmark_name": benchmark_name,
                "round": round_num,
                "evaluation_results": {
                    "metric_name": "failure_modes",
                    "scores": result_dict,
                    "idx": idx,
                },
                "metadata": trace_metadata,
                "raw_input": {
                    "history_length": len(str(history)),
                    "question": question,
                },
            }

            local_results_dir.mkdir(parents=True, exist_ok=True)
            output_file = local_results_dir / f"{trace_id}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Results saved: {output_file}")
            print(
                f"✓ Trace {trace_id} ({mas_name}) completed. Langfuse: {pipeline_trace_id}"
            )

        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            logger.error(
                f"Error processing task {trace_id}: {safe_error_msg}", exc_info=True
            )

            if failed_traces_ids:
                new_failed_traces.append(
                    {
                        "task_id": trace_id,
                        "task_index": idx + 1,
                        "mas_name": df.iloc[idx].get("mas_name", "unknown"),
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    }
                )
            else:
                failed_traces.append(
                    {
                        "task_id": trace_id,
                        "task_index": idx + 1,
                        "mas_name": df.iloc[idx].get("mas_name", "unknown"),
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    }
                )

            print(f"\n!  Failed task {idx + 1}/{len(df)}: {trace_id}")
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
                    f.write(f"Trace ID: {failed['task_id']}\n")
                    f.write(f"MAS: {failed.get('mas_name', 'unknown')}\n")
                    f.write(f"Index: {failed['task_index']}/{len(df)}\n")
                    f.write(f"Error Type: {failed['error_type']}\n")
                    f.write(f"Error Message: {failed['error']}\n")
                    f.write("-" * 80 + "\n\n")

            if len(failed_traces) > 0:
                logger.warning(
                    f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n"
                )

        elif new_failed_traces:
            with open(failed_file, "w") as f:
                for failed in new_failed_traces:
                    f.write(f"Task ID: {failed['task_id']}\n")
                    f.write(f"MAS: {failed.get('mas_name', 'unknown')}\n")
                    f.write(f"Index: {failed['task_index']}/{len(df)}\n")
                    f.write(f"Error Type: {failed['error_type']}\n")
                    f.write(f"Error Message: {failed['error']}\n")
                    f.write("-" * 80 + "\n\n")
            if len(new_failed_traces) > 0:
                logger.warning(
                    f"\n!  {len(new_failed_traces)} traces failed. Details saved to: {failed_file}\n"
                )

        if failed_traces_ids:
            logger.info(
                f"Completed evaluation: {len(df) - len(failed_traces)}/{len(df)} successful, {len(failed_traces)} failed"
            )
        elif new_failed_traces:
            logger.info(
                f"Completed evaluation: {len(df) - len(new_failed_traces)}/{len(df)} successful, {len(new_failed_traces)} failed"
            )


if __name__ == "__main__":
    json_file_path = "MAD_human_labelled_dataset.json"

    asyncio.run(
        main(save_folder="mast_human_dataset_eval", json_file_path=json_file_path)
    )
