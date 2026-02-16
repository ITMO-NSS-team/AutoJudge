import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import json
from pathlib import Path

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

from automas.meta_agents import GraphGenerator, PoolGenerator
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from automas.utils import get_logger
from maseval import get_langfuse_judge_client
import json
import pandas as pd

logger = get_logger(__name__)

taxonomy = """
├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related (fabricating tool outputs/capabilities)
│   ├── Information Processing
│   │   ├── Poor Information Retrieval (Tried to find information that was not relevant to the task)
│   │   └── Tool Output Misinterpretation (Made assumptions about the tool output or used the tool output in an incorrect context)
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification (Misunderstood the overall task or the local task)
│   │   ├── Tool Selection Errors (Used the wrong tool for the task)
│   └── Output Generation
│       ├── Formatting Errors (Errors with formatting and execution of code or structuring of output in a specific format)
│       └── Instruction Non-compliance (Failed to perform the task provided and instead did something else)
├── System Execution Errors
│   ├── Configuration
│   │   ├── Tool Definition Issues (The tool was not defined correctly by the user or contains some errors that make it inconsistent with its description. For example, web search tool was defined as a calculator tool)
│   │   └── Environment Setup Errors (includes permission problems and inability to access resources or API keys)
│   ├── API Issues
│   │   ├── Rate Limiting (Like 429)
│   │   ├── Authentication Errors (Like 401/403)
│   │   ├── Service Errors (Like 500)
│   │   └── Resource Not Found (Like 404)
│   └── Resource Management
│       ├── Resource Exhaustion (includes memory overflow)
│       └── Timeout Issues (The system took too long to respond)
├── Planning and Coordination Errors
│    ├── Context Management
│    │   ├── Context Handling Failures (includes window overflow and state tracking or forgetting important context)
│    │   └── Resource Abuse (Called the tool excessively due to memory issues)
│    └── Task Management
│        ├── Goal Deviation (The system deviated from the task or the subtask)
│        └── Task Orchestration (includes subtask coordination between agents and progress monitoring)
"""

output_schema = """
**OUTPUT FORMAT - STRICTLY REQUIRED:**
{{
    "errors": [
        {{
            "category": "[INSERT ERROR CATEGORY FROM TAXONOMY HERE]", # The category of the error
            "location": "[INSERT LOCATION OF ERROR HERE]", # The location of the error in the trace (span id)
            "evidence": "[INSERT EXTRACTED EVIDENCE HERE]",
            "description": "[INSERT DETAILED ERROR DESCRIPTION HERE]",
            "impact": "[INSERT IMPACT HERE]" # The impact of the error (HIGH, MEDIUM, LOW)
        }},
        ... # more errors
    ],
    "scores": [
        {{
            "reliability_score": 3, # The reliability score of the system (0-5)
            "reliability_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the reliability score
            "security_score": 5, # The security score of the system (0-5)
            "security_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the security score
            "instruction_adherence_score": 4, # The instruction adherence score of the system (0-5)
            "instruction_adherence_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the instruction adherence score
            "plan_opt_score": 3, # The plan optimality score of the system (0-5)
            "plan_opt_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the plan optimality score
            "overall": 3.75 # The overall score of the system (0-5)
        }}
    ]
}}

Example output:

{{
    "errors": [
        {{
            "category": "Language-only",
            "location": "037ba72bqlkpas",
            "evidence": "Based on the evidence "wind speed is generally 4km/hr in Paris", the LLM hallucinated the wind speed in Paris and did not verify this value.",
            "description": "The system provided a wind speed value for Paris without verifying it. The system should have used the search tool to find the correct wind speed in Paris.",
            "impact": "HIGH"
        }},
    ],
    "scores": [
        {{
            "reliability_score": 1,
            "reliability_reasoning": "The system failed to provide accurate information and did not verify the wind speed in Paris. The system should have used the search tool to find the correct wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 2,
            "instruction_adherence_reasoning": "The system did not follow instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 2,
            "plan_opt_reasoning": "The system's plan was not optimal because it did not incorporate the use of search tool effectively to validate information",
            "overall": 2.5
        }}
    ]
}}

If the trace has no errors, the output should be:
{{
    "errors": [],
    "scores": [
        {{
            "reliability_score": 5,
            "reliability_reasoning": "The system provided accurate information and verified the wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 5,
            "instruction_adherence_reasoning": "The system followed instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 5,
            "plan_opt_reasoning": "The system's plan was optimal because it incorporated the use of search tool effectively to validate information",
            "overall": 5
        }}
    ]
}}

- Ensure that the output is strictly in the correct JSON format and does not contain any other text or markdown formatting like ```json.
- Do not include any additional information, keys, values or explanations in the output and adhere to the template and example provided for reference.
- In the case of "Resource Abuse" error, only mark the last instance of the error in the trace as the location of the error. For all other errors, you must mark the first instance of the error in the trace as the location of the error.
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


async def main(save_folder: str, df):
    logger.info(f"===Starting Who&When evaluation===")
    
    pool_gen = PoolGenerator(output_schema=output_schema, taxonomy=taxonomy, examples=examples)
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
        if df.iloc[idx]["trace_id"] in done_traces:
            trace_id = df.iloc[idx]["trace_id"]
            logger.info(f"Task {trace_id} already processed, skipping...")
            continue
        
        task = df.iloc[idx]["trace_id"]
        logger.info(f"Processing task {idx + 1}/{len(df)}: {task}")
        serializable_results = {}
        
        try:
            trace_data = {
            "history": df.iloc[idx]["spans"],
            "question": "You should evaluate the trace.",
            "task_id": df.iloc[idx]["trace_id"],
            "trace_id": df.iloc[idx]["trace_id"],
            }
            q = trace_data["question"][:100]
            logger.debug(f"Parsed task query: {q}...")
            
            trace_metadata = {
                "task_id": trace_data["task_id"],
                "trace_id": trace_data["trace_id"]
                }
            
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
                name=f"evaluate_task_{task}",
                input={"task_id": df.iloc[idx]["trace_id"], "trace_id": df.iloc[idx]["trace_id"]},
                metadata=trace_metadata,
            ) as span:
                judge_client.update_current_trace(
                    tags=["trail_30", f"task_id:{task}"]
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
                                    "score": result_dict,
                                    "idx": idx,
                                    "task_id": df.iloc[idx]["trace_id"],
                                    "filename": df.iloc[idx]['filename']
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


def create_gaia_dataframe():
    gaia_dir = Path("trail-benchmark/benchmarking/data/GAIA")
    
    json_files = list(gaia_dir.glob("*.json"))
    
    if not json_files:
        return None
    
    files_to_process = json_files[:30]
    
    data_list = []
    
    for i, file_path in enumerate(files_to_process, 1):
        try:
            print(f"[{i:2d}/30] {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['filename'] = file_path.name
            data['file_path'] = str(file_path)
            
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
    asyncio.run(main(
        save_folder="who_and_when_gpt5_mini",
        df=df
    ))
