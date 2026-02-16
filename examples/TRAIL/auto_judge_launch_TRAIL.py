import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(".env")

from automas.meta_agents import GraphGenerator, PoolGenerator
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from automas.utils import get_logger
from maseval import get_langfuse_judge_client

logger = get_logger(__name__)
# sys.path.insert(0, str(Path(__file__).parent))

output_schema = """
**OUTPUT FORMAT - STRICTLY REQUIRED:**
You MUST return ONLY a valid JSON object with exactly this structure:
{
  \"scores\": [
    {
      \"reliability_score\": 0-5 float,
      \"reliability_reasoning\": \"string\",
      \"security_score\": 0-5 float,
      \"security_reasoning\": \"string\",
      \"instruction_adherence_score\": 0-5 float,
      \"instruction_adherence_reasoning\": \"string\",
      \"plan_opt_score\": 0-5 float,
      \"plan_opt_reasoning\": \"string\",
      \"overall\": 0-5 float
    }
  ]
}

**CRITICAL RULES:**
- Return ONLY the JSON object, nothing else
- NO markdown code fences (no ```json or ```)
- NO explanatory text before or after the JSON
- NO additional fields (no confidence, no metadata)
- \"scores\" must be a list with exactly 1 object
- Each *_score must be a number between 0 and 5 (decimals allowed, e.g. 2.5)
- Each *_reasoning must be a single string (concise, 1-3 sentences)
- \"overall\" MUST be the arithmetic mean of:
  reliability_score, security_score, instruction_adherence_score, plan_opt_score
  and rounded to 2 decimals.

**VALID EXAMPLE:**
{\"scores\": [{\"reliability_score\": 1.5, \"reliability_reasoning\": \"Tool usage errors reduced reliability.\", \"security_score\": 5, \"security_reasoning\": \"No security issues detected.\", \"instruction_adherence_score\": 2.5, \"instruction_adherence_reasoning\": \"Core instruction met but sub-instructions missed.\", \"plan_opt_score\": 2.5, \"plan_opt_reasoning\": \"Plan reasonable but adaptation poor.\", \"overall\": 2.88}]}

**INVALID EXAMPLES:**
- ```json{\"scores\": []}```  ← NO markdown fences
- Here is my assessment: {\"scores\": [...]}  ← NO extra text
- {\"score\": \"ideal\"}  ← wrong schema
- {\"scores\": [{\"overall\": 2.88}]}  ← missing required fields
"""

taxonomy = """
# Taxonomy
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

- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it. 
- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors").
- You must provide the output strictly in JSON format as is shown in the template and example below (do not wrap your output in markdown and do not output anything other than the JSON).
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
]
"""

async def main(
    trace_dir: Path, anno_dir: Path, dir_to_save=Path("mad_results"), num_traces: int | None = None
):
    """Run metrics on TRAIL datadset"""

    logger.info(f"Starting TRAIL evaluation:")
    
    pool_gen = PoolGenerator(
        output_schema=output_schema, taxonomy=taxonomy, examples=examples
    )
    graph_gen = GraphGenerator()
    judge_client = get_langfuse_judge_client()
    logger.info("Initialized generators and Langfuse clients")

    # Step 1: Get Langfuse client for downloading traces
    # This uses LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    print("=== Downloading traces from evaluation project ===")

    data_dir = Path(trace_dir)

    if num_traces is not None:
        all_traces = [json.load(open(f)) for f in data_dir.glob("*.json")][:num_traces]
    else:
        all_traces = [json.load(open(f)) for f in data_dir.glob("*.json")]

    def safe_load(f):
        try:
            with open(f, "r", encoding="utf-8") as file:
                return json.load(file)
        except:
            return {}

    anno = [safe_load(f) for f in Path(anno_dir).glob("*.json")]

    def extract_summary(span):
        # Build the summary for each span
        summary = {
            "span_id": span.get("span_id"),
            "span_name": span.get("span_name"),
            "span_attributes": span.get("span_attributes", {}),
            "child_spans": []
        }
        # Recurse into children if present
        for child in span.get("child_spans", []):
            summary["child_spans"].append(extract_summary(child))
        return summary

    # continue processing that was already started
    # ======================
    done_traces = []
    results_dir = dir_to_save

    if results_dir.exists():
        for res in os.listdir(results_dir):
            res_cropped = res.split(".")[0]
            done_traces.append(res_cropped)
    # ======================

    failed_traces = []

    if os.path.exists(dir_to_save):
        for file in os.listdir(dir_to_save):
            if file.endswith(".txt"):
                with open(os.path.join(dir_to_save, file), "r") as f:
                    for line in f:
                        if line.startswith("Task ID:"):
                            failed_traces.append(line.split(":")[1].strip())
 
    new_failed_traces = []

    # Run evaluation for each task
    for idx, task in enumerate(all_traces):
        if task["trace_id"] in done_traces:
            logger.info(f"Task {task['trace_id']} already processed, skipping...")
            continue

        if task["trace_id"] in failed_traces:
            logger.info(f"Task {task['trace_id']} already failed, skipping...")
            continue
        
        logger.info(f"Processing task {idx + 1}/{len(all_traces)}: {task['trace_id']}")
        serializable_results = {}

        try:
            trace_data = task["spans"]
            rewritten_json = [extract_summary(span) for span in trace_data]
            trace_data = str(json.dumps(rewritten_json, indent=2))

            annotation = [
                i for i in anno if i.get("trace_id", "") == task["spans"][0]["trace_id"]
            ]

            if annotation == []:
                continue
            annotation = annotation[0]

            # Prepare metadata for the trace
            trace_metadata = {
                "trace_id": task["trace_id"],
                "ЕTRAIL_trace": task["spans"],
            }

            judge_input = str({"history_for_evaluating": trace_data})
            
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

            # Create a parent span for all evaluations of this task
            # All metric evaluations will be grouped under this span
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{task['trace_id']}",
                input={"trace_id": task["trace_id"]},
                metadata=trace_metadata,
            ) as span:
                # Update the trace with tags (tags are set at trace level, not span level)
                judge_client.update_current_trace(
                    tags=["test", f"task_id:{task['trace_id']}"] #trail_launch_test_30_traces_rewritten_json_gemini_pool_generator
                )
                
                logger.info("Executing evaluation pipeline...")
                result, trace_id = await ainvoke_with_lf(pool, pipeline, judge_input, graph)
                logger.info(f"Pipeline execution completed. Trace ID: {trace_id}")
                
                span.update(output={"result": result, "trace_id": trace_id})
                span.end()

            result_dict = json.loads(result)

            score_obj = result_dict["scores"][0]
            serializable_results["summarizer_score"] = {
                "metric_name": "summarizer_score",
                "scores": [
                    {
                        "item_id": "reliability",
                        "score": score_obj["reliability_score"],
                        "justification": score_obj["reliability_reasoning"],
                    },
                    {
                        "item_id": "security",
                        "score": score_obj["security_score"],
                        "justification": score_obj["security_reasoning"],
                    },
                    {
                        "item_id": "instruction_adherence",
                        "score": score_obj["instruction_adherence_score"],
                        "justification": score_obj["instruction_adherence_reasoning"],
                    },
                    {
                        "item_id": "plan_opt",
                        "score": score_obj["plan_opt_score"],
                        "justification": score_obj["plan_opt_reasoning"],
                    },
                    {
                        "item_id": "overall_score",
                        "score": score_obj["overall"],
                        "justification": (
                            "Reliability: "
                            + score_obj["reliability_reasoning"]
                            + " Security: "
                            + score_obj["security_reasoning"]
                            + " Instruction adherence: "
                            + score_obj["instruction_adherence_reasoning"]
                            + " Plan optimization: "
                            + score_obj["plan_opt_reasoning"]
                        ),
                    },
                ],
            }

            output_dir = dir_to_save
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / Path(f"{task['trace_id']}.json")

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}\n\n")

            print(f"Result: {result}")
            print(f"Langfuse trace ID: {trace_id}")

        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            task_id = task["trace_id"]
            logger.error(f"Error processing task {task_id}: {safe_error_msg}", exc_info=True)
            
            failed_traces.append({
                "task_id": task_id,
                "task_index": idx + 1,
                "error": error_msg,
                "error_type": type(e).__name__
            })
            
            print(f"\n!  Failed task {idx + 1}/{len(all_traces)}: {task_id}")
            print(f"   Error: {error_msg}\n")
            continue 
    
    if new_failed_traces:
        output_dir = dir_to_save
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_file = output_dir / "failed_traces.txt"
        
        with open(failed_file, "w") as f:
            f.write(f"Failed traces: {len(failed_traces)} out of {len(all_traces)}\n")
            f.write("=" * 80 + "\n\n")
            
            for failed in new_failed_traces:
                f.write(f"Task ID: {failed['task_id']}\n")
                f.write(f"Index: {failed['task_index']}/{len(all_traces)}\n")
                f.write(f"Error Type: {failed['error_type']}\n")
                f.write(f"Error Message: {failed['error']}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.warning(f"\n!  {len(new_failed_traces)} traces failed. Details saved to: {failed_file}\n")
        print(f"\n!  {len(new_failed_traces)} traces failed. Details saved to: {failed_file}\n")
    
    logger.info(f"Completed evaluation: {len(all_traces) - len(new_failed_traces)}/{len(all_traces)} successful, {len(new_failed_traces)} failed")


if __name__ == "__main__":
    asyncio.run(
        main(
            trace_dir="/home/user/Desktop/AutoMAS/AutoJudge/trail-benchmark/benchmarking/data/GAIA",
            anno_dir="/home/user/Desktop/AutoMAS/AutoJudge/trail-benchmark/benchmarking/processed_annotations_gaia",
            dir_to_save=Path(__file__).parent / "results" / "trail_30_traces_rewritten_json_gemini_pool_generator",
            num_traces=5,
        )
    )
