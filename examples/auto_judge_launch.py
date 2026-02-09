import asyncio

from automas.meta_agents import GraphGenerator, PoolGenerator
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from maseval import get_langfuse_download_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(".env")

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

async def main(name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097"):
    pool_gen = PoolGenerator(output_schema=output_schema, taxonomy=taxonomy, examples=examples)
    graph_gen = GraphGenerator()
    lf = get_langfuse_download_client()

    traces_page1 = lf.api.trace.list(name=name, limit=50, page=1)
    
    parsed_traces = [parse_langfuse_task(lf.api.trace.get(task.id)) for task in traces_page1.data]
    
    for idx, query in tqdm(enumerate(parsed_traces)):
        judge_input = str({"query": query.user_query, "history_for_evaluating": query.agent_states})
        pool = await pool_gen.create_pool(judge_input)
        graph = await graph_gen.create_graph(pool, judge_input)

        builder = PipelineBuilder()
        pipeline = builder.create_from_pool(pool, graph).build()
        pipeline.to_mermaid_lr(visualize=True)

        result, trace_id = await ainvoke_with_lf(pool, pipeline, judge_input, graph)

        print(f"Result: {result}")
        print(f"Langfuse trace ID: {trace_id}")
        print(f"Launch № {idx} from {len(parsed_traces)}")


if __name__ == "__main__":
    asyncio.run(main())
