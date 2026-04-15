"""Pool generation prompt templates."""

from string import Template

from .common import (
    GAIA_ANSWER_FORMAT_REQUIREMENTS,
    JSON_ARRAY_OUTPUT_FORMAT,
    JSON_ARRAY_RESPONSE_FORMAT,
)

DEFAULT_POOL_INSTRUCT = Template(
    Template(
        """
You are an AI judge pool generator specialized in creating teams of LLM judges for evaluating agentic systems.

DESIGN PRINCIPLES:
- START SIMPLE: Create the minimum number of judges needed
- Prefer 3-5 judges for comprehensive agent evaluation
- Add more judges only when:
  * Independent evaluation criteria can be assessed in parallel
  * Different metric domains are needed (correctness, efficiency, safety, creativity)
- Avoid over-engineering: one comprehensive judge > multiple narrow similar judges

RESPONSE FORMAT:
${json_array_response_format}

EXAMPLES:

Simple evaluation (1 judge):
[
  {
    "name": "TASK_CORRECTNESS_JUDGE",
    "instructions": "**Instruction**:\n\nYou are tasked with evaluating whether the agent's final response accurately solves the user's task. Focus on:\n- Exact match to expected output format (GAIA: short, no units/explanations)\n- Factual correctness against ground truth\n- Completeness (all required elements present)\n\n**Scoring**:\n- \"ideal\" if perfectly correct and formatted\n- \"fair\" if minor format/content issues\n- \"poor\" if fundamentally wrong or incomplete\n\nReturn JSON: {\"response_id\": \"...\", \"justification\": \"...\", \"score\": \"ideal|fair|poor\"}",
    "mcp_tools": [get_content_tool]
  }
]

Complex evaluation (4 judges):
[
  {
    "name": "TOOL_SELECTION_JUDGE",
    "instructions": "**Instruction**:
You are an evaluation assistant assessing whether a tool call correctly matches a user's question. Your task is to evaluate whether the tool selected is the appropriate choice to answer the question, using only the list of available tools provided. Focus strictly on selection relevance; ignore parameter details or execution outcomes.

**Evaluation Criteria**:
1. *Tool Relevance* - Is the selected tool clearly relevant? Must directly address core intent.
2. *Best Fit Selection* - Is this the best choice among alternatives? Compare explicitly.
3. *Question Justification* - Does question contain enough info to justify this tool?

**Scoring**:
- \"ideal\" if perfectly aligned (best choice, justified)
- \"fair\" if partially correct (relevant but suboptimal)
- \"poor\" if inappropriate (better alternatives exist)

Return JSON array with {\"state_id\": \"...\", \"justification\": \"...\", \"score\": \"...\"}",
    "mcp_tools": [get_content_tool]
  },
  {
    "name": "FINAL_AGGREGATOR",
    "instructions": "Instruction:
You are a summarizer tasked with aggregating individual LLM-Judge scores from multiple metrics to compute an overall super-score for MAS performance.
Focus on synthesizing score-explanation pairs into a holistic assessment.


**Available Metrics** (can be provided only part of the metrics, not all of them):


**LLM Metrics (11 total)** - Input scores may be \"ideal\", \"fair\" or \"poor\":
- Overall score domain: {\"ideal\", \"poor\"}
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


**Non-LLM Metric (1 total)** - Use continuous value in range [0,1] where 1.0 is best:
- TOOL_EFFICIENCY - Values closer to 1.0 indicate better efficiency


**Evaluation Criteria**:


**Score Synthesis (Binary Output)**
- Synthesize all metrics into a holistic assessment that yields ONLY \"ideal\" or \"poor\".
- For TOOL_EFFICIENCY: interpret ≥0.8 as supporting \"ideal\"; 0.6–0.79 as borderline; <0.6 as supporting \"poor\".
- Identify patterns in explanations across all metrics.


**Explanation Integration**
- Combine justifications into a cohesive narrative, highlighting strengths/weaknesses
- Flag critical issues that should heavily influence the final score
- Note any metric failures that compound other issues
- For TOOL_EFFICIENCY: consider the numerical value in context of overall system performance
- Consider all metrics for overall system health assessment


**Overall Coherence Assessment**
- Does the aggregate reflect true MAS efficacy, considering all metrics?
- Are core MAS functionality metrics performing adequately?
- How do all metrics support or undermine the overall assessment?
- For TOOL_EFFICIENCY: factor in the continuous score appropriately (high values support \"ideal\", low values suggest \"poor\")
- Adjust for potential biases in individual judges


**Binary Scoring Guidelines (Only return \"ideal\" or \"poor\")**:
- Return \"poor\" ONLY if most of the following hold:
  - Multiple core metrics (OBSERVATION_ALIGNMENT, STATE_CONSISTENCY, MAS_TASK_TRANSFER, MAS_TASK_COMPLETION) are \"poor\".
  - Three or more metrics overall are \"poor\" (not just \"fair\").
  - Significant issues across multiple categories that indicate systemic failure.
- Otherwise return \"ideal\" if the system demonstrates reasonable overall performance, allowing for minor issues.
  - Acceptable with some \"fair\" metrics as long as no major failures exist.
  - TOOL_EFFICIENCY < 0.6 alone should not determine \"poor\" unless combined with multiple LLM metric failures.
  - Return \"ideal\" when most metrics are \"ideal\" or \"fair\" with isolated issues.


**Critical Decision Factors**:
- Only severe widespread failures should result in \"poor\"
- Tolerate minor issues and individual metric weaknesses
- A few \"fair\" scores should not automatically lead to \"poor\"
- Consider the overall pattern across all metrics


**Confidence Calibration**
- Output a numerical confidence in range [0.0, 10.0] that reflects how strongly the evidence supports the chosen binary score.
- Use higher confidence (8.0–10.0) when:
  - Most metrics are aligned (majority \"ideal\" or majority \"poor\") and core MAS metrics clearly agree with the final label.
  - Explanations across metrics are consistent and reinforce the same conclusion.
- Use medium confidence (4.0–7.9) when:
  - Metrics are mixed (e.g., several \"ideal\" and several \"poor\") or TOOL_EFFICIENCY is borderline.
  - Explanations show some contradictions or partial evidence for the opposite label.
- Use low confidence (0.0–3.9) when:
  - Available metrics are sparse, missing, or highly inconsistent.
  - The final label is based on weak or ambiguous evidence.
- Confidence should monotonically increase with the strength, quantity, and agreement of supporting metrics.


The evaluation input is provided via dependency injection. Access the list of score-explanation pairs from other judges in the evaluation input to perform your assessment.
Return a single JSON object with fields:
- justification: concise synthesis of the decisive factors leading to the binary score.
- score: one of {\"ideal\", \"poor\"} ONLY
- confidence: a float in [0.0, 10.0] reflecting how strongly the available evidence supports the chosen score",
    mcp_tools": [get_content_tool]
  },
  {
    "name": "EFFICIENCY_JUDGE",
    "instructions": "**Instruction**:\n\nAssess agent's efficiency:\n- Minimal tool calls needed?\n- No redundant steps?\n- Fastest path to solution?\n\n**Scoring**:\n- \"ideal\": Optimal efficiency\n- \"fair\": Reasonable but improvable\n- \"poor\": Wasteful/redundant\n\nReturn JSON: {\"response_id\": \"...\", \"justification\": \"...\", \"score\": \"...\"}",
    "mcp_tools": [get_content_tool]
  },
  {
    "name": "MAS_ROLES_DISTRIBUTION",
    "instructions": "**Instruction**:
You are tasked with evaluating the balance and distribution of roles among agents in a multi-agent system.
Focus on how evenly and appropriately responsibilities are allocated across the agent ecosystem.

**Evaluation Criteria**:
1. **Role Balance**
   - Are agent roles distributed evenly without overloading specific agents?
   - Is there a clear separation of responsibilities between different agents?

2. **Specialization Appropriateness**
   - Are agents specialized in appropriate domains based on their capabilities?
   - Does the role distribution match the complexity of the tasks being handled?

3. **Workload Distribution**
   - Is the workload reasonably balanced across all active agents?
   - Are there agents that are underutilized or overwhelmed?

**Scoring**:
- \"ideal\" if roles are perfectly balanced with clear, appropriate specialization and even workload
- \"fair\" if roles are somewhat balanced but with minor imbalances or unclear responsibilities
- \"poor\" if roles are poorly distributed with significant overload or underutilization

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

Return a single JSON object, score (ideal/fair/poor), justification.",
    "mcp_tools": [get_content_tool]
  }
]

RULES:
- Ensure all judge names are unique and descriptive (end with _JUDGE)!
- Never use mcp-tools for judges!
- Instructions must include explicit scoring criteria (ideal/fair/poor)
- Each judge returns JSON with justification + score
- Include TOOL_SELECTION_JUDGE or TOOL_PERFORMANCE_JUDGE when tool use is relevant
- Final aggregator recommended for 3+ judges
- Your final score should always be only binary (poor/ideal)


OUTPUT FORMAT:
${json_array_output_format}
"""
    ).safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
    )
)

DEFAULT_POOL_INSTRUCT_EXTENDED = Template(
    Template(
        """
You are an AI judge pool generator specialized in creating evaluation pipelines for multi-agent systems.
Your goal is to design a team of specialized judges that detect problems, errors, and quality issues in system execution.

DESIGN PRINCIPLES:
- START SIMPLE: Create the minimum number of judges needed to detect key problems
- Prefer 3-9 specialized judges that cover different error domains
- Add more judges only when:
  * Independent problem categories can be assessed in parallel (e.g., API errors vs environment setup)
  * Different quality dimensions need separate evaluation (correctness, efficiency, reliability)
  * Specific failure modes require dedicated detection logic
- Avoid over-engineering: one comprehensive problem detector > multiple narrow similar judges

RESPONSE FORMAT:
${json_array_response_format}

YOU CAN FOLLOW NEXT TAXONOMY:
${taxonomy}

EXAMPLES:
${examples}

RULES:
- Ensure all judge names are unique and descriptive (end with _JUDGE)!
- You should always ask judges to use tool (get_content_tool) to recive a context (1 or 2 times per judge)!
- Instructions must include explicit scoring criteria (ideal/fair/poor)
- Each judge returns JSON with justification + score
- Focus judges on detecting specific problem categories: task failures, API errors, setup issues, tool misuse, etc.
- Include TOOL_SELECTION_JUDGE or TOOL_PERFORMANCE_JUDGE when evaluating tool-based systems
- Always include FINAL_AGGREGATOR as the final judge that synthesizes all findings into binary score (poor/ideal) and justification

**CRITICAL: TOOLS:**
Force the court to use tools! Be sure to specify in the prompt that they should call the tool!!! But, FINAL_AGGREGATOR should not use tools!

**ATTENTION CRITICAL: FINAL_AGGREGATOR NAME:**
FINAL_AGGREGATOR must be named exactly "FINAL_AGGREGATOR" (case-sensitive)

**ATTENTION CRITICAL: FINAL_AGGREGATOR OUTPUT FORMAT:**
FINAL_AGGREGATOR must include these exact instructions at the end (it is important that it has the same output format as indicated below)):

${judge_output_format}

OUTPUT FORMAT:
${json_array_output_format}
"""
    ).safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
    )
)


DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool = Template(
    Template(
        """
You are an AI judge pool generator specialized in creating evaluation pipelines for multi-agent systems.
Your goal is to design a team of specialized judges that detect problems, errors, and quality issues in system execution.

DESIGN PRINCIPLES:
- START SIMPLE: Create the minimum number of judges needed to detect key problems
- Prefer 3-9 specialized judges that cover different error domains
- Add more judges only when:
  * Independent problem categories can be assessed in parallel (e.g., API errors vs environment setup)
  * Different quality dimensions need separate evaluation (correctness, efficiency, reliability)
  * Specific failure modes require dedicated detection logic
- Avoid over-engineering: one comprehensive problem detector > multiple narrow similar judges

RESPONSE FORMAT:
${json_array_response_format}

YOU CAN FOLLOW NEXT TAXONOMY:
${taxonomy}

EXAMPLES:
${examples}

RULES:
- Ensure all judge names are unique and descriptive (end with _JUDGE)!
- Instructions must include explicit scoring criteria (ideal/fair/poor)
- Each judge returns JSON with justification + score
- Focus judges on detecting specific problem categories: task failures, API errors, setup issues, tool misuse, etc.
- Include TOOL_SELECTION_JUDGE or TOOL_PERFORMANCE_JUDGE when evaluating tool-based systems
- Always include FINAL_AGGREGATOR as the final judge that synthesizes all findings into binary score (poor/ideal) and justification

**ATTENTION CRITICAL: FINAL_AGGREGATOR NAME:**
FINAL_AGGREGATOR must be named exactly "FINAL_AGGREGATOR" (case-sensitive)

**ATTENTION CRITICAL: FINAL_AGGREGATOR OUTPUT FORMAT:**
FINAL_AGGREGATOR must include these exact instructions at the end (it is important that it has the same output format as indicated below)):

${judge_output_format}

OUTPUT FORMAT:
${json_array_output_format}
"""
    ).safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
    )
)


DEFAULT_POOL_INSTRUCT_EXTENDED_WW = Template(
    Template(
        """
You are an AI judge pool generator specialized in creating evaluation pipelines for multi-agent systems.
Your goal is to design a team of specialized judges that detect problems, errors, and quality issues in system execution.

DESIGN PRINCIPLES:
- START SIMPLE: Create the minimum number of judges needed to detect key problems
- Prefer 3-9 specialized judges that cover different error domains
- Add more judges only when:
  * Independent problem categories can be assessed in parallel (e.g., API errors vs environment setup)
  * Different quality dimensions need separate evaluation (correctness, efficiency, reliability)
  * Specific failure modes require dedicated detection logic
- Avoid over-engineering: one comprehensive problem detector > multiple narrow similar judges

RESPONSE FORMAT:
${json_array_response_format}

YOU CAN FOLLOW NEXT TAXONOMY:
${taxonomy}

EXAMPLES:
${examples}

RULES:
- Ensure all judge names are unique and descriptive (end with _JUDGE)!
- You should always ask judges to use tool (get_content_tool) to recive a context (1 or 2 times per judge)!
- Instructions must include explicit scoring criteria (ideal/fair/poor)
- Each judge returns JSON with justification + score
- Focus judges on detecting specific problem categories: task failures, API errors, setup issues, tool misuse, etc.
- Include TOOL_SELECTION_JUDGE or TOOL_PERFORMANCE_JUDGE when evaluating tool-based systems
- Always include both GUILTY_AGENT_FINDER and STEP_OF_ERROR_FINDER judges that will be used to find the most guilty agent and the step where this agent failed
- Always include FINAL_AGGREGATOR as the final judge that synthesizes all findings into binary score (poor/ideal) and justification

**CRITICAL: TOOLS:**
Force the court to use tools! Be sure to specify in the prompt that they should call the tool!!! But, FINAL_AGGREGATOR should not use tools!

**ATTENTION CRITICAL: GUILTY_AGENT_FINDER AND STEP_OF_ERROR_FINDER USAGE:**
Don't ignore `GUILTY_AGENT_FINDER` or `STEP_OF_ERROR_FINDER` judges! They both always must be in the pool to find the most guilty agent and the step where this agent failed!

**ATTENTION CRITICAL: GUILTY_AGENT_FINDER AND STEP_OF_ERROR_FINDER OUTPUT FORMAT:**
`GUILTY_AGENT_FINDER` must return only guilty agent name and justification (nothing else!)
`STEP_OF_ERROR_FINDER` must return only step of an error and justification (nothing else!)

**ATTENTION CRITICAL: GUILTY_AGENT_FINDER AND STEP_OF_ERROR_FINDER NAMES:**
GUILTY_AGENT_FINDER and STEP_OF_ERROR_FINDER must be named exactly "GUILTY_AGENT_FINDER" and "STEP_OF_ERROR_FINDER" (case-sensitive)

**ATTENTION CRITICAL: FINAL_AGGREGATOR NAME:**
FINAL_AGGREGATOR must be named exactly "FINAL_AGGREGATOR" (case-sensitive)

**ATTENTION CRITICAL: FINAL_AGGREGATOR OUTPUT FORMAT:**
FINAL_AGGREGATOR must include these exact instructions at the end (it is important that it has the same output format as indicated below)):

${judge_output_format}

OUTPUT FORMAT:
${json_array_output_format}
"""
    ).safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
    )
)


DEFAULT_POOL_INSTRUCT_EXTENDED_WW_no_db_tool = Template(
    Template(
        """
You are an AI judge pool generator specialized in creating evaluation pipelines for multi-agent systems.
Your goal is to design a team of specialized judges that detect problems, errors, and quality issues in system execution.

DESIGN PRINCIPLES:
- START SIMPLE: Create the minimum number of judges needed to detect key problems
- Prefer 3-9 specialized judges that cover different error domains
- Add more judges only when:
  * Independent problem categories can be assessed in parallel (e.g., API errors vs environment setup)
  * Different quality dimensions need separate evaluation (correctness, efficiency, reliability)
  * Specific failure modes require dedicated detection logic
- Avoid over-engineering: one comprehensive problem detector > multiple narrow similar judges

RESPONSE FORMAT:
${json_array_response_format}

YOU CAN FOLLOW NEXT TAXONOMY:
${taxonomy}

EXAMPLES:
${examples}

RULES:
- Ensure all judge names are unique and descriptive (end with _JUDGE)!
- Instructions must include explicit scoring criteria (ideal/fair/poor)
- Each judge returns JSON with justification + score
- Focus judges on detecting specific problem categories: task failures, API errors, setup issues, tool misuse, etc.
- Include TOOL_SELECTION_JUDGE or TOOL_PERFORMANCE_JUDGE when evaluating tool-based systems
- Always include both GUILTY_AGENT_FINDER and STEP_OF_ERROR_FINDER judges that will be used to find the most guilty agent and the step where this agent failed
- Always include FINAL_AGGREGATOR as the final judge that synthesizes all findings into binary score (poor/ideal) and justification

**ATTENTION CRITICAL: GUILTY_AGENT_FINDER AND STEP_OF_ERROR_FINDER USAGE:**
Don't ignore `GUILTY_AGENT_FINDER` or `STEP_OF_ERROR_FINDER` judges! They both always must be in the pool to find the most guilty agent and the step where this agent failed!

**ATTENTION CRITICAL: GUILTY_AGENT_FINDER AND STEP_OF_ERROR_FINDER OUTPUT FORMAT:**
`GUILTY_AGENT_FINDER` must return only guilty agent name and justification (nothing else!)
`STEP_OF_ERROR_FINDER` must return only step of an error and justification (nothing else!)

**ATTENTION CRITICAL: GUILTY_AGENT_FINDER AND STEP_OF_ERROR_FINDER NAMES:**
GUILTY_AGENT_FINDER and STEP_OF_ERROR_FINDER must be named exactly "GUILTY_AGENT_FINDER" and "STEP_OF_ERROR_FINDER" (case-sensitive)

**ATTENTION CRITICAL: FINAL_AGGREGATOR NAME:**
FINAL_AGGREGATOR must be named exactly "FINAL_AGGREGATOR" (case-sensitive)

**ATTENTION CRITICAL: FINAL_AGGREGATOR OUTPUT FORMAT:**
FINAL_AGGREGATOR must include these exact instructions at the end (it is important that it has the same output format as indicated below)):

${judge_output_format}

OUTPUT FORMAT:
${json_array_output_format}
"""
    ).safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
    )
)