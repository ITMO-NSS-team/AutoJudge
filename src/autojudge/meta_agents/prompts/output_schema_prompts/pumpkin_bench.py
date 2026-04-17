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
- Here is my assessment: {\"score\": \"ideal\"}  ← NO extra text
- {\"score\": \"IDEAL\"}  ← must be lowercase"""
