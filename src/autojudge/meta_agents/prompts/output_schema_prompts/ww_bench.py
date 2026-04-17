
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