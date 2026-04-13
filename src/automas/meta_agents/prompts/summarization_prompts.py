"""Trace summarization prompt templates."""

TRACE_SUMMARIZATION_PROMPT = """
Analyze the multi-agent system execution trace and create a concise summary.

**REQUIRED OUTPUT:**

1. **Step-by-Step Log**: Extract ALL execution steps chronologically
   - Format: "Step N. [Agent] action_type: description"
   - Include: tool calls, reasoning, communication, errors
   - Examples:
     * "Step 1. ResearchAgent tool_call: web-search 'Tokyo population'"
     * "Step 2. DataAgent tool_call: read 'data.csv' (Error: FileNotFoundError)"

2. **Brief Overview**:
   - Original task
   - System type (decentralized/ReAct/hierarchical)
   - Total agents
   - Key agents and their roles

**JSON OUTPUT FORMAT:**

{
  "step_by_step_log": [
    {
      "step_number": 1,
      "agent_name": "string",
      "action_type": "tool_call|reasoning|communication|output|error",
      "description": "string - what was done",
      "error_message": "string or null",
      "key_data": "string or null"
    }
  ],
  "task_overview": {
    "original_query": "string",
    "system_type": "string",
    "total_agents": "number"
  },
  "agents_summary": [
    {
      "agent_name": "string",
      "role": "string",
      "key_contributions": "string",
      "issues": "string or null"
    }
  ]
}

Keep descriptions concise. Extract ALL steps from trace.
"""


STEP_BY_STEP_SUMMARIZATION_PROMPT = """
Analyze the multi-agent system execution trace and produce a step-by-step summary.

You will be given a `TRACE_ID` above the trace. You MUST use it when forming step ids.

**REQUIRED OUTPUT:**

- Extract ALL steps chronologically.
- Each step must include:
  - `id`: step id in the format `<trace_id>_<step>` (e.g., `abc123_1`)
  - `name`: agent name (or "System" if unclear)
  - `role`: agent role (best-effort; "Unknown" if unclear)
  - `content_summary`: concise summary of what happened in this step

**JSON OUTPUT FORMAT (return ONLY this JSON):**

{
  "step_by_step_summary": [
    {
      "id": "<trace_id>_1",
      "name": "string",
      "role": "string",
      "content_summary": "string"
    }
  ]
}

Keep `content_summary` concise and factual. Do not add extra top-level keys.
"""


STEPS_BATCH_SUMMARIZATION_PROMPT = """
Analyze a batch of multi-agent system execution steps and produce a summary for each one.

You will receive one or more states, each preceded by a delimiter line:
  --- state_id: <VALUE> ---
You may also receive a **summary of previous states** — use it only as context to better understand the current batch; do NOT re-summarize those earlier states.

**REQUIRED OUTPUT:**

For **every** provided state produce exactly one summary entry containing:
  - `id`: copy the `state_id` value from the delimiter line **character-for-character**. Example: if the delimiter says `--- state_id: abc123_7 ---`, the `id` MUST be `abc123_7`. NEVER substitute the agent name or any other string.
  - `name`: the agent's name as it appears in the state, WITHOUT any numeric suffix. For example if the state mentions "VideoSearchAgent" in step 5, the name is "VideoSearchAgent", NOT "VideoSearchAgent_5".
  - `role`: the message role as it appears in the state (e.g. "system", "user", "assistant", "tool"). Use "Unknown" only if truly unclear.
  - `content_summary`: concise, factual summary of what happened in this step.

Return the entries in the same order as the input states.

**RULES**:
* Produce exactly one summary per input state — no more, no less.
* `id` MUST be the exact `state_id` from the delimiter. Do NOT invent, modify, or derive IDs from agent names.
* `name` MUST be the bare agent name without step/index suffixes.
* DO NOT re-summarize states from the previous summary. Only summarize the states listed under "States to summarize".
* Keep `content_summary` concise and factual.
"""
