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
