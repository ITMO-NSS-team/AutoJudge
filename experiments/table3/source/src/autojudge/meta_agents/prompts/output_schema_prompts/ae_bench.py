taxonomy = """
You must analyze the trajectory using AgentErrorTaxonomy.

Identify the FIRST CRITICAL FAILURE (root cause), defined as:
The earliest step after which the task is unlikely to succeed.

For this failure, determine:

1) critical_failure_step:
   - The step index (1, 2, 3...) where the root cause occurs

2) critical_failure_module:
   - One of the following modules:
     - "plan"        (bad or inefficient planning, ignoring constraints)
     - "reflection"  (misinterpretation, wrong self-evaluation, wrong conclusions)
     - "action"      (incorrect or invalid action execution)
     - "memory"      (missing, hallucinated, or unused information)
     - "system"      (environment limits, step limit, external constraints)

3) failure_type (fine-grained classification):

   For "plan":
     - "inefficient_plan"
     - "constraint_ignorance"

   For "reflection":
     - "outcome_misinterpretation"
     - "progress_misjudge"

   For "action":
     - "invalid_action"
     - "wrong_argument"

   For "memory":
     - "missing_information"
     - "hallucinated_information"

   For "system":
     - "step_limit"
     - "environment_failure"

IMPORTANT:
- Focus on ROOT CAUSE, not the final failure
- Select ONLY ONE critical failure (the earliest decisive mistake)
"""

output_schema = """
**OUTPUT FORMAT - STRICTLY REQUIRED:**
You must return ONLY a valid JSON object with exactly these four fields:

{
  "critical_failure_step": "integer (1, 2, 3...)",
  "critical_failure_module": "plan | reflection | action | memory | system",
  "failure_type": "one of predefined types",
  "reason": "short explanation of why this is the root cause"
}

**CRITICAL RULES:**
- Return ONLY the JSON object, nothing else
- NO markdown code fences
- NO additional text before or after
- NO additional fields

**STEP RULE:**
- This is the position in 'history_for_evaluating'
- Use ONLY integers: "1", "2", "3", ...

**MODULE RULE:**
- Must be EXACTLY one of:
  "plan", "reflection", "action", "memory", "system"

**FAILURE TYPE RULE:**
- Must match the module:

plan:
  - "inefficient_plan"
  - "constraint_ignorance"

reflection:
  - "outcome_misinterpretation"
  - "progress_misjudge"

action:
  - "invalid_action"
  - "wrong_argument"

memory:
  - "missing_information"
  - "hallucinated_information"

system:
  - "step_limit"
  - "environment_failure"

**VALID EXAMPLE:**
{
  "critical_failure_step": "5",
  "critical_failure_module": "plan",
  "failure_type": "inefficient_plan",
  "reason": "The agent repeatedly explores low-probability locations without updating its search strategy."
}
"""
