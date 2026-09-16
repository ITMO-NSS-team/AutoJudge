# TRAIL taxonomy

## Reasoning Errors
- Hallucinations: Language-only
- Hallucinations: Tool-related (fabricating tool outputs/capabilities)
- Information Processing: Poor Information Retrieval (tried to find information that was not relevant to the task)
- Information Processing: Tool Output Misinterpretation (made assumptions about the tool output or used the tool output in an incorrect context)
- Decision Making: Incorrect Problem Identification (misunderstood the overall task or the local task)
- Decision Making: Tool Selection Errors (used the wrong tool for the task)
- Output Generation: Formatting Errors (errors with formatting and execution of code or structuring of output in a specific format)
- Output Generation: Instruction Non-compliance (failed to perform the task provided and instead did something else)

## System Execution Errors
- Configuration: Tool Definition Issues (the tool was not defined correctly or is inconsistent with its description)
- Configuration: Environment Setup Errors (permission problems and inability to access resources or API keys)
- API Issues: Rate Limiting (429)
- API Issues: Authentication Errors (401/403)
- API Issues: Service Errors (500)
- API Issues: Resource Not Found (404)
- Resource Management: Resource Exhaustion (includes memory overflow)
- Resource Management: Timeout Issues (the system took too long to respond)

## Planning and Coordination Errors
- Context Management: Context Handling Failures (window overflow, state tracking or forgetting important context)
- Context Management: Resource Abuse (called the tool excessively due to memory issues)
- Task Management: Goal Deviation (the system deviated from the task or the subtask)
- Task Management: Task Orchestration (subtask coordination between agents and progress monitoring)

Mark only the first instance of each error as its location, except Resource Abuse:
mark the last instance. Cite span or step IDs as location evidence.
