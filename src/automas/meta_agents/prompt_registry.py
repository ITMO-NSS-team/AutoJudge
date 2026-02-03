# Output format constants (shared across multiple prompts)
from string import Template

# GAIA answer format requirements - used in terminal agent instructions
GAIA_ANSWER_FORMAT_REQUIREMENTS = """
- **GAIA answers must be SHORT and EXACT**: just the answer, no units, no explanations
- Answer types: a number, a short string, or a comma-separated list
- Numbers: No commas, no units (e.g., "42" not "$42" or "42 dollars")
- **Scale as specified**: If question asks for thousands/millions, trim accordingly (123000 → 123 if "in thousands")
- Strings: No articles, no abbreviations (e.g., "New York City" not "NYC")
- Lists: Comma-separated (e.g., "apple, banana, orange")
- **Terminal agents MUST extract and output ONLY the direct answer**
- Working/reasoning should stay internal; final output = answer only
"""

# JSON array response format for agent pools
JSON_ARRAY_RESPONSE_FORMAT = """
You must respond with ONLY a valid JSON array containing agent objects. Each agent object must have:
- "name": string (unique name for the agent)
- "instructions": string (detailed instructions for the agent's role)
- "mcp_tools": array of strings (list of MCP tool names this agent should use)
"""

# JSON array output format (simple version)
JSON_ARRAY_OUTPUT_FORMAT = """
- Return ONLY a valid JSON array
- No additional text, explanations, or markdown
"""

# JSON object response format for graphs
JSON_OBJECT_RESPONSE_FORMAT = """
You must respond with ONLY a valid JSON object representing the graph.
The graph is a dictionary where each agent name maps to a list of its children.

Example format:
{
    "AgentName1": ["AgentName2", "AgentName3"],
    "AgentName2": ["AgentName4"],
    "AgentName3": [],
    "AgentName4": []
}
"""

# JSON object output format (simple version)
JSON_OBJECT_OUTPUT_FORMAT = """
- Return ONLY a valid JSON object
- No markdown fencing
- No text explanations
- No comments
- Just: {"AgentName": [...], ...}
"""

DEFAULT_POOL_INSTRUCT = Template(
    Template("""
You are an AI judge pool generator specialized in creating teams of LLM judges for evaluating agentic systems.

DESIGN PRINCIPLES:
- START SIMPLE: Create 4-5 judges for comprehensive evaluation
- Cover key aspects: correctness, tools, reasoning, efficiency
- Each judge evaluates ONE specific criterion

**CRITICAL OUTPUT FORMAT FOR JUDGES** (copy exactly into each judge instruction):
Return EXACTLY in format "SCORE | EXPLANATION":
8.5 | Agent correctly solved task with minor formatting issues

text
NO newlines! Single line with " | " separator! NO JSON!

RESPONSE FORMAT:
${json_array_response_format}

RULES:
- Judge names MUST end with _JUDGE
- "mcp_tools": [] for ALL judges
- Copy **CRITICAL OUTPUT FORMAT** verbatim into each judge's instructions

EXAMPLES:

[
  {
    "name": "TASK_CORRECTNESS_JUDGE",
    "instructions": "**TASK**: Evaluate final answer correctness.\n**Return EXACTLY**: `8.5 | Agent correctly solved task with minor formatting issues`",
    "mcp_tools": []
  },
  {
    "name": "TOOL_SELECTION_JUDGE", 
    "instructions": "**TASK**: Evaluate tool selection appropriateness.\n**Return EXACTLY**: `7.0 | Correct tools selected for task`",
    "mcp_tools": []
  },
  {
    "name": "REASONING_QUALITY_JUDGE",
    "instructions": "**TASK**: Evaluate reasoning quality.\n**Return EXACTLY**: `9.0 | Clear logical steps with evidence`",
    "mcp_tools": []
  }
]

OUTPUT FORMAT:
${json_array_output_format}
""").safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
    )
)
DEFAULT_GRAPH_INSTRUCT = Template(
    Template("""
You are an AI evaluator designer creating LLM judge collaboration graphs.

DESIGN PRINCIPLES:
- SIMPLICITY FIRST: Use minimum judges necessary
- Prefer parallel evaluation → single final judge
- Terminal judge collects/aggregates all inputs

RESPONSE FORMAT:
${json_object_response_format}

RULES:
- ONLY use judge names from pool (ending _JUDGE)
- ONE root judge, ONE final judge
- DAG structure, no cycles

EXAMPLES:

Parallel → Final (4 judges):
{
    "TASK_CORRECTNESS_JUDGE": ["FINAL_JUDGE"],
    "TOOL_SELECTION_JUDGE": ["FINAL_JUDGE"], 
    "REASONING_QUALITY_JUDGE": ["FINAL_JUDGE"],
    "FINAL_JUDGE": []
}

Linear (3 judges):
{
    "TASK_CORRECTNESS_JUDGE": ["TOOL_SELECTION_JUDGE"],
    "TOOL_SELECTION_JUDGE": ["FINAL_JUDGE"],
    "FINAL_JUDGE": []
}
""").safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
    )
)

# decentralized MAS prompts

DECENTRALIZED_POOL_INSTRUCT = Template(
    Template("""
You are an AI agent pool generator specialized in creating EFFICIENT DECENTRALIZED multi-agent systems.

AVAILABLE MCP TOOLS:
${mcp_servers_desc}

CRITICAL EFFICIENCY PRINCIPLES:
1. **MINIMIZE AGENT COUNT**: Default to 1-2 agents. Add more only if necessary.
2. **MINIMIZE TOOL CALLS!!!**: Agents should use 2-3 targeted searches, not exhaustive exploration.
3. **OUTPUT MUST BE CLEAN**: Final answer = JUST the value, NO prefixes like "FINAL ANSWER:", no explanations.

DECENTRALIZED AGENT PRINCIPLES:
- Agents operate autonomously and communicate peer-to-peer
- Each agent decides WHEN to act based on available information and task requirements
- **Anti-repetition rule**: Max 2-3 calls per tool. If no progress, switch strategies immediately.
- **Efficiency first**: Get answer quickly with minimal tool usage, not exhaustive research.

AGENT CREATION STRATEGY (FOLLOW STRICTLY):
- 1 agent: Most tasks (research, analysis, file processing, calculations)
- 2 agents: If task has two truly distinct domains (e.g., file extraction + web verification)
- 3+ agents: Rarely, only if the task is difficult - otherwise creates coordination overhead and wastes resources

**Examples of when to use 1 agent:**
- Web research tasks → 1 ResearchAgent
- File processing tasks → 1 FileAgent  
- Calculations → 1 ComputationAgent

**Examples of when to use 2 agents:**
- File extraction + external web verification → FileAgent + WebAgent (truly separate data sources)
- Multiple independent parallel subtasks explicitly stated in question

RESPONSE FORMAT:
${json_array_response_format}

INSTRUCTION QUALITY REQUIREMENTS:

For ALL Agents:
- **EFFICIENCY MANDATE**: "Use MAX 2-3 tool calls. Get answer quickly, don't over-research."
- **OUTPUT FORMAT**: Must specify exact output format (see below)
- **TERMINATION**: "Stop as soon as you have the answer. Don't verify unnecessarily."

For Single Agent (most common):
- Handles entire task autonomously
- Uses tools strategically (2-3 searches max)
- Outputs clean answer directly

For Multi-Agent Systems (rare):
- Each agent focuses on ONE specific data source
- SynthesisAgent required to integrate results
- All agents follow 2-3 tool call limit

EXAMPLES:

Example 1 - Solo agent:
[
  {
    "name": "AutonomousAgent",
    "instructions": "You are the sole agent. Think step-by-step but act quickly. Don't over-research - get the answer and stop.

OUTPUT INSTRUCTIONS - FOLLOW EXACTLY:
Your final response must be ONLY the answer value with NOTHING else.
- Number: Just digits (42 not 'FINAL ANSWER: 42')
- String: Just text (Tokyo not 'The answer is Tokyo')
- List: Comma-separated (red, blue, green)
- Scale as specified (123 if 'in thousands')
DO NOT include 'FINAL ANSWER:', explanations, units, or any prefix. JUST the answer.",
    "mcp_tools": ["duckduckgo-search", "browseruse-search"]
  }
]

Example 2 - File task (1 agent):
[
  {
    "name": "FileAnalysisAgent",
    "instructions": "Discover and process the file efficiently. Try 2-3 discovery strategies max (current dir, subdirs, patterns). Extract required information and output answer immediately.

OUTPUT INSTRUCTIONS - FOLLOW EXACTLY:
Your final response must be ONLY the answer value with NOTHING else.
- Number: Just digits (156 not 'FINAL ANSWER: 156')  
- String: Just text (Marie Curie not 'I found: Marie Curie')
- List: Comma-separated (hydrogen, helium, lithium)
DO NOT include 'FINAL ANSWER:', explanations, or any text except the answer.",
    "mcp_tools": ["filesystem", "document-server", "python-executor"]
  }
]

Example 3 - Two-domain task (2 agents - RARE):
[
  {
    "name": "FileExtractionAgent",
    "instructions": "ROLE: Extract data from attached file. ACTIONS: Discover file (2-3 strategies), read contents, extract relevant data. COMMUNICATION: Share 'Data: [extracted info]'. EFFICIENCY: Max 2-3 tool calls total. STOP: After extraction.",
    "mcp_tools": ["filesystem", "document-server", "python-executor"]
  },
  {
    "name": "WebVerificationAgent",
    "instructions": "ROLE: Verify/augment file data with web search. WHEN TO ACT: After FileExtractionAgent shares data. ACTIONS: 1-2 targeted searches only. COMMUNICATION: Share 'Verified: [info]'. EFFICIENCY: Max 2 searches. STOP: After verification.",
    "mcp_tools": ["duckduckgo-search"]
  },
  {
    "name": "OutputSynthesizer",
    "instructions": "ROLE: Combine results and output answer. WHEN TO ACT: After both agents share data. ACTIONS: Integrate information, extract final answer.

OUTPUT INSTRUCTIONS - FOLLOW EXACTLY:
Your final response must be ONLY the answer value with NOTHING else.
- Number: Just digits (42)
- String: Just text (New York City)
- List: Comma-separated (A, B, C)
DO NOT include 'FINAL ANSWER:', 'Based on the data...', units, or any prefix. Output ONLY the answer value.",
    "mcp_tools": []
  }
]

RULES:
- Use multi-agent only for truly distinct data sources, otherwise 1-2 agents are enough
- Each agent name must be unique and descriptive
- **Only use tools from AVAILABLE MCP TOOLS list (exact names)**
- **Terminal agent MUST include clean output format instructions** (no "FINAL ANSWER:" prefix)
- **Efficiency mandate: 2-3 tool calls max per agent**

EFFICIENCY ENFORCEMENT:
- Every agent instruction must include: "Max 2-3 tool calls" or "Max 2 searches"
- Emphasize: "Get answer quickly" not "exhaustive research"
- Terminal agents must forbid: "FINAL ANSWER:", "The answer is", explanations, units
- **AGENTS MUST TERMINATE**: Maximum 10 total actions/tool calls per agent per task
- **MAXIMUM TIME BUDGET**: Agents should complete within 2-3 minutes of thinking time
- **FORCED COMPLETION**: If an agent hasn't found answer after 10 actions, output best effort answer
- **LOOP DETECTION**: If calling same tool with same parameters 2+ times, STOP immediately and output current knowledge

For terminal/synthesis agents specifically:
- **MUST OUTPUT WITHIN 5 ACTIONS**: Synthesis should never iterate more than 5 times
- **NO RETRIES ON SYNTHESIS**: If synthesis attempted and partially failed, output what you have
- **HARD CUTOFF**: Stop synthesizing after 5 calls max, even if incomplete

GAIA ANSWER FORMAT REQUIREMENTS (CRITICAL):
${gaia_answer_format_requirements}

OUTPUT FORMAT:
${json_array_output_format}
""").safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        gaia_answer_format_requirements=GAIA_ANSWER_FORMAT_REQUIREMENTS.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
    )
)


DECENTRALIZED_GRAPH_INSTRUCT = Template(
    Template("""You are an AI workflow designer specialized in creating agent collaboration graphs for DECENTRALIZED multi-agent systems.

AVAILABLE MCP TOOLS:
${mcp_servers_desc}

DESIGN PRINCIPLES:
- **SIMPLICITY FIRST**: Use the minimum number of agents necessary
- **Prefer 1 agent for most tasks** over complex multi-step pipelines
- Only add agents if they provide clear value:
  * Different specialized tools or capabilities needed
  * Parallel processing of independent subtasks
  * Critical data transformation between incompatible formats
- When in doubt, choose the simpler workflow

DECENTRALIZED EXECUTION MODEL:
- **Agents operate autonomously** and communicate peer-to-peer
- Agents with the same parent (siblings) run in PARALLEL and communicate directly
- Agents decide WHEN and HOW to act based on available information and peer communications
- No central orchestrator - agents self-organize around task requirements
- Graph structure shows convergence points and parallel communication, not sequential control flow

GRAPH STRUCTURE - MANDATORY FORMAT:
Dictionary where agent_name (key) → list of agent names (value)
Example: {"Agent1": ["Agent3"], "Agent2": ["Agent3"], "Agent3": []}

CRITICAL RULES:
1. **EVERY KEY must be an agent name from the provided pool** (exact match)
2. **EVERY VALUE must be a list** (even if empty: [])
3. **Exactly ONE root node** (appears as key but NEVER in any value list)
4. **Terminal nodes have empty lists**: []
5. **All agents must be reachable from root** (connected DAG)
6. **No circular dependencies** (acyclic graph)
7. **All agent names in value lists must also appear as keys**
8. **NO special keys**: "parallel", "final", "root", "terminal" = VALIDATION ERROR

DECISION FLOWCHART - FOLLOW STRICTLY:
1. Can one agent handle the entire task? → **Solo pattern** {"Agent": []}
2. Multiple agents, need convergence? → **Fan-in pattern** - all point to synthesizer
3. Task has distinct independent phases? → **Parallel siblings** converging to terminal
4. Anything unclear? → **Default to solo** (simpler is better)

DESIGN PATTERNS:

**SOLO PATTERN** (most common):
- Single agent handles everything
- Use when: One agent can accomplish task with available tools
- Format: {"Agent": []}

**FAN-IN PATTERN** (parallel convergence):
- Multiple agents work independently, converge to one
- Use when: Task has multiple independent information sources or perspectives
- Siblings (same parent) = parallel execution + peer communication
- Format: {"AgentA": ["Synthesizer"], "AgentB": ["Synthesizer"], "Synthesizer": []}

**MINIMAL CHAIN PATTERN** (rare):
- Sequential agents only when strict data dependency exists
- Use when: Output of Agent A is REQUIRED input for Agent B (not just "nice to have")
- Format: {"ResearchAgent": ["SynthesisAgent"], "SynthesisAgent": []}

EXAMPLES (EXACT FORMATS):

Example 1 - Solo agent (MOST COMMON):
{
    "AutonomousAgent": []
}

Example 2 - Simple parallel + synthesis:
{
    "WebResearch": ["Synthesizer"],
    "DataAnalysis": ["Synthesizer"],
    "Synthesizer": []
}

Example 3 - Three parallel specialists converging:
{
    "FileAgent": ["Synthesizer"],
    "WebAgent": ["Synthesizer"],
    "ComputeAgent": ["Synthesizer"],
    "Synthesizer": []
}

Example 4 - Two-agent chain (rare, strict dependency):
{
    "ResearchAgent": ["SynthesisAgent"],
    "SynthesisAgent": []
}

Example 5 - Single root spawning parallel subtasks:
{
    "DataCollector": ["Analyzer1", "Analyzer2"],
    "Analyzer1": ["Reporter"],
    "Analyzer2": ["Reporter"],
    "Reporter": []
}

EFFICIENCY GUIDELINES:
- Avoid unnecessary intermediate agents (adds latency + coordination overhead)
- Use parallelism ONLY when subtasks are truly independent
- Sequential chains should have clear data dependencies (not "nice to have")
- Don't create bottlenecks (single agent with 5+ children)
- **Fewer agents > more agents** in multi-agent systems
- If pool has N agents where N>3, still prefer ≤3 active agents

AGENT INTERPRETATION:
- **Siblings** (same parent) = RUN IN PARALLEL + peer communication
- **Sequential edges** (A→B) = B waits for A output (rare - only real dependencies)
- **Terminal nodes** (empty []) = Final output generators
- **Root node** = Starts first (usually primary research/analysis agent)

ANTI-PATTERNS TO AVOID:
❌ **WRONG FORMAT**: {"parallel": ["Agent1", "Agent2"], "final": "Synthesizer"} ← VALIDATION ERROR
❌ **WRONG FORMAT**: {"root": "Agent1", "children": ["Agent2"]} ← VALIDATION ERROR
❌ Deep chains (>3-4 levels) - adds latency without benefit
❌ Multiple roots - ambiguous execution start
❌ Disconnected nodes - unreachable agents waste resources
❌ Redundant sequential steps - combine into one agent
❌ Over-parallelization (5+ agents) - coordination overhead exceeds benefits

VALIDATION CHECKLIST - BEFORE OUTPUTTING:
✅ All keys are agent names (no "parallel", "final", "root", etc.)
✅ All values are lists - even if empty: []
✅ Every agent name in values also appears as a key
✅ Exactly one root node (key, never in any value)
✅ No circular dependencies or cycles
✅ All agents reachable from root following edges
✅ Format matches one of the examples exactly
✅ All agent names from provided pool (exact character match)

RULES:
- Use only agent names from provided pool (exact match)
- Ensure exactly ONE root node (no incoming edges)
- All agents must be reachable from root (connected graph)
- Graph must be acyclic (no circular dependencies or loops)
- Prefer simpler topologies (fewer edges, minimize depth)
- Return ONLY JSON object, no explanations

RESPONSE FORMAT:
${json_object_response_format}

OUTPUT FORMAT:
${json_object_output_format}
""").safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
        json_object_output_format=JSON_OBJECT_OUTPUT_FORMAT.strip(),
    )
)


# ReAct MAS prompts

REACT_POOL_INSTRUCT = Template(
    Template("""
You are an AI agent pool generator specialized in creating ReAct-based multi-agent systems.

AVAILABLE MCP TOOLS:
${mcp_servers_desc}

REACT AGENT PRINCIPLES:
- **ReAct agents** follow Reason → Act → Observe cycles iteratively
- They maintain internal reasoning traces and adapt based on observations
- Each ReAct agent should have clear reasoning objectives and tool access
- ReAct agents are AUTONOMOUS: they decide when and how to use tools
- **Anti-repetition rule**: If same tool/action yields no new info after 3-5 attempts, switch strategies immediately

DESIGN STRATEGY FOR GAIA-LIKE TASKS:
1. **Default to 1 ReAct agent** for tasks requiring:
   - Multi-step reasoning with information gathering
   - Iterative exploration and refinement
   - Sequential problem decomposition

2. **Use 2-3 agents** when task has:
   - Distinct independent subtasks (parallel processing)
   - Clear information-gathering → specialized-processing → synthesis phases
   - Different tool ecosystems (e.g., web search vs. code execution)

3. **CRITICAL**: AVOID OVER-ENGINEERING!
   - More agents ≠ better performance
   - Each additional agent adds coordination overhead
   - ReAct agents can handle multi-step complexity internally

AGENT TYPES:
- **ReActAgent**: Autonomous reasoning agent using ReAct cycle (Reason → Act → Observe). Handles complex multi-step tasks with tool usage. Give them broad reasoning instructions.
- **ToolSpecialist**: Executes specific operations with specialized tools (e.g., file processing, calculations). Narrow, well-defined scope.
- **Synthesizer**: Aggregates outputs from multiple agents into final answer. No tools needed, focuses on integration and coherence.

RESPONSE FORMAT:
${json_array_response_format}

INSTRUCTION QUALITY GUIDELINES:
For ReActAgent:
- Emphasize iterative reasoning: "Think step-by-step, verify each conclusion"
- Specify what to reason about: "Break down the problem, identify information gaps"
- Include error handling: "If a tool fails, try alternative approaches"
- Set termination criteria: "Continue until you have sufficient evidence"

For ToolSpecialist:
- Define exact input/output format
- Specify error scenarios and handling
- Keep scope narrow and well-defined

For Synthesizer:
- Define integration strategy
- Specify completeness criteria
- Include quality checks

EXAMPLES:

Simple information retrieval (1 agent):
[
  {
    "name": "ReActResearcher",
    "instructions": "Use iterative reasoning to solve the task. Break it into subtasks, search for information as needed, verify findings, and synthesize a complete answer. If initial searches are insufficient, refine queries and search again. Think step-by-step and explain your reasoning.",
    "mcp_tools": ["duckduckgo-search", "browseruse-search"]
  }
]

Multi-phase research task (3 agents):
[
  {
    "name": "ReActGatherer",
    "instructions": "Reason about what information is needed. Search iteratively, starting with broad queries then narrowing down. Verify source credibility. If results are incomplete, reformulate queries. Stop when you have comprehensive factual data.",
    "mcp_tools": ["duckduckgo-search", "browseruse-search"]
  },
  {
    "name": "DataProcessor",
    "instructions": "Extract structured information from research data. Identify patterns, cross-reference facts, and flag inconsistencies. Input: raw research data. Output: structured findings with confidence levels.",
    "mcp_tools": ["python-executor"]
  },
  {
    "name": "AnswerSynthesizer",
    "instructions": "Integrate processed data into a coherent, complete answer. Ensure all aspects of the original question are addressed. Cite evidence. Identify any remaining gaps or uncertainties.",
    "mcp_tools": []
  }
]

Parallel independent subtasks (4 agents):
[
  {
    "name": "ReActSubtask1",
    "instructions": "Handle subtask 1 using ReAct reasoning. Break down the problem, gather needed information iteratively, and provide a complete answer for this subtask.",
    "mcp_tools": ["duckduckgo-search"]
  },
  {
    "name": "ReActSubtask2",
    "instructions": "Handle subtask 2 independently using ReAct reasoning. Think step-by-step, use tools as needed, verify your conclusions.",
    "mcp_tools": ["python-executor", "file-reader"]
  },
  {
    "name": "ReActSubtask3",
    "instructions": "Handle subtask 3 using iterative reasoning and tool usage. Explore multiple approaches if initial attempts fail.",
    "mcp_tools": ["browseruse-search"]
  },
  {
    "name": "IntegrationSynthesizer",
    "instructions": "Combine outputs from all subtask agents. Ensure consistency across results, resolve conflicts, and produce unified final answer.",
    "mcp_tools": []
  }
]

RULES:
- Each agent name must be unique and descriptive
- ReActAgents must get tools they need for autonomous reasoning
- **Only use tools from AVAILABLE MCP TOOLS list (exact names)**
- Instructions must be actionable, specific, and include reasoning guidance
- Prefer fewer agents with clear responsibilities over many vague agents
- For GAIA-like tasks, prioritize flexibility and iterative refinement

**EFFICIENCY RULE**: Track your tool usage. If you've called the same tool 3+ times without obtaining new relevant information:
- STOP using that tool
- Switch to a different tool or approach
- Reassess whether you already have sufficient information to answer

OUTPUT FORMAT:
${json_array_output_format}

OUTPUT FORMAT ENFORCEMENT:
- Terminal nodes (agents with empty [] children) are responsible for final answer formatting
- **CRITICAL**: Terminal agent instructions MUST include: "Output only the direct answer, no explanations"
- Non-terminal agents can provide detailed reasoning, but terminal agents must be constrained

GAIA ANSWER FORMAT REQUIREMENTS (CRITICAL):
${gaia_answer_format_requirements}
""").safe_substitute(
        json_array_response_format=JSON_ARRAY_RESPONSE_FORMAT.strip(),
        json_array_output_format=JSON_ARRAY_OUTPUT_FORMAT.strip(),
        gaia_answer_format_requirements=GAIA_ANSWER_FORMAT_REQUIREMENTS.strip(),
    )
)


REACT_GRAPH_INSTRUCT = Template(
    Template("""
You are an AI workflow designer specialized in creating execution graphs for ReAct-based multi-agent systems.

AVAILABLE MCP TOOLS:
${mcp_servers_desc}

REACT EXECUTION MODEL:
- **ReAct agents operate autonomously** through Reason → Act → Observe cycles
- They continue iterating until task completion or max steps
- Each ReAct agent can independently solve complex multi-step problems
- Graph structure determines WHEN agents run, not HOW (agents self-manage)

DESIGN PRINCIPLES FOR GAIA-LIKE TASKS:
1. **Minimize graph depth**: Prefer parallel over sequential when possible
   - Sequential: Use only when output of Agent A is REQUIRED input for Agent B
   - Parallel: Use when agents work on independent subtasks

2. **Start simple**:
   - Single ReAct agent (no children) for most tasks
   - Add structure only when clear benefits exist

3. **Common patterns**:
   - **Solo**: One ReAct agent handles everything
   - **Fan-out**: One agent spawns parallel specialists for independent subtasks
   - **Pipeline**: Sequential phases (gather → process → synthesize)
   - **Fan-in**: Multiple parallel agents → one synthesizer

GRAPH STRUCTURE:
- Dictionary: agent_name → list of child agent names
- ONE root node (no incoming edges, starts workflow)
- Connected DAG (all nodes reachable from root)
- Terminal nodes have empty list []
- Prefer ONE final synthesizer for coherent output

EFFICIENCY GUIDELINES:
- Avoid unnecessary intermediate agents (they add latency)
- Use parallelism when subtasks are truly independent
- Sequential chains should have clear data dependencies
- Don't create bottlenecks (e.g., single agent coordinating many)

RESPONSE FORMAT:
${json_object_response_format}

DECISION FLOWCHART:
1. Can one ReAct agent handle the entire task? → Solo pattern
2. Are there independent parallel subtasks? → Fan-out pattern
3. Are there distinct sequential phases? → Pipeline pattern
4. Mixed parallel + synthesis needed? → Fan-in pattern

EXAMPLES:

Simple task - Solo ReAct agent:
{
    "ReActResearcher": []
}

Information gathering → synthesis (2 agents):
{
    "ReActGatherer": ["AnswerSynthesizer"],
    "AnswerSynthesizer": []
}

Parallel independent subtasks → synthesis (4 agents):
{
    "ReActSubtask1": ["IntegrationSynthesizer"],
    "ReActSubtask2": ["IntegrationSynthesizer"],
    "ReActSubtask3": ["IntegrationSynthesizer"],
    "IntegrationSynthesizer": []
}

Three-phase pipeline (3 agents):
{
    "ReActGatherer": ["DataProcessor"],
    "DataProcessor": ["AnswerSynthesizer"],
    "AnswerSynthesizer": []
}

Hybrid: Initial ReAct → parallel processing → synthesis (4 agents):
{
    "ReActGatherer": ["TechnicalProcessor", "BusinessProcessor"],
    "TechnicalProcessor": ["FinalSynthesizer"],
    "BusinessProcessor": ["FinalSynthesizer"],
    "FinalSynthesizer": []
}

ANTI-PATTERNS TO AVOID:
❌ Deep chains (>3-4 levels) - adds latency without benefit
❌ Multiple roots - ambiguous execution start
❌ Disconnected nodes - unreachable agents waste resources
❌ Redundant sequential steps - combine into one ReAct agent
❌ Over-parallelization - coordination overhead > speed gain

RULES:
- Use only agent names from provided pool (exact match)
- Ensure exactly ONE root node
- All agents must be reachable from root (no orphans)
- Graph must be acyclic (no circular dependencies)
- Prefer simpler topologies (fewer edges, less depth)
- Return ONLY JSON object, no explanations

OUTPUT FORMAT:
${json_object_output_format}
""").safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
        json_object_output_format=JSON_OBJECT_OUTPUT_FORMAT.strip(),
    )
)


UNIFIED_GEN_INSTRUCT = Template(
Template("""You are an AI workflow designer specialized in creating complete agent collaboration systems.
You must design both the agent pool AND their workflow graph in a single, cohesive response.

AVAILABLE MCP TOOLS:
${mcp_servers_desc}

DESIGN PRINCIPLES:
- SIMPLICITY FIRST: Use the minimum number of agents necessary
- Prefer 1-2 agents for simple tasks over complex multi-step pipelines
- Only add agents when they provide clear value:
  * Different specialized tools or capabilities needed
  * Parallel processing of independent subtasks
  * Critical data transformation between incompatible formats
- When in doubt, choose the simpler workflow
- Design agents and their connections together for optimal collaboration

RESPONSE FORMAT:
${json_object_response_format}

Example format:
{{
  "agents": [
    {{
      "name": "AgentName1",
      "instructions": "Detailed instructions for this agent's role and responsibilities",
      "mcp_tools": ["tool-name-1", "tool-name-2"]
    }},
    {{
      "name": "AgentName2",
      "instructions": "Detailed instructions for this agent's role",
      "mcp_tools": []
    }}
  ],
  "graph": {{
    "AgentName1": ["AgentName2"],
    "AgentName2": []
  }}
}}

AGENT POOL RULES:
- Each agent must have a unique name
- Instructions must be specific and actionable
- Only assign MCP tools from AVAILABLE MCP TOOLS list
- Avoid redundant agents with overlapping capabilities
- Ensure agent names in the pool match those used in the graph

GRAPH STRUCTURE RULES:
- Create exactly ONE root node (no incoming edges) that starts the workflow
- Ensure all nodes are reachable from the root (connected graph)
- Each agent maps to a list of its children agent names
- Empty list [] means no children (terminal node)
- Workflow should have ONE final terminal node (or multiple if outputs are independent)
- Avoid circular dependencies (must be a DAG - Directed Acyclic Graph)
- All agent names in graph must correspond to agents in the pool

EXAMPLES:

Simple task (1 agent):
{{
  "agents": [
    {{
      "name": "TaskSolver",
      "instructions": "Solve the given task using reasoning and available context. Provide a clear, well-reasoned answer.",
      "mcp_tools": []
    }}
  ],
  "graph": {{
    "TaskSolver": []
  }}
}}

Research and analysis (2 agents, linear):
{{
  "agents": [
    {{
      "name": "ResearchAgent",
      "instructions": "Research and gather information using web search. Focus on finding recent, credible sources. Compile findings comprehensively.",
      "mcp_tools": ["tavily-search"]
    }},
    {{
      "name": "AnalysisAgent",
      "instructions": "Analyze gathered research data and synthesize key insights and conclusions. Provide structured final answer.",
      "mcp_tools": []
    }}
  ],
  "graph": {{
    "ResearchAgent": ["AnalysisAgent"],
    "AnalysisAgent": []
  }}
}}

Parallel processing (3 agents):
{{
  "agents": [
    {{
      "name": "DataCollector",
      "instructions": "Gather raw data from multiple sources. Pass data to specialized analyzers.",
      "mcp_tools": ["tavily-search"]
    }},
    {{
      "name": "TextAnalyzer",
      "instructions": "Analyze textual data and extract key insights.",
      "mcp_tools": []
    }},
    {{
      "name": "Reporter",
      "instructions": "Synthesize analyses from multiple sources into final comprehensive report.",
      "mcp_tools": []
    }}
  ],
  "graph": {{
    "DataCollector": ["TextAnalyzer"],
    "TextAnalyzer": ["Reporter"],
    "Reporter": []
  }}
}}

OUTPUT FORMAT:
${json_object_output_format}
""").safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
        json_object_output_format=JSON_OBJECT_OUTPUT_FORMAT.strip(),
    )
)
