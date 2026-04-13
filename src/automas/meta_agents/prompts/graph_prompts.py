"""Graph generation prompt templates."""

from string import Template

from .common import (
    JSON_OBJECT_OUTPUT_FORMAT,
    JSON_OBJECT_RESPONSE_FORMAT,
)

DEFAULT_GRAPH_INSTRUCT = Template(
    Template(
        """
You are an AI workflow designer specialized in creating evaluation pipelines for multi-agent systems.
Your goal is to design an optimal flow of judges that efficiently detect problems, errors, and quality issues.

AVAILABLE MCP TOOLS:
None

DESIGN PRINCIPLES:
- SIMPLICITY FIRST: Use the minimum number of judges necessary to detect all critical problems
- Prefer 1-2 specialized judges for simple evaluation over complex multi-step pipelines
- Only add parallel judges if they detect independent problem categories:
  * Different error domains (API errors vs environment setup vs task completion)
  * Separate quality dimensions requiring simultaneous assessment
  * Independent failure modes that don't depend on each other
- When in doubt, choose the simpler workflow that still catches all major issues
- FINAL_AGGREGATOR must ALWAYS be present as the final terminal node that synthesizes all findings
- All evaluation paths must eventually lead to FINAL_AGGREGATOR for final problem assessment
- If GUILTY_AGENT_FINDER and STEP_OF_ERROR_FINDER are present in the pool, they must be ordered exactly like this: GUILTY_AGENT_FINDER -> STEP_OF_ERROR_FINDER

RESPONSE FORMAT:
${json_object_response_format}

RULES:
- Select only judges necessary to detect relevant problems (subset allowed)
- Create exactly ONE root node (no incoming edges) that starts the evaluation pipeline
- Ensure all nodes are reachable from the root (connected graph)
- Each judge maps to a list of judge names (its children in the pipeline)
- Empty list [] means no children (terminal node)
- FINAL_AGGREGATOR must be the single terminal node that receives all evaluation results
- Avoid circular dependencies (DAG - Directed Acyclic Graph)
- Return ONLY the JSON object, no additional text

CRITICAL: USE ONLY NAMES OF THE JUDGES WHICH ARE IN THE POOL! DON'T USE ANY OTHER NAMES!

EXAMPLES:

Simple task (2 agent):
{
    "MAS_TASK_COMPLETION_JUDGE": ["FINAL_AGGREGATOR"],
    "FINAL_AGGREGATOR": []
}

Linear workflow (3 agents):
{
    "MAS_TASK_COMPLETION_JUDGE": ["MAS_COMPLEXITY_JUDGE"],
    "MAS_COMPLEXITY_JUDGE": ["FINAL_AGGREGATOR"],
    "FINAL_AGGREGATOR": []
}

Parallel processing (5 agents):
{
    "TOOL_PERFORMANCE_JUDGE": ["MAS_ENVIRONMENT_SETUP_JUDGE", "MAS_API_ISSUES_JUDGE", "MAS_TASK_COMPLETION_JUDGE"],
    "MAS_ENVIRONMENT_SETUP_JUDGE": ["FINAL_AGGREGATOR"],
    "MAS_API_ISSUES_JUDGE": ["FINAL_AGGREGATOR"],
    "MAS_TASK_COMPLETION_JUDGE": ["FINAL_AGGREGATOR"],
    "FINAL_AGGREGATOR": []
}
"""
    ).safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
    )
)


DECENTRALIZED_GRAPH_INSTRUCT = Template(
    Template(
        """You are an AI workflow designer specialized in creating agent collaboration graphs for DECENTRALIZED multi-agent systems.

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
"""
    ).safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
        json_object_output_format=JSON_OBJECT_OUTPUT_FORMAT.strip(),
    )
)


REACT_GRAPH_INSTRUCT = Template(
    Template(
        """
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
"""
    ).safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
        json_object_output_format=JSON_OBJECT_OUTPUT_FORMAT.strip(),
    )
)
