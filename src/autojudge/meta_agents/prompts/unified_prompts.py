"""Unified generation prompt templates (pool + graph in one shot)."""

from string import Template

from .common import JSON_OBJECT_OUTPUT_FORMAT, JSON_OBJECT_RESPONSE_FORMAT

UNIFIED_GEN_INSTRUCT = Template(
    Template(
        """You are an AI workflow designer specialized in creating complete agent collaboration systems.
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
"""
    ).safe_substitute(
        json_object_response_format=JSON_OBJECT_RESPONSE_FORMAT.strip(),
        json_object_output_format=JSON_OBJECT_OUTPUT_FORMAT.strip(),
    )
)
