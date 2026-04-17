from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from .pipeline.node import AgentNode


class AgentPool:
    """Pool of agents for reuse in different pipelines."""

    def __init__(self, agents: Optional[List[AgentNode]] = None):
        """Initialize agent pool with optional list of agents."""
        self._agents: Dict[str, AgentNode] = {}
        self._agents_by_name: Dict[str, AgentNode] = {}

        if agents:
            for agent in agents:
                self.add_agent(agent)

    def add_agent(self, agent: AgentNode) -> AgentPool:
        """Add agent to pool. Returns self for chaining."""
        self._agents[agent.id] = agent
        self._agents_by_name[agent.name] = agent
        return self

    def get_agent(self, identifier: str) -> AgentNode:
        """Get agent by name or ID."""
        if identifier in self._agents_by_name:
            return self._agents_by_name[identifier]
        elif identifier in self._agents:
            return self._agents[identifier]
        else:
            raise KeyError(f"Agent '{identifier}' not found in pool")

    def has_agent(self, identifier: str) -> bool:
        """Check if agent exists in pool."""
        return identifier in self._agents_by_name or identifier in self._agents

    def __add__(self, other: AgentPool) -> AgentPool:
        """Combine two agent pools. Returns new pool with agents from both pools."""
        combined_agents = list(self._agents.values()) + list(other._agents.values())
        return AgentPool(combined_agents)

    def __iadd__(self, other: AgentPool) -> AgentPool:
        """Add agents from another pool to current pool."""
        for agent in other._agents.values():
            self.add_agent(agent)

        return self

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[AgentNode]:
        return iter(self._agents.values())

    def __contains__(self, identifier: str) -> bool:
        return self.has_agent(identifier)

    def __repr__(self) -> str:
        agent_names = list(self._agents_by_name.keys())
        return f"AgentPool({len(agent_names)} agents): {agent_names})"

    @property
    def full_agents_data(self) -> List[Dict[str, Any]]:
        """Return list of all agents and their meta-data."""
        return [
            {
                "name": agent.name,
                "id": agent.id,
                "instructions": agent.instructions,
                "model": agent.model,
                "mcp_tools": agent.mcp_tools,
            }
            for agent in self._agents.values()
        ]


class DefaultAgents:
    """Commonly used agent pools."""

    @staticmethod
    def create_research_pool() -> AgentPool:
        """Create pool of agents for research tasks."""
        agents = [
            AgentNode(
                name="Researcher",
                instructions="Collect data and information on given topic. Use search for up-to-date information.",
                mcp_tools=["web-search"],
            ),
            AgentNode(
                name="Summarizer",
                instructions="Create concise and informative summaries based on data analysis.",
            ),
        ]
        return AgentPool(agents)

    @staticmethod
    def create_planning_pool() -> AgentPool:
        """Create pool of agents for planning and management."""
        agents = [
            AgentNode(
                name="Planner",
                instructions="Create detailed plans and strategies for task execution.",
            ),
            AgentNode(
                name="TaskBreaker",
                instructions="Break complex tasks into smaller, manageable subtasks.",
            ),
        ]
        return AgentPool(agents)

    @staticmethod
    def create_sandbox_pool() -> AgentPool:
        """Create pool of agents for sandbox-based code execution and data analysis."""
        agents = [
            AgentNode(
                name="DataScientist",
                instructions="Perform data analysis, machine learning, and statistical computations using Python in E2B sandboxes. Create visualizations, analyze datasets, and provide insights.",
                mcp_tools=["e2b-sandbox"],
            ),
            AgentNode(
                name="ScriptTester",
                instructions="Test and validate Python scripts in isolated E2B sandbox environments. Run unit tests, integration tests, and perform code quality checks.",
                mcp_tools=["e2b-sandbox"],
            ),
            AgentNode(
                name="ComputationalAgent",
                instructions="Perform complex computational tasks, mathematical calculations, and algorithmic operations in E2B sandboxes. Solve numerical problems and run simulations.",
                mcp_tools=["e2b-sandbox"],
            ),
        ]
        return AgentPool(agents)


# Categories
RESEARCH_POOL = DefaultAgents.create_research_pool()
PLANNING_POOL = DefaultAgents.create_planning_pool()
SANDBOX_POOL = DefaultAgents.create_sandbox_pool()

# Universal pool
ALL_AGENTS_POOL = RESEARCH_POOL + PLANNING_POOL + SANDBOX_POOL
