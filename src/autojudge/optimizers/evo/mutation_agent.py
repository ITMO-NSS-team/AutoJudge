import copy
from typing import Optional

from autojudge import AgentPool
from autojudge.meta_agents.base import DEFAULT_MODEL, BaseMetaAgent
from autojudge.pipeline.types import GraphDict
from autojudge.utils.logger import get_logger

from .types import GraphOperationResult

logger = get_logger(__name__)


class MutationAgent(BaseMetaAgent):
    def __init__(
        self,
        agents_pool: AgentPool,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )
        self.agent_pool = agents_pool

    def _get_system_prompt(self) -> str:
        return """You are an expert in optimizing multi-agent systems through evolutionary algorithms.

You are given an agent graph in the form of Dict[str, List[str]], where the key is the agent name and the value is a list of agents to which it passes data.

Your task is to propose ONE useful graph mutation:
1. Add / remove / redirect edges
2. Add / remove agents (ONLY from the provided AVAILABLE AGENTS list)
3. Keep DAG validity and explain reasoning.
4. Result graph MUST have only one root (i.e., exactly one node with no incoming edges)
5. Result graph MUST be not empty

CRITICAL CONSTRAINT: You MUST ONLY use agent names that are explicitly listed in the "AVAILABLE AGENTS" section.
Do NOT invent new agent names or use agents not provided in the available list.

Return the mutated graph in the same format.

EXAMPLES:

Simple task (1 agent):
{{
    "TaskSolver": []
}}

Linear workflow (3 agents):
{{
    "DataCollector": ["Analyzer"],
    "Analyzer": ["Reporter"],
    "Reporter": []
}}

Parallel processing (4 agents):
{{
    "DataCollector": ["Analyzer1", "Analyzer2"],
    "Analyzer1": ["Reporter"],
    "Analyzer2": ["Reporter"],
    "Reporter": []
}}"""

    def _get_output_type(self):
        return GraphOperationResult

    def _validate_graph(self, graph: GraphDict, available_agents: AgentPool) -> GraphDict:
        """Validate and filter graph to only include agents from available pool."""
        allowed = {agent.name for agent in available_agents} | {
            agent.id for agent in available_agents
        }

        validated = {}
        for parent, children in graph.items():
            if parent not in allowed:
                logger.warning(
                    f"Agent '{parent}' in mutated graph not found in available agents. Skipping."
                )
                continue

            invalid_children = [c for c in children if c not in allowed]
            if invalid_children:
                logger.warning(f"Removed invalid children {invalid_children} from agent '{parent}'")

            validated[parent] = [c for c in children if c in allowed]

        if not validated and graph:
            logger.warning("Validation resulted in empty graph. Returning original graph.")
            return graph

        return validated if validated else graph

    async def perform_mutation(
        self,
        graph: GraphDict,
        task_description: str,
        current_score: Optional[float] = None,
    ) -> GraphDict:
        user_prompt = f"""
TASK: {task_description}

AVAILABLE AGENTS to add: {self.agent_pool.full_agents_data}

Agent graph for mutation:
{graph}

Current score: {current_score if current_score else "unknown"}

Propose a useful mutation.

IMPORTANT: You MUST only use agents from the AVAILABLE AGENTS list above.
Do not invent new agent names or use agents not in the list."""

        try:
            result = await self._run_agent(user_prompt)
            return self._validate_graph(result.graph, self.agent_pool)
        except Exception:
            logger.warning("Mutation error. Returning original graph.", exc_info=True)
            return copy.deepcopy(graph)
