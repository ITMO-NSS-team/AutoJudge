import copy
from typing import Optional

from autojudge.meta_agents.base import DEFAULT_MODEL, BaseMetaAgent
from autojudge.pipeline.types import GraphDict
from autojudge.utils.logger import get_logger

from .types import GraphOperationResult

logger = get_logger(__name__)


class CrossoverAgent(BaseMetaAgent):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )

    def _get_system_prompt(self) -> str:
        return """You are an expert in optimizing multi-agent systems through evolutionary algorithms.

Given two parent agent graphs, produce an offspring that combines the best parts of both while keeping DAG validity.

CRITICAL CONSTRAINT: You MUST ONLY use agents that appear in the parent graphs.
Do NOT invent new agent names or use agents not present in Parent 1 or Parent 2.

Explain your reasoning for the crossover strategy.


EXAMPLES OF OUTPUT FORMAT:

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

    def _validate_graph(
        self, graph: GraphDict, parent1: GraphDict, parent2: GraphDict
    ) -> GraphDict:
        """Validate and filter graph to only include agents from parent graphs."""
        allowed = set(parent1.keys()) | set(parent2.keys())
        for parent_graph in [parent1, parent2]:
            for children in parent_graph.values():
                allowed.update(children)

        validated = {}
        for parent, children in graph.items():
            if parent not in allowed:
                logger.warning(
                    f"Agent '{parent}' in crossover result not found in parent graphs. Skipping."
                )
                continue

            invalid_children = [c for c in children if c not in allowed]
            if invalid_children:
                logger.warning(
                    f"Removed invalid children {invalid_children} from agent '{parent}'"
                )

            validated[parent] = [c for c in children if c in allowed]

        if not validated and graph:
            logger.warning(
                "Validation resulted in empty graph. Returning original graph."
            )
            return graph

        return validated if validated else graph

    async def perform_crossover(
        self,
        parent1: GraphDict,
        parent2: GraphDict,
        task_description: str,
        score1: Optional[float] = None,
        score2: Optional[float] = None,
    ) -> GraphDict:
        user_prompt = f"""
TASK: {task_description}

Parent 1 (score: {score1 if score1 else "unknown"}):
{parent1}

Parent 2 (score: {score2 if score2 else "unknown"}):
{parent2}

Create an offspring by combining the best parts of both parents.

IMPORTANT: You MUST only use agents that appear in Parent 1 or Parent 2.
Do not invent new agent names or use agents not present in the parent graphs."""

        try:
            result = await self._run_agent(user_prompt)
            return self._validate_graph(result.graph, parent1, parent2)
        except Exception:
            logger.warning(
                "Crossover error. Returning copy of best parent.", exc_info=True
            )
            if score1 and score2:
                return copy.deepcopy(parent1 if score1 > score2 else parent2)
            return copy.deepcopy(parent1)
