from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from autojudge import AgentPool
from autojudge.judge import Judge
from autojudge.meta_agents.graph_gen import GraphDict


class MASOptimizer(ABC):
    """Base class for multi-agent system optimizers"""

    @abstractmethod
    def optimize(
        self, initial_graph: GraphDict, agent_pool: AgentPool, judge: Judge, **kwargs
    ) -> Tuple[GraphDict, float, List[Dict]]:
        """
        Optimizes the agent graph

        Args:
            initial_graph: Initial graph from meta-agent
            agent_pool: List of available agents for mutations
            judge: Object with evaluate(graph) -> float method
            **kwargs: Additional optimization parameters

        Returns:
            Tuple[best_graph, its_score, evolution_history]
        """
        pass
