from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from automas.pipeline.types import GraphDict


class GraphOperationResult(BaseModel):
    """Result of a graph operation (mutation or crossover)."""

    graph: GraphDict = Field(description="Resulting agent graph after operation")
    explanation: str = Field(description="Explanation of the applied operation")


def validate_graph(graph: GraphDict, allowed_agents: set[str]) -> GraphDict:
    """Filter graph to only include allowed agents."""
    validated = {}
    for parent, children in graph.items():
        if parent not in allowed_agents:
            continue
        validated[parent] = [c for c in children if c in allowed_agents]
    return validated if validated else graph


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    best_score: float
    avg_score: float
    best_graph: GraphDict
    best_response: Optional[str] = None
