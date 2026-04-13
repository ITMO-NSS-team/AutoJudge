from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Union, overload

from .node import AgentNode
from .pipeline import Pipeline
from .types import GraphDict

if TYPE_CHECKING:
    from ..agent_pool import AgentPool


class PipelineBuilder:
    """Builder for creating DAG pipeline."""

    def __init__(self):
        self.nodes: Dict[str, AgentNode] = {}
        self.nodes_by_name: Dict[str, AgentNode] = {}

    @overload
    def add_node(self, node: AgentNode) -> AgentNode: ...

    @overload
    def add_node(self, node: List[AgentNode]) -> List[AgentNode]: ...

    def add_node(
        self, node: Union[AgentNode, List[AgentNode]]
    ) -> Union[AgentNode, List[AgentNode]]:
        """Adds a node or list of nodes to the graph."""
        if isinstance(node, list):
            for n in node:
                self.nodes[n.id] = n  # type: ignore
                self.nodes_by_name[n.name] = n  # type: ignore
            return node
        else:
            self.nodes[node.id] = node
            self.nodes_by_name[node.name] = node
            return node

    def add_from_pool(self, pool: AgentPool) -> List[AgentNode]:
        """Add all agents from pool to builder."""
        agents_to_add = [agent for agent in pool]
        return self.add_node(agents_to_add)

    def create_from_pool(self, pool: AgentPool, graph: GraphDict) -> PipelineBuilder:
        """Create graph from agent pool and adjacency list."""
        # Collect all agent names/IDs referenced in the graph
        referenced_agents = set(graph.keys())
        for children in graph.values():
            referenced_agents.update(children)

        # Add DEEP COPIES of agents to prevent race conditions when multiple
        # graphs are built in parallel from the same pool
        agents_to_add = [
            copy.deepcopy(agent)
            for agent in pool
            if agent.name in referenced_agents or agent.id in referenced_agents
        ]
        self.add_node(agents_to_add)

        self.add_edges_from_dict(graph)
        return self

    def _find_node(self, identifier: str) -> AgentNode:
        """Find node by name or ID."""
        # Try by name first, then by ID
        if identifier in self.nodes_by_name:
            return self.nodes_by_name[identifier]
        elif identifier in self.nodes:
            return self.nodes[identifier]
        else:
            raise KeyError(f"Node '{identifier}' not found (tried both name and ID)")

    def add_edge(self, parent: AgentNode, child: AgentNode) -> PipelineBuilder:
        """Create parent -> child relationship. Parent executes before child."""
        parent.add_child(child)
        return self

    def add_edges_from_dict(self, graph: GraphDict) -> PipelineBuilder:
        """Create edges from adjacency list (dict mapping node -> list of children)."""
        # Validate that all referenced nodes exist
        for parent_id, children in graph.items():
            self._find_node(parent_id)
            for child_id in children:
                self._find_node(child_id)

        # Create edges
        for parent_id, children in graph.items():
            parent_node = self._find_node(parent_id)
            for child_id in children:
                child_node = self._find_node(child_id)
                parent_node.add_child(child_node)

        # Validate that no cycles were introduced
        self._validate_acyclic()

        return self

    def _validate_acyclic(self) -> None:
        """
        Validate that graph is acyclic using DFS.
        https://www.geeksforgeeks.org/dsa/detect-cycle-direct-graph-using-colors/

        WHITE - unvisited
        GRAY - currently processing
        BLACK - fully processed
        """
        if not self.nodes:
            return

        # DFS-based cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}

        def dfs(node_id: str) -> bool:
            if colors[node_id] == GRAY:  # Back edge found - cycle detected
                return True
            if colors[node_id] == BLACK:  # Already processed
                return False

            colors[node_id] = GRAY

            # Check all children
            for child in self.nodes[node_id].children:
                if dfs(child.id):
                    return True

            colors[node_id] = BLACK
            return False

        # Check for cycles starting from each unvisited node
        for node_id in self.nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    raise RuntimeError(
                        f"Graph has cycles! Cycle detected involving node '{node_id}'"
                    )

    def _topological_sort(self) -> List[AgentNode]:
        """
        Topological sort to determine execution order (entry nodes first, terminal node last).
        https://www.geeksforgeeks.org/dsa/topological-sorting/
        """
        # Count incoming edges for each node
        in_degree = {node_id: len(node.parents) for node_id, node in self.nodes.items()}

        # Queue with nodes having no incoming edges (entry nodes)
        queue = [node for node in self.nodes.values() if in_degree[node.id] == 0]
        result = []

        while queue:
            # Take node without dependencies
            node = queue.pop(0)
            result.append(node)

            # Decrease incoming edge count for children
            for child in node.children:
                in_degree[child.id] -= 1
                if in_degree[child.id] == 0:
                    queue.append(child)

        if len(result) != len(self.nodes):
            raise RuntimeError("Graph has cycles!")

        return result

    def build(self) -> Pipeline:
        """Creates a ready-to-use pipeline."""
        if not self.nodes:
            raise RuntimeError("No nodes in the graph")

        # Validate graph and get execution order
        execution_order = self._topological_sort()
        return Pipeline(execution_order)
