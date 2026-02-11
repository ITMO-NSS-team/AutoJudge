from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

from automas.utils.logger import get_artifact_path, get_logger

from .node import AgentNode
from .node_session import NodeExecution, NodeSession
from .types import CostBreakdown, ExecutionLevels, NodeTrace, PipelineTrace

logger = get_logger()


class Pipeline:
    """Pipeline for executing DAG of agents."""

    def __init__(
        self,
        execution_order: list[AgentNode],
    ):
        self.execution_order = execution_order
        self.node_session = NodeSession()

        # Pipeline trace
        self._trace: PipelineTrace

        # Compute execution levels for parallel execution
        self._execution_levels = self._compute_execution_levels()

    def _compute_execution_levels(self) -> ExecutionLevels:
        """
        Group nodes by execution levels for concurrent execution.
        """
        if not self.execution_order:
            return []

        # Compute level for each node
        levels_map: dict[str, int] = {}

        def compute_level(node: AgentNode) -> int:
            """Recursively compute the execution level of a node."""
            if node.id in levels_map:
                return levels_map[node.id]

            # Entry nodes (no parents) are at level 0
            if not node.parents:
                levels_map[node.id] = 0
                return 0

            # Level = max(parent levels) + 1
            parent_levels = [compute_level(parent) for parent in node.parents]
            level = max(parent_levels) + 1
            levels_map[node.id] = level
            return level

        # Compute levels for all nodes
        for node in self.execution_order:
            compute_level(node)

        # Group nodes by their levels
        max_level = max(levels_map.values()) if levels_map else 0
        levels: ExecutionLevels = [[] for _ in range(max_level + 1)]

        for node in self.execution_order:
            level = levels_map[node.id]
            levels[level].append(node)

        return levels

    async def _execute_node(self, node: AgentNode) -> Any:
        """Execute a single node."""
        # Check if already executed
        if node.id in self.node_session.node_executions:
            logger.debug(f"Node '{node.name}' already executed, returning cached result")
            return self.node_session.node_executions[node.id].output

        # Validate that all dependencies are satisfied
        self.node_session.validate_dependencies(node)

        # Build input for this node
        node_input = self.node_session.get_input_for_node(node)

        # Execute agent
        logger.info(f"Executing node: {node.name} (id: {node.id})")
        agent = node.build_agent()
        result = await agent.run(node_input)
        logger.info(f"Completed node: {node.name} (id: {node.id})")

        # Store usage in node for cost tracking
        node._usage = result.usage()

        # Store node trace
        node_trace = NodeTrace(
            node_id=node.id,
            node_name=node.name,
            model=node.model,
            message_history=result.all_messages(),
            usage=result.usage(),
        )

        if self._trace is not None:
            self._trace.node_traces.append(node_trace)

        # Store node result in session
        execution = NodeExecution(
            node_id=node.id,
            node_name=node.name,
            output=result.output,
        )
        self.node_session.add_node_execution(execution)

        return result.output

    async def _execute_level(self, level_nodes: list[AgentNode]) -> None:
        """Execute all nodes in a level concurrently using asyncio.gather()."""
        if not level_nodes:
            return

        # Log which nodes will be executed concurrently
        node_names = [f"{node.name} (id: {node.id})" for node in level_nodes]
        if len(level_nodes) > 1:
            logger.info(f"Executing {len(level_nodes)} nodes concurrently: {', '.join(node_names)}")

        # Execute all nodes concurrently
        tasks = [self._execute_node(node) for node in level_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions and log them
        errors = []
        for node, result in zip(level_nodes, results):
            if isinstance(result, Exception):
                logger.error(f"Node '{node.name}' failed with error: {result}")
                errors.append((node.name, result))

        # If any node failed, raise an exception with details
        if errors:
            error_details = "; ".join([f"{name}: {err}" for name, err in errors])
            raise RuntimeError(
                f"Failed to execute level with {len(errors)} error(s): {error_details}"
            )

    async def ainvoke(self, query: Any) -> Any:
        """Asynchronous pipeline execution."""
        # Set original query in node session
        query_str = str(query)
        self.node_session.set_original_query(query_str)

        # Initialize pipeline trace
        self._trace = PipelineTrace(
            session_id=self.node_session.session_id,
            original_query=query_str,
            node_traces=[],
            final_output=None,
        )

        logger.info("Starting pipeline execution")

        # Execute nodes level by level (nodes within a level run concurrently)
        for level_idx, level_nodes in enumerate(self._execution_levels):
            logger.info(f"Executing level {level_idx} with {len(level_nodes)} node(s)")
            await self._execute_level(level_nodes)
            logger.info(f"Completed level {level_idx}")

        # Return result of the final node (last in execution order)
        final_node = self.execution_order[-1]
        final_output = self.node_session.node_executions[final_node.id].output

        # Update trace with final output
        self._trace.final_output = final_output

        logger.info(f"Pipeline completed. Final result from: {final_node.name}")
        print(final_output)
        return final_output

    def invoke(self, query: Any) -> Any:
        """Synchronous pipeline execution."""
        return asyncio.run(self.ainvoke(query))

    @property
    def trace(self) -> Optional[PipelineTrace]:
        """Get pipeline execution trace."""
        return self._trace

    @property
    def input_tokens(self) -> int:
        """Total input tokens used across all nodes."""
        return sum(node.input_tokens for node in self.execution_order)

    @property
    def output_tokens(self) -> int:
        """Total output tokens generated across all nodes."""
        return sum(node.output_tokens for node in self.execution_order)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all nodes."""
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> CostBreakdown:
        """Total cost breakdown for pipeline execution."""
        total_input = 0.0
        total_output = 0.0

        for node in self.execution_order:
            node_cost = node.cost
            total_input += node_cost.input_price
            total_output += node_cost.output_price

        return CostBreakdown(
            input_price=total_input,
            output_price=total_output,
            total_price=total_input + total_output,
        )

    def to_mermaid_lr(self, visualize: bool = True, output_path: Optional[str] = None) -> str:
        """Generate Mermaid diagram in LR format. Optionally save as PNG image."""
        lines = ["graph LR"]

        # Create node map
        node_map = {node.id: node for node in self.execution_order}

        # Collect all edges
        edges = []
        all_nodes = set()

        for node in self.execution_order:
            all_nodes.add(node.id)

            # Add edges from this node to its children
            for child in node.children:
                edges.append((node.id, child.id))
                all_nodes.add(child.id)

        # If no edges, show standalone nodes
        if not edges:
            for node in self.execution_order:
                safe_name = node.name.replace(" ", "_")
                lines.append(f"    {node.id}[{safe_name}]")
        else:
            # Add edges
            for parent_id, child_id in edges:
                parent_node = node_map[parent_id]
                child_node = node_map[child_id]

                # Create safe node identifiers and labels
                parent_safe = parent_node.name.replace(" ", "_")
                child_safe = child_node.name.replace(" ", "_")

                lines.append(f"    {parent_id}[{parent_safe}] --> {child_id}[{child_safe}]")

        mermaid_graph = "\n".join(lines)

        # Generate visualization if requested
        if visualize:
            self._save_mermaid_graph(mermaid_graph, output_path)

        return mermaid_graph

    def _save_mermaid_graph(self, graph: str, output_path: Optional[str] = None) -> None:
        """Convert mermaid graph to PNG and save it in session directory."""

        try:
            # Use session directory if no path specified
            if output_path is None:
                output_file = get_artifact_path("pipeline_graph.png")
            else:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)

            # Encode graph to base64
            graphbytes = graph.encode("utf8")
            base64_bytes = base64.urlsafe_b64encode(graphbytes)
            base64_string = base64_bytes.decode("ascii")

            # Get image from mermaid.ink API
            response = requests.get(f"https://mermaid.ink/img/{base64_string}", timeout=30)
            response.raise_for_status()

            # Save image
            img = Image.open(io.BytesIO(response.content))
            img.save(output_file, "PNG")

            logger.info(f"Saved pipeline graph to {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate or save graph image: {e}")
