import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .node import AgentNode


@dataclass
class NodeExecution:
    """Node execution result with metadata."""

    node_id: str
    node_name: str
    output: Any
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class NodeSessionError(Exception):
    """Base exception for NodeSession errors."""

    pass


class NodeSession:
    """Session that manages state and context between pipeline nodes."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"node_session_{uuid.uuid4().hex[:8]}"
        self.node_executions: Dict[str, NodeExecution] = {}
        self.original_query: str = ""

    def set_original_query(self, query: str) -> None:
        """Set the original pipeline query."""
        self.original_query = query

    def add_node_execution(self, execution: NodeExecution) -> None:
        """Add node execution result."""
        self.node_executions[execution.node_id] = execution

    def validate_dependencies(self, node: AgentNode) -> None:
        """Validate that all parent nodes have been executed."""
        missing = []
        for parent in node.parents:
            if parent.id not in self.node_executions:
                missing.append(f"{parent.name} ({parent.id})")

        if missing:
            raise NodeSessionError(
                f"Node '{node.name}' missing dependencies: {', '.join(missing)}"
            )

    def _collect_parent_results(self, node: AgentNode) -> List[Dict[str, Any]]:
        """Collect parent node results with metadata."""
        parent_results: List[Dict[str, Any]] = []
        for parent in node.parents:
            exec_result = self.node_executions[parent.id]
            parent_results.append(
                {
                    "node_id": parent.id,
                    "node_name": parent.name,
                    "node_role": parent.instructions,
                    "output": exec_result.output,
                }
            )
        return parent_results

    def _collect_children_info(self, node: AgentNode) -> List[Dict[str, str]]:
        """Collect child node metadata."""
        children_info: List[Dict[str, str]] = []
        for child in node.children:
            children_info.append(
                {
                    "node_id": child.id,
                    "node_name": child.name,
                    "node_role": child.instructions,
                }
            )
        return children_info

    def _generate_task_instruction(self, node: AgentNode) -> str:
        """Generate task instruction based on node position."""
        if not node.parents:
            # Entry node - gets original query
            return self.original_query
        elif len(node.parents) == 1:
            # Single parent - process its result
            return (
                f"Process the result from the previous node and complete your task. "
                f"Original query: {self.original_query}"
            )
        else:
            # Multiple parents - synthesis required
            return (
                f"Synthesize results from {len(node.parents)} previous nodes and complete your task. "
                f"Original query: {self.original_query}"
            )

    def get_input_for_node(self, node: AgentNode) -> str:
        """Generate structured JSON input for a node with complete pipeline context."""
        # Entry node: simplified format without context overhead
        if node.is_entry_node:
            return self.original_query

        # Collect data
        parent_results = self._collect_parent_results(node)
        children_info = self._collect_children_info(node)
        task_instruction = self._generate_task_instruction(node)

        # Build structured input
        structured_input = {
            "pipeline_context": {
                "session_id": self.session_id,
                "original_query": self.original_query,
                "current_node": {
                    "node_id": node.id,
                    "node_name": node.name,
                    "node_role": node.instructions,
                    "is_entry_node": node.is_entry_node,
                    "is_terminal_node": node.is_terminal_node,
                },
                "parent_nodes": parent_results,
                "child_nodes": children_info,
            },
            "task": task_instruction,
        }

        # Format as readable JSON with instructions
        json_str = json.dumps(structured_input, indent=2, ensure_ascii=False)

        instruction = f"""You are executing as part of a multi-agent pipeline. Below is your structured input in JSON format:

{json_str}

# Instructions:
- The "task" field contains your primary objective
- "parent_nodes" provides results from previous agents - use as context
- "child_nodes" shows which agents will receive your output
- If "is_terminal_node" is true, provide a complete, user-ready final result
- Format your output appropriately for consumption by child nodes or as final result"""

        return instruction
