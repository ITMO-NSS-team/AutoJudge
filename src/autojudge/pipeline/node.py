from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
from pydantic_ai import Agent, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from autojudge.db.db_tools import get_content_tool
from autojudge.pipeline.types import UsageTrackingMixin
from autojudge.utils.langfuse_utils import setup_langfuse_instrumentation

load_dotenv(".env")

setup_langfuse_instrumentation()

AGENT_NODE_TEMPERATURE = float(os.getenv("AGENT_NODE_TEMPERATURE", "0.1"))
print('Judge temperature = ', AGENT_NODE_TEMPERATURE)

@dataclass
class AgentNode(UsageTrackingMixin):
    """Agent node in the DAG."""

    name: str
    instructions: str
    model: str = os.getenv("AGENT_NODE_MODEL", "google/gemini-2.5-flash")
    api_key: Optional[str] = field(default=None, repr=False)
    mcp_tools: List[str] = field(default_factory=list)
    use_tools: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    children: List["AgentNode"] = field(default_factory=list)
    parents: List["AgentNode"] = field(default_factory=list)
    _usage: Optional[RunUsage] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("No API key provided")

    def build_agent(self) -> Agent:
        print('Judge model: ', self.model)
        model = OpenAIChatModel(
            self.model,
            provider=OpenRouterProvider(api_key=self.api_key or ""),
            settings={"temperature": AGENT_NODE_TEMPERATURE},
        )

        tools = [get_content_tool] if self.use_tools else []

        return Agent(
            name=self.name,
            model=model,
            # tools=tools,
            instructions=self.instructions,
            retries=3,
            instrument=True,
        )

    def add_child(self, child: "AgentNode") -> None:
        """Adds a dependent child (child depends on self)."""
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    @property
    def is_entry_node(self) -> bool:
        """Entry point node - receives original query (no parents)."""
        return len(self.parents) == 0

    @property
    def is_terminal_node(self) -> bool:
        """Terminal node - produces final result (no children)."""
        return len(self.children) == 0

    def __eq__(self, other) -> bool:
        """Compare nodes by their unique ID."""
        if not isinstance(other, AgentNode):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on unique ID for use in sets and dicts."""
        return hash(self.id)