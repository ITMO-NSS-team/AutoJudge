from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from typing import List, Optional

from .prompt_registry import TRACE_SUMMARIZATION_PROMPT


class ExecutionStep(BaseModel):
    """Single step in the execution log."""
    
    step_number: int = Field(description="Sequential step number")
    agent_name: str = Field(description="Agent performing this step")
    action_type: str = Field(description="Type of action (tool_call/reasoning/communication/observation/output/error)")
    description: str = Field(description="Detailed description of what was done/attempted")
    error_message: Optional[str] = Field(default=None, description="Error details if any")
    key_data: Optional[str] = Field(default=None, description="Important data from this step")


class TaskOverview(BaseModel):
    """Overview of the task and system."""
    
    original_query: str = Field(description="The user's original question/task")
    system_type: str = Field(description="Type of MAS (decentralized/ReAct/hierarchical/etc)")
    total_agents: int = Field(description="Total number of agents")


class AgentSummary(BaseModel):
    """Summary of a single agent's activity."""
    
    agent_name: str = Field(description="Name of the agent")
    role: str = Field(description="Agent's role/specialization")
    key_contributions: str = Field(description="Main contributions to task completion")
    issues: Optional[str] = Field(default=None, description="Any problems encountered")


class TraceSummary(BaseModel):
    """Complete trace summarization result."""
    
    step_by_step_log: List[ExecutionStep] = Field(description="Chronological log of all execution steps")
    task_overview: TaskOverview
    agents_summary: List[AgentSummary]


class TraceInput(BaseModel):
    """Input for trace summarization."""
    
    trace: str = Field(description="Raw execution trace of the multi-agent system")


class TraceSummarizer:
    """Agent for summarizing multi-agent system execution traces."""

    def __init__(
        self,
        model: OpenAIChatModel | str | None = None,
    ):
        """Initialize the trace summarizer.

        Args:
            model: OpenAI model instance for summarization
        """
        self.model = model
        self.prompt_template = TRACE_SUMMARIZATION_PROMPT

        # Create the pydantic AI agent with dependency injection
        self.agent = Agent(
            model=self.model,
            name="TraceSummarizer",
            instrument=True,
            output_type=TraceSummary,
            deps_type=TraceInput,
        )

        # Add system prompt that includes the trace data
        @self.agent.system_prompt
        def get_system_prompt(ctx: RunContext[TraceInput]) -> str:
            """Generate system prompt with trace data."""
            trace_input = ctx.deps
            
            return f"""{self.prompt_template}

**TRACE:**

{trace_input.trace}
"""

    async def summarize(self, trace: str) -> TraceSummary:
        """Summarize the execution trace.
        
        Args:
            trace: Raw execution trace string
            
        Returns:
            TraceSummary object with structured analysis
        """
        trace_input = TraceInput(trace=trace)
        result = await self.agent.run("Analyze the trace.", deps=trace_input)
        return result.output
