from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

from .prompts import STEP_BY_STEP_SUMMARIZATION_PROMPT


class StepByStepSummaryItem(BaseModel):
    """One step in the step-by-step summary."""

    id: str = Field(description="Step id in format <trace_id>_<step> (e.g., abc123_1)")
    name: str = Field(description="Agent name")
    role: str = Field(description="Agent role")
    content_summary: str = Field(
        description="Concise summary of what happened in this step"
    )


class StepByStepSummary(BaseModel):
    """Step-by-step summary of the execution trace."""

    step_by_step_summary: List[StepByStepSummaryItem] = Field(
        description="Chronological step-by-step summary"
    )


class TraceInput(BaseModel):
    """Input for trace summarization."""

    trace_id: str = Field(description="Unique trace id to use for step ids")
    trace: str = Field(description="Raw execution trace of the multi-agent system")


class StepByStepSummarizer:
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
        self.prompt_template = STEP_BY_STEP_SUMMARIZATION_PROMPT

        self.agent = Agent(
            model=self.model,
            name="TraceSummarizer",
            instrument=True,
            output_type=StepByStepSummary,
            deps_type=TraceInput,
        )

        @self.agent.system_prompt
        def get_system_prompt(ctx: RunContext[TraceInput]) -> str:
            """Generate system prompt with trace data."""
            trace_input = ctx.deps

            return f"""{self.prompt_template}

**TRACE_ID:**

{trace_input.trace_id}

**TRACE:**

{trace_input.trace}
"""

    async def summarize(self, trace: str, trace_id: str = "trace") -> StepByStepSummary:
        """Summarize the execution trace.

        Args:
            trace: Raw execution trace string
            trace_id: Unique trace id used to build step ids (<trace_id>_<step>)

        Returns:
            StepByStepSummary object with structured analysis
        """
        trace_input = TraceInput(trace_id=trace_id, trace=trace)
        result = await self.agent.run("Analyze the trace.", deps=trace_input)
        return result.output
