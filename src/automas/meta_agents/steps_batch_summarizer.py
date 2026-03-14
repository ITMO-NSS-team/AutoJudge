from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from typing import List, Optional

from .prompt_registry import STEPS_BATCH_SUMMARIZATION_PROMPT


class StepsBatchSummaryItem(BaseModel):
    """One step in the step-by-step summary."""

    id: str = Field(description="Step id (given)")
    name: str = Field(description="Agent name (as it is in state representation)")
    role: str = Field(description="Agent role")
    content_summary: str = Field(
        description="Concise summary of what happened in this step"
    )


class StepsBatchSummaryOutput(BaseModel):
    """Batch output containing summaries for all provided steps."""

    steps_batch_summaries: List[StepsBatchSummaryItem] = Field(
        description="List of step summaries, one per input state, in the same order"
    )


class StepBatchInput(BaseModel):
    """Input for batch step summarization."""

    task_id: str = Field(description="Task-level identifier shared by all states")
    states: List[dict] = Field(
        description="Ordered list of state representations to summarize"
    )
    state_ids: List[str] = Field(
        description="Ordered list of state ids corresponding to each state"
    )
    previous_summary: Optional[str] = Field(
        default=None,
        description="Optional textual summary of earlier states for context",
    )


class StepsBatchSummarizer:
    """Agent for summarizing a batch of multi-agent system execution steps."""

    def __init__(
        self,
        model: OpenAIChatModel | str | None = None,
    ):
        self.model = model
        self.prompt_template = STEPS_BATCH_SUMMARIZATION_PROMPT

        self.agent = Agent(
            model=self.model,
            name="StepsBatchSummarizer",
            instrument=True,
            output_type=StepsBatchSummaryOutput,
            deps_type=StepBatchInput,
        )

        @self.agent.system_prompt
        def get_system_prompt(ctx: RunContext[StepBatchInput]) -> str:
            batch = ctx.deps

            steps_block = ""
            for sid, state in zip(batch.state_ids, batch.states):
                steps_block += f"\n--- state_id: {sid} ---\n{state}\n"

            prev_block = ""
            if batch.previous_summary:
                prev_block = (
                    f"\n**Summary of previous states (for context only — do NOT "
                    f"re-summarize these):**\n\n{batch.previous_summary}\n"
                )

            return (
                f"{self.prompt_template}"
                f"{prev_block}"
                f"\n**States to summarize:**\n{steps_block}"
            )

    async def summarize(
        self,
        states: List[dict],
        state_ids: List[str],
        task_id: str,
        previous_summary: Optional[str] = None,
    ) -> StepsBatchSummaryOutput:
        """Summarize a batch of execution steps.

        Args:
            states: Ordered list of raw state dicts to summarize.
            state_ids: Matching list of unique ids for each state.
            task_id: Task-level identifier.
            previous_summary: Optional summary of states that came before
                              this batch (provides context but should not
                              be re-summarized).

        Returns:
            List of StepSummaryItem objects, one per input state.
        """
        batch_input = StepBatchInput(
            task_id=task_id,
            states=states,
            state_ids=state_ids,
            previous_summary=previous_summary,
        )
        result = await self.agent.run("Analyze the steps.", deps=batch_input)
        return result.output.step_summaries
