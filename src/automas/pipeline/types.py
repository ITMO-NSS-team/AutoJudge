from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from genai_prices import Usage, calc_price
from pydantic import BaseModel
from pydantic_ai import RunUsage
from pydantic_ai.messages import ModelMessage

if TYPE_CHECKING:
    from .node import AgentNode

from automas.utils.logger import get_logger

logger = get_logger()

# Graph structure: node name/id -> list of children names/ids
GraphDict = Dict[str, List[str]]

# Execution levels: nodes grouped by dependency level for concurrent execution
ExecutionLevels = List[List["AgentNode"]]


class NodeTrace(BaseModel):
    """Trace of single node execution"""

    node_id: str
    node_name: str
    model: str
    message_history: List[ModelMessage]
    usage: Optional[RunUsage] = None


class PipelineTrace(BaseModel):
    """Trace of entire pipeline execution"""

    session_id: str
    original_query: str
    node_traces: List[NodeTrace]
    final_output: Any


class CostBreakdown(BaseModel):
    """Price of entire execution"""

    total_price: float
    input_price: float
    output_price: float


class UsageTrackingMixin:
    """Mixin providing token usage and cost tracking properties.

    Requires implementing class to have:
    - self._usage: Optional[RunUsage] - usage data from last run
    - self.model: str - model name (e.g., "anthropic/claude-sonnet-4")

    Provides properties:
    - input_tokens: int - total input tokens from last run
    - output_tokens: int - total output tokens from last run
    - total_tokens: int - sum of input and output tokens
    - cost: CostBreakdown - cost breakdown (input_price, output_price, total_price)
    """

    _usage: Optional[RunUsage]
    model: str

    @property
    def input_tokens(self) -> int:
        """Total input tokens from last run."""
        if self._usage is None:
            return 0
        return getattr(self._usage, "input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Total output tokens from last run."""
        if self._usage is None:
            return 0
        return getattr(self._usage, "output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) from last run."""
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> CostBreakdown:
        """Cost breakdown for last run."""
        if self._usage is None:
            return CostBreakdown(input_price=0.0, output_price=0.0, total_price=0.0)

        try:
            input_tok = getattr(self._usage, "input_tokens", 0)
            output_tok = getattr(self._usage, "output_tokens", 0)

            usage = Usage(
                input_tokens=input_tok,
                output_tokens=output_tok,
            )

            provider_id = self.model.split("/")[0]  # type: ignore
            model = self.model.split("/")[1]  # type: ignore

            price_data = calc_price(usage, model_ref=model, provider_id=provider_id)

            return CostBreakdown(
                input_price=float(price_data.input_price),
                output_price=float(price_data.output_price),
                total_price=float(price_data.input_price) + float(price_data.output_price),
            )

        except Exception as e:
            logger.debug(f"Could not calculate price for model {self.model}: {e}")
            return CostBreakdown(input_price=0.0, output_price=0.0, total_price=0.0)
