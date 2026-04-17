from unittest.mock import Mock, patch

import pytest
from pydantic_ai import RunUsage

from autojudge import AgentNode
from autojudge.pipeline.pipeline import Pipeline


class TestPipelineCost:
    """Test pipeline cost calculation"""

    def test_cost_property_returns_zero_without_trace(self, simple_node):
        """Should return zero costs when pipeline hasn't been executed"""
        node = simple_node()
        pipeline = Pipeline(execution_order=[node])

        cost = pipeline.cost

        assert cost.input_price == 0.0
        assert cost.output_price == 0.0
        assert cost.total_price == 0.0

    @patch("autojudge.pipeline.types.calc_price")
    def test_cost_calculates_single_node(self, mock_calc_price, simple_node):
        """Should calculate cost for single node execution"""
        mock_result = Mock()
        mock_result.input_price = 0.01
        mock_result.output_price = 0.02
        mock_result.total_price = 0.03
        mock_calc_price.return_value = mock_result

        node = simple_node()
        pipeline = Pipeline(execution_order=[node])

        usage = RunUsage(input_tokens=1000, output_tokens=500)
        node._usage = usage

        cost = pipeline.cost

        assert cost.input_price == 0.01
        assert cost.output_price == 0.02
        assert cost.total_price == 0.03

    @patch("autojudge.pipeline.types.calc_price")
    def test_cost_sums_multiple_nodes(self, mock_calc_price, simple_node):
        """Should sum costs across multiple nodes"""
        mock_result1 = Mock()
        mock_result1.input_price = 0.01
        mock_result1.output_price = 0.02

        mock_result2 = Mock()
        mock_result2.input_price = 0.03
        mock_result2.output_price = 0.04

        mock_calc_price.side_effect = [mock_result1, mock_result2]

        node1 = simple_node(name="Agent1")
        node2 = simple_node(name="Agent2")
        pipeline = Pipeline(execution_order=[node1, node2])

        usage1 = RunUsage(input_tokens=1000, output_tokens=500)
        usage2 = RunUsage(input_tokens=2000, output_tokens=1000)

        node1._usage = usage1
        node2._usage = usage2

        cost = pipeline.cost

        assert cost.input_price == 0.04
        assert cost.output_price == 0.06
        assert cost.total_price == 0.10

    def test_cost_handles_node_without_usage(self, simple_node):
        """Should skip nodes without usage data"""
        node = simple_node()
        pipeline = Pipeline(execution_order=[node])

        node._usage = None

        cost = pipeline.cost

        assert cost.input_price == 0.0
        assert cost.output_price == 0.0
        assert cost.total_price == 0.0

    @patch("autojudge.pipeline.types.calc_price")
    def test_cost_extracts_provider_from_model(self, mock_calc_price, simple_node):
        """Should extract provider_id from model string"""
        mock_result = Mock()
        mock_result.input_price = 0.01
        mock_result.output_price = 0.02

        mock_calc_price.return_value = mock_result

        node = simple_node(model="google/gemini-2.5-flash")
        pipeline = Pipeline(execution_order=[node])

        usage = RunUsage(input_tokens=1000, output_tokens=500)
        node._usage = usage

        pipeline.cost

        call_args = mock_calc_price.call_args
        assert call_args[1]["provider_id"] == "google"
        assert call_args[1]["model_ref"] == "google/gemini-2.5-flash"

    @patch("autojudge.pipeline.types.calc_price")
    def test_cost_uses_openai_as_default_provider(self, mock_calc_price, simple_node):
        """Should use 'openai' as default provider when no slash in model"""
        mock_result = Mock()
        mock_result.input_price = 0.01
        mock_result.output_price = 0.02

        mock_calc_price.return_value = mock_result

        node = simple_node(model="gpt-4")
        pipeline = Pipeline(execution_order=[node])

        usage = RunUsage(input_tokens=1000, output_tokens=500)
        node._usage = usage

        pipeline.cost

        call_args = mock_calc_price.call_args
        assert call_args[1]["provider_id"] == "openai"

    @patch("autojudge.pipeline.types.calc_price")
    def test_cost_handles_calc_price_exception(self, mock_calc_price, simple_node):
        """Should handle exceptions from calc_price and continue"""
        mock_calc_price.side_effect = Exception("Model not found")

        node = simple_node(model="unknown/model")
        pipeline = Pipeline(execution_order=[node])

        usage = RunUsage(input_tokens=1000, output_tokens=500)
        node._usage = usage

        cost = pipeline.cost

        assert cost.input_price == 0.0
        assert cost.output_price == 0.0
        assert cost.total_price == 0.0


class TestPipelineCostIntegration:
    """Integration tests for cost calculation with real pipeline execution"""

    @pytest.mark.asyncio
    async def test_single_node_pipeline_cost(self):
        """Should calculate cost for single node pipeline execution"""
        node = AgentNode(
            name="Summarizer", instructions="Summarize the input in one sentence"
        )
        pipeline = Pipeline(execution_order=[node])

        await pipeline.ainvoke("Test query for cost calculation")

        assert pipeline.trace is not None
        assert len(pipeline.trace.node_traces) == 1

        cost = pipeline.cost

        assert isinstance(cost.input_price, float)
        assert isinstance(cost.output_price, float)
        assert isinstance(cost.total_price, float)
        assert cost.input_price >= 0.0
        assert cost.output_price >= 0.0
        assert cost.total_price >= 0.0
        assert cost.total_price == cost.input_price + cost.output_price

        assert pipeline.input_tokens > 0
        assert pipeline.output_tokens > 0
        assert pipeline.total_tokens == pipeline.input_tokens + pipeline.output_tokens
