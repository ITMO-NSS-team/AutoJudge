import pytest

from autojudge.pipeline.pipeline import Pipeline
from autojudge.pipeline.types import NodeTrace, PipelineTrace


class TestNodeTrace:
    """Test NodeTrace dataclass structure"""

    def test_node_trace_creation(self):
        """Should create NodeTrace with all required fields"""
        node_trace = NodeTrace(
            node_id="node1",
            node_name="TestAgent",
            message_history=[],
        )

        assert node_trace.node_id == "node1"
        assert node_trace.node_name == "TestAgent"
        assert node_trace.message_history == []


class TestPipelineTrace:
    """Test PipelineTrace dataclass structure"""

    def test_pipeline_trace_creation(self, sample_node_trace):
        """Should create PipelineTrace with all required fields"""
        pipeline_trace = PipelineTrace(
            session_id="session_123",
            original_query="What is the weather?",
            node_traces=[sample_node_trace],
            final_output="Final result",
        )

        assert pipeline_trace.session_id == "session_123"
        assert pipeline_trace.original_query == "What is the weather?"
        assert len(pipeline_trace.node_traces) == 1
        assert pipeline_trace.node_traces[0] == sample_node_trace
        assert pipeline_trace.final_output == "Final result"


class TestPipelineTraceCollection:
    """Test trace collection via pipeline.trace property"""

    def test_unexecuted_pipeline_has_no_trace(self, simple_node):
        """Should return None when pipeline has not been executed"""
        node = simple_node(name="TestAgent", instructions="Test instructions")
        pipeline = Pipeline(execution_order=[node])

        assert pipeline.trace is None

    @pytest.mark.asyncio
    async def test_executed_pipeline_has_trace(self, simple_node):
        """Should return PipelineTrace after pipeline execution"""
        node = simple_node()
        pipeline = Pipeline(execution_order=[node])

        # Execute pipeline
        result = await pipeline.ainvoke("Say hello")

        # Get trace via property
        trace = pipeline.trace

        # Verify trace structure
        assert trace is not None
        assert isinstance(trace, PipelineTrace)
        assert trace.session_id == pipeline.node_session.session_id
        assert trace.original_query == "Say hello"
        assert len(trace.node_traces) == 1
        assert trace.final_output == result

    @pytest.mark.asyncio
    async def test_multi_node_pipeline_trace(self, simple_node):
        """Should collect traces from all nodes in execution order"""
        node1 = simple_node(name="Agent1", instructions="You are agent 1")
        node2 = simple_node(name="Agent2", instructions="You are agent 2")
        node1.add_child(node2)

        pipeline = Pipeline(execution_order=[node1, node2])
        result = await pipeline.ainvoke("Test query")

        trace = pipeline.trace

        assert trace is not None
        assert len(trace.node_traces) == 2
        assert trace.node_traces[0].node_name == "Agent1"
        assert trace.node_traces[1].node_name == "Agent2"
        assert trace.final_output == result

    @pytest.mark.asyncio
    async def test_node_trace_contains_message_history(self, simple_node):
        """Should capture message history from PydanticAI RunResult"""
        node = simple_node(name="Agent", instructions="Reply with: Hello!")
        pipeline = Pipeline(execution_order=[node])
        await pipeline.ainvoke("Say hello")

        trace = pipeline.trace

        assert trace is not None
        node_trace = trace.node_traces[0]
        assert len(node_trace.message_history) > 0
        # Should contain at least request and response
        assert any(msg.kind == "request" for msg in node_trace.message_history)
        assert any(msg.kind == "response" for msg in node_trace.message_history)

    @pytest.mark.asyncio
    async def test_trace_available_for_judge_evaluation(self, simple_node):
        """Should provide trace data suitable for Judge evaluation"""
        node = simple_node(name="Agent", instructions="Repeat the input")
        pipeline = Pipeline(execution_order=[node])
        query = "Test input"
        result = await pipeline.ainvoke(query)

        trace = pipeline.trace

        # Verify trace has all data needed for evaluation
        assert trace is not None
        assert trace.session_id is not None
        assert trace.original_query == query
        assert len(trace.node_traces) == 1
        assert trace.final_output == result
        assert trace.node_traces[0].message_history is not None
