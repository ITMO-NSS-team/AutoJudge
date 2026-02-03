import pytest

from automas.pipeline.node import AgentNode
from automas.pipeline.types import NodeTrace, PipelineTrace

TEST_MODEL = "openai/gpt-oss-20b:free"


@pytest.fixture
def test_model():
    """Free model to use in tests"""
    return TEST_MODEL


@pytest.fixture
def simple_node():
    """Factory fixture for creating simple AgentNode"""

    def _create_node(
        name: str = "TestAgent",
        instructions: str = "You are a helpful assistant",
        model: str = TEST_MODEL,
    ):
        return AgentNode(
            name=name,
            instructions=instructions,
            model=model,
        )

    return _create_node


@pytest.fixture
def sample_node_trace():
    """Sample NodeTrace for testing"""
    return NodeTrace(
        model=TEST_MODEL,
        node_id="node1",
        node_name="TestAgent",
        message_history=[],
    )


@pytest.fixture
def sample_pipeline_trace(sample_node_trace):
    """Sample PipelineTrace for testing"""
    return PipelineTrace(
        session_id="test_session",
        original_query="Test query",
        node_traces=[sample_node_trace],
        final_output="Test output",
    )
