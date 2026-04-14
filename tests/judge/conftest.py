import pytest

from autojudge.judge.base import Judge, JudgeResult
from autojudge.pipeline.types import NodeTrace, PipelineTrace

TEST_MODEL = "openai/gpt-oss-120b"


@pytest.fixture
def test_model():
    """Model for testing"""
    return TEST_MODEL


@pytest.fixture
def sample_node_trace():
    """Sample NodeTrace for testing"""
    return NodeTrace(
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


class MockJudge(Judge):
    """Mock Judge implementation for testing"""

    def __init__(self, return_score: str = "good"):
        self.return_score = return_score
        self.evaluated_traces = []

    async def evaluate(self, trace: PipelineTrace) -> JudgeResult:
        self.evaluated_traces.append(trace)
        return JudgeResult(
            score=self.return_score,
            justification=f"Mock evaluation of {len(trace.node_traces)} nodes",
        )


@pytest.fixture
def mock_judge():
    """Factory fixture for creating MockJudge"""

    def _create_judge(return_score: str = "good"):
        return MockJudge(return_score=return_score)

    return _create_judge
