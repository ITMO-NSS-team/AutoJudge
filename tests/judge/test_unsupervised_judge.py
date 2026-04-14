import pytest

from autojudge.judge.base import JudgeResult
from autojudge.judge.unsupervised_judge import UnsupervisedJudge
from autojudge.pipeline.types import NodeTrace, PipelineTrace


class TestUnsupervisedJudgeInitialization:
    """Test UnsupervisedJudge initialization"""

    def test_init_with_default_parameters(self):
        """Should initialize with default model and temperature"""
        judge = UnsupervisedJudge()

        assert judge.agent is not None
        assert judge.agent.name == "UnsupervisedJudge"

    def test_init_with_custom_model(self, test_model):
        """Should accept custom model parameter"""
        judge = UnsupervisedJudge(model=test_model)

        assert judge.agent is not None

    def test_init_with_custom_temperature(self):
        """Should accept custom temperature parameter"""
        judge = UnsupervisedJudge(temperature=0.5)

        assert judge.agent is not None


class TestUnsupervisedJudgeEvaluation:
    """Test UnsupervisedJudge evaluation logic"""

    @pytest.mark.asyncio
    async def test_evaluate_returns_judge_result(self, test_model, sample_pipeline_trace):
        """Should return JudgeResult instance"""
        judge = UnsupervisedJudge(model=test_model)

        result = await judge.evaluate(sample_pipeline_trace)

        assert isinstance(result, JudgeResult)

    @pytest.mark.asyncio
    async def test_evaluate_returns_valid_score(self, test_model, sample_pipeline_trace):
        """Should return score from valid ScoreValue set"""
        judge = UnsupervisedJudge(model=test_model)
        valid_scores = {"ideal", "good", "fair", "poor", "bad"}

        result = await judge.evaluate(sample_pipeline_trace)

        assert result.score in valid_scores

    @pytest.mark.asyncio
    async def test_evaluate_includes_justification(self, test_model, sample_pipeline_trace):
        """Should include non-empty justification"""
        judge = UnsupervisedJudge(model=test_model)

        result = await judge.evaluate(sample_pipeline_trace)

        assert result.justification
        assert len(result.justification) > 0

    @pytest.mark.asyncio
    async def test_evaluate_with_single_node_trace(self, test_model):
        """Should evaluate pipeline with single node"""
        judge = UnsupervisedJudge(model=test_model)
        trace = PipelineTrace(
            session_id="test",
            original_query="Simple query",
            node_traces=[NodeTrace(node_id="node1", node_name="Agent1", message_history=[])],
            final_output="Output",
        )

        result = await judge.evaluate(trace)

        assert isinstance(result, JudgeResult)

    @pytest.mark.asyncio
    async def test_evaluate_with_multiple_node_traces(self, test_model):
        """Should evaluate pipeline with multiple interconnected nodes"""
        judge = UnsupervisedJudge(model=test_model)
        trace = PipelineTrace(
            session_id="test",
            original_query="Complex query",
            node_traces=[
                NodeTrace(node_id="node1", node_name="Agent1", message_history=[]),
                NodeTrace(node_id="node2", node_name="Agent2", message_history=[]),
                NodeTrace(node_id="node3", node_name="Agent3", message_history=[]),
            ],
            final_output="Output",
        )

        result = await judge.evaluate(trace)

        assert isinstance(result, JudgeResult)

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_node_traces(self, test_model):
        """Should handle pipeline with no node traces"""
        judge = UnsupervisedJudge(model=test_model)
        trace = PipelineTrace(
            session_id="test", original_query="Query", node_traces=[], final_output="Output"
        )

        result = await judge.evaluate(trace)

        assert isinstance(result, JudgeResult)
        # Likely should return "bad" or "poor" for empty traces


class TestUnsupervisedJudgeEvaluationCriteria:
    """Test UnsupervisedJudge evaluates based on complexity criteria"""

    @pytest.mark.asyncio
    async def test_evaluates_agent_density(self, test_model):
        """Should consider number of agents in evaluation"""
        judge = UnsupervisedJudge(model=test_model)

        # Pipeline with appropriate density
        balanced_trace = PipelineTrace(
            session_id="test",
            original_query="Moderate complexity task",
            node_traces=[
                NodeTrace(node_id=f"node{i}", node_name=f"Agent{i}", message_history=[])
                for i in range(3)
            ],
            final_output="Output",
        )

        result = await judge.evaluate(balanced_trace)

        assert isinstance(result, JudgeResult)
        # Justification should mention density or agent count
        assert any(
            keyword in result.justification.lower() for keyword in ["agent", "density", "number"]
        )

    @pytest.mark.asyncio
    async def test_evaluates_interconnection_quality(self, test_model, sample_pipeline_trace):
        """Should consider agent connections in evaluation"""
        judge = UnsupervisedJudge(model=test_model)

        result = await judge.evaluate(sample_pipeline_trace)

        # Justification should mention connections or communication
        assert any(
            keyword in result.justification.lower()
            for keyword in ["connection", "interconnect", "communication", "relationship"]
        )

    @pytest.mark.asyncio
    async def test_evaluates_system_scalability(self, test_model, sample_pipeline_trace):
        """Should consider scalability in evaluation"""
        judge = UnsupervisedJudge(model=test_model)

        result = await judge.evaluate(sample_pipeline_trace)

        # Justification should mention scalability or maintainability
        assert any(
            keyword in result.justification.lower()
            for keyword in ["scalability", "scalable", "maintainable", "extensible", "architecture"]
        )

    @pytest.mark.asyncio
    async def test_different_traces_produce_different_evaluations(self, test_model):
        """Should produce different evaluations for significantly different traces"""
        judge = UnsupervisedJudge(model=test_model)

        simple_trace = PipelineTrace(
            session_id="simple",
            original_query="Simple task",
            node_traces=[NodeTrace(node_id="node1", node_name="Agent1", message_history=[])],
            final_output="Output",
        )

        complex_trace = PipelineTrace(
            session_id="complex",
            original_query="Complex multi-step task",
            node_traces=[
                NodeTrace(node_id=f"node{i}", node_name=f"Agent{i}", message_history=[])
                for i in range(10)
            ],
            final_output="Output",
        )

        result_simple = await judge.evaluate(simple_trace)
        result_complex = await judge.evaluate(complex_trace)

        # Results should differ (either score or justification)
        assert (
            result_simple.score != result_complex.score
            or result_simple.justification != result_complex.justification
        )


class TestUnsupervisedJudgeConsistency:
    """Test UnsupervisedJudge consistency and reliability"""

    @pytest.mark.asyncio
    async def test_multiple_evaluations_with_temperature_zero(
        self, test_model, sample_pipeline_trace
    ):
        """Should produce consistent results with temperature=0.0"""
        judge = UnsupervisedJudge(model=test_model, temperature=0.0)

        result1 = await judge.evaluate(sample_pipeline_trace)
        result2 = await judge.evaluate(sample_pipeline_trace)

        # With temperature 0, should be deterministic
        assert result1.score == result2.score

    @pytest.mark.asyncio
    async def test_evaluate_accepts_all_trace_types(self, test_model):
        """Should handle various PipelineTrace configurations"""
        judge = UnsupervisedJudge(model=test_model)

        traces = [
            # Minimal trace
            PipelineTrace(session_id="min", original_query="Q", node_traces=[], final_output=""),
            # Standard trace
            PipelineTrace(
                session_id="std",
                original_query="Query",
                node_traces=[NodeTrace(node_id="n", node_name="A", message_history=[])],
                final_output="Out",
            ),
            # Large trace
            PipelineTrace(
                session_id="large",
                original_query="Complex query with many details",
                node_traces=[
                    NodeTrace(node_id=f"node{i}", node_name=f"Agent{i}", message_history=[])
                    for i in range(20)
                ],
                final_output="Detailed output",
            ),
        ]

        for trace in traces:
            result = await judge.evaluate(trace)
            assert isinstance(result, JudgeResult)
