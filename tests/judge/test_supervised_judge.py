import pytest

from automas.judge.base import JudgeResult
from automas.judge.supervised_judge import SupervisedJudge
from automas.pipeline.types import NodeTrace, PipelineTrace


class TestSupervisedJudgeInitialization:
    """Test SupervisedJudge initialization"""

    def test_init_with_default_parameters(self):
        """Should initialize with default model and temperature"""
        judge = SupervisedJudge()

        assert judge.agent is not None
        assert judge.agent.name == "SupervisedJudge"

    def test_init_with_custom_model(self, test_model):
        """Should accept custom model parameter"""
        judge = SupervisedJudge(model=test_model)

        assert judge.agent is not None

    def test_init_with_custom_temperature(self):
        """Should accept custom temperature parameter"""
        judge = SupervisedJudge(temperature=0.5)

        assert judge.agent is not None

    def test_init_ground_truth_is_none_by_default(self):
        """Should initialize with ground_truth as None"""
        judge = SupervisedJudge()

        assert judge.ground_truth is None


class TestSupervisedJudgeGroundTruthManagement:
    """Test SupervisedJudge ground truth management"""

    def test_set_ground_truth(self):
        """Should set ground truth for evaluation"""
        judge = SupervisedJudge()
        ground_truth = "Expected answer to the question"

        judge.set_ground_truth(ground_truth)

        assert judge.ground_truth == ground_truth

    def test_set_ground_truth_overrides_previous(self):
        """Should override previous ground truth when set again"""
        judge = SupervisedJudge()

        judge.set_ground_truth("First ground truth")
        assert judge.ground_truth == "First ground truth"

        judge.set_ground_truth("Second ground truth")
        assert judge.ground_truth == "Second ground truth"

    def test_set_ground_truth_accepts_empty_string(self):
        """Should accept empty string as ground truth"""
        judge = SupervisedJudge()

        judge.set_ground_truth("")

        assert judge.ground_truth == ""

    def test_set_ground_truth_accepts_multiline_text(self):
        """Should accept multiline text as ground truth"""
        judge = SupervisedJudge()
        multiline_gt = """This is a multiline
ground truth example
with several lines"""

        judge.set_ground_truth(multiline_gt)

        assert judge.ground_truth == multiline_gt


class TestSupervisedJudgeEvaluation:
    """Test SupervisedJudge evaluation logic"""

    @pytest.mark.asyncio
    async def test_evaluate_returns_judge_result(self, test_model, sample_pipeline_trace):
        """Should return JudgeResult instance"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Expected output")

        result = await judge.evaluate(sample_pipeline_trace)

        assert isinstance(result, JudgeResult)

    @pytest.mark.asyncio
    async def test_evaluate_returns_valid_score(self, test_model, sample_pipeline_trace):
        """Should return score from valid ScoreValue set"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Expected output")
        valid_scores = {"ideal", "good", "fair", "poor", "bad"}

        result = await judge.evaluate(sample_pipeline_trace)

        assert result.score in valid_scores

    @pytest.mark.asyncio
    async def test_evaluate_includes_justification(self, test_model, sample_pipeline_trace):
        """Should include non-empty justification"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Expected output")

        result = await judge.evaluate(sample_pipeline_trace)

        assert result.justification
        assert len(result.justification) > 0

    @pytest.mark.asyncio
    async def test_evaluate_with_single_node_trace(self, test_model):
        """Should evaluate pipeline with single node"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Simple expected output")

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
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Complex expected output")

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
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Expected output")

        trace = PipelineTrace(
            session_id="test", original_query="Query", node_traces=[], final_output="Output"
        )

        result = await judge.evaluate(trace)

        assert isinstance(result, JudgeResult)
        # Likely should return "bad" or "poor" for empty traces

    @pytest.mark.asyncio
    async def test_evaluate_without_ground_truth_raises_error(
        self, test_model, sample_pipeline_trace
    ):
        """Should raise error when evaluating without ground truth"""
        judge = SupervisedJudge(model=test_model)
        # Not setting ground truth

        with pytest.raises(Exception):  # Should raise ValueError or similar
            await judge.evaluate(sample_pipeline_trace)


class TestSupervisedJudgeEvaluationCriteria:
    """Test SupervisedJudge evaluates based on GT consistency criteria"""

    @pytest.mark.asyncio
    async def test_evaluates_consistency_with_gt(self, test_model):
        """Should consider consistency with ground truth in evaluation"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("The capital of France is Paris")

        trace = PipelineTrace(
            session_id="test",
            original_query="What is the capital of France?",
            node_traces=[NodeTrace(node_id="node1", node_name="Agent1", message_history=[])],
            final_output="The capital of France is Paris",
        )

        result = await judge.evaluate(trace)

        assert isinstance(result, JudgeResult)
        # Justification should mention GT or consistency
        assert any(
            keyword in result.justification.lower()
            for keyword in ["ground truth", "gt", "consistency", "match", "align"]
        )

    @pytest.mark.asyncio
    async def test_evaluates_state_consistency(self, test_model, sample_pipeline_trace):
        """Should consider state consistency between execution steps"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Expected answer")

        result = await judge.evaluate(sample_pipeline_trace)

        # Justification should mention state or flow
        assert any(
            keyword in result.justification.lower()
            for keyword in ["state", "flow", "step", "execution", "coherence"]
        )

    @pytest.mark.asyncio
    async def test_evaluates_role_distribution(self, test_model, sample_pipeline_trace):
        """Should consider role distribution and coordination"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Expected answer")

        result = await judge.evaluate(sample_pipeline_trace)

        # Justification should mention roles, tools, or agents
        assert any(
            keyword in result.justification.lower()
            for keyword in ["role", "tool", "agent", "coordination", "responsibility"]
        )

    @pytest.mark.asyncio
    async def test_evaluates_system_completeness(self, test_model, sample_pipeline_trace):
        """Should consider system completeness and coverage"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("Complete comprehensive answer")

        result = await judge.evaluate(sample_pipeline_trace)

        # Justification should mention completeness or coverage
        assert any(
            keyword in result.justification.lower()
            for keyword in ["complete", "coverage", "comprehensive", "aspect", "edge case"]
        )

    @pytest.mark.asyncio
    async def test_perfect_match_produces_ideal_score(self, test_model):
        """Should produce 'ideal' score for perfect GT match with flawless execution"""
        judge = SupervisedJudge(model=test_model)
        expected_output = "Paris is the capital of France"
        judge.set_ground_truth(expected_output)

        trace = PipelineTrace(
            session_id="test",
            original_query="What is the capital of France?",
            node_traces=[
                NodeTrace(node_id="node1", node_name="KnowledgeAgent", message_history=[])
            ],
            final_output=expected_output,
        )

        result = await judge.evaluate(trace)

        # Should be ideal or at least good
        assert result.score in ["ideal", "good"]

    @pytest.mark.asyncio
    async def test_complete_mismatch_produces_bad_score(self, test_model):
        """Should produce 'bad' or 'poor' score for complete GT mismatch"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("The capital of France is Paris")

        trace = PipelineTrace(
            session_id="test",
            original_query="What is the capital of France?",
            node_traces=[NodeTrace(node_id="node1", node_name="Agent1", message_history=[])],
            final_output="The capital of Germany is Berlin",  # Wrong answer
        )

        result = await judge.evaluate(trace)

        # Should be bad or poor
        assert result.score in ["bad", "poor"]


class TestSupervisedJudgeDifferentGroundTruths:
    """Test SupervisedJudge with different ground truths produces different evaluations"""

    @pytest.mark.asyncio
    async def test_different_ground_truths_affect_evaluation(self, test_model):
        """Should produce different evaluations for different ground truths on same trace"""
        trace = PipelineTrace(
            session_id="test",
            original_query="What is 2+2?",
            node_traces=[NodeTrace(node_id="node1", node_name="MathAgent", message_history=[])],
            final_output="4",
        )

        # First evaluation with correct GT
        judge1 = SupervisedJudge(model=test_model, temperature=0.0)
        judge1.set_ground_truth("4")
        result1 = await judge1.evaluate(trace)

        # Second evaluation with incorrect GT
        judge2 = SupervisedJudge(model=test_model, temperature=0.0)
        judge2.set_ground_truth("5")
        result2 = await judge2.evaluate(trace)

        # Results should differ significantly
        assert result1.numeric_score != result2.numeric_score
        assert result1.numeric_score > result2.numeric_score  # Correct GT should score higher

    @pytest.mark.asyncio
    async def test_same_ground_truth_produces_consistent_results(self, test_model):
        """Should produce consistent results with same GT and temperature=0.0"""
        judge = SupervisedJudge(model=test_model, temperature=0.0)
        judge.set_ground_truth("Expected answer")

        trace = PipelineTrace(
            session_id="test",
            original_query="Question",
            node_traces=[NodeTrace(node_id="node1", node_name="Agent1", message_history=[])],
            final_output="Expected answer",
        )

        result1 = await judge.evaluate(trace)
        result2 = await judge.evaluate(trace)

        # With temperature 0, should be deterministic
        assert result1.score == result2.score


class TestSupervisedJudgeConsistency:
    """Test SupervisedJudge consistency and reliability"""

    @pytest.mark.asyncio
    async def test_multiple_evaluations_with_temperature_zero(
        self, test_model, sample_pipeline_trace
    ):
        """Should produce consistent results with temperature=0.0"""
        judge = SupervisedJudge(model=test_model, temperature=0.0)
        judge.set_ground_truth("Consistent ground truth")

        result1 = await judge.evaluate(sample_pipeline_trace)
        result2 = await judge.evaluate(sample_pipeline_trace)

        # With temperature 0, should be deterministic
        assert result1.score == result2.score

    @pytest.mark.asyncio
    async def test_evaluate_accepts_all_trace_types(self, test_model):
        """Should handle various PipelineTrace configurations"""
        judge = SupervisedJudge(model=test_model)
        judge.set_ground_truth("General ground truth")

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

    @pytest.mark.asyncio
    async def test_ground_truth_persists_across_evaluations(self, test_model):
        """Should maintain ground truth across multiple evaluations"""
        judge = SupervisedJudge(model=test_model)
        gt = "Persistent ground truth"
        judge.set_ground_truth(gt)

        trace1 = PipelineTrace(
            session_id="t1",
            original_query="Q1",
            node_traces=[NodeTrace(node_id="n1", node_name="A1", message_history=[])],
            final_output="Out1",
        )
        trace2 = PipelineTrace(
            session_id="t2",
            original_query="Q2",
            node_traces=[NodeTrace(node_id="n2", node_name="A2", message_history=[])],
            final_output="Out2",
        )

        await judge.evaluate(trace1)
        assert judge.ground_truth == gt

        await judge.evaluate(trace2)
        assert judge.ground_truth == gt


class TestSupervisedJudgeEdgeCases:
    """Test SupervisedJudge edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_evaluate_with_very_long_ground_truth(self, test_model):
        """Should handle very long ground truth text"""
        judge = SupervisedJudge(model=test_model)
        long_gt = "Expected answer. " * 1000  # Very long text
        judge.set_ground_truth(long_gt)

        trace = PipelineTrace(
            session_id="test",
            original_query="Question",
            node_traces=[NodeTrace(node_id="n", node_name="A", message_history=[])],
            final_output="Short output",
        )

        result = await judge.evaluate(trace)
        assert isinstance(result, JudgeResult)

    @pytest.mark.asyncio
    async def test_evaluate_with_special_characters_in_gt(self, test_model):
        """Should handle special characters in ground truth"""
        judge = SupervisedJudge(model=test_model)
        special_gt = "Answer with special chars: @#$%^&*()[]{}|\\<>?/~`"
        judge.set_ground_truth(special_gt)

        trace = PipelineTrace(
            session_id="test",
            original_query="Question",
            node_traces=[NodeTrace(node_id="n", node_name="A", message_history=[])],
            final_output="Regular output",
        )

        result = await judge.evaluate(trace)
        assert isinstance(result, JudgeResult)

    @pytest.mark.asyncio
    async def test_evaluate_with_unicode_in_gt(self, test_model):
        """Should handle Unicode characters in ground truth"""
        judge = SupervisedJudge(model=test_model)
        unicode_gt = "Ответ на русском языке: Париж 🇫🇷 是巴黎"
        judge.set_ground_truth(unicode_gt)

        trace = PipelineTrace(
            session_id="test",
            original_query="Question",
            node_traces=[NodeTrace(node_id="n", node_name="A", message_history=[])],
            final_output="Paris",
        )

        result = await judge.evaluate(trace)
        assert isinstance(result, JudgeResult)
