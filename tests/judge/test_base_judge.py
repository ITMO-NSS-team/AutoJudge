import pytest

from automas.judge.base import Judge, JudgeResult


class TestJudgeResult:
    """Test JudgeResult dataclass"""

    def test_judge_result_creation_minimal(self):
        """Should create JudgeResult with required fields only"""
        result = JudgeResult(
            score="ideal",
            justification="Perfect execution",
        )

        assert result.score == "ideal"
        assert result.justification == "Perfect execution"

    def test_judge_result_creation_with_metadata(self):
        """Should create JudgeResult with metadata"""
        result = JudgeResult(score="good", justification="Good execution")

        assert result.score == "good"

    def test_judge_result_valid_scores(self):
        """Should accept valid score values"""
        valid_scores = ["ideal", "good", "fair", "poor", "bad"]

        for score in valid_scores:
            result = JudgeResult(score=score, justification="Test")
            assert result.score == score

    def test_judge_result_numeric_score_mapping(self):
        """Should correctly map categorical scores to numeric values (0-1 scale)"""
        test_cases = [
            ("ideal", 1.0),
            ("good", 0.75),
            ("fair", 0.5),
            ("poor", 0.25),
            ("bad", 0.0),
        ]

        for score, expected_numeric in test_cases:
            result = JudgeResult(score=score, justification="Test")
            assert result.numeric_score == expected_numeric

    def test_judge_result_invalid_score_rejected(self):
        """Should reject invalid score values"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            JudgeResult(score="excellent", justification="Test")


class TestJudgeInterface:
    """Test Judge abstract interface"""

    @pytest.mark.asyncio
    async def test_judge_evaluate_returns_result(self, mock_judge, sample_pipeline_trace):
        """Should return JudgeResult from evaluate method"""
        judge = mock_judge(return_score="ideal")

        result = await judge.evaluate(sample_pipeline_trace)

        assert isinstance(result, JudgeResult)
        assert result.score == "ideal"
        assert "Mock evaluation" in result.justification

    @pytest.mark.asyncio
    async def test_judge_receives_complete_trace(self, mock_judge, sample_pipeline_trace):
        """Should receive complete PipelineTrace in evaluate method"""
        judge = mock_judge()

        await judge.evaluate(sample_pipeline_trace)

        # Verify judge received the trace
        assert len(judge.evaluated_traces) == 1
        assert judge.evaluated_traces[0] == sample_pipeline_trace

    @pytest.mark.asyncio
    async def test_judge_can_be_used_independently(self, mock_judge, sample_pipeline_trace):
        """Should work as standalone module without Pipeline"""
        judge = mock_judge(return_score="good")

        result = await judge.evaluate(sample_pipeline_trace)

        assert result.score == "good"
        assert isinstance(result, JudgeResult)

    def test_judge_is_abstract(self):
        """Should not allow direct instantiation of Judge base class"""
        with pytest.raises(TypeError):
            Judge()

    @pytest.mark.asyncio
    async def test_multiple_judges_can_evaluate_same_trace(self, mock_judge, sample_pipeline_trace):
        """Should allow multiple Judge implementations to evaluate same trace"""
        judge1 = mock_judge(return_score="ideal")
        judge2 = mock_judge(return_score="good")

        result1 = await judge1.evaluate(sample_pipeline_trace)
        result2 = await judge2.evaluate(sample_pipeline_trace)

        assert result1.score == "ideal"
        assert result2.score == "good"
        assert result1.score != result2.score
