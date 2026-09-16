"""Graph generation reads the adjacency list as JSON text. No provider requests."""
import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from autojudge.agent_pool import AgentPool
from autojudge.meta_agents.graph_gen import GraphGenerator, parse_graph
from autojudge.pipeline.node import AgentNode

CHAIN = {
    "TOOL_JUDGE": ["EVIDENCE_JUDGE"],
    "EVIDENCE_JUDGE": ["FINAL_AGGREGATOR"],
    "FINAL_AGGREGATOR": [],
}


def pool():
    return AgentPool([
        AgentNode(name=name, instructions="judge", api_key="test-key", use_tools=False)
        for name in ("TOOL_JUDGE", "EVIDENCE_JUDGE", "FINAL_AGGREGATOR")
    ])


def test_plain_json_object():
    assert parse_graph('{"A": ["B"], "B": []}') == {"A": ["B"], "B": []}


def test_markdown_fenced_json():
    assert parse_graph('```json\n{"A": ["B"], "B": []}\n```') == {"A": ["B"], "B": []}
    assert parse_graph('``\n{"A": []}\n``') == {"A": []}


def test_dict_passes_through_and_null_children_become_empty():
    assert parse_graph({"A": ["B"], "B": None}) == {"A": ["B"], "B": []}


@pytest.mark.parametrize(
    "output",
    ['{}', '   ', 'not json', '[]', '{"A": "B"}', '{"A": [1]}', 42],
)
def test_unusable_responses_are_rejected(output):
    with pytest.raises(ValueError):
        parse_graph(output)


async def test_create_graph_reads_a_textual_graph(monkeypatch):
    """A model that cannot emit an open-ended map still drives the pipeline."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def answer(messages, info):
        assert not info.output_tools, "the graph must not depend on structured output"
        return ModelResponse(parts=[TextPart(
            '```json\n{"TOOL_JUDGE": ["EVIDENCE_JUDGE"], '
            '"EVIDENCE_JUDGE": ["FINAL_AGGREGATOR"], "FINAL_AGGREGATOR": []}\n```'
        )])

    monkeypatch.setattr(
        "autojudge.meta_agents.base.BaseMetaAgent._create_model",
        lambda self: FunctionModel(answer),
    )
    with models.override_allow_model_requests(False):
        graph = await GraphGenerator().create_graph(pool(), "Evaluate the trace")
    assert graph == CHAIN


async def test_empty_model_response_is_reported(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "autojudge.meta_agents.base.BaseMetaAgent._create_model",
        lambda self: FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("{}")])),
    )
    with models.override_allow_model_requests(False):
        with pytest.raises(ValueError, match="non-empty JSON object"):
            await GraphGenerator().create_graph(pool(), "Evaluate the trace")
