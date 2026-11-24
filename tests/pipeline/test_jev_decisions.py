import json

import httpx
import pytest
from pydantic_ai.models.test import TestModel

from autojudge.meta_agents.decision_pool import DecisionPoolGenerator, direct_decision_pool
from autojudge.meta_agents.graph_gen import get_parallel_graph
from autojudge.pipeline import PipelineBuilder
from autojudge.pipeline.decisions import JEV_MODEL, DecisionTrace

TRACE = DecisionTrace(
    query="What is 2 + 2?",
    history_for_evaluating=[
        {"role": "Planner (thought)", "content": "Add 2 and 2."},
        {"name": "Solver", "content": "The answer is 5."},
    ],
)


@pytest.fixture(autouse=True)
def isolate_gateway_configuration(monkeypatch):
    monkeypatch.delenv("OMNIROUTE_BASE_URL", raising=False)
    monkeypatch.delenv("OMNIROUTE_DECISIONS_URL", raising=False)
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_DECISIONS_URL", raising=False)


@pytest.mark.asyncio
async def test_meta_agent_plan_runs_specialists_then_jev_aggregator(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        DecisionPoolGenerator,
        "_create_model",
        lambda self: TestModel(
            custom_output_args={
                "specialists": [
                    {
                        "name": "ARITHMETIC",
                        "question": "Which message first miscalculates the sum?",
                    },
                    {
                        "name": "PLANNING",
                        "question": "Which message first proposes an invalid plan?",
                    },
                ]
            }
        ),
    )
    generator = DecisionPoolGenerator()
    pool = await generator.create_pool(TRACE)
    pipeline = PipelineBuilder().create_from_pool(pool, get_parallel_graph(pool)).build()
    requests = []

    def decide(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/api/alpha/decisions"
        assert body["model"] == JEV_MODEL
        assert body["state"]["trace"] == TRACE.model_dump()
        question = body["questions"]["attribution"]
        criteria = question["criteria"]
        assert criteria["step_1"] == "Message 1 by Planner"
        findings = body["state"]["prior_judgments"]
        if findings:
            assert len(requests) == 3
            by_judge = {item["node_name"]: item["output"] for item in findings}
            assert by_judge["ARITHMETIC"]["candidate"]["step"] == 2
            assert by_judge["PLANNING"]["candidate"] is None
            assert set(criteria) == {"step_1", "step_2"}
            answer = "step_2"
        else:
            assert set(criteria) == {"step_1", "step_2", "none"}
            answer = "none" if "invalid plan" in question["instructions"] else "step_2"
        return httpx.Response(
            200,
            json={
                "model": JEV_MODEL + "-20261014",
                "answers": {
                    "attribution": {
                        "type": "choice",
                        "choice": answer,
                        "probabilities": {key: float(key == answer) for key in criteria},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 0.0001},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(decide)) as client:
        verdict = await pipeline.ainvoke(TRACE.model_dump(), decision_client=client)
    assert verdict == {"agent": "Solver", "step": 2, "message": TRACE.history_for_evaluating[1]}
    assert len(requests) == len(pipeline.trace.node_traces) == 3
    assert all(node.decision["model"].startswith(JEV_MODEL) for node in pipeline.trace.node_traces)
    assert pipeline.cost.total_price == pytest.approx(0.0003)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_direct_single_candidate_requires_confirmation(monkeypatch, accepted):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    pool = direct_decision_pool()
    pipeline = PipelineBuilder().create_from_pool(pool, get_parallel_graph(pool)).build()
    trace = TRACE.model_dump()
    trace["history_for_evaluating"] = trace["history_for_evaluating"][1:]
    calls = []

    def decide(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["state"]["prior_judgments"] == []
        assert body["questions"]["attribution"]["type"] == "noul"
        return httpx.Response(
            200,
            json={
                "model": JEV_MODEL,
                "answers": {"attribution": {"type": "noul", "noul": 0.5 if accepted else 0.4}},
                "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.00001},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(decide)) as client:
        if accepted:
            result = await pipeline.ainvoke(trace, decision_client=client)
            assert (result["agent"], result["step"]) == ("Solver", 1)
        else:
            with pytest.raises(RuntimeError, match="rejected the only candidate"):
                await pipeline.ainvoke(trace, decision_client=client)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_omniroute_decisions_use_gateway_url_and_key(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "gateway-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "direct-test")
    monkeypatch.setenv("OMNIROUTE_DECISIONS_URL", "https://gateway.example/custom/decisions")
    pool = direct_decision_pool()
    pipeline = PipelineBuilder().create_from_pool(pool, get_parallel_graph(pool)).build()

    def decide(request):
        assert str(request.url) == "https://gateway.example/custom/decisions"
        assert request.headers["authorization"] == "Bearer gateway-test"
        return httpx.Response(
            200,
            json={
                "model": JEV_MODEL,
                "answers": {
                    "attribution": {
                        "type": "choice",
                        "choice": "step_2",
                        "probabilities": {"step_1": 0.1, "step_2": 0.9},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 0.0001},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(decide)) as client:
        result = await pipeline.ainvoke(TRACE.model_dump(), decision_client=client)
    assert result["agent"] == "Solver"


def test_gateway_cannot_fall_back_to_direct_decisions(monkeypatch):
    from autojudge.model_connection import decision_connection

    monkeypatch.setenv("OMNIROUTE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("OPENROUTER_API_KEY", "direct-test")
    with pytest.raises(RuntimeError, match="Set OMNIROUTE_DECISIONS_URL"):
        decision_connection()
    monkeypatch.setenv("OMNIROUTE_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")
    with pytest.raises(ValueError, match="same origin"):
        decision_connection()


@pytest.mark.asyncio
async def test_chat_provider_uses_omniroute_configuration(monkeypatch):
    from autojudge import model_connection

    monkeypatch.setenv("OMNIROUTE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "gateway-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "direct-test")

    def chat(request):
        assert str(request.url) == "https://gateway.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer gateway-test"
        return httpx.Response(200, json={"object": "chat.completion", "choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(chat)) as client:
        monkeypatch.setattr(model_connection, "model_http_client", lambda: client)
        provider = model_connection.chat_provider(model_connection.chat_api_key())
        await provider.client.chat.completions.create(
            model="example/model", messages=[{"role": "user", "content": "OK?"}]
        )


def test_explicit_openrouter_decisions_with_omniroute_chat(monkeypatch):
    from autojudge.model_connection import (
        OPENROUTER_DECISIONS_URL,
        chat_api_key,
        decision_connection,
    )

    monkeypatch.setenv("OMNIROUTE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "gateway-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "direct-test")
    monkeypatch.setenv("OPENROUTER_DECISIONS_URL", OPENROUTER_DECISIONS_URL + " ")
    assert chat_api_key() == "gateway-test"
    assert decision_connection() == (OPENROUTER_DECISIONS_URL, "direct-test")

    monkeypatch.setenv("OPENROUTER_DECISIONS_URL", "https://gateway.example/decisions")
    with pytest.raises(ValueError, match="official OpenRouter endpoint"):
        decision_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sequence", "expected_delays", "ok"),
    [
        (["read", 429, 200], [1, 3], True),
        ([401], [], False),
        ([503, 503, 503], [1, 2], False),
    ],
)
async def test_jev_retry_policy_records_requests(monkeypatch, sequence, expected_delays, ok):
    from autojudge.pipeline import decisions

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(decisions.asyncio, "sleep", sleep)
    pool = direct_decision_pool()
    pipeline = PipelineBuilder().create_from_pool(pool, get_parallel_graph(pool)).build()
    bodies = []

    def respond(request):
        outcome = sequence[len(bodies)]
        bodies.append(request.content)
        if outcome == "read":
            raise httpx.ReadError("temporary network failure", request=request)
        if outcome != 200:
            return httpx.Response(outcome, headers={"Retry-After": "3"} if outcome == 429 else {})
        return httpx.Response(
            200,
            json={
                "model": JEV_MODEL,
                "answers": {
                    "attribution": {
                        "type": "choice",
                        "choice": "step_2",
                        "probabilities": {"step_1": 0.1, "step_2": 0.9},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 0.0001},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        if ok:
            result = await pipeline.ainvoke(TRACE.model_dump(), decision_client=client)
            assert result["step"] == 2
        else:
            with pytest.raises(RuntimeError, match="HTTPStatusError"):
                await pipeline.ainvoke(TRACE.model_dump(), decision_client=client)
    assert delays == expected_delays
    assert len(bodies) == len(sequence)
    assert len(set(bodies)) == 1
    attempts = pipeline.execution_order[0].attempts
    assert len(attempts) == len(sequence)
    assert sum("usage" in attempt for attempt in attempts) == int(ok)
    if ok:
        assert attempts[0]["error"] == "ReadError"
        assert attempts[1]["status_code"] == 429
        assert pipeline.trace.node_traces[0].usage.requests == 3


@pytest.mark.asyncio
async def test_jev_caps_concurrent_specialists_and_passes_all_findings(monkeypatch):
    import asyncio

    from autojudge.agent_pool import AgentPool
    from autojudge.pipeline.decisions import DECISION_CONCURRENCY, FINAL_AGGREGATOR, DecisionNode

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    specialist_count = DECISION_CONCURRENCY + 2
    pool = AgentPool(
        [
            *[
                DecisionNode(name=f"SPECIALIST_{i}", instructions="Find a mistake.")
                for i in range(specialist_count)
            ],
            DecisionNode(name=FINAL_AGGREGATOR, instructions="Find the root cause.", final=True),
        ]
    )
    pipeline = PipelineBuilder().create_from_pool(pool, get_parallel_graph(pool)).build()
    first_wave = asyncio.Event()
    release = asyncio.Event()
    active = peak = 0

    async def respond(request):
        nonlocal active, peak
        body = json.loads(request.content)
        findings = body["state"]["prior_judgments"]
        criteria = body["questions"]["attribution"]["criteria"]
        if findings:
            assert len(findings) == specialist_count
        else:
            active += 1
            peak = max(peak, active)
            if active == DECISION_CONCURRENCY:
                first_wave.set()
            try:
                await release.wait()
            finally:
                active -= 1
        return httpx.Response(
            200,
            json={
                "model": JEV_MODEL,
                "answers": {
                    "attribution": {
                        "type": "choice",
                        "choice": "step_2",
                        "probabilities": {option: float(option == "step_2") for option in criteria},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 0.0001},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        running = asyncio.create_task(pipeline.ainvoke(TRACE.model_dump(), decision_client=client))
        try:
            await asyncio.wait_for(first_wave.wait(), timeout=5)
            assert peak == DECISION_CONCURRENCY
        finally:
            release.set()
            result = await running
    assert result["step"] == 2
    assert peak == DECISION_CONCURRENCY
