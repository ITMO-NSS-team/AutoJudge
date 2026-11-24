"""Offline checks for specialist abstention and exact Meta plan budgets."""

import json
from typing import Any

import httpx
import pytest

from autojudge.jev.client import JEV_MODEL
from autojudge.jev.contracts import (
    EvaluationCase,
    EvaluationSpec,
    EvidenceEvent,
    JudgePlan,
    MetricSpec,
    SpecialistPlan,
    TargetSpec,
)
from autojudge.jev.engine import JevEvaluator
from autojudge.jev.planner import JudgePlanGenerator, PlanGenerationError


def _mandatory_event_target() -> tuple[EvaluationCase, EvaluationSpec]:
    case = EvaluationCase(
        case_id="failed-attempt",
        task="Find the first decisive mistake",
        events=[
            EvidenceEvent(id="event_1", index=1, actor="Alice", content="First event"),
            EvidenceEvent(id="event_2", index=2, actor="Bob", content="Second event"),
        ],
    )
    spec = EvaluationSpec(
        benchmark_id="attribution",
        taxonomy="Find the first causal mistake",
        targets=[
            TargetSpec(
                id="origin_message",
                question="Which event first caused the failure?",
                source="events",
                allow_none=False,
            )
        ],
        metrics=[
            MetricSpec(
                id="step_accuracy",
                target="origin_message",
                projection="index",
                label="step",
            )
        ],
    )
    return case, spec


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_choice", "expected_status"),
    [("c_2", "ok"), ("none", "error")],
)
async def test_specialist_can_abstain_but_mandatory_final_cannot(
    monkeypatch: pytest.MonkeyPatch, final_choice: str, expected_status: str
) -> None:
    monkeypatch.setattr(
        "autojudge.jev.client.decision_connection",
        lambda: ("https://example.test/api/alpha/decisions", "test-key"),
    )
    case, spec = _mandatory_event_target()
    plan = JudgePlan(
        specialists=[
            SpecialistPlan(name="SCOPE_A", target_id="origin_message", question="Scope A check."),
            SpecialistPlan(name="SCOPE_B", target_id="origin_message", question="Scope B check."),
        ]
    )
    seen: set[str] = set()

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        question = payload["questions"]["q"]
        criteria = question["criteria"]
        instructions = question["instructions"]
        if "Scope A check." in instructions:
            stage, choice = "scope_a", "none"
            assert set(criteria) == {"c_1", "c_2", "none"}
            assert payload["state"]["prior_judgments"] == []
        elif "Scope B check." in instructions:
            stage, choice = "scope_b", "c_1"
            assert set(criteria) == {"c_1", "c_2", "none"}
            assert payload["state"]["prior_judgments"] == []
        else:
            stage, choice = "final", final_choice
            assert set(criteria) == {"c_1", "c_2"}
            assert [
                item["status"] for item in payload["state"]["prior_judgments"]
            ] == ["none", "ok"]
            assert [
                item["selected_ids"] for item in payload["state"]["prior_judgments"]
            ] == [[], ["event_1"]]
        seen.add(stage)
        probabilities = {
            key: float(key == (choice if choice in criteria else "c_2")) for key in criteria
        }
        return httpx.Response(
            200,
            json={
                "model": JEV_MODEL,
                "answers": {
                    "q": {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": probabilities,
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.01},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        result = await JevEvaluator(transport, max_attempts=1).evaluate(case, spec, plan)

    assert seen == {"scope_a", "scope_b", "final"}
    assert [item["answer"]["status"] for item in result["findings"]] == ["none", "ok"]
    assert result["findings"][1]["answer"]["selected_ids"] == ["event_1"]
    assert result["status"] == expected_status
    final_answer = result["answers"]["origin_message"]
    assert final_answer["status"] == expected_status
    if expected_status == "ok":
        assert final_answer["selected_ids"] == ["event_2"]
    else:
        assert final_answer["selected_ids"] == []
        assert result["calls"][-1]["status"] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize(("review_choice", "expected"), [("c_3", "event_3"), ("none", "event_1"), ("error", "event_1")])
async def test_review_uses_shortlist_and_keeps_first_verdict_on_failure(
    monkeypatch: pytest.MonkeyPatch, review_choice: str, expected: str
) -> None:
    monkeypatch.setattr(
        "autojudge.jev.client.decision_connection",
        lambda: ("https://example.test/api/alpha/decisions", "test-key"),
    )
    case, spec = _mandatory_event_target()
    case.events.extend(
        EvidenceEvent(id=f"event_{index}", index=index, actor="Alice", content=f"Evidence {index}")
        for index in (3, 4)
    )
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        question = payload["questions"]["q"]
        seen.append(set(question["criteria"]))
        assert len(payload["state"]["events"]) == 4
        assert payload["state"]["prior_judgments"] == []
        if len(seen) == 1:
            choice = "c_1"
            probabilities = {"c_1": 0.45, "c_2": 0.14, "c_3": 0.40, "c_4": 0.01}
        else:
            if review_choice == "error":
                return httpx.Response(503)
            choice = review_choice
            probabilities = {key: float(key == choice) for key in question["criteria"]}
        return httpx.Response(
            200,
            json={"model": JEV_MODEL, "answers": {"q": {
                "type": "choice", "choice": choice, "probabilities": probabilities,
            }}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        result = await JevEvaluator(transport, max_attempts=1).evaluate(case, spec, review="reviewed")

    assert seen == [{"c_1", "c_2", "c_3", "c_4"}, {"c_1", "c_2", "c_3", "none"}]
    assert result["answers"]["origin_message"]["selected_ids"] == [expected]
    assert result["calls"][-1]["stage"] == "final:origin_message:review"


@pytest.mark.asyncio
@pytest.mark.parametrize(("challenger_score", "expected"), [(0.75, "event_2"), (0.54, "event_1")])
async def test_evidence_requires_a_supported_challenger(
    monkeypatch: pytest.MonkeyPatch, challenger_score: float, expected: str
) -> None:
    monkeypatch.setattr(
        "autojudge.jev.client.decision_connection",
        lambda: ("https://example.test/api/alpha/decisions", "test-key"),
    )
    case, spec = _mandatory_event_target()

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        questions = payload["questions"]
        if "q" in questions:
            answers = {"q": {
                "type": "choice", "choice": "c_1", "probabilities": {"c_1": 0.6, "c_2": 0.4},
            }}
        else:
            assert set(questions) == {"c_1", "c_2"}
            assert len(payload["state"]["events"]) == 2
            answers = {
                key: {
                    "type": "choice",
                    "choice": "origin" if score >= 0.5 else "downstream",
                    "probabilities": {"origin": score, "downstream": 1 - score, "other": 0.0},
                }
                for key, score in {"c_1": 0.42, "c_2": challenger_score}.items()
            }
        return httpx.Response(200, json={"model": JEV_MODEL, "answers": answers})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        result = await JevEvaluator(transport, max_attempts=1).evaluate(case, spec, review="evidence")

    assert result["answers"]["origin_message"]["selected_ids"] == [expected]
    assert result["calls"][-1]["stage"] == "final:origin_message:evidence"


@pytest.mark.asyncio
@pytest.mark.parametrize("returned_count", [2, 4])
async def test_meta_plan_rejects_wrong_exact_specialist_count(
    monkeypatch: pytest.MonkeyPatch, returned_count: int
) -> None:
    case, spec = _mandatory_event_target()
    prompts: list[dict[str, Any]] = []

    async def fake_run_agent(self: JudgePlanGenerator, prompt: str) -> dict[str, Any]:
        prompts.append(json.loads(prompt))
        return {
            "specialists": [
                {
                    "name": f"SCOPE_{index}",
                    "target_id": "origin_message",
                    "question": f"Check {index}.",
                }
                for index in range(returned_count)
            ]
        }

    monkeypatch.setattr(JudgePlanGenerator, "_run_agent", fake_run_agent)
    # create_plan itself needs no provider; bypass BaseMetaAgent's provider setup.
    generator = object.__new__(JudgePlanGenerator)

    with pytest.raises(PlanGenerationError, match=f"{returned_count} specialists; expected 3"):
        await generator.create_plan(case, spec, specialist_count=3)

    assert len(prompts) == 1
    assert prompts[0]["specialist_count"] == 3
    assert prompts[0]["required_target_ids"] == ["origin_message"]


@pytest.mark.asyncio
async def test_oversized_first_request_fails_before_specialist_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "autojudge.jev.client.decision_connection",
        lambda: ("https://example.test/api/alpha/decisions", "test-key"),
    )
    case, spec = _mandatory_event_target()
    plan = JudgePlan(
        specialists=[
            SpecialistPlan(name="SCOPE_A", target_id="origin_message", question="Scope A check."),
            SpecialistPlan(name="SCOPE_B", target_id="origin_message", question="Scope B check."),
        ]
    )

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Request escaped preflight: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as transport:
        result = await JevEvaluator(transport, max_state_chars=10).evaluate(case, spec, plan)

    assert result["status"] == "error"
    assert result["error"].startswith("request preflight final:origin_message:")
    assert result["calls"] == []
    assert result["cost_complete"] is True
    assert result["cost_usd"] == 0.0
