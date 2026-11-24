"""Offline checks for Jev transport accounting and conversion integrity."""

import json
from pathlib import Path
from runpy import run_path
from typing import Any

import httpx
import pytest

from autojudge.jev.client import JEV_MODEL, JevClient
from autojudge.jev.normalization import publish_conversion_manifest


QUESTIONS: dict[str, dict[str, Any]] = {
    "judgment": {
        "type": "choice",
        "instructions": "Choose the supported answer.",
        "criteria": {"yes": "Yes", "no": "No"},
    }
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_receipt", "recorded_cost", "cost_complete"),
    [(0.02, 0.05, True), (None, 0.03, False)],
)
async def test_retry_ledger_keeps_receipts_from_malformed_200(
    monkeypatch: pytest.MonkeyPatch,
    first_receipt: float | None,
    recorded_cost: float,
    cost_complete: bool,
) -> None:
    monkeypatch.setattr(
        "autojudge.jev.client.decision_connection",
        lambda: ("https://example.test/api/alpha/decisions", "test-key"),
    )
    malformed: dict[str, Any] = {"model": JEV_MODEL, "answers": {}}
    if first_receipt is not None:
        malformed["usage"] = {"cost": first_receipt}
    valid = {
        "id": "second-response",
        "model": JEV_MODEL,
        "answers": {
            "judgment": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.8, "no": 0.2},
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.03},
    }
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/api/alpha/decisions"
        return httpx.Response(200, json=malformed if len(requests) == 1 else valid)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        result = await JevClient(
            transport, max_attempts=2, retry_delay_seconds=0
        ).decide(state={"case_id": "one"}, questions=QUESTIONS, stage="judge")

    assert len(requests) == 2
    assert result["status"] == "ok"
    assert result["response_id"] == "second-response"
    assert result["answers"]["judgment"]["choice"] == "yes"
    attempts = result["attempts"]
    assert [attempt["number"] for attempt in attempts] == [1, 2]
    assert [attempt["status_code"] for attempt in attempts] == [200, 200]
    assert attempts[0]["cost_usd"] == first_receipt
    assert attempts[0]["error"] is not None
    assert attempts[1]["cost_usd"] == pytest.approx(0.03)
    assert attempts[1]["error"] is None
    assert result["recorded_cost_usd"] == pytest.approx(recorded_cost)
    assert result["cost_complete"] is cost_complete
    if cost_complete:
        assert result["cost_usd"] == pytest.approx(recorded_cost)
    else:
        assert result["cost_usd"] is None


def test_runner_rejects_missing_or_stale_conversion_manifest(tmp_path: Path) -> None:
    runner_path = Path(__file__).resolve().parents[2] / "examples/jev/run_experiment.py"
    verify = run_path(str(runner_path))["_verify_conversion_manifest"]
    source_path = tmp_path / "source.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    references_path = tmp_path / "references.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    original_case = '{"case_id":"one"}\n'
    cases_path.write_text(original_case, encoding="utf-8")
    references_path.write_text(
        '{"case_id":"one","labels":{"result":true}}\n', encoding="utf-8"
    )
    errors_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing conversion manifest"):
        verify(cases_path, references_path, allow_unmanifested=False)
    assert verify(cases_path, references_path, allow_unmanifested=True) is None

    publish_conversion_manifest(
        tmp_path,
        converter="unit-test",
        sources=[("input", source_path)],
        cases=1,
        errors=0,
        limit=None,
    )
    manifest_path = tmp_path / "manifest.json"
    assert verify(cases_path, references_path, allow_unmanifested=False) == manifest_path

    cases_path.write_text('{"case_id":"altered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs for cases.jsonl"):
        verify(cases_path, references_path, allow_unmanifested=True)

    cases_path.write_text(original_case, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["cases.jsonl"]["rows"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row count differs for cases.jsonl"):
        verify(cases_path, references_path, allow_unmanifested=False)
