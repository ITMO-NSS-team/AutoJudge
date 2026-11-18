"""Check the original AJ graph with a local deterministic model transport."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_table3_executes_six_generated_judges_and_fixed_aggregator(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "table3_ablation", root / "scripts/who_and_when/table3_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = module.p
    inp = {
        "query": "Find an item",
        "messages": [
            {"step": 1, "agent": "Planner", "message": {"content": "Look for the wrong item"}},
            {"step": 2, "agent": "Worker", "message": {"content": "The wrong item was found"}},
        ],
    }
    p.write_json(tmp_path / "inputs/case.json", inp)
    calls = []

    class Client:
        path = tmp_path
        key = ""
        config = {"model": "google/gemini-2.5-flash", "prepare_temperature": 0.3, "max_attempts": 3}
        manifest = {"run_id": "offline-test"}

        async def complete(self, task, arm, stage, system, user, schema, supplied_input):
            calls.append((stage, system, user))
            assert supplied_input == inp
            assert "mistake_agent" not in user
            if stage == "prepare_autojudge":
                value = {
                    "judges": [
                        {"name": f"ROLE_{i}_JUDGE", "instructions": "Inspect evidence"}
                        for i in range(6)
                    ]
                }
            elif stage.startswith("judge_"):
                value = {"score": "poor", "justification": "Wrong item", "evidence_steps": [1]}
            else:
                value = {"agent": "Planner", "step": 1, "reason": "Wrong target"}
            p.validate_output("judge" if stage.startswith("judge_") else stage, value, schema, inp)
            return value

    result = await module.evaluate_using_aj(
        Client(),
        {"task_id": "case", "input_sha256": p.digest(inp)},
        "autojudge",
        p.prompt_snapshot(),
    )
    assert result["status"] == "success"
    assert result["prediction"]["step"] == 1
    assert len(calls) == 8
    assert [s for s, _, _ in calls] == ["prepare_autojudge"] + [
        f"judge_{i}" for i in range(1, 7)
    ] + ["overall"]
    assert len(result["findings"]) == 6
    assert result["execution"]["pipeline_class"] == "autojudge.pipeline.pipeline.Pipeline"
    assert p.OVERALL.strip() in calls[-1][1]
    assert "Wrong item" in calls[-1][2]


def load_table3_module():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "table3_ablation", root / "scripts/who_and_when/table3_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_cost_ceiling_is_not_reported_as_actual_cost():
    module = load_table3_module()
    records = [
        {"cost_usd": 0.1, "cost_source": "openrouter_usage"},
        {"cost_usd": 0.3, "cost_source": "reserved_upper_bound_not_actual"},
    ]
    result = module.accounting(records)
    assert result["known_cost_usd"] == pytest.approx(0.1)
    assert result["budget_debit_usd"] == pytest.approx(0.4)
    assert result["cost_upper_bound_usd"] == pytest.approx(0.4)
    assert result["unknown_cost_calls"] == 1
    assert result["total_cost_usd"] is None


def test_lost_request_retains_unknown_actual_charge_and_bounds_retry_budget():
    module = load_table3_module()
    client = object.__new__(module.ConservativeClient)
    saved = []
    client.save = lambda record: saved.append(dict(record))
    record = {
        "state": "unknown",
        "cost_usd": None,
        "error_type": "RemoteProtocolError",
        "reserved_usd": 0.5,
    }
    assert client.bound_lost(record)
    assert record["original_state"] == "unknown"
    assert record["actual_cost_usd"] is None
    assert record["cost_usd"] == 0.5
    assert record["cost_source"] == "reserved_upper_bound_not_actual"
    assert not client.bound_lost(record)
    assert len(saved) == 1
