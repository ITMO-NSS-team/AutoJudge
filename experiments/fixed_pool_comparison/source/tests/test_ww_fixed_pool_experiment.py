"""All model/API responses are simulated with httpx.MockTransport. No keys or live APIs."""

import asyncio
import json
import hashlib
import shutil
import zipfile

import httpx
import pytest

from examples.who_and_when.fixed_pool.cli import main
from examples.who_and_when.fixed_pool.protocol import (
    ARMS,
    FIXED_NAMES,
    MODEL,
    ROOT,
    SCORING,
    digest,
    encode,
    load_config,
    load_run,
    make_input,
    prompt_snapshot,
    read_json,
    schemas,
    source_hashes,
    validate_output,
    write_json,
)
from examples.who_and_when.fixed_pool.report import build_report, paired_interval
from examples.who_and_when.fixed_pool.runner import (
    HaltRun,
    call_records,
    resolve_lost_call,
    run,
    usage_fields,
)

SECRET = "TEST-SECRET-NEVER-PERSIST"
CHARGE = 0.000155


@pytest.fixture
def frozen(tmp_path):
    """Small, labeled fixture separate from the pinned public dataset."""
    row = {
        "question": "Inspect the output",
        "question_ID": "fixture",
        "history": [
            {"role": "user", "name": "User", "content": "Please inspect"},
            {"role": "assistant", "name": "Researcher", "content": "Unverified answer"},
        ],
        "mistake_agent": "Researcher",
        "mistake_step": "1",
        "mistake_reason": "GOLD_SENTINEL",
    }
    inp = make_input(row)
    config = load_config()
    config["bootstrap_samples"] = 200
    prompts = prompt_snapshot()
    refs = {
        "fixture": {"agent": "Researcher", "step_zero_based": 1, "annotation_out_of_range": False}
    }
    protocol = {
        "version": 4,
        "config": config,
        "scoring": SCORING,
        "arms": list(ARMS),
        "prompts_sha256": digest(prompts),
        "code_hashes": source_hashes(),
    }
    manifest = {
        "phase": "pilot",
        "protocol_sha256": digest(protocol),
        "references_sha256": digest(refs),
        "entries": [
            {"task_id": "fixture", "input_sha256": digest(inp), "cohort": "hand_dev", "messages": 2}
        ],
    }
    manifest["run_id"] = digest(manifest)
    for name, value in [
        ("manifest", manifest),
        ("protocol", protocol),
        ("prompts", prompts),
        ("references", refs),
    ]:
        write_json(tmp_path / f"{name}.json", value)
    write_json(tmp_path / "inputs/fixture.json", inp)
    return tmp_path


class Router:
    def __init__(self, invalid_first=False, timeout_first=False, model=MODEL, missing_cost=False):
        self.posts = []
        self.invalid_first = invalid_first
        self.timeout_first = timeout_first
        self.model = model
        self.missing_cost = missing_cost

    def __call__(self, request):
        if request.method == "GET" and request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "endpoints": [
                            {
                                "tag": "google-ai-studio",
                                "provider_name": "Google AI Studio",
                                "pricing": {"prompt": "0.0000003", "completion": "0.0000025"},
                            }
                        ]
                    }
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "total_cost": CHARGE,
                        "native_tokens_prompt": 100,
                        "native_tokens_completion": 50,
                        "provider_name": "Google AI Studio",
                        "model": self.model,
                    }
                },
            )
        body = json.loads(request.content)
        self.posts.append(body)
        assert "GOLD_SENTINEL" not in encode(body)
        assert "mistake_step" not in encode(body)
        assert "ground_truth" not in encode(body)
        assert body["model"] == MODEL and "tools" not in body
        assert body["provider"]["allow_fallbacks"] is False
        assert body["reasoning"]["max_tokens"] == 1024
        if self.timeout_first and len(self.posts) == 1:
            raise httpx.ReadTimeout(f"Lost response {SECRET}", request=request)
        category = body["response_format"]["json_schema"]["name"]
        if category == "prepare":
            if "CONTROLLED EXPERIMENT:" in body["messages"][0]["content"]:
                value = {
                    "judges": [
                        {"name": f"ROLE_{i}_JUDGE", "instructions": f"Inspect aspect {i}"}
                        for i in range(6)
                    ]
                }
                if self.invalid_first and len(self.posts) == 1:
                    value["judges"].pop()
            else:
                value = {
                    "evidence": [
                        {"judge": name, "facts": [{"step": 2, "fact": "Unverified answer"}]}
                        for name in FIXED_NAMES
                    ]
                }
        elif category == "judge":
            value = {"score": "poor", "justification": "Unverified answer", "evidence_steps": [2]}
        else:
            value = {"agent": "Researcher", "step": 2, "reason": "Unverified answer"}
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 20},
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        if not self.missing_cost:
            usage["cost"] = CHARGE
        return httpx.Response(
            200,
            json={
                "id": f"gen-{len(self.posts)}",
                "provider": "Google AI Studio",
                "model": self.model,
                "usage": usage,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": encode(value),
                            "reasoning": "Fixture reasoning text",
                        },
                    }
                ],
            },
        )


def execute(path, router, cap=1):
    asyncio.run(run(path, SECRET, cap, httpx.MockTransport(router)))


class StreamingRouter(Router):
    def __call__(self, request):
        response = super().__call__(request)
        if request.method != "POST":
            return response
        assert json.loads(request.content)["stream"] is True
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        events = [": OPENROUTER PROCESSING\n\n"]
        for i in range(2):
            chunk = {k: v for k, v in data.items() if k not in {"usage", "choices"}}
            half = len(text) // 2
            chunk["choices"] = [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "role": "assistant",
                        "content": text[:half] if i == 0 else text[half:],
                        "reasoning": "Fixture reasoning " if i == 0 else "text",
                        "reasoning_details": [{"index": 0, "text": str(i)}],
                    },
                }
            ]
            events.append("data: " + json.dumps(chunk) + "\n\n")
        events.append(
            "data: "
            + json.dumps(
                {
                    "id": data["id"],
                    "usage": data["usage"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            + "\n\n"
        )
        events.append("data: [DONE]\n\n")
        return httpx.Response(
            200,
            text="".join(events),
            headers={"content-type": "text/event-stream", "x-generation-id": data["id"]},
        )


def test_streams_preserve_all_wire_events_and_reconstruct_valid_answers(frozen):
    router = StreamingRouter()
    execute(frozen, router)
    report = build_report(frozen)
    assert report["complete"]
    assert report["accounting_all_attempts"]["known_cost_usd"] == pytest.approx(16 * CHARGE)
    records = call_records(frozen, report["run_id"])
    assert len(records) == 16
    for r in records:
        assert r["stream_done"] and r["response_assembled_from_stream"]
        assert r["response"]["choices"][0]["message"]["reasoning"] == "Fixture reasoning text"
        wire = (frozen / r["stream_file"]).read_text()
        assert "OPENROUTER PROCESSING" in wire and "[DONE]" in wire
        assert "reasoning_details" in wire and SECRET not in wire
    execute(frozen, router)
    assert len(router.posts) == 16


def test_stream_disconnect_retains_generation_id_and_partial_wire(frozen):
    router = Router()

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"id":"gen-lost","choices":[{"index":0,"delta":{"content":"{"}}]}\n\n'
            raise httpx.RemoteProtocolError("Connection lost")

    def handler(request):
        if request.method == "GET":
            return router(request)
        return httpx.Response(
            200,
            stream=BrokenStream(),
            headers={"content-type": "text/event-stream", "x-generation-id": "gen-lost"},
        )

    with pytest.raises(HaltRun, match="Network error"):
        execute(frozen, handler)
    manifest = read_json(frozen / "manifest.json")
    records = call_records(frozen, manifest["run_id"])
    assert len(records) == 1
    r = records[0]
    assert r["state"] == "unknown" and r["cost_usd"] is None
    assert r["response"]["id"] == "gen-lost" and not r["stream_done"]
    assert r["response"]["choices"][0]["message"]["content"] == "{"
    assert '"id":"gen-lost"' in (frozen / r["stream_file"]).read_text()
    # Native preflight resolves the saved generation's cost before any subsequent call.
    execute(frozen, Router())
    report = build_report(frozen)
    assert report["complete"] and report["accounting_all_attempts"]["unknown_cost_calls"] == 0


def test_ssl_failure_is_retained_as_unresolved_without_replay(frozen):
    import ssl

    router = Router()

    def fail(request):
        if request.method == "POST":
            raise ssl.SSLError("DECRYPTION_FAILED_OR_BAD_RECORD_MAC")
        return router(request)

    with pytest.raises(HaltRun, match="Network error"):
        execute(frozen, fail)
    manifest = read_json(frozen / "manifest.json")
    records = call_records(frozen, manifest["run_id"])
    assert len(records) == 1
    assert records[0]["state"] == "unknown"
    assert records[0]["error_type"] == "SSLError"
    assert records[0]["cost_usd"] is None
    with pytest.raises(HaltRun, match="Unreconciled"):
        execute(frozen, fail)


def test_templates_preserved_and_inputs_never_include_labels(frozen):
    snapshot = prompt_snapshot()
    assert tuple(j["name"] for j in snapshot["fixed_judges"]) == FIXED_NAMES
    assert "**Scoring**" in snapshot["fixed_judges"][0]["instructions"]
    inp = read_json(frozen / "inputs/fixture.json")
    assert set(inp) == {"query", "messages"}
    assert inp["messages"][0]["step"] == 1
    assert "GOLD_SENTINEL" not in encode(inp)
    canonical = make_input(
        {
            "question": "x",
            "history": [{"role": "Orchestrator (-> WebSurfer)", "content": "handoff"}],
        }
    )
    assert canonical["messages"][0]["agent"] == "Orchestrator"
    assert canonical["messages"][0]["message"]["role"] == "Orchestrator (-> WebSurfer)"


def test_existing_aj_generator_builder_pipeline_and_session_are_used(frozen, monkeypatch):
    from examples.who_and_when.fixed_pool.aj_bridge import import_aj

    AgentPool, PoolGenerator, _, AgentNode, NodeSession, _, PipelineBuilder = import_aj()
    from autojudge.pipeline.pipeline import Pipeline
    from autojudge.meta_agents import graph_gen

    calls = {"generator": 0, "builder": 0, "pipeline": 0, "session": 0, "graph": 0}
    original_generate = PoolGenerator.create_pool
    original_build = PipelineBuilder.create_from_pool
    original_invoke = Pipeline.ainvoke
    original_input = NodeSession.get_input_for_node
    original_graph = graph_gen.get_parallel_graph

    async def generate(self, *args, **kwargs):
        calls["generator"] += 1
        pool = await original_generate(self, *args, **kwargs)
        assert isinstance(pool, AgentPool) and len(pool) == 6
        return pool

    def build(self, pool, graph):
        calls["builder"] += 1
        assert len(pool) == 7 and all(isinstance(node, AgentNode) for node in pool)
        return original_build(self, pool, graph)

    async def invoke(self, query):
        calls["pipeline"] += 1
        assert type(self) is Pipeline
        assert isinstance(self.node_session, NodeSession)
        assert all(not node.use_tools for node in self.execution_order)
        return await original_invoke(self, query)

    def session_input(self, node):
        calls["session"] += 1
        return original_input(self, node)

    def graph(pool):
        calls["graph"] += 1
        return original_graph(pool)

    monkeypatch.setattr(PoolGenerator, "create_pool", generate)
    monkeypatch.setattr(PipelineBuilder, "create_from_pool", build)
    monkeypatch.setattr(Pipeline, "ainvoke", invoke)
    monkeypatch.setattr(NodeSession, "get_input_for_node", session_input)
    monkeypatch.setattr(graph_gen, "get_parallel_graph", graph)
    execute(frozen, Router())
    assert calls == {"generator": 1, "builder": 2, "pipeline": 2, "session": 14, "graph": 2}


def test_end_to_end_eight_calls_per_arm_and_resume_is_free(frozen):
    router = Router()
    execute(frozen, router)
    assert len(router.posts) == 16
    execute(frozen, router)
    assert len(router.posts) == 16
    report = build_report(frozen)
    assert report["complete"]
    for arm in ARMS:
        result = report["groups"]["all"]["arms"][arm]
        assert result["metrics"]["agent"]["accuracy_percent"] == 100
        assert result["metrics"]["step"]["accuracy_percent"] == 100
        assert result["accounting"]["attempts"] == 8
    assert report["accounting_all_attempts"]["known_cost_usd"] == pytest.approx(16 * CHARGE)
    assert report["accounting_all_attempts"]["completion_tokens"] == 16 * 50
    assert report["accounting_all_attempts"]["reasoning_tokens"] == 16 * 20
    # Both aggregators use the same prompt and AJ's native six-parent NodeSession input.
    final = [p for p in router.posts if p["response_format"]["json_schema"]["name"] == "overall"]
    assert final[0]["messages"][0] == final[1]["messages"][0]
    for request in final:
        text = request["messages"][1]["content"]
        context, _ = json.JSONDecoder().raw_decode(text.split("JSON format:\n\n", 1)[1])
        assert len(context["pipeline_context"]["parent_nodes"]) == 6
        assert "Original query:" in context["task"]
    assert all(SECRET not in p.read_text() for p in frozen.rglob("*.json"))


def test_invalid_pool_retry_charged_once_and_resume_does_not_repeat(frozen):
    # Force AJ first for the first-call invalid pool fixture.
    router = Router(invalid_first=True)
    # Whether AJ or fixed comes first, make the first preparation invalid explicitly.
    original = router.__call__

    def invalid(request):
        response = original(request)
        if request.method == "POST" and len(router.posts) == 1:
            data = response.json()
            data["choices"][0]["message"]["content"] = "{}"
            return httpx.Response(200, json=data)
        return response

    execute(frozen, invalid)
    report = build_report(frozen)
    assert len(router.posts) == 17
    assert report["accounting_all_attempts"]["known_cost_usd"] == pytest.approx(17 * CHARGE)
    assert any(r["state"] == "invalid" for r in call_records(frozen, report["run_id"]))
    execute(frozen, invalid)
    assert len(router.posts) == 17


def test_budget_gate_prevents_any_paid_request(frozen):
    router = Router()
    with pytest.raises(HaltRun, match="Budget"):
        execute(frozen, router, cap=0.000001)
    assert not router.posts


def test_main_requires_completed_pilot_before_network(frozen):
    manifest = read_json(frozen / "manifest.json")
    manifest["phase"] = "main"
    manifest["run_id"] = digest({k: v for k, v in manifest.items() if k != "run_id"})
    write_json(frozen / "manifest.json", manifest)
    router = Router()
    with pytest.raises(HaltRun, match="requires --pilot-dir"):
        execute(frozen, router)
    assert not router.posts


def test_parallel_budget_pause_resumes_completed_nodes_without_rebilling(frozen):
    router = Router()

    async def slow(request):
        if request.method == "POST":
            await asyncio.sleep(0.005)
        return router(request)

    with pytest.raises(HaltRun):
        execute(frozen, slow, cap=0.02)
    assert 0 < len(router.posts) < 16
    partial = build_report(frozen)
    assert partial["provisional"] and partial["accounting_all_attempts"]["unknown_cost_calls"] == 0
    execute(frozen, slow, cap=1)
    assert len(router.posts) == 16
    assert build_report(frozen)["accounting_all_attempts"]["known_cost_usd"] == pytest.approx(
        16 * CHARGE
    )


def test_exhausted_judge_still_charged_and_failed_trace_stays_in_denominator(frozen):
    router = Router()

    def truncated(request):
        response = router(request)
        if request.method == "POST":
            body = json.loads(request.content)
            if "Evaluate complexity and interconnectedness" in body["messages"][0]["content"]:
                data = response.json()
                data["choices"][0]["finish_reason"] = "length"
                return httpx.Response(200, json=data)
        return response

    execute(frozen, truncated)
    report = build_report(frozen)
    fixed = report["groups"]["all"]["arms"]["fixed"]
    assert fixed["n"] == fixed["failed"] == 1
    assert fixed["metrics"]["agent"]["accuracy_percent"] == 0
    assert fixed["accounting"]["attempts"] == 9
    assert fixed["accounting"]["known_cost_usd"] == pytest.approx(9 * CHARGE)
    assert report["complete"]


def test_timeout_is_not_replayed_until_explicit_reconciliation(frozen):
    router = Router(timeout_first=True)
    with pytest.raises(HaltRun, match="billing uncertain"):
        execute(frozen, router)
    with pytest.raises(HaltRun, match="Unreconciled"):
        execute(frozen, router)
    assert len(router.posts) == 1
    manifest = read_json(frozen / "manifest.json")
    record = call_records(frozen, manifest["run_id"])[0]
    assert record["cost_usd"] is None
    assert all(SECRET not in p.read_text() for p in frozen.rglob("*.json"))
    resolve_lost_call(frozen, record["call_id"], CHARGE, "Verified against fake billing fixture")
    execute(frozen, router)
    report = build_report(frozen)
    assert report["complete"] and len(router.posts) == 17
    assert report["accounting_all_attempts"]["manual_cost_calls"] == 1
    assert report["accounting_all_attempts"]["known_cost_usd"] == pytest.approx(17 * CHARGE)


def test_usage_cost_recovered_from_generation_metadata(frozen):
    router = Router(missing_cost=True)
    execute(frozen, router)
    report = build_report(frozen)
    assert report["accounting_all_attempts"]["unknown_cost_calls"] == 0
    assert report["accounting_all_attempts"]["known_cost_usd"] == pytest.approx(16 * CHARGE)


def test_reported_charge_on_error_is_never_overwritten_with_zero(frozen):
    router = Router()

    def charged_error(request):
        if request.method == "POST":
            router.posts.append(json.loads(request.content))
            return httpx.Response(
                400,
                json={"error": {"message": "Rejected after accounting"}, "usage": {"cost": 0.01}},
            )
        return router(request)

    with pytest.raises(HaltRun, match="HTTP 400"):
        execute(frozen, charged_error)
    report = build_report(frozen)
    assert report["accounting_all_attempts"]["known_cost_usd"] == 0.01
    assert report["accounting_all_attempts"]["pre_generation_rejections"] == 0


def test_model_substitution_halts_and_cannot_be_bypassed_by_resume(frozen):
    router = Router(model="other/model")
    with pytest.raises(HaltRun, match="Unexpected response model"):
        execute(frozen, router)
    with pytest.raises(HaltRun, match="Unexpected response model"):
        execute(frozen, router)
    assert len(router.posts) == 1


def test_manifest_changes_refused_before_network(frozen):
    inp = read_json(frozen / "inputs/fixture.json")
    inp["query"] = "changed"
    write_json(frozen / "inputs/fixture.json", inp)
    with pytest.raises(ValueError, match="Input changed"):
        load_run(frozen)


def test_schema_rejects_bool_step_and_missing_fixed_roles(frozen):
    inp = read_json(frozen / "inputs/fixture.json")
    schema = schemas(inp)
    with pytest.raises(ValueError):
        validate_output(
            "overall", {"agent": "Researcher", "step": True, "reason": "x"}, schema["overall"], inp
        )
    duplicated = {"evidence": [{"judge": FIXED_NAMES[0], "facts": []} for _ in range(6)]}
    with pytest.raises(ValueError, match="one evidence entry"):
        validate_output("prepare_fixed", duplicated, schema["prepare_fixed"], inp)


def test_report_uses_manifest_denominator_and_exact_agent_matching(frozen):
    manifest = read_json(frozen / "manifest.json")
    common = {"run_id": manifest["run_id"], "input_sha256": manifest["entries"][0]["input_sha256"]}
    write_json(
        frozen / "results/autojudge/fixture.json",
        {**common, "status": "success", "prediction": {"agent": "User", "step": 2, "reason": "x"}},
    )
    write_json(frozen / "results/fixed/fixture.json", {**common, "status": "failed"})
    report = build_report(frozen)
    arms = report["groups"]["all"]["arms"]
    assert arms["autojudge"]["metrics"]["agent"]["correct"] == 0
    assert arms["autojudge"]["metrics"]["step"]["correct"] == 1
    assert arms["fixed"]["n"] == 1 and arms["fixed"]["coverage"] == 0
    assert report["groups"]["all"]["autojudge_minus_fixed"]["step"]["delta_pp"] == 100
    (frozen / "results/fixed/fixture.json").unlink()
    report = build_report(frozen)
    assert report["provisional"] and arms["fixed"]["n"] == 1


def test_paired_bootstrap_and_reasoning_not_double_counted():
    interval = paired_interval([1, 1, 1], 100, 42)
    assert interval["delta_pp"] == 100 and interval["ci95_pp"] == [100, 100]
    fields = usage_fields(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": CHARGE,
                "completion_tokens_details": {"reasoning_tokens": 20},
            }
        }
    )
    assert fields["completion_tokens"] == 50 and fields["reasoning_tokens"] == 20


def test_dry_run_requires_no_key_and_sends_no_request(frozen, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(httpx.AsyncClient, "send", lambda *a, **k: pytest.fail("Network forbidden"))
    assert main(["run", "--output-dir", str(frozen), "--dry-run"]) == 0


def test_archive_preserves_all_raw_answers_native_traces_source_and_checksums(frozen, monkeypatch):
    from examples.who_and_when.fixed_pool import archive

    router = StreamingRouter()
    execute(frozen, router)
    protocol = read_json(frozen / "protocol.json")
    for name in protocol["code_hashes"]:
        target = frozen / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(frozen / "source_snapshot_manifest.json", protocol["code_hashes"])
    write_json(frozen / "environment.json", {"python": "fixture"})
    write_json(frozen / "provenance.json", {"git_commit": "fixture"})
    data = b"public dataset fixture"
    monkeypatch.setattr(archive, "FILE_HASHES", {"fixture": hashlib.sha256(data).hexdigest()})
    (frozen / "dataset_snapshot").mkdir()
    (frozen / "dataset_snapshot/fixture.parquet").write_bytes(data)
    output = frozen.parent / (frozen.name + ".zip")
    audit = archive.export_archive(frozen, output)
    assert audit["storage_ok"] and audit["execution_complete"]
    assert audit["total_attempts"] == 16
    with zipfile.ZipFile(output) as z:
        files = z.namelist()
        assert len([n for n in files if n.startswith("experiment/calls/")]) == 16
        assert len([n for n in files if n.startswith("experiment/streams/")]) == 16
        assert len([n for n in files if n.startswith("experiment/pipeline_traces/")]) == 2
        for name, expected in json.loads(z.read("experiment/SHA256SUMS.json")).items():
            assert hashlib.sha256(z.read("experiment/" + name)).hexdigest() == expected
        for name in files:
            assert SECRET.encode() not in z.read(name)
            if name.startswith("experiment/calls/"):
                r = json.loads(z.read(name))
                assert (
                    r["response"]["choices"][0]["message"]["reasoning"] == "Fixture reasoning text"
                )
    # A credential-like string is rejected before making a second archive.
    (frozen / "console.log").write_text("sk-or-v1-" + "a" * 64)
    bad = frozen.parent / "credential-check.zip"
    with pytest.raises(ValueError, match="Credential-like"):
        archive.export_archive(frozen, bad)
    assert not bad.exists()
