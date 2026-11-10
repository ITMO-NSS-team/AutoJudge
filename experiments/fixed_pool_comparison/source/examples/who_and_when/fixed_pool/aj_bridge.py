"""Run the repository's AJ classes; inject only a journalled PydanticAI model transport."""

from __future__ import annotations

import json
import inspect
import os
import sys
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from .protocol import ROOT, digest, encode, read_json, schemas, write_json
from .runner import HaltRun, redact


def import_aj():
    # Runtime-only: prepare/dry-run/report do not import AJ or load its .env.
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    previous = os.environ.get("AUTOJUDGE_DISABLE_TRACING")
    os.environ["AUTOJUDGE_DISABLE_TRACING"] = "1"
    try:
        from autojudge.agent_pool import AgentPool
        from autojudge.meta_agents.pool_gen import PoolGenerator
        from autojudge.meta_agents.graph_gen import get_parallel_graph
        from autojudge.pipeline.node import AgentNode
        from autojudge.pipeline.node_session import NodeSession
        from autojudge.pipeline.pipeline import PipelineExecutionError
        from autojudge.pipeline.pipeline_builder import PipelineBuilder

        for cls in (AgentPool, PoolGenerator, AgentNode, NodeSession, PipelineBuilder):
            if not Path(inspect.getfile(cls)).resolve().is_relative_to(ROOT / "src"):
                raise HaltRun(
                    "AJ was imported from another checkout; refusing mixed source versions"
                )
    finally:
        if previous is None:
            os.environ.pop("AUTOJUDGE_DISABLE_TRACING", None)
        else:
            os.environ["AUTOJUDGE_DISABLE_TRACING"] = previous
    return (
        AgentPool,
        PoolGenerator,
        get_parallel_graph,
        AgentNode,
        NodeSession,
        PipelineExecutionError,
        PipelineBuilder,
    )


def tracked_model(client, task: str, arm: str, stage: str, inp: dict) -> FunctionModel:
    """Adapt the existing PydanticAI agents to the shared OpenRouter/checkpoint layer."""
    schema = schemas(inp)["judge" if stage.startswith("judge_") else stage]

    async def request(messages, info):
        if info.function_tools:
            raise HaltRun("Tools must be disabled in this full-trace experiment")
        systems, users = [], []
        for message in messages:
            for part in message.parts:
                if part.part_kind == "system-prompt":
                    systems.append(part.content)
                elif part.part_kind == "user-prompt":
                    users.append(part.content)
        if info.instructions:
            systems.append(info.instructions)
        if len(users) != 1 or not isinstance(users[0], str):
            raise HaltRun("Unexpected AJ conversation shape; refusing hidden extra model turns")
        value = await client.complete(task, arm, stage, "\n\n".join(systems), users[0], schema, inp)
        # Usage belongs to this stage only. Avoid reparsing the whole growing run
        # journal after every call in the 126-trace experiment.
        records = []
        for attempt in range(1, client.config["max_attempts"] + 1):
            call_id = digest([client.manifest["run_id"], task, arm, stage, attempt])
            file = client.path / "calls" / f"{call_id}.json"
            if file.exists():
                record = read_json(file)
                if record["call_id"] != call_id or record["run_id"] != client.manifest["run_id"]:
                    raise HaltRun("Stage usage record does not match this run")
                records.append(record)
        usage = RequestUsage(
            input_tokens=sum(r.get("prompt_tokens") or 0 for r in records),
            output_tokens=sum(r.get("completion_tokens") or 0 for r in records),
            details={"reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in records)},
        )
        if info.output_tools:
            # PoolGenerator's native output is list[AgentSchema]; PydanticAI wraps root lists.
            if stage != "prepare_autojudge" or len(info.output_tools) != 1:
                raise HaltRun("Unexpected AJ structured-output interface")
            tool = info.output_tools[0]
            properties = tool.parameters_json_schema.get("properties", {})
            if list(properties) != ["response"]:
                raise HaltRun("PydanticAI root-list wrapper changed; inspect before running")
            parts = [ToolCallPart(tool.name, {"response": value["judges"]})]
        else:
            parts = [TextPart(encode(value))]
        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=client.config["model"],
            provider_name="openrouter",
            finish_reason="stop",
        )

    return FunctionModel(request, model_name=client.config["model"])


async def evaluate_using_aj(client, entry: dict, arm: str, prompts: dict) -> dict:
    (
        AgentPool,
        PoolGenerator,
        get_parallel_graph,
        AgentNode,
        NodeSession,
        PipelineExecutionError,
        PipelineBuilder,
    ) = import_aj()
    task = entry["task_id"]
    inp = read_json(client.path / "inputs" / f"{task}.json")
    if digest(inp) != entry["input_sha256"]:
        raise HaltRun(f"Frozen input changed during execution: {task}")

    def factory(stage):
        # Plain function closures survive PipelineBuilder's deep copy without copying HTTP clients.
        return lambda: tracked_model(client, task, arm, stage, inp)

    if arm == "autojudge":
        generator = PoolGenerator(
            model=client.config["model"],
            temperature=client.config["prepare_temperature"],
            use_summary=False,
            model_override=factory("prepare_autojudge")(),
            system_prompt_override=prompts["common"] + prompts["meta"],
            judge_model=client.config["model"],
            retries=0,
            instrument=False,
        )
        pool = await generator.create_pool(encode({"trace": inp}))
        judges = [{"name": node.name, "instructions": node.instructions} for node in pool]
        prepared = {"judges": judges}
    else:
        extractor = AgentNode(
            name="EVIDENCE_EXTRACTOR",
            model=client.config["model"],
            use_tools=False,
            instructions=prompts["common"] + prompts["extractor"],
            model_factory=factory("prepare_fixed"),
            retries=0,
            instrument=False,
        )
        result = await extractor.build_agent().run(
            encode({"trace": inp, "fixed_roles": prompts["fixed_judges"]})
        )
        prepared = json.loads(result.output)
        judges = prompts["fixed_judges"]
        pool = AgentPool(
            [
                AgentNode(
                    name=j["name"],
                    instructions=j["instructions"],
                    model=client.config["model"],
                    use_tools=False,
                )
                for j in judges
            ]
        )

    for i, node in enumerate(pool):
        node.instructions = prompts["common"] + node.instructions + prompts["judge_wrapper"]
        node.model = client.config["model"]
        node.model_factory = factory(f"judge_{i + 1}")
        node.retries, node.instrument, node.use_tools = 0, False, False
        node.id = digest([client.manifest["run_id"], task, arm, node.name])[:16]
    overall = AgentNode(
        name="FINAL_AGGREGATOR",
        instructions=prompts["common"] + prompts["overall"],
        model=client.config["model"],
        use_tools=False,
        model_factory=factory("overall"),
        retries=0,
        instrument=False,
        id=digest([client.manifest["run_id"], task, arm, "FINAL_AGGREGATOR"])[:16],
    )
    pool.add_agent(overall)
    graph = get_parallel_graph(pool)
    pipeline = PipelineBuilder().create_from_pool(pool, graph).build()
    pipeline.node_session = NodeSession(
        session_id="fixed_pool_" + digest([client.manifest["run_id"], task, arm])[:16]
    )
    query = {"trace": inp}
    if arm == "fixed":
        query["prepared_observations"] = prepared["evidence"]
    try:
        prediction = json.loads(await pipeline.ainvoke(query))
    except PipelineExecutionError as exc:
        # AJ already waited for every paid sibling node and retained the original failures.
        errors = [error for _, error in exc.node_errors]
        raise next((error for error in errors if isinstance(error, HaltRun)), errors[0]) from exc
    finally:
        if pipeline.trace is not None:
            write_json(
                client.path / "pipeline_traces" / arm / f"{task}.json",
                redact(pipeline.trace.model_dump(mode="json"), client.key),
            )
    findings = [
        {
            "name": node.name,
            "instructions": node.instructions,
            "findings": json.loads(pipeline.node_session.node_executions[node.id].output),
        }
        for node in pipeline.execution_order
        if node.name != "FINAL_AGGREGATOR"
    ]
    return {
        "status": "success",
        "prediction": prediction,
        "judges": judges,
        "prepared": prepared,
        "findings": findings,
        "execution": {
            "pipeline_class": "autojudge.pipeline.pipeline.Pipeline",
            "pool_generator": "autojudge.meta_agents.pool_gen.PoolGenerator"
            if arm == "autojudge"
            else None,
            "builder": "autojudge.pipeline.pipeline_builder.PipelineBuilder",
            "node_session": "autojudge.pipeline.node_session.NodeSession",
            "session_id": pipeline.node_session.session_id,
            "graph": graph,
        },
    }
