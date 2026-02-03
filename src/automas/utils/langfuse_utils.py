import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional

from dotenv import load_dotenv
from langfuse import get_client, observe
from pydantic_ai import Agent

if TYPE_CHECKING:
    from automas.agent_pool import AgentPool
    from automas.pipeline.pipeline import Pipeline

load_dotenv()

REQUIRED_LANGFUSE_VARS = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
LANGFUSE_FLUSH_DELAY = 1.5

_instrumentation_enabled = False
langfuse = None


def check_langfuse_env() -> bool:
    return all(os.getenv(var) for var in REQUIRED_LANGFUSE_VARS)


def setup_langfuse_instrumentation() -> None:
    global _instrumentation_enabled, langfuse

    if _instrumentation_enabled:
        return

    if check_langfuse_env():
        langfuse = get_client()
        Agent.instrument_all()
        _instrumentation_enabled = True


def is_instrumentation_enabled() -> bool:
    return _instrumentation_enabled


@observe(name="automas_execution_task_")
async def _execute_pipeline_with_metadata(
    pool: "AgentPool", pipeline: "Pipeline", query: Any, graph_dict: dict
) -> dict:
    result = await pipeline.ainvoke(query)

    return {
        "pool": pool.full_agents_data,
        "pipeline": {
            "session_id": pipeline.node_session.session_id,
            "num_nodes": len(pipeline.execution_order),
            "execution_levels": len(pipeline._execution_levels),
        },
        "mermaid_graph": pipeline.to_mermaid_lr(visualize=False),
        "dict_graph": graph_dict,
        "pipeline_tokens": {
            "input_tokens": pipeline.input_tokens,
            "output_tokens": pipeline.output_tokens,
            "total_tokens": pipeline.total_tokens,
        },
        "pipeline_cost": pipeline.cost.model_dump(),
        "question": str(query),
        "response": str(result),
    }


async def ainvoke_with_lf(
    pool: "AgentPool", pipeline: "Pipeline", query: Any, graph_dict: dict
) -> tuple[Any, Optional[str]]:
    if not is_instrumentation_enabled():
        result = await pipeline.ainvoke(query)
        return result, None

    client = get_client()
    trace_id = client.create_trace_id(seed=f"{pipeline.node_session.session_id}")

    output = await _execute_pipeline_with_metadata(
        pool, pipeline, query, graph_dict, langfuse_trace_id=trace_id
    )
    client.flush()
    await asyncio.sleep(LANGFUSE_FLUSH_DELAY)
    return output.get("response"), trace_id
