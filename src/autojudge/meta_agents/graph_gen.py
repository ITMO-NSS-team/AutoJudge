import json
import os
import re
from typing import TYPE_CHECKING, Any, Optional

from autojudge.pipeline.pipeline_builder import PipelineBuilder
from autojudge.pipeline.types import GraphDict
from autojudge.utils.logger import get_logger

from .base import BaseMetaAgent
from .prompts import DEFAULT_GRAPH_INSTRUCT

from autojudge.agent_pool import AgentPool

logger = get_logger()


def get_parallel_graph(agent_pool: AgentPool) -> GraphDict:
    graph_dict: GraphDict = {}
    _agents_info = agent_pool.full_agents_data

    for agent in _agents_info:
        if agent.get("name") != "FINAL_AGGREGATOR":
            graph_dict[agent.get("name")] = ["FINAL_AGGREGATOR"]

    graph_dict["FINAL_AGGREGATOR"] = []

    return graph_dict


def parse_graph(output: Any) -> GraphDict:
    """Read the adjacency list the model returned as JSON text.

    The graph is an open-ended map, which several providers cannot express as a
    structured-output schema: their function-call schema keeps only declared
    properties, so the model can answer nothing but an empty object. The prompt
    already asks for a bare JSON object, so it is parsed from text instead.
    """
    if isinstance(output, dict):
        graph = output
    else:
        if not isinstance(output, str):
            raise ValueError(f"Graph response is not JSON, got {type(output).__name__}")
        text = output.strip().lstrip("﻿")
        fenced = re.fullmatch(
            r"`{2,}\s*(?:json)?\s*(.*?)\s*`{2,}", text, flags=re.IGNORECASE | re.DOTALL
        )
        if fenced:
            text = fenced.group(1).strip()
        try:
            graph = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Graph response is not valid JSON: {error}") from None
    if not isinstance(graph, dict) or not graph:
        raise ValueError("Graph response must be a non-empty JSON object")
    parsed: GraphDict = {}
    for name, children in graph.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Graph keys must be agent names")
        if children is None:
            children = []
        if not isinstance(children, list) or any(
            not isinstance(child, str) for child in children
        ):
            raise ValueError(f"Children of '{name}' must be a list of agent names")
        parsed[name] = children
    return parsed


class GraphGenerator(BaseMetaAgent):
    def __init__(
        self,
        model: str = os.getenv("GRAPH_GEN_MODEL", "google/gemini-2.5-flash"),
        temperature: float = 0.3,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )

    def _get_system_prompt(self) -> str:
        return DEFAULT_GRAPH_INSTRUCT.substitute()

    def _get_output_type(self):
        # Text, not a structured map: see parse_graph.
        return str

    def _validate_response(
        self, graph_dict: GraphDict, agent_pool: "AgentPool"
    ) -> None:
        logger.debug(f"Validating graph response: {graph_dict}")

        agent_names = [agent.name for agent in agent_pool]
        agent_names_set = set(agent_names)

        for agent_name in graph_dict:
            if agent_name not in agent_names_set:
                logger.error(f"Unknown agent '{agent_name}' in graph")
                raise ValueError(
                    f"Unknown agent '{agent_name}' in graph. "
                    f"Available agents: {', '.join(agent_names)}"
                )

        for agent_name, children in graph_dict.items():
            if not isinstance(children, list):
                raise ValueError(
                    f"Children for agent '{agent_name}' must be a list, got {type(children)}"
                )

            for child_name in children:
                if child_name not in agent_names_set:
                    raise ValueError(
                        f"Unknown child agent '{child_name}' referenced by '{agent_name}'. "
                        f"Available agents: {', '.join(agent_names)}"
                    )

                if child_name == agent_name:
                    raise ValueError(f"Agent '{agent_name}' cannot connect to itself")

    def _validate_graph(self, agent_pool: "AgentPool", graph_dict: GraphDict) -> None:
        try:
            builder = PipelineBuilder()
            builder.create_from_pool(agent_pool, graph_dict)
        except KeyError as e:
            raise ValueError(f"Graph structure validation failed: {e}") from e
        except RuntimeError as e:
            raise ValueError(f"Graph validation failed: {e}") from e

    async def create_graph(
        self,
        agent_pool: "AgentPool",
        task_description: str,
        context: Optional[str] = None,
    ) -> GraphDict:
        if len(agent_pool) == 0:
            raise ValueError("Agent pool cannot be empty")

        logger.info(f"Generating workflow graph for {len(agent_pool)} agents")
        logger.debug(f"Task: {task_description[:100]}...")

        agents_info = agent_pool.full_agents_data

        user_prompt = f"""TASK: {task_description}

AGENTS IN POOL:
{agents_info}

Design the workflow graph for the given task and agent pool.
"""

        if context:
            user_prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{context}"

        graph_dict = parse_graph(await self._run_agent(user_prompt))

        self._validate_response(graph_dict, agent_pool)
        self._validate_graph(agent_pool, graph_dict)

        logger.info("Successfully created pipeline")

        return graph_dict
