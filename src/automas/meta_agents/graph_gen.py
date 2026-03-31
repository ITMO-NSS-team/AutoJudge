from typing import TYPE_CHECKING, Optional

from automas.mcp.registry import get_server_descriptions
from automas.pipeline.pipeline_builder import PipelineBuilder
from automas.pipeline.types import GraphDict
from automas.utils.logger import get_logger

from .base import DEFAULT_MODEL, BaseMetaAgent
from .prompt_registry import DEFAULT_GRAPH_INSTRUCT, DEFAULT_POOL_INSTRUCT_DATASET
import os

if TYPE_CHECKING:
    from automas.agent_pool import AgentPool

logger = get_logger()


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
        mcp_servers_desc = get_server_descriptions()
        return DEFAULT_GRAPH_INSTRUCT.substitute(mcp_servers_desc=mcp_servers_desc)

    def _get_output_type(self):
        return GraphDict

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
            # TODO: Implement graph validation logic
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

        graph_dict = await self._run_agent(user_prompt)

        self._validate_response(graph_dict, agent_pool)
        self._validate_graph(agent_pool, graph_dict)

        logger.info("Successfully created pipeline")

        return graph_dict
