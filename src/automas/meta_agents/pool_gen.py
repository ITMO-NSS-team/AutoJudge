import inspect
import os
from typing import List, Optional

from pydantic import BaseModel

from automas.agent_pool import AgentPool
from automas.mcp.registry import get_server_descriptions
from automas.pipeline.node import AgentNode
from automas.utils import get_logger

from .base import DEFAULT_MODEL, BaseMetaAgent
from .prompt_registry import DEFAULT_POOL_INSTRUCT_EXTENDED, DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool

logger = get_logger()


class AgentSchema(BaseModel):
    name: str
    instructions: str
    mcp_tools: List[str] = []
    model: str = os.getenv("AGENT_NODE_MODEL", "google/gemini-2.5-flash")
    use_tools: bool = True


class PoolGenerator(BaseMetaAgent):
    def __init__(
        self,
        model: str = os.getenv("POOL_GEN_MODEL", "google/gemini-2.5-flash"),
        temperature: float = 0.3,
        output_schema: str = "",
        taxonomy: str = "",
        examples: str = "",
    ):
        print(
            f"Initializing PoolGenerator with model={model}, temperature={temperature}"
        )
        self.schema = output_schema
        self.taxonomy = taxonomy
        self.examples = examples

        # Auto-detect instruction set from caller's imports
        caller_frame = inspect.currentframe()
        if caller_frame and caller_frame.f_back:
            caller_globals = caller_frame.f_back.f_globals
            self.use_tools = "DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool" not in caller_globals
            self.pool_instruct = (
                DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool
                if not self.use_tools
                else DEFAULT_POOL_INSTRUCT_EXTENDED
            )
            if not self.use_tools:
                logger.info("Detected no-tools mode from caller imports")
        else:
            self.use_tools = True
            self.pool_instruct = DEFAULT_POOL_INSTRUCT_EXTENDED
            logger.warning("Could not detect caller frame; defaulting to use_tools=True")

        super().__init__(
            model=model,
            temperature=temperature,
        )

    def _get_system_prompt(self) -> str:
        mcp_servers_desc = get_server_descriptions()
        return self.pool_instruct.substitute(
            mcp_servers_desc=mcp_servers_desc,
            taxonomy=self.taxonomy,
            judge_output_format=self.schema,
            examples=self.examples,
        )

    def _get_output_type(self):
        return list[AgentSchema]

    def _create_agents(self, agent_schemas: List[AgentSchema]) -> List[AgentNode]:
        if not agent_schemas:
            raise ValueError("No valid agents generated")

        logger.info(
            f"Creating {len(agent_schemas)} agents: {[s.name for s in agent_schemas]}"
        )

        return [
            AgentNode(
                name=schema.name,
                instructions=schema.instructions,
                model=schema.model,
                mcp_tools=schema.mcp_tools,
                use_tools=self.use_tools,
            )
            for schema in agent_schemas
        ]

    async def create_pool(
        self, task_description: str, context: Optional[str] = None
    ) -> AgentPool:
        user_prompt = f"TASK: {task_description}"

        if context:
            user_prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{context}"

        agent_schemas = await self._run_agent(user_prompt)

        agents = self._create_agents(agent_schemas)

        logger.info(f"Successfully created agent pool with {len(agents)} agents")

        return AgentPool(agents)
