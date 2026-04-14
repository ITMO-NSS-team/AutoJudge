import os
from string import Template
from typing import List, Optional

from pydantic import BaseModel

from autojudge.agent_pool import AgentPool
from autojudge.mcp.registry import get_server_descriptions
from autojudge.pipeline.node import AgentNode
from autojudge.utils import get_logger

from .base import BaseMetaAgent
from .prompts import DEFAULT_POOL_INSTRUCT_EXTENDED

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
        use_tools: bool = True,
        prompt_template: Optional[Template] = None,
    ):
        self.schema = output_schema
        self.taxonomy = taxonomy
        self.examples = examples
        self.use_tools = use_tools
        self.pool_instruct = prompt_template or DEFAULT_POOL_INSTRUCT_EXTENDED

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
