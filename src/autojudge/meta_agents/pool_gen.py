import os
from typing import List, Optional
from pydantic import BaseModel
from pydantic_ai.models import Model
from autojudge.agent_pool import AgentPool
from autojudge.pipeline.node import AgentNode
from autojudge.utils import get_logger
from dotenv import load_dotenv

from .base import BaseMetaAgent
from .prompts import (
    DEFAULT_POOL_INSTRUCT_EXTENDED,
    DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool,
)

logger = get_logger()
load_dotenv(".env")


class AgentSchema(BaseModel):
    name: str
    instructions: str
    mcp_tools: List[str] = []
    model: str = os.getenv("AGENT_NODE_MODEL", "google/gemini-2.5-flash")
    use_tools: bool = True  # added from new version

class PoolGenerator(BaseMetaAgent):
    def __init__(
        self,
        model: str = os.getenv("POOL_GEN_MODEL", "deepseek/deepseek-v4-pro"),
        temperature: float = float(os.getenv("POOL_GEN_TEMPERATURE", 0.3)),
        output_schema: str = "",
        taxonomy: str = "",
        examples: str = "",
        use_summary: bool = False,
        model_override: Optional[Model] = None,
        system_prompt_override: Optional[str] = None,
        judge_model: Optional[str] = None,
        retries: int = 3,
        instrument: bool = True,
    ):
        configured_temperature = os.getenv("POOL_GEN_TEMPERATURE")
        if configured_temperature is not None and float(configured_temperature) != temperature:
            print('ATTENTION: Found another PoolGenerator temperature in config file: ', os.getenv("POOL_GEN_TEMPERATURE"))
        print(
            f"Initializing PoolGenerator with model={model}, temperature={temperature}, summary: {use_summary}"
        )
        self.summary = use_summary
        self.use_tools = use_summary  # derived: summary mode implies tool-enabled prompt
        self.schema = output_schema
        self.taxonomy = taxonomy
        self.examples = examples
        self.system_prompt_override = system_prompt_override
        self.judge_model = judge_model
        super().__init__(
            model=model,
            temperature=temperature,
            model_override=model_override,
            retries=retries,
            instrument=instrument,
        )

    def _get_system_prompt(self) -> str:
        if self.system_prompt_override is not None:
            return self.system_prompt_override
        if not self.summary:
            return DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool.substitute(
                taxonomy=self.taxonomy,
                judge_output_format=self.schema,
                examples=self.examples,
            )
        else:
            return DEFAULT_POOL_INSTRUCT_EXTENDED.substitute(
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
                model=self.judge_model or schema.model,
                mcp_tools=[schema.mcp_tools],  # kept original wrapping
                use_tools=self.use_tools,       # added from new version
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
