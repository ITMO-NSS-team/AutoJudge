from typing import List

from pydantic import BaseModel

from autojudge.agent_pool import AgentPool
from autojudge.mcp.registry import get_server_descriptions
from autojudge.pipeline.node import AgentNode
from autojudge.pipeline.types import GraphDict
from autojudge.utils.logger import get_logger

from .base import DEFAULT_MODEL, BaseMetaAgent
from .prompts import UNIFIED_GEN_INSTRUCT
from .pool_gen import AgentSchema


logger = get_logger()


class UnifiedPoolAndGraph(BaseModel):
    """Combined schema for agent pool and graph structure."""
    agents: List[AgentSchema]
    graph: GraphDict


class UnifiedGenerator(BaseMetaAgent):
    """Meta-agent that generates both agent pool and workflow graph in a single iteration."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )

    def _get_system_prompt(self) -> str:
        mcp_servers_desc = get_server_descriptions()
        return UNIFIED_GEN_INSTRUCT.substitute(mcp_servers_desc=mcp_servers_desc)

    def _get_output_type(self):
        return UnifiedPoolAndGraph

    def _validate_response(self, response: UnifiedPoolAndGraph) -> None:
        """Validate the unified response structure."""
        logger.debug(f"Validating unified response with {len(response.agents)} agents")
        
        if not response.agents:
            raise ValueError("No agents generated")
        
        # Check for unique agent names
        agent_names = [agent.name for agent in response.agents]
        if len(agent_names) != len(set(agent_names)):
            raise ValueError("Agent names must be unique")
        
        agent_names_set = set(agent_names)
        
        # Validate graph references valid agents
        for agent_name in response.graph:
            if agent_name not in agent_names_set:
                logger.error(f"Unknown agent '{agent_name}' in graph")
                raise ValueError(
                    f"Unknown agent '{agent_name}' in graph. "
                    f"Available agents: {', '.join(agent_names)}"
                )
        
        # Validate children references
        for agent_name, children in response.graph.items():
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

    def _create_agents(self, agent_schemas: List[AgentSchema]) -> List[AgentNode]:
        """Create AgentNode instances from schemas."""
        logger.info(f"Creating {len(agent_schemas)} agents: {[s.name for s in agent_schemas]}")

        return [
            AgentNode(
                name=schema.name,
                instructions=schema.instructions,
                model=schema.model,
                mcp_tools=schema.mcp_tools,
            )
            for schema in agent_schemas
        ]

    async def create_pool_and_graph(
        self, 
        task_description: str
    ) -> tuple[AgentPool, GraphDict]:
        """
        Generate both agent pool and workflow graph in a single iteration.
        
        Args:
            task_description: Description of the task to solve
            
        Returns:
            Tuple of (AgentPool, GraphDict) ready to be used in pipeline
        """
        logger.info("Generating unified agent pool and workflow graph")
        logger.debug(f"Task: {task_description[:100]}...")

        user_prompt = f"TASK: {task_description}"

        response = await self._run_agent(user_prompt)
        
        # Validate the response
        self._validate_response(response)
        
        # Create agent pool
        agents = self._create_agents(response.agents)
        agent_pool = AgentPool(agents)
        
        logger.info(
            f"Successfully created unified pool with {len(agents)} agents "
            f"and graph with {len(response.graph)} nodes"
        )
        
        return agent_pool, response.graph
