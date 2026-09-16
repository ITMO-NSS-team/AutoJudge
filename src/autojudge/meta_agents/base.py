import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from autojudge.pipeline.types import UsageTrackingMixin
from autojudge.utils.langfuse_utils import setup_langfuse_instrumentation

load_dotenv()
setup_langfuse_instrumentation()

DEFAULT_MODEL = os.getenv("DEFAULT_META_MODEL", "anthropic/claude-sonnet-4")
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 3


class BaseMetaAgent(UsageTrackingMixin, ABC):
    """Base class for all meta-agents with common initialization and configuration."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        self.api_key = os.getenv("OPENROUTER_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError(
                "No OPENROUTER_API_KEY provided. Set it in environment or pass as argument."
            )

        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"Temperature must be in [0.0, 1.0], got {temperature}")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

        self._model = self._create_model()
        self.agent = self._create_agent()
        self._usage: Optional[RunUsage] = None

    def _create_model(self) -> OpenAIChatModel:
        settings = ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        return OpenAIChatModel(
            self.model,
            provider=OpenRouterProvider(api_key=self.api_key),
            settings=settings,
        )

    def _get_mcp_tools(self) -> list[str]:
        """Override to specify MCP tools for this agent. Returns empty list by default."""
        return []

    def _create_agent(self) -> Agent:
        system_prompt = self._get_system_prompt()
        output_type = self._get_output_type()

        agent = Agent(
            name=self.__class__.__name__,
            model=self._model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=self.retries,
        )
        # pydantic-ai 2.x accepts instrumentation as an attribute, not a kwarg.
        agent.instrument = True
        return agent

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Returns system prompt for the agent."""
        pass

    @abstractmethod
    def _get_output_type(self) -> Type[BaseModel] | Type[dict] | Type[list]:
        """Returns output type for structured responses."""
        pass

    async def _run_agent(self, prompt: str) -> Any:
        """Run agent with given prompt and return structured output."""
        result = await self.agent.run(prompt)
        # pydantic-ai 1.x exposes usage as a method, 2.x as a property.
        self._usage = result.usage() if callable(result.usage) else result.usage
        return result.output
