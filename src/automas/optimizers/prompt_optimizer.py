from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from automas.agent_pool import AgentPool
from automas.meta_agents.base import BaseMetaAgent
from automas.pipeline.node import AgentNode
from automas.pipeline.pipeline import Pipeline
from automas.utils.logger import get_logger, save_json

from pydantic import BaseModel

logger = get_logger()


class OptimizedPrompt(BaseModel):
    optimized_prompt: str


class PromptOptimizer(BaseMetaAgent, ABC):
    """Abstract base class for prompt optimizers."""

    def __init__(self, system_prompt: Optional[str] = None, **kwargs):
        self._system_prompt = system_prompt
        super().__init__(**kwargs)

    def _get_system_prompt(self):
        return self._system_prompt

    def _get_output_type(self):
        return OptimizedPrompt

    @abstractmethod
    async def aoptimize(self, prompt: str, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def optimize(self, prompt: str, *args, **kwargs) -> str:
        pass


class HyPEPromptOptimizer(PromptOptimizer):
    """HyPE prompt optimizer implementation."""

    def __init__(self, *args, **kwargs):
        system_prompt = (
            "You are an expert prompt engineer. Your only task is to "
            "generate a hypothetical instructional system prompt that would help "
            "an agent inside a multi-agent pipeline effectively perform the same "
            "underlying task as the original instruction. You will be provided "
            "with the agent's system instruction, name, and information about whether it is terminal.\n"
            "### HARD CONSTRAINTS ###\n"
            "1. LANGUAGE:\n"
            "   - Output MUST be in the EXACT SAME LANGUAGE as the agent's original instruction.\n"
            "2. CONTENT:\n"
            "   - Output ONLY the hypothetical instructional prompt - do NOT answer the original query directly.\n"
            "   - The hypothetical prompt must solve the same task as the original instruction.\n"
            "   - If the original instruction contains any code snippets, you must include it in the final prompt.\n"
            "3. TECHNICAL PRESERVATION:\n"
            "   - Code blocks must be preserved with original syntax and formatting.\n"
            "   - Variables, placeholders ({{var}}), and technical terms kept unchanged.\n"
            "   - Markdown and special formatting replicated precisely.\n"
            "4. OUTPUT QUALITY:\n"
            "   - For intermediate agents: output should be immediately usable by the next agent in the pipeline.\n"
            "   - For terminal agents: output should be the complete final solution for the user.\n"
            "   - Preserve ALL explicit output format requirements from the original instruction.\n"
            "TEMPLATE POLICY (SHORT):\n"
            "   - If the provided general_task or original agent instruction contains an explicit output template or example-format block (e.g. a tag pair or templated example), COPY THAT TEMPLATE VERBATIM into the generated hypothetical prompt (including its exact tags and content).\n"
            "   - If NO explicit template/example is present in the inputs, DO NOT invent any example outputs, sample values, example-format lines, or tags.\n"
            "### YOUR OUTPUT FORMAT ###\n"
            "Return ONLY a valid JSON object with the following format:\n"
            '{ "optimized_prompt": "<your hypothetical instructional system prompt here>" }\n'
        )
        super().__init__(system_prompt=system_prompt, *args, **kwargs)

    async def aoptimize(self, agent: AgentNode, general_task: str) -> str:
        role_specific = ""
        if agent.is_terminal_node:
            role_specific = (
                "**TERMINAL AGENT: Ensure the output is the complete final solution with all formatting requirements from the general task preserved.**\n"
                "- The final answer must be FIRST\n"
                "- If there are previous agents, do NOT include their logs, narrative, or intermediate text in the final answer. Extract needed data from previous outputs and present only the required final content.\n"
                "- Do not return high-level plans without the final output.\n"
                "- After the final validated output you may include one <reasoning> block.\n"
                f"General task: {general_task}"
            )
        else:
            role_specific = (
                "**INTERMEDIATE AGENT: Ensure the output is concrete and immediately usable by the next agent in the pipeline.**:\n"
                "- Do not return high-level plans without data."
            )

        query = f"""
        ### INPUT ###
        Original system instruction: {agent.instructions}
        Agent's name: {agent.name}
        {role_specific}
        ### OUTPUT ###
        Hypothetical Instructional Prompt:
        """

        output = await self._run_agent(query)
        return output.optimized_prompt

    def optimize(self, agent: AgentNode) -> str:
        return asyncio.run(self.aoptimize(agent))


async def optimize_pipeline_prompts(
    pipeline: Pipeline,
    pool: AgentPool,
    optimizer: PromptOptimizer,
    general_task: str = None,
) -> Path:
    """Optimizes the agent system prompts from the pipeline.

    Saves the optimization results as a JSON file with timestamp.
    The `general_task` is used for retrieving the output format for
    the terminal agent node.
    """

    logger.info("Starting agent prompts optimization")

    tasks = [
        optimizer.aoptimize(node, general_task)
        for node in pipeline.execution_order
    ]
    optimized_prompts = await asyncio.gather(*tasks, return_exceptions=True)

    prompt_optimization_data = {"general_task": general_task}

    pool_agents_by_name = {agent.name: agent for agent in pool}
    for node, opt_result in zip(pipeline.execution_order, optimized_prompts):
        original = node.instructions
        if isinstance(opt_result, Exception):
            logger.warning(
                f"Prompt optimizer failed for agent {node.name}: {opt_result}"
            )
            optimized = original
        else:
            optimized = opt_result
            logger.debug(
                f"Agent '{node.name}' original instructions: {original[:200]}"
            )
            logger.debug(
                f"Agent '{node.name}' optimized instructions: {optimized[:200]}"
            )

        prompt_optimization_data[f"agent_{node.id}_{node.name}"] = {
            "success": not isinstance(opt_result, Exception),
            "original": original,
            "optimized": optimized,
            "error": (
                str(opt_result) if isinstance(opt_result, Exception) else None
            ),
        }

        node.instructions = optimized
        pool_agents_by_name[node.name].instructions = optimized

    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
    filename = f"agent_prompts_{timestamp}.json"
    filepath = save_json(prompt_optimization_data, filename)

    return filepath
