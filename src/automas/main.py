import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from automas.utils.logger import get_logger

from .meta_agents import GraphGenerator, PoolGenerator
from .pipeline import PipelineBuilder
from .utils.langfuse_utils import ainvoke_with_lf

if TYPE_CHECKING:
    from automas.agent_pool import AgentPool
    from automas.pipeline import Pipeline

try:
    from .judge import MasevalJudge  # type: ignore

    MASEVAL_AVAILABLE = True
except ImportError:
    MASEVAL_AVAILABLE = False
    MasevalJudge = None  # type: ignore

logger = get_logger()


class AutoMAS:
    def __init__(self):
        self.pool_gen = PoolGenerator()
        self.graph_gen = GraphGenerator()
        self._pipeline: Optional["Pipeline"] = None

    @property
    def pipeline(self) -> Optional["Pipeline"]:
        return self._pipeline

    async def arun(
        self,
        query: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        full_query = query
        if file_path:
            full_query = f"File path: {file_path}\n{full_query}"

        pool: AgentPool = await self.pool_gen.create_pool(query)
        graph_dict = await self.graph_gen.create_graph(pool, query)

        builder = PipelineBuilder()
        self._pipeline = builder.create_from_pool(pool, graph_dict).build()
        answer = await self._pipeline.ainvoke(full_query)

        result = {
            "answer": str(answer),
            "pool": pool.full_agents_data,
            "graph": self._pipeline.to_mermaid_lr(),
        }

        return result

    def run(
        self,
        query: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return asyncio.run(self.arun(query, file_path))

    async def arun_with_judge(
        self,
        query: str,
        max_iter: int = 3,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not MASEVAL_AVAILABLE:
            raise RuntimeError(
                "MasevalJudge is not available. "
                "Please install the maseval dependency group: uv sync --group maseval"
            )

        logger.info(f"Starting arun_with_judge: max_iter={max_iter}, query_length={len(query)}")

        judge = MasevalJudge()  # type: ignore
        full_query = query
        if file_path:
            full_query = f"File path: {file_path}\n{full_query}"
            logger.info(f"File path included: {file_path}")

        experience: list[str] = []

        for iteration in range(max_iter):
            logger.info(f"Iteration {iteration + 1}/{max_iter} started")

            context = "\n\n".join(experience) if experience else None
            if context:
                logger.info(
                    f"Using context from previous iterations (length: {len(context)} chars)"
                )

            logger.info("Creating agent pool")
            pool = await self.pool_gen.create_pool(full_query, context=context)
            logger.info(f"Pool created with {len(pool.full_agents_data)} agents")

            logger.info("Generating execution graph")
            graph_dict = await self.graph_gen.create_graph(pool, full_query, context=context)
            logger.info(f"Graph generated:\n{graph_dict}")

            builder = PipelineBuilder()
            self._pipeline = builder.create_from_pool(pool, graph_dict).build()
            mmd_graph = self._pipeline.to_mermaid_lr()
            logger.info(f"Pipeline graph:\n{mmd_graph}")

            logger.info("Executing pipeline...")
            result, trace_id = await ainvoke_with_lf(pool, self._pipeline, full_query, graph_dict)

            if not trace_id:
                raise ValueError("Trace ID is None")

            logger.info(f"Pipeline executed, trace_id: {trace_id}")
            logger.info(
                f"Pipeline result: {str(result)[:500]}{'...' if len(str(result)) > 500 else ''}"
            )

            logger.info("Evaluating result with judge...")
            judge_result = await judge.evaluate_by_trace_id(trace_id)
            logger.info(
                f"Iteration {iteration + 1} - Judge evaluation: "
                f"score={judge_result.score}, numeric={judge_result.numeric_score}"
            )
            logger.info(f"Judge justification: {judge_result.justification}")

            if judge_result.score in ["ideal", "good"]:
                logger.info(
                    f"Optimal solution found at iteration {iteration + 1}. "
                    f"Stopping early (score: {judge_result.score})"
                )
                return {
                    "answer": str(result),
                    "pool": pool.full_agents_data,
                    "graph": mmd_graph,
                    "trace_id": trace_id,
                    "judge_score": judge_result.score,
                    "iterations": iteration + 1,
                }

            experience.append(f"Iteration {iteration + 1} feedback:\n{judge_result.justification}")

        logger.warning(
            f"Maximum iterations ({max_iter}) reached without finding optimal solution. "
            f"Final score: {judge_result.score}"
        )
        logger.info(f"Final judge justification: {judge_result.justification}")

        return {
            "answer": str(result),
            "pool": pool.full_agents_data,
            "graph": self._pipeline.to_mermaid_lr(),  # type: ignore
            "trace_id": trace_id,
            "judge_score": judge_result.score,
            "iterations": max_iter,
        }

    def run_with_judge(
        self,
        query: str,
        max_iter: int = 3,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return asyncio.run(self.arun_with_judge(query, max_iter, file_path))
