from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, cast
from copy import deepcopy
from datetime import datetime

from autojudge.agent_pool import AgentPool
from autojudge.optimizers.base import MASOptimizer
from autojudge.judge.unsupervised_judge import UnsupervisedJudge
from autojudge.pipeline.pipeline_builder import PipelineBuilder
from autojudge.pipeline.types import CostBreakdown, GraphDict
from autojudge.utils.logger import get_logger
from autojudge.optimizers.telemetry import (
    AgentPoolNeighbour,
    AgentPoolNeighborhood,
    EvaluationTelemetry,
    PoolCandidate,
    PoolEvaluationRecord,
    clone_pool,
    normalise_quality,
)

LOGGER = get_logger()


DEFAULT_SHAKE_AMPLITUDES: tuple[int, ...] = (1, 2, 3)
DEFAULT_ITERATION_LIMIT = 25


NeighbourKind = Literal["data", "formulation"]


@dataclass(slots=True, frozen=True)
class NeighbourhoodDefinition:
    radius: float | None = None
    metric: Callable[[Mapping[str, object], Mapping[str, object]], float] | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)

    def metric_for(self) -> Callable[[Mapping[str, object], Mapping[str, object]], float]:
        if self.metric is None:
            return lambda _a, _b: 0.0
        return self.metric


@dataclass(frozen=True, slots=True)
class BasicVNSConfiguration:
    """Immutable configuration for :class:`BasicVNS` using default VNS settings."""
    shake_amplitudes: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_SHAKE_AMPLITUDES
    )
    iteration_limit: int = DEFAULT_ITERATION_LIMIT

    def as_payload(self) -> dict[str, object]:
        return {
            "shake_amplitudes": list(self.shake_amplitudes),
            "iteration_limit": self.iteration_limit,
        }


@dataclass(frozen=True, slots=True)
class BasicVNSResult:
    """Aggregated result produced by :class:`BasicVNS`."""
    best_record: PoolEvaluationRecord
    history: tuple[PoolEvaluationRecord, ...]
    budget_consumed: float
    termination_reason: str
    configuration: BasicVNSConfiguration


PipelineBuilderFactory = Callable[[], PipelineBuilder]


def _graph_from_entry(entry: Mapping[str, Sequence[str]] | GraphDict | None) -> GraphDict | None:
    if entry is None:
        return None
    if isinstance(entry, Mapping):
        return {
            str(node): [str(child) for child in cast(Iterable[str], children)]
            for node, children in entry.items()
        }
    return cast(GraphDict, entry)


async def evaluate_agent_pool(
    pool: AgentPool,
    *,
    query: str,
    judge: UnsupervisedJudge,
    pipeline_builder_factory: PipelineBuilderFactory,
    graph: Mapping[str, Sequence[str]] | GraphDict | None = None,
) -> EvaluationTelemetry:
    """Execute a pipeline for the given pool and collect telemetry."""
    graph_dict = _graph_from_entry(graph)
    builder = pipeline_builder_factory()
    if graph_dict is not None:
        if hasattr(builder, "create_from_pool"):
            builder.create_from_pool(pool, graph_dict)
        else:
            default_builder = PipelineBuilder()
            default_builder.create_from_pool(pool, graph_dict)
            builder = default_builder
    else:
        builder.add_from_pool(pool)
    pipeline = builder.build()
    start = perf_counter()
    await pipeline.ainvoke(query)
    latency = perf_counter() - start
    trace = pipeline.trace
    final_output = trace.final_output
    judge_result = await judge.evaluate(trace)
    quality = normalise_quality(judge_result)
    raw_cost = pipeline.cost
    if isinstance(raw_cost, CostBreakdown):
        cost_breakdown = raw_cost
    else:
        cost_breakdown = None
        dump_method = getattr(raw_cost, "model_dump", None)
        if callable(dump_method):
            dumped_cost = dump_method()
            if dumped_cost is not None:
                cost_breakdown = CostBreakdown.model_validate(dumped_cost)
    total_cost = float(cost_breakdown.total_price)
    cost_payload = {key: float(value) for key, value in cost_breakdown.model_dump().items()}
    tokens: Mapping[str, int] | None = None
    if all(hasattr(pipeline, attr) for attr in ("input_tokens", "output_tokens", "total_tokens")):
        tokens = {
            "input_tokens": int(getattr(pipeline, "input_tokens")),
            "output_tokens": int(getattr(pipeline, "output_tokens")),
            "total_tokens": int(getattr(pipeline, "total_tokens")),
        }
    mermaid_graph: str | None = None
    if hasattr(pipeline, "to_mermaid_lr"):
        try:
            mermaid_graph = pipeline.to_mermaid_lr(visualize=False)
        except TypeError:
            mermaid_graph = pipeline.to_mermaid_lr()
    metrics: dict[str, float] = {
        "quality": quality.value,
        "cost": total_cost,
        "pipeline_cost": total_cost,
    }
    if latency >= 0:
        metrics["latency_seconds"] = float(latency)
    if tokens is not None:
        metrics.update(
            {
                "input_tokens": float(tokens["input_tokens"]),
                "output_tokens": float(tokens["output_tokens"]),
                "total_tokens": float(tokens["total_tokens"]),
            }
        )
    telemetry = EvaluationTelemetry(
        quality=quality,
        cost=total_cost,
        budget_spent=total_cost,
        trace=trace,
        judge_result=judge_result,
        timestamp=datetime.now(),
        result_output=final_output,
        latency_seconds=latency,
        pipeline_tokens=tokens,
        pipeline_cost=cost_payload,
        mermaid_graph=mermaid_graph,
        metrics=metrics,
    )
    LOGGER.info(
        f"Evaluated pool size={len(pool)} quality={quality.value:.4f} "
        f"cost={total_cost:.4f} reason=evaluation"
    )
    return telemetry


def _swap_order(pool: AgentPool, amplitude: int, rng: random.Random) -> AgentPool:
    agents = [deepcopy(agent) for agent in pool]
    if len(agents) < 2:
        return AgentPool(agents)
    swaps = min(amplitude, len(agents))
    indices = list(range(len(agents)))
    for _ in range(swaps):
        i, j = rng.sample(indices, 2)
        agents[i], agents[j] = agents[j], agents[i]
    return AgentPool(agents)


def _tweak_instructions(pool: AgentPool, amplitude: int, rng: random.Random) -> AgentPool:
    agents = [deepcopy(agent) for agent in pool]
    if not agents:
        return AgentPool(agents)
    count = min(amplitude, len(agents))
    targets = rng.sample(range(len(agents)), count)
    for index in targets:
        agent = agents[index]
        agent.instructions = f"{agent.instructions}\n\nRefinement level {amplitude}"
    return AgentPool(agents)


_TOOL_LIBRARY: tuple[str, ...] = ("web-search", "code-exec", "summary-audit")


def _tweak_tools(pool: AgentPool, amplitude: int, rng: random.Random) -> AgentPool:
    agents = [deepcopy(agent) for agent in pool]
    if not agents:
        return AgentPool(agents)
    count = min(amplitude, len(agents))
    targets = rng.sample(range(len(agents)), count)
    for index, tool_index in zip(targets, rng.choices(range(len(_TOOL_LIBRARY)), k=len(targets))):
        agent = agents[index]
        tool = _TOOL_LIBRARY[tool_index]
        tools = list(agent.mcp_tools)
        if tool in tools:
            tools.remove(tool)
        else:
            tools.append(tool)
        agent.mcp_tools = tools
    return AgentPool(agents)


def _default_neighbourhood(*, rng: random.Random | None = None) -> AgentPoolNeighborhood:
    operators = (
        AgentPoolNeighbour("swap-order", _swap_order),
        AgentPoolNeighbour("instruction-refine", _tweak_instructions),
        AgentPoolNeighbour("tool-adjust", _tweak_tools),
    )
    return AgentPoolNeighborhood(operators, rng=rng)

class BasicVNS(MASOptimizer):
    """Implementation of the Basic Variable Neighbourhood Search algorithm."""
    def __init__(
        self,
        *,
        query: str,
        judge: UnsupervisedJudge,
        pipeline_builder_factory: PipelineBuilderFactory | None = None,
        neighbourhood: AgentPoolNeighborhood | None = None,
        configuration: BasicVNSConfiguration | None = None,
        rng: random.Random | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._query = query
        self._judge = judge
        self._builder_factory = pipeline_builder_factory or PipelineBuilder
        self._configuration = configuration or BasicVNSConfiguration()
        self._neighbourhood = neighbourhood or _default_neighbourhood(rng=rng)
        self._rng = rng or random.Random()
        self._time_source = time_source or perf_counter
        self._history: list[PoolEvaluationRecord] = []
        self._last_result: BasicVNSResult | None = None
        self._budget: float | None = None
        self._time_limit: float | None = None
        self._max_retries = 2

    async def optimize(
        self,
        initial_graph: GraphDict,
        agent_pool: AgentPool,
        judge: UnsupervisedJudge,
        **kwargs: Any,
    ) -> tuple[GraphDict, float, list[dict[str, Any]]]:
        self._judge = judge
        history_reporter = cast(
            Callable[[Iterable[PoolEvaluationRecord]], Sequence[Any]] | None,
            kwargs.get("history_reporter"),
        )
        budget_value = kwargs.get("budget", self._budget)
        time_limit_value = kwargs.get("time_limit", self._time_limit)
        budget = float(budget_value)
        time_limit = float(time_limit_value)
        self._budget = budget
        self._time_limit = time_limit
        pool = kwargs.get("initial_pool", agent_pool)
        start_time = self._time_source()
        LOGGER.info("Starting BasicVNS search reason=start")
        current_candidate = PoolCandidate(clone_pool(pool))
        current_record = await self._evaluate_with_retry(current_candidate, stage="initial")
        history: list[PoolEvaluationRecord] = [current_record]
        budget_consumed = current_record.telemetry.cost
        best_record = current_record
        shake_amplitudes = self._configuration.shake_amplitudes
        k_index = 0
        iteration = 0
        termination_reason = "running"
        if budget_consumed >= budget:
            termination_reason = "budget-exhausted"
        while termination_reason == "running" and iteration < self._configuration.iteration_limit:
            elapsed = self._time_source() - start_time
            if elapsed >= time_limit:
                termination_reason = "timeout"
                LOGGER.info(
                    f"Time limit reached after {elapsed:.2f}s iterations={iteration} reason=timeout"
                )
                break
            if budget_consumed >= budget:
                termination_reason = "budget-exhausted"
                LOGGER.info(
                    f"Budget exhausted spent={budget_consumed:.4f} iteration={iteration} reason=budget"
                )
                break
            iteration += 1
            k = shake_amplitudes[k_index]
            LOGGER.info(f"Iteration {iteration} using k={k} reason=iterate")
            neighbours = self._neighbourhood.neighbours(current_record.candidate.pool, k)
            if not neighbours:
                LOGGER.info(f"No neighbours generated for k={k} reason=no-neighbour")
                k_index = (k_index + 1) % len(shake_amplitudes)
                continue
            shaken_candidate = self._rng.choice(neighbours)
            shaken_record = await self._evaluate_with_retry(
                shaken_candidate, stage=f"shake-k{k}"
            )
            history.append(shaken_record)
            budget_consumed += shaken_record.telemetry.cost
            if shaken_record.telemetry.quality.value > best_record.telemetry.quality.value:
                best_record = shaken_record
            improved_record = await self._local_search(
                shaken_record,
                history,
                start_time,
                time_limit,
                budget,
                budget_consumed,
            )
            budget_consumed = sum(record.telemetry.cost for record in history)
            if improved_record.telemetry.quality.value > current_record.telemetry.quality.value:
                LOGGER.info(
                    f"Accepted improved candidate quality={improved_record.telemetry.quality.value:.4f} "
                    f"previous={current_record.telemetry.quality.value:.4f} reason=improve"
                )
                current_record = improved_record
                if current_record.telemetry.quality.value > best_record.telemetry.quality.value:
                    best_record = current_record
                k_index = 0
                continue
            LOGGER.info(f"No improvement at k={k} advancing neighbourhood reason=no-improvement")
            k_index = (k_index + 1) % len(shake_amplitudes)
        if termination_reason == "running":
            termination_reason = "iteration-limit"
            LOGGER.info(
                f"Maximum iterations {self._configuration.iteration_limit} reached reason=iteration-limit"
            )
        result = BasicVNSResult(
            best_record=best_record,
            history=tuple(history),
            budget_consumed=sum(record.telemetry.cost for record in history),
            termination_reason=termination_reason,
            configuration=self._configuration,
        )
        self._history = history
        self._last_result = result
        LOGGER.info(
            f"BasicVNS finished termination={termination_reason} "
            f"budget={result.budget_consumed:.4f} reason=complete"
        )
        history_payload = (
            list(history_reporter(result.history))
            if history_reporter is not None
            else list(result.history)
        )
        best_score = result.best_record.telemetry.quality.value
        graph_copy = deepcopy(initial_graph)
        return graph_copy, best_score, history_payload

    @property
    def evaluation_history(self) -> tuple[PoolEvaluationRecord, ...]:
        return tuple(self._history)

    @property
    def last_result(self) -> BasicVNSResult | None:
        return self._last_result

    @property
    def configuration(self) -> BasicVNSConfiguration:
        return self._configuration

    async def _evaluate(self, candidate: PoolCandidate) -> PoolEvaluationRecord:
        telemetry = await evaluate_agent_pool(
            candidate.pool,
            query=self._query,
            judge=self._judge,
            pipeline_builder_factory=self._builder_factory,
        )
        return PoolEvaluationRecord(candidate.clone(), telemetry)

    async def _evaluate_with_retry(
        self, candidate: PoolCandidate, *, stage: str
    ) -> PoolEvaluationRecord:
        attempt = 0
        backoff = 0.05
        while attempt < self._max_retries:
            try:
                return await self._evaluate(candidate)
            except Exception as exc:
                attempt += 1
                LOGGER.warning(
                    f"Evaluation failed stage={stage} attempt={attempt} error={exc} "
                    f"reason=pipeline-error"
                )
                if attempt >= self._max_retries:
                    LOGGER.error(
                        f"Exhausted retries stage={stage} attempts={attempt} "
                        f"reason=retry-exhausted"
                    )
                    raise
                await asyncio.sleep(backoff * attempt)

    async def _local_search(
        self,
        seed_record: PoolEvaluationRecord,
        history: list[PoolEvaluationRecord],
        start_time: float,
        time_limit: float,
        budget: float,
        budget_consumed: float,
    ) -> PoolEvaluationRecord:
        best = seed_record
        elapsed = self._time_source() - start_time
        if elapsed >= time_limit or budget_consumed >= budget:
            return best
        neighbours = self._neighbourhood.neighbours(
            seed_record.candidate.pool, self._configuration.shake_amplitudes[0]
        )
        for neighbour in neighbours:
            elapsed = self._time_source() - start_time
            if elapsed >= time_limit:
                LOGGER.info("Stopping local search due to timeout reason=timeout")
                break
            if budget_consumed >= budget:
                LOGGER.info("Stopping local search due to budget reason=budget")
                break
            record = await self._evaluate_with_retry(
                neighbour, stage=f"local-search-k{self._configuration.shake_amplitudes[0]}"
            )
            history.append(record)
            budget_consumed += record.telemetry.cost
            if record.telemetry.quality.value > best.telemetry.quality.value:
                LOGGER.info(
                    f"Local search improved quality={record.telemetry.quality.value:.4f} "
                    f"previous={best.telemetry.quality.value:.4f} reason=local-improve"
                )
                best = record
                break
        return best