from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from types import MappingProxyType
from typing import Any, cast

from automas.agent_pool import AgentPool
from automas.judge.base import Judge
from automas.judge.unsupervised_judge import UnsupervisedJudge
from automas.optimizers.base import MASOptimizer
from automas.optimizers.telemetry import EvaluationTelemetry, clone_pool
from automas.optimizers.vns import (
    NeighbourhoodDefinition,
    NeighbourKind,
    evaluate_agent_pool,
)
from automas.pipeline.pipeline_builder import PipelineBuilder
from automas.pipeline.types import GraphDict
from automas.utils.logger import get_logger

LOGGER = get_logger()


PipelineBuilderFactory = Callable[[], PipelineBuilder]
Translator = Callable[["Landscape", "Landscape"], "Landscape"]


def _normalise_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    if mapping is None:
        return MappingProxyType({})
    return MappingProxyType({str(key): value for key, value in dict(mapping).items()})


@dataclass(slots=True, frozen=True)
class Landscape:
    dataset_id: str
    formulation_id: str
    pool: AgentPool
    dataset_metadata: Mapping[str, object] = field(default_factory=dict)
    formulation_metadata: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool", clone_pool(self.pool))
        object.__setattr__(self, "dataset_metadata", _normalise_mapping(self.dataset_metadata))
        object.__setattr__(
            self, "formulation_metadata", _normalise_mapping(self.formulation_metadata)
        )
        object.__setattr__(self, "metadata", _normalise_mapping(self.metadata))

    @property
    def identifier(self) -> tuple[str, str]:
        return self.dataset_id, self.formulation_id

    def clone(self) -> Landscape:
        return Landscape(
            dataset_id=self.dataset_id,
            formulation_id=self.formulation_id,
            pool=self.pool,
            dataset_metadata=self.dataset_metadata,
            formulation_metadata=self.formulation_metadata,
            metadata=self.metadata,
        )

    def with_pool(self, pool: AgentPool) -> Landscape:
        return Landscape(
            dataset_id=self.dataset_id,
            formulation_id=self.formulation_id,
            pool=pool,
            dataset_metadata=self.dataset_metadata,
            formulation_metadata=self.formulation_metadata,
            metadata=self.metadata,
        )

    def with_metadata(self, metadata: Mapping[str, object]) -> Landscape:
        return Landscape(
            dataset_id=self.dataset_id,
            formulation_id=self.formulation_id,
            pool=self.pool,
            dataset_metadata=self.dataset_metadata,
            formulation_metadata=self.formulation_metadata,
            metadata=metadata,
        )


@dataclass(slots=True, frozen=True)
class LandscapeNeighbour:
    key: tuple[str, str]
    distance: float
    _meta_space: "LandscapeMetaSpace" = field(repr=False, compare=False)

    @property
    def landscape(self) -> "Landscape":
        return self._meta_space.require(self.key)


@dataclass(slots=True)
class LandscapeNeighborhood:
    meta_space: "LandscapeMetaSpace"
    origin: tuple[str, str]
    kind: NeighbourKind
    definition: NeighbourhoodDefinition | None
    _cache: tuple[LandscapeNeighbour, ...] | None = field(default=None, init=False, repr=False)

    @property
    def parameters(self) -> Mapping[str, object]:
        if self.definition is None:
            return MappingProxyType({})
        return self.definition.parameters

    def neighbours(self) -> tuple[LandscapeNeighbour, ...]:
        if self._cache is None:
            self._cache = self._compute()
        return self._cache

    def within_amplitude(self, amplitude: int) -> tuple[LandscapeNeighbour, ...]:
        neighbours = self.neighbours()
        if self.definition is None or self.definition.radius is None:
            return neighbours
        radius = max(1, amplitude) * float(self.definition.radius)
        return tuple(neighbour for neighbour in neighbours if neighbour.distance <= radius)

    def _compute(self) -> tuple[LandscapeNeighbour, ...]:
        if self.definition is None:
            LOGGER.info(
                f"No neighbourhood for origin={self.origin} phase={self.kind} reason=no-definition"
            )
            return tuple()
        origin_metadata = self.meta_space.metadata_for(self.origin, self.kind)
        if origin_metadata is None:
            return tuple()
        metric = self.definition.metric_for()
        neighbours: list[LandscapeNeighbour] = []
        for key in self.meta_space.keys():
            if key == self.origin:
                continue
            target_metadata = self.meta_space.metadata_for(key, self.kind)
            if target_metadata is None:
                continue
            distance = float(metric(origin_metadata, target_metadata))
            neighbours.append(
                LandscapeNeighbour(key=key, distance=distance, _meta_space=self.meta_space)
            )
        neighbours.sort(key=lambda item: (item.distance, item.key))
        return tuple(neighbours)


@dataclass(slots=True)
class LandscapeMetaSpace:
    _registry: dict[tuple[str, str], "Landscape"] = field(default_factory=dict)
    _definitions: dict[tuple[str, str], dict[NeighbourKind, NeighbourhoodDefinition | None]] = (
        field(default_factory=dict)
    )

    def register(
        self,
        landscape: "Landscape",
        *,
        data_neighbourhood: NeighbourhoodDefinition | None = None,
        formulation_neighbourhood: NeighbourhoodDefinition | None = None,
    ) -> None:
        key = landscape.identifier
        self._registry[key] = landscape.clone()
        self._definitions[key] = {
            "data": data_neighbourhood,
            "formulation": formulation_neighbourhood,
        }

    def require(self, key: tuple[str, str]) -> "Landscape":
        try:
            stored = self._registry[key]
        except KeyError as exc:
            raise KeyError(f"Landscape '{key}' is not registered") from exc
        return stored.clone()

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._registry.keys())

    def neighbourhood_for(self, landscape: "Landscape", kind: NeighbourKind) -> LandscapeNeighborhood:
        definition = self._definitions.get(landscape.identifier, {}).get(kind)
        return LandscapeNeighborhood(
            meta_space=self,
            origin=landscape.identifier,
            kind=kind,
            definition=definition,
        )

    def metadata_for(
        self, key: tuple[str, str], kind: NeighbourKind
    ) -> Mapping[str, object] | None:
        landscape = self._registry.get(key)
        if landscape is None:
            return None
        if kind == "data":
            return landscape.dataset_metadata
        return landscape.formulation_metadata

    def __bool__(self) -> bool:
        return bool(self._registry)


@dataclass(slots=True, frozen=True)
class LandscapeEvaluationRecord:
    landscape: Landscape
    telemetry: EvaluationTelemetry

    @property
    def identifier(self) -> tuple[str, str]:
        return self.landscape.identifier


@dataclass(slots=True, frozen=True)
class PhaseSchedule:
    amplitudes: Sequence[int]
    stability: Sequence[int] | None = None

    def __post_init__(self) -> None:
        amplitudes = tuple(int(value) for value in self.amplitudes)
        stability = (
            tuple(int(value) for value in self.stability)
            if self.stability is not None
            else tuple(1 for _ in amplitudes)
        )
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "stability", stability)

    @property
    def amplitude_count(self) -> int:
        return len(cast(Sequence[int], self.amplitudes))

    def amplitude_for(self, index: int) -> int:
        amplitudes = cast(tuple[int, ...], self.amplitudes)
        return amplitudes[min(index, len(amplitudes) - 1)]

    def stability_for(self, index: int) -> int:
        stability = cast(tuple[int, ...], self.stability)
        if len(stability) == 1:
            return stability[0]
        return stability[min(index, len(stability) - 1)]


class PhaseProgress(str, Enum):
    HOLD = "hold"
    AMPLITUDE = "amplitude"
    RESET = "reset"


@dataclass(slots=True)
class PhaseState:
    schedule: PhaseSchedule
    k_index: int = 0
    s_remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    @property
    def amplitude(self) -> int:
        return self.schedule.amplitude_for(self.k_index)

    def reset(self) -> None:
        self.k_index = 0
        self.s_remaining = self.schedule.stability_for(self.k_index)

    def record_failure(self) -> PhaseProgress:
        self.s_remaining -= 1
        if self.s_remaining > 0:
            return PhaseProgress.HOLD
        if self.k_index + 1 < self.schedule.amplitude_count:
            self.k_index += 1
            self.s_remaining = self.schedule.stability_for(self.k_index)
            return PhaseProgress.AMPLITUDE
        self.reset()
        return PhaseProgress.RESET


@dataclass(slots=True, frozen=True)
class VariableLandscapeSearchResult:
    best_record: LandscapeEvaluationRecord
    history: tuple[LandscapeEvaluationRecord, ...]
    budget_consumed: float
    elapsed_seconds: float
    termination_reason: str


class VariableLandscapeSearch:
    _phase_order: tuple[NeighbourKind, ...] = ("data", "formulation")

    def __init__(
        self,
        *,
        meta_space: LandscapeMetaSpace,
        query: str,
        judge: UnsupervisedJudge,
        pipeline_builder_factory: PipelineBuilderFactory,
        phase_schedules: Mapping[NeighbourKind, PhaseSchedule],
        translator: Translator | None = None,
        time_source: Callable[[], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._meta_space = meta_space
        self._query = query
        self._judge = judge
        self._pipeline_builder_factory = pipeline_builder_factory
        self._phase_schedules = {phase: phase_schedules[phase] for phase in self._phase_order}
        self._translator = translator or (lambda _origin, candidate: candidate.clone())
        self._time_source = time_source or perf_counter
        self._rng = rng or random.Random()
        self._phase_states = {
            phase: PhaseState(self._phase_schedules[phase]) for phase in self._phase_order
        }

    async def run(
        self,
        start: Landscape,
        *,
        budget: float,
        time_limit: float | None = None,
    ) -> VariableLandscapeSearchResult:
        limit = float("inf") if time_limit is None else float(time_limit)
        start_time = self._time_source()
        history: list[LandscapeEvaluationRecord] = []
        LOGGER.info(f"Evaluating seed landscape={start.identifier} reason=initial-evaluation")
        current_record = await self._evaluate(start.clone())
        history.append(current_record)
        best_record = current_record
        budget_consumed = current_record.telemetry.budget_spent
        state_reason = "running"
        if budget_consumed >= budget:
            state_reason = "budget-exhausted"
            LOGGER.info(
                f"Budget exhausted during initial evaluation budget={budget_consumed:.4f} reason=budget-exhausted"
            )
        elif self._time_exceeded(start_time, limit):
            state_reason = "time-exhausted"
            LOGGER.info("Time limit reached during initial evaluation reason=time-exhausted")
        phase_index = 0
        empty_visits = 0
        max_empty_visits = max(1, len(self._meta_space.keys()) * len(self._phase_order))
        while state_reason == "running":
            if budget_consumed >= budget:
                state_reason = "budget-exhausted"
                LOGGER.info(
                    f"Budget exhausted budget={budget_consumed:.4f} reason=budget-exhausted"
                )
                break
            if self._time_exceeded(start_time, limit):
                state_reason = "time-exhausted"
                LOGGER.info("Time limit reached reason=time-exhausted")
                break
            phase = self._phase_order[phase_index % len(self._phase_order)]
            state = self._phase_states[phase]
            neighbourhood = self._meta_space.neighbourhood_for(current_record.landscape, phase)
            candidates = neighbourhood.within_amplitude(state.amplitude)
            if not candidates:
                LOGGER.info(
                    f"Sequential change due to empty neighbourhood phase={phase} reason=no-neighbours"
                )
                state.reset()
                phase_index += 1
                empty_visits += 1
                if empty_visits >= max_empty_visits:
                    state_reason = "no-neighbours"
                continue
            empty_visits = 0
            target = self._rng.choice(candidates)
            translated = self._translator(current_record.landscape, target.landscape)
            LOGGER.info(
                f"Shaking from={current_record.identifier} to={translated.identifier} "
                f"phase={phase} amplitude={state.amplitude} distance={target.distance:.4f} reason=shake"
            )
            shaken_record = await self._evaluate(translated)
            history.append(shaken_record)
            budget_consumed += shaken_record.telemetry.budget_spent
            if shaken_record.telemetry.quality.value > best_record.telemetry.quality.value:
                best_record = shaken_record
            if budget_consumed >= budget:
                state_reason = "budget-exhausted"
                LOGGER.info(
                    f"Budget exhausted after shake budget={budget_consumed:.4f} reason=budget-exhausted"
                )
                break
            if self._time_exceeded(start_time, limit):
                state_reason = "time-exhausted"
                LOGGER.info("Time limit reached after shake reason=time-exhausted")
                break
            improved_record: LandscapeEvaluationRecord | None = None
            if shaken_record.telemetry.quality.value > current_record.telemetry.quality.value:
                improved_record = shaken_record
            else:
                ordered = [candidate for candidate in candidates if candidate.key != target.key]
                ordered.sort(key=lambda item: (item.distance, item.key))
                for candidate in ordered:
                    translated_candidate = self._translator(
                        current_record.landscape, candidate.landscape
                    )
                    LOGGER.info(
                        f"Local improvement attempt from={current_record.identifier} to={translated_candidate.identifier} "
                        f"phase={phase} amplitude={state.amplitude} distance={candidate.distance:.4f} reason=improve"
                    )
                    evaluation = await self._evaluate(translated_candidate)
                    history.append(evaluation)
                    budget_consumed += evaluation.telemetry.budget_spent
                    if evaluation.telemetry.quality.value > best_record.telemetry.quality.value:
                        best_record = evaluation
                    if budget_consumed >= budget:
                        state_reason = "budget-exhausted"
                        LOGGER.info(
                            f"Budget exhausted during improvement budget={budget_consumed:.4f} reason=budget-exhausted"
                        )
                        break
                    if self._time_exceeded(start_time, limit):
                        state_reason = "time-exhausted"
                        LOGGER.info("Time limit reached during improvement reason=time-exhausted")
                        break
                    if evaluation.telemetry.quality.value > current_record.telemetry.quality.value:
                        improved_record = evaluation
                        break
                if state_reason != "running":
                    break
            if improved_record is not None:
                current_record = improved_record
                if improved_record.telemetry.quality.value > best_record.telemetry.quality.value:
                    best_record = improved_record
                state.reset()
                phase_index += 1
                LOGGER.info(
                    f"Accepted improvement phase={phase} quality={current_record.telemetry.quality.value:.4f} reason=improvement"
                )
                continue
            LOGGER.info(
                f"No improvement in phase={phase} amplitude={state.amplitude} reason=no-improvement"
            )
            progress = state.record_failure()
            if progress == PhaseProgress.AMPLITUDE:
                LOGGER.info(
                    f"Increased amplitude phase={phase} next={state.amplitude} reason=amplitude-increase"
                )
            elif progress == PhaseProgress.HOLD:
                LOGGER.info(
                    f"Maintaining amplitude phase={phase} remaining={state.s_remaining} reason=stability-hold"
                )
            else:
                LOGGER.info(
                    f"Sequential change after failure phase={phase} reason=sequential-change"
                )
            phase_index += 1
        elapsed = self._time_source() - start_time
        LOGGER.info(
            f"VariableLandscapeSearch finished termination={state_reason} budget={budget_consumed:.4f} "
            f"elapsed={elapsed:.4f} reason=complete"
        )
        return VariableLandscapeSearchResult(
            best_record=best_record,
            history=tuple(history),
            budget_consumed=budget_consumed,
            elapsed_seconds=elapsed,
            termination_reason=state_reason,
        )

    async def _evaluate(self, landscape: Landscape) -> LandscapeEvaluationRecord:
        telemetry = await evaluate_agent_pool(
            landscape.pool,
            query=self._query,
            judge=self._judge,
            pipeline_builder_factory=self._pipeline_builder_factory,
            graph=self._graph_payload(landscape),
        )
        LOGGER.info(
            f"Evaluated landscape={landscape.identifier} quality={telemetry.quality.value:.4f} "
            f"cost={telemetry.cost:.4f} reason=evaluation"
        )
        return LandscapeEvaluationRecord(landscape=landscape.clone(), telemetry=telemetry)

    @staticmethod
    def _graph_payload(landscape: Landscape) -> Mapping[str, Sequence[str]] | GraphDict | None:
        entry = landscape.metadata.get("graph")
        if entry is None:
            return None
        if isinstance(entry, Mapping):
            return {
                str(node): [str(child) for child in cast(Sequence[str], children)]
                for node, children in dict(entry).items()
            }
        return cast(GraphDict, entry)

    def _time_exceeded(self, start_time: float, limit: float) -> bool:
        return self._time_source() - start_time >= limit


@dataclass(slots=True, frozen=True)
class BasicVLSPhaseConfiguration:
    schedules: Mapping[NeighbourKind, PhaseSchedule]

    def __post_init__(self) -> None:
        prepared: dict[NeighbourKind, PhaseSchedule] = {}
        for phase in ("data", "formulation"):
            prepared[cast(NeighbourKind, phase)] = self.schedules[phase]
        object.__setattr__(self, "schedules", MappingProxyType(prepared))

    @classmethod
    def default(cls) -> "BasicVLSPhaseConfiguration":
        schedule = PhaseSchedule(amplitudes=(1, 2, 3), stability=(1,))
        return cls(schedules={"data": schedule, "formulation": schedule})

    def schedule_for(self, phase: NeighbourKind) -> PhaseSchedule:
        return self.schedules[phase]

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for phase, schedule in self.schedules.items():
            payload[phase] = {
                "K": list(schedule.amplitudes),
                "S": list(schedule.stability) if schedule.stability is not None else [],
            }
        return payload


@dataclass(slots=True, frozen=True)
class BasicVLSResult:
    search_result: VariableLandscapeSearchResult
    configuration: BasicVLSPhaseConfiguration

    @property
    def best_record(self) -> LandscapeEvaluationRecord:
        return self.search_result.best_record

    @property
    def best_landscape(self) -> Landscape:
        return self.best_record.landscape

    @property
    def best_telemetry(self) -> EvaluationTelemetry:
        return self.best_record.telemetry

    @property
    def history(self) -> tuple[LandscapeEvaluationRecord, ...]:
        return self.search_result.history

    @property
    def budget_consumed(self) -> float:
        return self.search_result.budget_consumed

    @property
    def termination_reason(self) -> str:
        return self.search_result.termination_reason

    @property
    def elapsed_seconds(self) -> float:
        return self.search_result.elapsed_seconds


class BasicVLS(MASOptimizer):
    def __init__(
        self,
        *,
        judge: UnsupervisedJudge,
        meta_space: LandscapeMetaSpace,
        query: str,
        budget: float,
        time_limit: float | None,
        pipeline_builder_factory: PipelineBuilderFactory | None = None,
        configuration: BasicVLSPhaseConfiguration | None = None,
        initial_landscape: Landscape | None = None,
        translator: Translator | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._judge = judge
        self._meta_space = meta_space
        self._query = query
        self._budget = float(budget)
        self._time_limit = time_limit
        self._pipeline_builder_factory = pipeline_builder_factory or PipelineBuilder
        self._configuration = configuration or BasicVLSPhaseConfiguration.default()
        self._initial_landscape = (
            initial_landscape.clone() if initial_landscape is not None else None
        )
        self._translator = translator or self._default_translator
        self._history: list[LandscapeEvaluationRecord] = []
        self._last_result: BasicVLSResult | None = None
        self._time_source = time_source or perf_counter
        self._max_retries = 2

    async def optimize(
        self,
        initial_graph: GraphDict,
        agent_pool: AgentPool,
        judge: Judge,
        **kwargs: Any,
    ) -> tuple[GraphDict, float, list[dict[str, Any]]]:
        if not isinstance(judge, UnsupervisedJudge):
            raise TypeError("BasicVLS requires an UnsupervisedJudge instance")
        history_reporter = cast(
            Callable[[Iterable[LandscapeEvaluationRecord]], Sequence[Any]] | None,
            kwargs.get("history_reporter"),
        )
        meta_space = cast(LandscapeMetaSpace | None, kwargs.get("meta_space", self._meta_space))
        if meta_space is None:
            raise ValueError("meta_space must be provided")
        query = cast(str | None, kwargs.get("query", self._query))
        if query is None:
            raise ValueError("query must be provided")
        budget = float(kwargs.get("budget", self._budget))
        if budget <= 0:
            raise ValueError("budget must be positive")
        time_limit = cast(float | None, kwargs.get("time_limit", self._time_limit))
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive or None")
        landscape_override = cast(
            Landscape | None, kwargs.get("initial_landscape", self._initial_landscape)
        )
        if landscape_override is None:
            raise ValueError("initial_landscape must be provided")
        start_landscape = landscape_override.with_pool(agent_pool)
        graph_payload = {
            str(node): [str(child) for child in cast(Sequence[str], children)]
            for node, children in initial_graph.items()
        }
        metadata = dict(start_landscape.metadata)
        metadata["graph"] = graph_payload
        start_landscape = start_landscape.with_metadata(metadata)
        self._judge = judge
        self._meta_space = meta_space
        self._query = query
        self._budget = budget
        self._time_limit = time_limit
        self._initial_landscape = start_landscape.clone()
        LOGGER.info("Launching BVLS query reason=start")
        search = VariableLandscapeSearch(
            meta_space=self._meta_space,
            query=self._query,
            judge=self._judge,
            pipeline_builder_factory=self._pipeline_builder_factory,
            phase_schedules=self._configuration.schedules,
            translator=self._translator,
        )
        search_result = await self._run_search_with_retry(
            search,
            start_landscape,
            budget=self._budget,
            time_limit=self._resolve_time_limit(),
        )
        result = BasicVLSResult(search_result=search_result, configuration=self._configuration)
        self._history.extend(result.history)
        LOGGER.info(
            f"BVLS finished termination={result.termination_reason} budget={result.budget_consumed:.4f} "
            f"elapsed={result.elapsed_seconds:.4f} reason=complete"
        )
        self._last_result = result
        best_graph = self._extract_graph(result.best_landscape.metadata.get("graph"), initial_graph)
        history_payload = (
            list(history_reporter(result.history))
            if history_reporter is not None
            else list(result.history)
        )
        best_score = result.best_record.telemetry.quality.value
        return best_graph, best_score, history_payload

    @property
    def evaluation_history(self) -> tuple[LandscapeEvaluationRecord, ...]:
        return tuple(self._history)

    @property
    def last_result(self) -> BasicVLSResult | None:
        return self._last_result

    @property
    def configuration(self) -> BasicVLSPhaseConfiguration:
        return self._configuration

    async def run(
        self,
        *,
        initial_landscape: Landscape | None = None,
        budget: float | None = None,
        time_limit: float | None = None,
    ) -> BasicVLSResult:
        start_landscape = (initial_landscape or self._initial_landscape).clone() if (
            initial_landscape or self._initial_landscape
        ) else None
        if start_landscape is None:
            raise ValueError("initial_landscape must be provided")
        resolved_budget = float(budget if budget is not None else self._budget)
        if resolved_budget <= 0:
            raise ValueError("budget must be positive")
        limit = time_limit if time_limit is not None else self._time_limit
        if limit is not None and limit <= 0:
            raise ValueError("time_limit must be positive or None")
        LOGGER.info(
            f"Starting BVLS run from={start_landscape.identifier} budget={resolved_budget:.4f} reason=run-start"
        )
        search = VariableLandscapeSearch(
            meta_space=self._meta_space,
            query=self._query,
            judge=self._judge,
            pipeline_builder_factory=self._pipeline_builder_factory,
            phase_schedules=self._configuration.schedules,
            translator=self._translator,
        )
        search_result = await self._run_search_with_retry(
            search,
            start_landscape,
            budget=resolved_budget,
            time_limit=limit,
        )
        result = BasicVLSResult(search_result=search_result, configuration=self._configuration)
        self._history.extend(result.history)
        LOGGER.info(
            f"BVLS run complete termination={result.termination_reason} budget={result.budget_consumed:.4f} reason=run-complete"
        )
        self._last_result = result
        return result

    async def _run_search_with_retry(
        self,
        search: VariableLandscapeSearch,
        start_landscape: Landscape,
        *,
        budget: float,
        time_limit: float | None,
    ) -> VariableLandscapeSearchResult:
        attempts = 0
        start_time = self._time_source()
        while True:
            elapsed = self._time_source() - start_time
            remaining = None if time_limit is None else time_limit - elapsed
            if remaining is not None and remaining <= 0:
                LOGGER.error(
                    f"Time limit exhausted before search completion elapsed={elapsed:.4f} reason=time-exhausted"
                )
                raise TimeoutError("Time limit exhausted before completing search")
            try:
                return await search.run(start_landscape, budget=budget, time_limit=remaining)
            except Exception as exc:
                attempts += 1
                LOGGER.warning(
                    f"Search attempt failed attempt={attempts} error={exc} reason=pipeline-error"
                )
                if attempts > self._max_retries:
                    LOGGER.error(
                        f"Exhausted search retries attempts={attempts} reason=retry-exhausted"
                    )
                    raise

    def _resolve_time_limit(self) -> float | None:
        if self._time_limit is None:
            return None
        return float(self._time_limit)

    @staticmethod
    def _extract_graph(
        entry: Mapping[str, Sequence[str]] | GraphDict | None,
        fallback: GraphDict,
    ) -> GraphDict:
        if entry is None:
            return deepcopy(fallback)
        if isinstance(entry, Mapping):
            return {
                str(node): [str(child) for child in cast(Sequence[str], children)]
                for node, children in dict(entry).items()
            }
        return deepcopy(cast(GraphDict, entry))

    @staticmethod
    def _default_translator(origin: Landscape, candidate: Landscape) -> Landscape:
        if "graph" in candidate.metadata:
            return candidate.clone()
        metadata = dict(candidate.metadata)
        graph_entry = origin.metadata.get("graph")
        if graph_entry is not None:
            metadata["graph"] = graph_entry
        return candidate.with_metadata(metadata)
