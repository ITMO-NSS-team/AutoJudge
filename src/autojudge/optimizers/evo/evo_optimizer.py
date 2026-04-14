import asyncio
import copy
import os
import random
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from autojudge import ALL_AGENTS_POOL, AgentPool
from autojudge.judge import Judge
from autojudge.meta_agents.base import DEFAULT_MODEL
from autojudge.optimizers.base import MASOptimizer
from autojudge.optimizers.evo.types import GenerationStats
from autojudge.pipeline.pipeline_builder import PipelineBuilder
from autojudge.pipeline.types import GraphDict
from autojudge.utils.logger import get_logger

from .crossover_agent import CrossoverAgent
from .mutation_agent import MutationAgent
from ...utils.langfuse_utils import ainvoke_with_lf

load_dotenv()

logger = get_logger()


class LLMEvoOptimizer(MASOptimizer):
    """LLM-based evolutionary optimizer."""

    def __init__(
        self,
        agents_pool: AgentPool,
        mutation_model: str = DEFAULT_MODEL,
        crossover_model: str = DEFAULT_MODEL,
        mutation_rate: float = 1.0,
        crossover_rate: float = 1.0,
    ):
        self.api_key = os.getenv("OPENROUTER_API_KEY") or ""
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.extended_agents_pool = agents_pool + ALL_AGENTS_POOL
        self.mutation_agent = MutationAgent(
            model=mutation_model, agents_pool=self.extended_agents_pool
        )
        self.crossover_agent = CrossoverAgent(model=crossover_model)

        # Track pipeline execution costs during evaluation
        self._pipeline_input_tokens = 0
        self._pipeline_output_tokens = 0
        self._pipeline_total_cost = 0.0

        self.history: List[GenerationStats] = []

    @property
    def input_tokens(self) -> int:
        """Total input tokens from both mutation and crossover agents."""
        return self.mutation_agent.input_tokens + self.crossover_agent.input_tokens

    @property
    def output_tokens(self) -> int:
        """Total output tokens from both mutation and crossover agents."""
        return self.mutation_agent.output_tokens + self.crossover_agent.output_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens from both agents."""
        return self.input_tokens + self.output_tokens

    @property
    def cost(self):
        """Aggregate cost from both agents."""
        from autojudge.pipeline.types import CostBreakdown

        mutation_cost = self.mutation_agent.cost
        crossover_cost = self.crossover_agent.cost

        return CostBreakdown(
            input_price=mutation_cost.input_price + crossover_cost.input_price,
            output_price=mutation_cost.output_price + crossover_cost.output_price,
            total_price=mutation_cost.total_price + crossover_cost.total_price,
        )

    @property
    def mutation_tokens(self) -> dict:
        """Tokens used by mutation agent."""
        return {
            "input_tokens": self.mutation_agent.input_tokens,
            "output_tokens": self.mutation_agent.output_tokens,
            "total_tokens": self.mutation_agent.total_tokens,
        }

    @property
    def mutation_cost(self):
        """Cost of mutation agent."""
        return self.mutation_agent.cost

    @property
    def crossover_tokens(self) -> dict:
        """Tokens used by crossover agent."""
        return {
            "input_tokens": self.crossover_agent.input_tokens,
            "output_tokens": self.crossover_agent.output_tokens,
            "total_tokens": self.crossover_agent.total_tokens,
        }

    @property
    def crossover_cost(self):
        """Cost of crossover agent."""
        return self.crossover_agent.cost

    @property
    def pipeline_tokens(self) -> dict:
        """Total tokens used by pipeline executions during evaluation."""
        return {
            "input_tokens": self._pipeline_input_tokens,
            "output_tokens": self._pipeline_output_tokens,
            "total_tokens": self._pipeline_input_tokens + self._pipeline_output_tokens,
        }

    @property
    def pipeline_cost(self):
        """Total cost of pipeline executions during evaluation."""
        from autojudge.pipeline.types import CostBreakdown

        return CostBreakdown(
            input_price=0.0,  # Not tracking input/output separately for pipelines
            output_price=0.0,
            total_price=self._pipeline_total_cost,
        )

    async def _initialize_population(
        self,
        initial_graph: GraphDict,
        pop_size: int,
        task_description: str,
        available_agents: AgentPool,
    ) -> List[GraphDict]:
        """Creates initial population."""
        population = [copy.deepcopy(initial_graph)]

        tasks = [
            self.mutation_agent.perform_mutation(initial_graph, task_description, available_agents)  # type: ignore
            for _ in range(pop_size - 1)
        ]

        mutated_graphs = await asyncio.gather(*tasks, return_exceptions=True)
        for i, g in enumerate(mutated_graphs):
            if isinstance(g, Exception):
                logger.error(f"Initialization error {i}: {g}", exc_info=g)
                g = copy.deepcopy(initial_graph)
            population.append(g)
            logger.info(f"Initialization: created graph {i + 1}/{pop_size - 1}")

        return population

    def _select_parents(
        self,
        population: List[GraphDict],
        scores: List[float],
        num_parents: int = 2,
    ) -> List[Tuple[GraphDict, float]]:
        """Tournament selection of parents."""
        parents = []
        tournament_size = 3

        for _ in range(num_parents):
            indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            sub_scores = [scores[i] for i in indices]
            winner_idx = indices[sub_scores.index(max(sub_scores))]
            parents.append((copy.deepcopy(population[winner_idx]), scores[winner_idx]))

        return parents

    async def _evaluate_graph(
        self,
        graph: GraphDict,
        agent_pool: AgentPool,
        judge: Judge,
        task_description: str,
    ) -> Tuple[float, Optional[str]]:
        """Evaluate a graph by building and running a pipeline, then judging the trace."""
        try:
            # Build pipeline from graph
            builder = PipelineBuilder()
            builder.create_from_pool(agent_pool, graph)
            pipeline = builder.build()

            # Execute pipeline
            response, trace_id = await ainvoke_with_lf(agent_pool, pipeline, task_description, graph)
            logger.info(f"Pipeline response: {response}")

            # Track pipeline costs
            self._pipeline_input_tokens += pipeline.input_tokens
            self._pipeline_output_tokens += pipeline.output_tokens
            self._pipeline_total_cost += pipeline.cost.total_price
            logger.debug(
                f"Pipeline evaluation cost: ${pipeline.cost.total_price:.6f} "
                f"({pipeline.total_tokens} tokens)"
            )

            # Get trace from pipeline
            trace = pipeline.trace
            if trace is None:
                logger.error(f"Pipeline execution did not produce a trace for graph: {graph}")
                return float("-inf"), None

            trace.session_id = trace_id

            # Evaluate trace with judge
            result = await judge.evaluate(trace)

            # Return numeric score and response
            logger.debug(f"Pipeline evaluated: {result.numeric_score}")
            return result.numeric_score, response
        except Exception as e:
            logger.error(f"Error evaluating graph: {e}", exc_info=True)
            return float("-inf"), None

    async def _evaluate_population(
        self,
        population: List[GraphDict],
        agent_pool: AgentPool,
        judge: Judge,
        task_description: str,
    ) -> Tuple[List[float], List[Optional[str]]]:
        """Evaluate all graphs in the current population asynchronously."""
        eval_tasks = [
            self._evaluate_graph(graph, agent_pool, judge, task_description) for graph in population
        ]
        results = await asyncio.gather(*eval_tasks)
        scores, responses = zip(*results)

        for i, score in enumerate(scores):
            if score == float("-inf"):
                logger.error(f"  Graph {i + 1}: evaluation failed")
            else:
                logger.info(f"  Graph {i + 1}: {score:.4f}")

        return list(scores), list(responses)

    def _compute_generation_stats(
        self,
        generation: int,
        population: List[GraphDict],
        scores: List[float],
        responses: List[Optional[str]],
    ) -> GenerationStats:
        """Compute best and average scores for the generation."""
        best_idx = scores.index(max(scores))
        valid_scores = [s for s in scores if s != float("-inf")]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        logger.info(f"Best score: {scores[best_idx]:.4f}, Average: {avg_score:.4f}")

        return GenerationStats(
            generation=generation,
            best_score=scores[best_idx],
            avg_score=avg_score,
            best_graph=copy.deepcopy(population[best_idx]),
            best_response=responses[best_idx],
        )

    def _apply_elitism(
        self,
        population: List[GraphDict],
        scores: List[float],
        elitism: int,
    ) -> List[GraphDict]:
        """Select top individuals to carry forward."""
        elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elitism]
        elites = [copy.deepcopy(population[i]) for i in elite_indices]
        logger.debug(f"Selected {len(elites)} elites for next generation")
        return elites

    async def _generate_offspring(
        self,
        population: List[GraphDict],
        scores: List[float],
        target_size: int,
        task_description: str,
        agent_pool: AgentPool,
    ) -> List[GraphDict]:
        """Fill the next generation via crossover/mutation/copy."""
        new_population = []
        while len(new_population) < target_size:
            op = random.random()
            if op < self.crossover_rate:
                parents = self._select_parents(population, scores, 2)
                child = await self.crossover_agent.perform_crossover(
                    parent1=parents[0][0],
                    parent2=parents[1][0],
                    task_description=task_description,
                    score1=parents[0][1],
                    score2=parents[1][1],
                )
                new_population.append(child)
            elif op < self.crossover_rate + self.mutation_rate:
                parents = self._select_parents(population, scores, 1)
                mutated = await self.mutation_agent.perform_mutation(
                    parents[0][0],
                    task_description,
                    parents[0][1],
                )
                new_population.append(mutated)
            else:
                parents = self._select_parents(population, scores, 1)
                new_population.append(copy.deepcopy(parents[0][0]))

        return new_population[:target_size]

    async def _run_generation(
        self,
        generation: int,
        population: List[GraphDict],
        agent_pool: AgentPool,
        judge: Judge,
        task_description: str,
        elitism: int,
        pop_size: int,
    ) -> Tuple[List[GraphDict], GenerationStats]:
        """Run one generation of evaluation and evolution."""
        scores, responses = await self._evaluate_population(
            population, agent_pool, judge, task_description
        )
        gen_stats = self._compute_generation_stats(generation, population, scores, responses)

        # Early stop if perfect score reached
        if gen_stats.best_score >= 1.0:
            logger.info(f"Perfect score reached in generation {generation}")
            return population, gen_stats

        # Build next generation
        elites = self._apply_elitism(population, scores, elitism)
        offspring = await self._generate_offspring(
            population, scores, pop_size - len(elites), task_description, agent_pool
        )
        next_population = elites + offspring
        return next_population, gen_stats

    async def optimize(
        self,
        initial_graph: GraphDict,
        agent_pool: AgentPool,
        judge: Judge,
        task_description: str,
        pop_size: int = 3,
        num_generations: int = 3,
        elitism: int = 2,
    ) -> Tuple[GraphDict, float, List[GenerationStats]]:
        """Asynchronous evolutionary optimization with LLM."""
        # Evaluate initial graph
        logger.info("Evaluating initial graph")
        initial_score, initial_response = await self._evaluate_graph(
            initial_graph, agent_pool, judge, task_description
        )
        initial_stats = GenerationStats(
            generation=0,
            best_score=initial_score,
            avg_score=initial_score,
            best_graph=copy.deepcopy(initial_graph),
            best_response=initial_response,
        )
        self.history.append(initial_stats)

        if initial_score >= 1.0:
            logger.info("Initial graph already perfect. Exiting early.")
            return initial_graph, initial_score, self.history

        population = await self._initialize_population(
            initial_graph, pop_size, task_description, self.extended_agents_pool
        )

        for gen in range(1, num_generations + 1):
            population, gen_stats = await self._run_generation(
                gen,
                population,
                self.extended_agents_pool,
                judge,
                task_description,
                elitism,
                pop_size,
            )
            self.history.append(gen_stats)

            # Log cumulative costs after each generation
            logger.info(
                f"Generation {gen} - Cumulative costs: "
                f"Optimizer=${self.cost.total_price:.6f}, "
                f"Pipelines=${self._pipeline_total_cost:.6f}, "
                f"Total=${self.cost.total_price + self._pipeline_total_cost:.6f}"
            )

            if gen_stats.best_score >= 1.0:
                return gen_stats.best_graph, gen_stats.best_score, self.history

        best_generation = max(self.history, key=lambda x: x.best_score)

        # Log final costs
        logger.info("Evolution complete - Final costs:")
        logger.info(f"  Optimizer (Mutation + Crossover): ${self.cost.total_price:.6f}")
        logger.info(f"  Pipeline Evaluations: ${self._pipeline_total_cost:.6f}")
        logger.info(f"  Grand Total: ${self.cost.total_price + self._pipeline_total_cost:.6f}")

        return best_generation.best_graph, best_generation.best_score, self.history
