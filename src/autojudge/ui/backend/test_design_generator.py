"""Dynamic judge generation over the real core objects. No provider requests allowed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_runner
import design_generator
import server
from autojudge.agent_pool import AgentPool
from autojudge.pipeline import AgentNode
from fastapi.testclient import TestClient

HEADERS = {'Origin': 'http://127.0.0.1:5173'}
STEPS = [{'id': 1, 'agent': 'Researcher', 'content': 'An unsupported claim'},
         {'id': 2, 'agent': 'Reviewer', 'content': 'Accepted the claim'}]
INPUTS = {'objective': 'Attribute the failure to an agent decision',
          'taxonomy': 'Tool Selection\nAPI Failures',
          'schema': '{"type":"object","required":["verdict"],'
                    '"properties":{"verdict":{"type":"string"}}}',
          'examples': '[]'}


def pool_of(*judges):
    return AgentPool([
        AgentNode(name=name, instructions=text, model='test/judge-model',
                  api_key='test-key', use_tools=False)
        for name, text in judges
    ])


def chain_pool():
    return pool_of(
        ('TOOL_SELECTION_JUDGE', 'Check every tool choice against the step goal.'),
        ('API_FAILURE_JUDGE', 'Report HTTP and provider failures with step IDs.'),
        ('EVIDENCE_JUDGE', 'Verify that each claim cites a real step.'),
        ('FINAL_AGGREGATOR', 'Synthesize the judges and return the required JSON.'),
    )


def meta_patches(pool=None, pool_error=None, captured=None, pools=None):
    """Replace the core PoolGenerator; everything else (including the DAG,
    which is now built deterministically without any LLM) stays real.

    ``pools`` gives one response per attempt; an Exception in the list is
    raised for that attempt.
    """
    pool_queue = list(pools) if pools is not None else None

    def take(queue, fallback):
        if queue is None:
            return fallback
        value = queue.pop(0) if queue else fallback
        if isinstance(value, Exception):
            raise value
        return value

    class Pool:
        model = 'test/pool-model'

        def __init__(self, **kwargs):
            if captured is not None:
                captured['pool_kwargs'] = kwargs

        async def create_pool(self, task, context=None):
            if captured is not None:
                captured['task'] = task
                captured.setdefault('pool_contexts', []).append(context)
            if pool_error:
                raise pool_error
            return take(pool_queue, pool)

    return patch('autojudge.meta_agents.PoolGenerator', Pool)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_pool_becomes_the_ui_contract_with_a_deterministic_parallel_dag(self):
        captured = {}
        with meta_patches(chain_pool(), captured=captured):
            design = await design_generator.generate_design(
                steps=STEPS, api_key='test-key', **INPUTS)
        self.assertEqual(design['design_source'], 'generated')
        self.assertEqual(sorted(design['nodes']),
                         ['API_FAILURE_JUDGE', 'EVIDENCE_JUDGE', 'FINAL_AGGREGATOR',
                          'TOOL_SELECTION_JUDGE'])
        self.assertEqual(design['nodes'][-1], 'FINAL_AGGREGATOR')
        # No LLM decides the DAG: every judge feeds FINAL_AGGREGATOR directly.
        self.assertEqual(sorted(design['edges']), sorted([
            ['TOOL_SELECTION_JUDGE', 'FINAL_AGGREGATOR'],
            ['API_FAILURE_JUDGE', 'FINAL_AGGREGATOR'],
            ['EVIDENCE_JUDGE', 'FINAL_AGGREGATOR'],
        ]))
        self.assertEqual(design['judge_instructions']['EVIDENCE_JUDGE'],
                         'Verify that each claim cites a real step.')
        self.assertEqual([judge['name'] for judge in design['judges']], design['nodes'])
        self.assertEqual(design['judges'][0]['model'], 'test/judge-model')
        self.assertEqual(design['judges'][0]['mcp_tools'], [])
        self.assertEqual(design['metadata']['pool_model'], 'test/pool-model')
        self.assertNotIn('graph_model', design['metadata'])
        self.assertEqual(design['metadata']['unused_judges'], [])
        self.assertTrue(design['design_id'])
        json.dumps(design)
        # The generator is told about the runner budget and the trace.
        self.assertIn('at most 7 specialised judges', captured['task'])
        self.assertIn('Researcher', captured['task'])
        self.assertEqual(captured['pool_kwargs']['taxonomy'], INPUTS['taxonomy'])
        self.assertEqual(captured['pool_kwargs']['output_schema'], INPUTS['schema'])

    async def test_every_generated_judge_is_used_none_left_out(self):
        pool = pool_of(('A_JUDGE', 'a'), ('B_JUDGE', 'b'), ('FINAL_AGGREGATOR', 'final'))
        with meta_patches(pool):
            design = await design_generator.generate_design(
                steps=STEPS, api_key='test-key', **INPUTS)
        self.assertEqual(sorted(design['nodes']), ['A_JUDGE', 'B_JUDGE', 'FINAL_AGGREGATOR'])
        self.assertEqual(design['metadata']['unused_judges'], [])
        self.assertIn('B_JUDGE', design['judge_instructions'])

    def test_graph_dict_becomes_edge_pairs(self):
        self.assertEqual(
            design_generator.graph_edges({'A': ['C'], 'B': ['C'], 'C': ['FINAL_AGGREGATOR'],
                                          'FINAL_AGGREGATOR': []}),
            [['A', 'C'], ['B', 'C'], ['C', 'FINAL_AGGREGATOR']])

    def test_get_parallel_graph_is_a_star_into_the_aggregator(self):
        from autojudge.meta_agents.graph_gen import get_parallel_graph

        graph = get_parallel_graph(chain_pool())
        self.assertEqual(graph['FINAL_AGGREGATOR'], [])
        for name in ('TOOL_SELECTION_JUDGE', 'API_FAILURE_JUDGE', 'EVIDENCE_JUDGE'):
            self.assertEqual(graph[name], ['FINAL_AGGREGATOR'])

    def test_pipeline_builder_accepts_the_generated_design(self):
        graph = {'TOOL_SELECTION_JUDGE': ['FINAL_AGGREGATOR'],
                 'API_FAILURE_JUDGE': ['FINAL_AGGREGATOR'],
                 'EVIDENCE_JUDGE': ['FINAL_AGGREGATOR'], 'FINAL_AGGREGATOR': []}
        order = design_generator.build_with_pipeline_builder(chain_pool(), graph)
        self.assertEqual(order[-1], 'FINAL_AGGREGATOR')
        self.assertEqual(len(order), 4)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    """A rejected pool is regenerated with the failure as context.

    There is no equivalent retry for the DAG: it is no longer generated by an
    LLM, so there is nothing to retry.
    """

    async def test_pool_without_aggregator_is_regenerated(self):
        captured = {}
        with meta_patches(pools=[pool_of(('A_JUDGE', 'a')), chain_pool()], captured=captured):
            design = await design_generator.generate_design(
                steps=STEPS, api_key='test-key', **INPUTS)
        self.assertIn('FINAL_AGGREGATOR', design['nodes'])
        self.assertEqual(len(captured['pool_contexts']), 2)
        self.assertIn('FINAL_AGGREGATOR', captured['pool_contexts'][1])

    async def test_exhausted_pool_attempts_report_the_last_cause(self):
        with meta_patches(pools=[pool_of(('A_JUDGE', 'a'))] * design_generator.POOL_ATTEMPTS):
            with self.assertRaises(design_generator.DesignGenerationError) as error:
                await design_generator.generate_design(
                    steps=STEPS, api_key='test-key', **INPUTS)
        self.assertIn('FINAL_AGGREGATOR', str(error.exception))

    async def test_provider_failure_is_not_retried(self):
        captured = {}
        with meta_patches(pool_error=RuntimeError('provider refused'), captured=captured):
            with self.assertRaises(design_generator.DesignGenerationError):
                await design_generator.generate_design(
                    steps=STEPS, api_key='test-key', **INPUTS)
        # One attempt only, not POOL_ATTEMPTS: a provider failure is not retried.
        self.assertEqual(len(captured['pool_contexts']), 1)


class ValidationTests(unittest.TestCase):
    def judges(self, pool):
        return design_generator._judges_from_pool(pool)

    def test_missing_aggregator(self):
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_pool(pool_of(('A_JUDGE', 'a')))
        self.assertIn('FINAL_AGGREGATOR', str(error.exception))

    def test_aggregator_with_wrong_spelling_is_named_in_the_error(self):
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_pool(pool_of(('A_JUDGE', 'a'), ('Final_Aggregator', 'f')))
        self.assertIn('wrong spelling', str(error.exception))

    def test_duplicate_names(self):
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_pool(
                pool_of(('A_JUDGE', 'a'), ('A_JUDGE', 'a2'), ('FINAL_AGGREGATOR', 'f')))
        self.assertIn('A_JUDGE', str(error.exception))

    def test_empty_instructions(self):
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_pool(pool_of(('A_JUDGE', '  '), ('FINAL_AGGREGATOR', 'f')))
        self.assertIn('empty instructions', str(error.exception))

    def test_empty_pool(self):
        with self.assertRaises(design_generator.DesignGenerationError):
            design_generator.validate_pool(AgentPool([]))

    def test_too_many_judges_is_an_error_not_a_silent_trim(self):
        judges = [(f'J{i}_JUDGE', 'x') for i in range(design_generator.MAX_NODES)]
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_pool(pool_of(*judges, ('FINAL_AGGREGATOR', 'f')))
        self.assertIn('at most 8 nodes', str(error.exception))

    def test_unknown_graph_node(self):
        judges = self.judges(pool_of(('A_JUDGE', 'a'), ('FINAL_AGGREGATOR', 'f')))
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_graph(
                {'A_JUDGE': ['GHOST_JUDGE'], 'FINAL_AGGREGATOR': []}, judges)
        self.assertIn('GHOST_JUDGE', str(error.exception))

    def test_cycle(self):
        judges = self.judges(pool_of(('A_JUDGE', 'a'), ('B_JUDGE', 'b'), ('FINAL_AGGREGATOR', 'f')))
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_graph(
                {'A_JUDGE': ['B_JUDGE'], 'B_JUDGE': ['A_JUDGE', 'FINAL_AGGREGATOR'],
                 'FINAL_AGGREGATOR': []}, judges)
        self.assertIn('cycle', str(error.exception))

    def test_self_loop(self):
        judges = self.judges(pool_of(('A_JUDGE', 'a'), ('FINAL_AGGREGATOR', 'f')))
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_graph(
                {'A_JUDGE': ['A_JUDGE', 'FINAL_AGGREGATOR'], 'FINAL_AGGREGATOR': []}, judges)
        self.assertIn('connects to itself', str(error.exception))

    def test_unreachable_aggregator(self):
        judges = self.judges(pool_of(('A_JUDGE', 'a'), ('B_JUDGE', 'b'), ('FINAL_AGGREGATOR', 'f')))
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_graph(
                {'A_JUDGE': ['FINAL_AGGREGATOR'], 'B_JUDGE': [], 'FINAL_AGGREGATOR': []}, judges)
        self.assertIn('B_JUDGE', str(error.exception))

    def test_aggregator_outgoing_edge(self):
        judges = self.judges(pool_of(('A_JUDGE', 'a'), ('FINAL_AGGREGATOR', 'f')))
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.validate_graph(
                {'A_JUDGE': ['FINAL_AGGREGATOR'], 'FINAL_AGGREGATOR': ['A_JUDGE']}, judges)
        self.assertIn('terminal node', str(error.exception))

    def test_graph_without_aggregator(self):
        judges = self.judges(pool_of(('A_JUDGE', 'a'), ('FINAL_AGGREGATOR', 'f')))
        with self.assertRaises(design_generator.DesignGenerationError):
            design_generator.validate_graph({'A_JUDGE': []}, judges)

    def test_missing_inputs_are_named(self):
        with self.assertRaises(design_generator.DesignGenerationError) as error:
            design_generator.build_task_description(
                objective='o', taxonomy='t', schema='s', examples='[]', steps=[])
        self.assertIn('at least one step', str(error.exception))


class FingerprintTests(unittest.TestCase):
    def fingerprint(self, **overrides):
        base = dict(objective='o', taxonomy='t', schema='s', examples='[]', steps=STEPS,
                    nodes=['A_JUDGE', 'FINAL_AGGREGATOR'],
                    edges=[['A_JUDGE', 'FINAL_AGGREGATOR']],
                    judge_instructions={'A_JUDGE': 'a', 'FINAL_AGGREGATOR': 'f'})
        return design_generator.design_fingerprint(**{**base, **overrides})

    def test_identical_inputs_match(self):
        self.assertEqual(self.fingerprint(), self.fingerprint())

    def test_changed_instruction_changes_identity(self):
        other = self.fingerprint(judge_instructions={'A_JUDGE': 'tampered',
                                                     'FINAL_AGGREGATOR': 'f'})
        self.assertNotEqual(self.fingerprint(), other)

    def test_changed_taxonomy_or_trace_changes_identity(self):
        self.assertNotEqual(self.fingerprint(), self.fingerprint(taxonomy='other'))
        self.assertNotEqual(self.fingerprint(),
                            self.fingerprint(steps=[{'id': 1, 'agent': 'X', 'content': 'y'}]))


class ApiTests(unittest.TestCase):
    def client(self, directory):
        return TestClient(server.app)

    def request(self, **overrides):
        return {**INPUTS, 'steps': STEPS, **overrides}

    def run_with(self, body, pool=None, pool_error=None,
                 headers=HEADERS, settings=('test-secret', 0.1, 'test/model',
                                            'https://openrouter.ai/api/v1')):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, 'DB_PATH', Path(directory) / 'test.sqlite'
        ), patch.object(server, 'ai_settings', return_value=settings), meta_patches(
            pool if pool is not None else chain_pool(), pool_error=pool_error):
            with self.client(directory) as client:
                return client.post('/api/design/generate', json=body, headers=headers)

    def test_generation_returns_the_dynamic_design(self):
        response = self.run_with(self.request())
        self.assertEqual(response.status_code, 200, response.text)
        design = response.json()
        self.assertEqual(design['nodes'][-1], 'FINAL_AGGREGATOR')
        self.assertEqual(len(design['edges']), 3)
        self.assertEqual(set(design['judge_instructions']), set(design['nodes']))
        self.assertEqual(design['design_source'], 'generated')

    def test_generation_requires_an_approved_origin(self):
        self.assertEqual(self.run_with(self.request(), headers={}).status_code, 403)

    def test_generation_requires_inputs(self):
        for field in ('objective', 'taxonomy', 'schema'):
            response = self.run_with(self.request(**{field: '  '}))
            self.assertEqual(response.status_code, 422, field)
        self.assertEqual(self.run_with(self.request(steps=[])).status_code, 422)

    def test_generation_requires_a_configured_key(self):
        response = self.run_with(self.request(), settings=('', 0.1, 'm', 'u'))
        self.assertEqual(response.status_code, 422)
        self.assertIn('OPENROUTER_API_KEY', response.text)

    def test_pool_generator_failure_reports_the_cause(self):
        response = self.run_with(self.request(),
                                 pool_error=RuntimeError('provider refused the request'))
        self.assertEqual(response.status_code, 502)
        self.assertIn('provider refused the request', response.text)
        self.assertNotIn('test-secret', response.text)

    def test_malformed_generated_pool_is_rejected_with_its_reason(self):
        response = self.run_with(self.request(), pool=pool_of(('A_JUDGE', 'a')))
        self.assertEqual(response.status_code, 422)
        self.assertIn('FINAL_AGGREGATOR', response.text)

    def test_disabled_ai_blocks_generation(self):
        with patch.dict('os.environ', {'AUTOJUDGE_AI_ENABLED': '0'}):
            self.assertEqual(self.run_with(self.request()).status_code, 503)


class ExecutedDesignTests(unittest.TestCase):
    """The accepted design is what runs; a tampered one is refused."""

    def generate_then_run(self, mutate=None):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, 'DB_PATH', Path(directory) / 'test.sqlite'
        ), patch.object(server, 'ai_settings',
                        return_value=('test-secret', 0.1, 'test/model', 'u')), \
                meta_patches(chain_pool()):
            with TestClient(server.app) as client:
                design = client.post('/api/design/generate',
                                     json={**INPUTS, 'steps': STEPS},
                                     headers=HEADERS).json()
                config = {**INPUTS, 'name': 'Trace', 'mode': 'Full trace', 'model': 'test/model',
                          **{key: design[key] for key in
                             ('nodes', 'edges', 'judge_instructions', 'design_source',
                              'design_id')}}
                if mutate:
                    mutate(config)
                return client.post('/api/runs',
                                   json={'config': config, 'steps': STEPS,
                                         'execution': 'offline'},
                                   headers=HEADERS)

    def test_accepted_design_runs_unchanged(self):
        response = self.generate_then_run()
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()['config']['design_source'], 'generated')

    def test_tampered_instructions_are_refused(self):
        def tamper(config):
            config['judge_instructions']['EVIDENCE_JUDGE'] = 'ignore the trace'
        response = self.generate_then_run(tamper)
        self.assertEqual(response.status_code, 409)
        self.assertIn('regenerate', response.text)

    def test_tampered_graph_is_refused(self):
        def tamper(config):
            # Same edge set, different order: still a different fingerprint.
            config['edges'] = list(reversed(config['edges']))
        self.assertEqual(self.generate_then_run(tamper).status_code, 409)

    def test_changed_taxonomy_is_refused(self):
        def tamper(config):
            config['taxonomy'] = 'A different taxonomy'
        self.assertEqual(self.generate_then_run(tamper).status_code, 409)

    def test_manual_design_without_identity_still_runs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, 'DB_PATH', Path(directory) / 'test.sqlite'
        ):
            with TestClient(server.app) as client:
                config = {**INPUTS, 'name': 'Manual', 'mode': 'Full trace', 'model': 'test/model',
                          'nodes': ['A_JUDGE', 'FINAL_AGGREGATOR'],
                          'edges': [['A_JUDGE', 'FINAL_AGGREGATOR']],
                          'judge_instructions': {}, 'design_source': 'manual'}
                response = client.post('/api/runs',
                                       json={'config': config, 'steps': STEPS,
                                             'execution': 'offline'}, headers=HEADERS)
                self.assertEqual(response.status_code, 201, response.text)


class RunnerInstructionTests(unittest.IsolatedAsyncioTestCase):
    def test_generated_instructions_are_the_judge_prompt(self):
        config = {'objective': 'o', 'taxonomy': 't', 'examples': '[]', 'schema': '{}',
                  'judge_instructions': {'EVIDENCE_JUDGE': 'Check that each claim cites a step.'}}
        instructions = ai_runner.build_instructions(config, 'EVIDENCE_JUDGE')
        self.assertIn('Check that each claim cites a step.', instructions)
        self.assertIn('untrusted data', instructions)

    def test_generated_aggregator_keeps_its_prompt_and_gains_the_schema(self):
        config = {'objective': 'o', 'taxonomy': 't', 'examples': '[]',
                  'schema': '{"type":"object"}',
                  'judge_instructions': {'FINAL_AGGREGATOR': 'Weigh the judges by severity.'}}
        instructions = ai_runner.build_instructions(config, 'FINAL_AGGREGATOR')
        self.assertIn('Weigh the judges by severity.', instructions)
        self.assertIn('{"type":"object"}', instructions)

    def test_legacy_fallback_without_generated_instructions(self):
        config = {'objective': 'o', 'taxonomy': 't', 'examples': '[]', 'schema': '{"type":"a"}'}
        judge = ai_runner.build_instructions(config, 'SOME_JUDGE')
        self.assertIn('You are the evaluation judge SOME_JUDGE', judge)
        self.assertNotIn('Judge instructions:', judge)
        self.assertIn('{"type":"a"}', ai_runner.build_instructions(config, 'FINAL_AGGREGATOR'))

    async def test_generated_instructions_reach_the_created_agent(self):
        from pydantic_ai import models
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        seen = []

        def answer(messages, info):
            seen.append(str(messages[0]))
            return ModelResponse(parts=[TextPart('{"verdict":"ok"}')])

        config = {'nodes': ['EVIDENCE_JUDGE', 'FINAL_AGGREGATOR'],
                  'edges': [['EVIDENCE_JUDGE', 'FINAL_AGGREGATOR']],
                  'model': 'test/model', 'objective': 'o', 'taxonomy': 't', 'examples': '[]',
                  'schema': '{"type":"object","required":["verdict"],'
                            '"properties":{"verdict":{"type":"string"}}}',
                  'judge_instructions': {'EVIDENCE_JUDGE': 'UNIQUE-JUDGE-MARKER-A',
                                         'FINAL_AGGREGATOR': 'UNIQUE-JUDGE-MARKER-B'}}
        with models.override_allow_model_requests(False), patch(
            'socket.socket.connect', side_effect=AssertionError('Network forbidden')
        ):
            await ai_runner.run(config, STEPS, 'fake-key', 0.1, lambda n, o: None,
                                FunctionModel(answer))
        self.assertIn('UNIQUE-JUDGE-MARKER-A', seen[0])
        self.assertIn('UNIQUE-JUDGE-MARKER-B', seen[1])


if __name__ == '__main__':
    unittest.main()
