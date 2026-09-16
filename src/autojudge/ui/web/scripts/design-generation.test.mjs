import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  designKey,
  designSourceHint,
  designStateLabel,
  nextDesignState,
  runBlockedReason,
} from '../src/designState.ts';
import { parseDesignConfig } from '../src/imports.ts';

const inputs = { objective: 'Attribute the failure', taxonomy: 'Tool Selection', schema: '{"type":"object"}', examples: '[]' };
const steps = [{ id: 1, agent: 'Researcher', content: 'claim' }];
const workspace = readFileSync(new URL('../src/Workspace.tsx', import.meta.url), 'utf8');

test('a generated design cannot run before it is generated', () => {
  assert.match(runBlockedReason('generated', 'not_generated', false), /Run Meta Agent/);
  assert.match(runBlockedReason('generated', 'generating', false), /still being generated/);
  assert.match(runBlockedReason('generated', 'failed', false), /Retry generation/);
});

test('a generated design still needs explicit acceptance', () => {
  assert.match(runBlockedReason('generated', 'generated', false), /Use this pool/);
  assert.equal(runBlockedReason('generated', 'generated', true), '');
});

test('manual and imported designs run without generation', () => {
  assert.equal(runBlockedReason('manual', 'not_generated', false), '');
  assert.equal(runBlockedReason('imported', 'not_generated', false), '');
});

test('changing any design input marks the generated design stale and blocks the run', () => {
  const snapshot = designKey(inputs, steps);
  assert.equal(nextDesignState('generated', snapshot, snapshot), 'generated');
  for (const [field, value] of [['taxonomy', 'Hallucination'], ['objective', 'Other'], ['schema', '{"type":"array"}'], ['examples', '[{"a":1}]']]) {
    const changed = designKey({ ...inputs, [field]: value }, steps);
    assert.equal(nextDesignState('generated', changed, snapshot), 'stale', field);
  }
  const changedTrace = designKey(inputs, [{ id: 2, agent: 'Other', content: 'x' }]);
  assert.equal(nextDesignState('generated', changedTrace, snapshot), 'stale');
  assert.match(runBlockedReason('generated', 'stale', true), /Regenerate the Meta Agent/);
});

test('regeneration or restored inputs clear the stale state', () => {
  const snapshot = designKey(inputs, steps);
  assert.equal(nextDesignState('stale', snapshot, snapshot), 'generated');
  // A fresh generation installs a new snapshot, so the new state is current.
  const regenerated = designKey({ ...inputs, taxonomy: 'Hallucination' }, steps);
  assert.equal(nextDesignState('generated', regenerated, regenerated), 'generated');
});

test('no snapshot means no staleness verdict', () => {
  assert.equal(nextDesignState('not_generated', designKey(inputs, steps), null), 'not_generated');
});

test('state and source are labelled for the user', () => {
  assert.equal(designStateLabel('generated', true), 'Accepted');
  assert.equal(designStateLabel('generated', false), 'Review required');
  assert.equal(designStateLabel('stale', false), 'Stale');
  assert.match(designSourceHint('generated'), /meta-agent/);
  assert.match(designSourceHint('imported'), /without judge generation/);
  assert.match(designSourceHint('manual'), /Run Meta Agent/);
});

test('imported designs keep an arbitrary DAG', () => {
  const design = parseDesignConfig(JSON.stringify({
    nodes: ['A_JUDGE', 'B_JUDGE', 'FINAL_AGGREGATOR'],
    edges: [['A_JUDGE', 'B_JUDGE'], ['B_JUDGE', 'FINAL_AGGREGATOR']],
    judge_instructions: { A_JUDGE: 'check tools', FINAL_AGGREGATOR: 'aggregate' },
    judges: [{ name: 'A_JUDGE', instructions: 'check tools', model: 'x/y', mcp_tools: [] }],
  }));
  assert.deepEqual(design.nodes, ['A_JUDGE', 'B_JUDGE', 'FINAL_AGGREGATOR']);
  assert.deepEqual(design.edges, [['A_JUDGE', 'B_JUDGE'], ['B_JUDGE', 'FINAL_AGGREGATOR']]);
  assert.equal(design.judge_instructions.A_JUDGE, 'check tools');
  assert.equal(design.judges?.[0].model, 'x/y');
});

test('an exported run snapshot can be imported as a design', () => {
  const design = parseDesignConfig(JSON.stringify({
    id: 'abc', config: { nodes: ['J', 'FINAL_AGGREGATOR'], edges: [['J', 'FINAL_AGGREGATOR']], judge_instructions: { J: 'x' } },
  }));
  assert.deepEqual(design.nodes, ['J', 'FINAL_AGGREGATOR']);
});

test('invalid imported designs are rejected with their reason', () => {
  const cases = [
    ['{}', /nodes/],
    ['{"nodes":["A","A","FINAL_AGGREGATOR"]}', /unique/],
    ['{"nodes":["A"]}', /FINAL_AGGREGATOR/],
    ['{"nodes":["A","FINAL_AGGREGATOR"],"edges":[["A","GHOST"]]}', /not in "nodes"/],
    ['{"nodes":["A","FINAL_AGGREGATOR"],"edges":[["FINAL_AGGREGATOR","A"]]}', /terminal judge/],
    ['{"nodes":["A","FINAL_AGGREGATOR"],"judge_instructions":{"GHOST":"x"}}', /unknown judge/],
    ['not json', /JSON/],
  ];
  for (const [payload, pattern] of cases) {
    assert.throws(() => parseDesignConfig(payload), pattern, payload);
  }
});

test('the workspace no longer forces the static demo pool onto runtime state', () => {
  // The restored workspace config and cloned runs must keep their own judges.
  assert.doesNotMatch(workspace, /nodes:\s*roles,\s*\n\s*edges:\s*usesFixedJudges/);
  assert.doesNotMatch(workspace, /nodes:\s*structuredClone\(roles\)/);
  assert.match(workspace, /"\/design\/generate",\s*"POST"/);
  assert.match(workspace, /design_source:\s*"generated"/);
});

test('the run request carries the accepted design identity', () => {
  assert.match(workspace, /disabled=\{launching \|\| !connected \|\| runBlocked\}/);
  assert.match(workspace, /if \(runBlocked\) \{/);
});
