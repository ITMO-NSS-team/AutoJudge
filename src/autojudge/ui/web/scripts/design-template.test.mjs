import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { exampleOutputSchema, exampleTaxonomy } from '../src/designTemplate.ts';
import { parseDesignFile } from '../src/imports.ts';

test('example output schema documents the README errors/scores format', () => {
  assert.match(exampleOutputSchema, /"errors":/);
  assert.match(exampleOutputSchema, /"scores":/);
  assert.match(exampleOutputSchema, /HIGH\|MEDIUM\|LOW/);
  // Placeholders like "0-5" are not valid JSON, so this is a free-form
  // format instruction, not a strict schema, and must not parse as JSON.
  assert.throws(() => JSON.parse(exampleOutputSchema));
  // The runner still accepts it, as text, through the free-form fallback.
  assert.equal(parseDesignFile('schema', exampleOutputSchema), exampleOutputSchema);
});
test('example taxonomy matches the README reasoning/execution/planning tree', () => {
  assert.match(exampleTaxonomy, /Reasoning Errors/);
  assert.match(exampleTaxonomy, /Hallucinations/);
  assert.match(exampleTaxonomy, /System Execution Errors/);
  assert.match(exampleTaxonomy, /Planning and Coordination Errors/);
  assert.equal(parseDesignFile('taxonomy', exampleTaxonomy), exampleTaxonomy);
});
test('judge model is selected only through environment settings', () => {
  const workspace = readFileSync(new URL('../src/Workspace.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(workspace, /label="Judge model"/);
  assert.doesNotMatch(workspace, /model:\s*"openrouter\/auto"/);
  assert.match(workspace, /AGENT_NODE_MODEL/);
});
