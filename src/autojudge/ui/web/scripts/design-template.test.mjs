import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { exampleOutputSchema, exampleTaxonomy } from '../src/designTemplate.ts';
import { parseDesignFile } from '../src/imports.ts';

test('example template uses final attribution contract, not specialist score', () => {
  const schema=JSON.parse(parseDesignFile('schema',JSON.stringify(exampleOutputSchema)));
  assert.deepEqual(schema.required,['agent','step','reason']);
  assert.equal(schema.additionalProperties,false);
  const pattern=new RegExp(schema.properties.step.pattern);
  assert.ok(pattern.test('4'));
  for(const value of ['0','-1','4.5','step 4'])assert.equal(pattern.test(value),false);
});
test('example taxonomy contains judge and final attribution guidance', () => {
  assert.match(exampleTaxonomy, /Search integrity/);
  assert.match(exampleTaxonomy, /Final attribution/);
});
test('judge model is selected only through environment settings', () => {
  const workspace = readFileSync(new URL('../src/Workspace.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(workspace, /label="Judge model"/);
  assert.doesNotMatch(workspace, /model:\s*"openrouter\/auto"/);
  assert.match(workspace, /AGENT_NODE_MODEL/);
});
