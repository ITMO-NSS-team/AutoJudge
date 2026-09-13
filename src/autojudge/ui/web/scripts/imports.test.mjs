import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeTrace, parseDesignFile } from '../src/imports.ts';

test('archived turns keep actions and thoughts, exclude answer labels', () => {
  const data = { gold_answer: 'DO NOT LEAK', mistake_reason: 'DO NOT LEAK', turns: [{ role: 'Manager', thought: 'reasoning', action: 'tool()', content: '' }] };
  const result = normalizeTrace(JSON.stringify(data));
  assert.equal(result[0].id, 1);
  assert.equal(result[0].agent, 'Manager');
  assert.match(result[0].content, /reasoning/);
  assert.match(result[0].content, /tool\(\)/);
  assert.doesNotMatch(JSON.stringify(result), /DO NOT LEAK/);
});
test('JSONL, steps, messages and BOM are supported', () => {
  assert.equal(normalizeTrace('\uFEFF{"steps":[{"id":5,"content":"hello"}]}')[0].id, 5);
  assert.equal(normalizeTrace('{"messages":[{"role":"user","content":"hi"}]}')[0].agent, 'user');
  assert.equal(normalizeTrace('{"content":"a"}\n{"content":"b"}').length, 2);
});
test('invalid trace is rejected', () => {
  for (const raw of ['', '[]', '[{"id":1},{"id":1}]', '[{"id":"invalid"}]']) assert.throws(() => normalizeTrace(raw));
});
test('schema validation preserves valid input', () => {
  assert.equal(JSON.parse(parseDesignFile('schema', '{"type":"object"}')).type, 'object');
  for (const raw of ['', '{}', '[]', '{broken', '{"type":"object","$ref":"https://example.com"}']) assert.throws(() => parseDesignFile('schema', raw));
});
test('taxonomy import trims text and rejects empty files', () => {
  assert.equal(parseDesignFile('taxonomy', '\uFEFF  # Taxonomy\n\n- unsupported_claim  '), '# Taxonomy\n\n- unsupported_claim');
  assert.throws(() => parseDesignFile('taxonomy', '   '));
});
test('bundled traces fit AI input limit and do not include ground truth', () => {
  for (let i = 1; i <= 3; i++) {
    const raw = readFileSync(new URL(`../public/test-data/trace-0${i}.json`, import.meta.url), 'utf8');
    const steps = normalizeTrace(raw);
    assert.ok(steps.length >= 4);
    assert.ok(Buffer.byteLength(JSON.stringify(steps)) < 100000);
    assert.doesNotMatch(raw, /"gold_answer"|"mistake_reason"|"mistake_step"/);
  }
});
