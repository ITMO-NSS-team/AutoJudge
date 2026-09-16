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
test('raw nested OpenTelemetry spans become chronological trace steps', () => {
  const raw = {
    trace_id: 'trace-1',
    spans: [{
      timestamp: '2026-01-01T00:00:02Z', span_id: 'root', span_name: 'root', service_name: 'orchestrator',
      span_attributes: {}, logs: [], events: [], child_spans: [{
        timestamp: '2026-01-01T00:00:01Z', span_id: 'child', parent_span_id: 'root', span_name: 'llm', service_name: 'agent',
        span_attributes: {
          'llm.model_name': 'test/model',
          'llm.input_messages.0.message.role': 'user',
          'llm.input_messages.0.message.content': 'task',
          'llm.output_messages.0.message.role': 'assistant',
          'llm.output_messages.0.message.content': 'answer',
          'input.value': 'duplicated task',
          'output.value': 'duplicated answer',
        }, logs: [], events: [], child_spans: [],
      }],
    }],
  };
  const steps = normalizeTrace(JSON.stringify(raw));
  assert.equal(steps.length, 2);
  assert.equal(steps[0].agent, 'test/model');
  assert.match(steps[0].content, /"content":"task"/);
  assert.match(steps[0].content, /"content":"answer"/);
  assert.doesNotMatch(steps[0].content, /duplicated task|duplicated answer/);
  assert.match(steps[0].content, /"depth":1/);
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

test('few-shot presets are valid JSON arrays of trace/output pairs', async () => {
  const { fewShotPresets } = await import('../src/fewShotPresets.ts');
  assert.ok(fewShotPresets.length >= 1);
  for (const preset of fewShotPresets) {
    assert.ok(preset.id && preset.name && preset.description);
    assert.ok(Array.isArray(preset.examples) && preset.examples.length >= 1);
    for (const example of preset.examples) {
      assert.ok(Array.isArray(example.trace) && example.trace.length >= 1);
      assert.equal(typeof example.output, 'object');
      assert.ok(example.output !== null && !Array.isArray(example.output));
      for (const step of example.trace) {
        assert.equal(typeof step.id, 'number');
        assert.equal(typeof step.agent, 'string');
        assert.equal(typeof step.content, 'string');
      }
    }
    const serialized = JSON.stringify(preset.examples);
    assert.ok(Buffer.byteLength(serialized) < 100000, 'preset must stay small');
  }
});
