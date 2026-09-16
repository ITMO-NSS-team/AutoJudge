import test from 'node:test';
import assert from 'node:assert/strict';
import { api } from '../src/apiClient.ts';

test('API client accepts a successful 204 response without JSON', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async () => new Response(null, { status: 204 });

  assert.equal(await api('/runs/test-run', 'DELETE'), undefined);
});

test('API client surfaces the backend cause, not the JSON envelope', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async () =>
    Response.json({ detail: 'PoolGenerator request failed (401)' }, { status: 502 });

  await assert.rejects(api('/design/generate', 'POST', {}), {
    message: 'PoolGenerator request failed (401)',
  });
});

test('API client falls back to the raw body when it is not JSON', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async () => new Response('Bad gateway', { status: 502 });

  await assert.rejects(api('/health'), { message: 'Bad gateway' });
});

test('API client still parses successful JSON responses', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async () => Response.json({ status: 'ok' });

  assert.deepEqual(await api('/health'), { status: 'ok' });
});
