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

test('API client still parses successful JSON responses', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async () => Response.json({ status: 'ok' });

  assert.deepEqual(await api('/health'), { status: 'ok' });
});
