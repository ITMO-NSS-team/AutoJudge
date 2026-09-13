export function parseDesignFile(kind: 'schema' | 'taxonomy', input: string): string {
  const text = input.replace(/^\uFEFF/, '').trim();
  if (!text) throw Error('The file is empty.');
  if (kind === 'taxonomy') return text;
  const schema = JSON.parse(text);
  if (!schema || Array.isArray(schema) || schema.type !== 'object') throw Error('The output schema must describe a JSON object.');
  const check = (value: unknown): void => {
    if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) {
      if (['$ref', '$dynamicRef', '$recursiveRef'].includes(key)) throw Error('Schema references are not yet supported by the runner.');
      check(child);
    }
  };
  check(schema);
  return JSON.stringify(schema, null, 2);
}

export function normalizeTrace(text: string) {
  const clean = text.replace(/^\uFEFF/, '').trim();
  let data;
  try { data = JSON.parse(clean); } catch { data = clean.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line)); }
  const arr = Array.isArray(data) ? data : data?.steps ?? data?.turns ?? data?.messages;
  if (!Array.isArray(arr) || !arr.length) throw Error('Expected an array, a steps/turns/messages object, or JSONL.');
  const normalized = arr.map((value: unknown, i: number) => {
    const o = value && typeof value === 'object' ? value as Record<string, unknown> : {};
    const parts = ['thought', 'content', 'action', 'tool_calls'].filter(key => o[key] !== undefined && o[key] !== '').map(key => {
      const content = typeof o[key] === 'string' ? o[key] : JSON.stringify(o[key]);
      return key === 'content' ? content : `[${key}]\n${content}`;
    });
    return { id: Number(o.id ?? i + 1), agent: String(o.agent ?? o.role ?? 'Agent'), content: parts.length ? parts.join('\n\n') : JSON.stringify(value) };
  });
  if (normalized.some(s => !Number.isFinite(s.id)) || new Set(normalized.map(s => s.id)).size !== normalized.length) throw Error('Step IDs must be numeric and unique.');
  return normalized;
}
