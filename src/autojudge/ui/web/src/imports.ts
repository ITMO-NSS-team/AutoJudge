function extractFencedJson(text: string): string {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  return fence ? fence[1].trim() : text;
}

function taxonomyFromJson(data: unknown): string {
  if (Array.isArray(data)) {
    if (!data.length || data.some((item) => typeof item !== 'string' || !item.trim()))
      throw Error('A JSON taxonomy must be a non-empty array of category strings.');
    return ['# Taxonomy', '', ...data.map((item) => `- ${item.trim()}`)].join('\n');
  }
  if (data && typeof data === 'object') {
    const entries = Object.entries(data as Record<string, unknown>);
    if (!entries.length) throw Error('The JSON taxonomy object is empty.');
    const lines = ['# Taxonomy'];
    for (const [key, value] of entries) {
      if (typeof value === 'string' && value.trim()) lines.push('', `## ${key}`, `- ${value.trim()}`);
      else if (Array.isArray(value) && value.length && value.every((item) => typeof item === 'string' && item.trim()))
        lines.push('', `## ${key}`, ...value.map((item) => `- ${item.trim()}`));
      else throw Error('JSON taxonomy values must be category strings or arrays of them.');
    }
    return lines.join('\n');
  }
  throw Error('A JSON taxonomy must be an array of categories or an object of sections.');
}

export function parseDesignFile(kind: 'schema' | 'taxonomy', input: string): string {
  const text = input.replace(/^\uFEFF/, '').trim();
  if (!text) throw Error('The file is empty.');
  if (kind === 'taxonomy') {
    // Markdown passes through unchanged; JSON arrays/objects become Markdown.
    try { return taxonomyFromJson(JSON.parse(text)); } catch (e) {
      if (text.startsWith('[') || text.startsWith('{')) throw e;
      return text;
    }
  }
  // Schema: accept a JSON Schema (raw JSON, or JSON inside a Markdown code
  // fence), or any free-form text/Markdown used as a format instruction for
  // the FINAL_AGGREGATOR. Free-form text is returned unchanged.
  try {
    const candidate = extractFencedJson(text);
    const schema = JSON.parse(candidate);
    if (schema && typeof schema === 'object' && !Array.isArray(schema) && (schema as { type?: string }).type === 'object') {
      const check = (value: unknown): void => {
        if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) {
          if (['$ref', '$dynamicRef', '$recursiveRef'].includes(key)) throw Error('Schema references are not yet supported by the runner.');
          check(child);
        }
      };
      check(schema);
      return JSON.stringify(schema, null, 2);
    }
  } catch { /* not a JSON Schema — fall through to free-form text */ }
  return text;
}

export type ImportedDesign = {
  nodes: string[];
  edges: string[][];
  judge_instructions: Record<string, string>;
  judges?: { name: string; instructions: string; model?: string; mcp_tools?: string[] }[];
};

// Accepts a design exported from this workspace or written by hand, so an
// existing pipeline can run without calling the judge generator.
export function parseDesignConfig(input: string): ImportedDesign {
  const text = input.replace(/^﻿/, '').trim();
  if (!text) throw Error('The file is empty.');
  let data: unknown;
  try { data = JSON.parse(extractFencedJson(text)); } catch { throw Error('A design file must contain JSON.'); }
  const source = (data && typeof data === 'object' && !Array.isArray(data) ? data : {}) as Record<string, unknown>;
  const design = (source.config && typeof source.config === 'object' ? source.config : source) as Record<string, unknown>;
  const nodes = design.nodes;
  if (!Array.isArray(nodes) || !nodes.length || nodes.some((n) => typeof n !== 'string' || !n.trim()))
    throw Error('A design must list non-empty judge names in "nodes".');
  if (new Set(nodes as string[]).size !== nodes.length) throw Error('Judge names must be unique.');
  if (!(nodes as string[]).includes('FINAL_AGGREGATOR')) throw Error('A design must include FINAL_AGGREGATOR.');
  const rawEdges = design.edges ?? [];
  if (!Array.isArray(rawEdges)) throw Error('"edges" must be an array of [from, to] pairs.');
  const edges = rawEdges.map((edge) => {
    if (!Array.isArray(edge) || edge.length !== 2 || edge.some((n) => typeof n !== 'string'))
      throw Error('Every edge must be a [from, to] pair of judge names.');
    const [from, to] = edge as string[];
    if (!(nodes as string[]).includes(from) || !(nodes as string[]).includes(to))
      throw Error(`Edge ${from} → ${to} references a judge that is not in "nodes".`);
    if (from === 'FINAL_AGGREGATOR') throw Error('FINAL_AGGREGATOR must be the terminal judge.');
    return [from, to];
  });
  const rawInstructions = design.judge_instructions;
  const instructions: Record<string, string> = {};
  if (rawInstructions !== undefined) {
    if (!rawInstructions || typeof rawInstructions !== 'object' || Array.isArray(rawInstructions))
      throw Error('"judge_instructions" must map judge names to instructions.');
    for (const [name, value] of Object.entries(rawInstructions as Record<string, unknown>)) {
      if (typeof value !== 'string') throw Error(`Instructions for ${name} must be text.`);
      if (!(nodes as string[]).includes(name)) throw Error(`Instructions reference unknown judge ${name}.`);
      instructions[name] = value;
    }
  }
  const judges = Array.isArray(design.judges)
    ? (design.judges as Record<string, unknown>[])
        .filter((judge) => judge && typeof judge.name === 'string' && (nodes as string[]).includes(judge.name as string))
        .map((judge) => ({
          name: String(judge.name),
          instructions: typeof judge.instructions === 'string' ? judge.instructions : instructions[String(judge.name)] ?? '',
          model: typeof judge.model === 'string' ? judge.model : undefined,
          mcp_tools: Array.isArray(judge.mcp_tools) ? judge.mcp_tools.map(String) : undefined,
        }))
    : undefined;
  return { nodes: nodes as string[], edges, judge_instructions: instructions, judges };
}

export function normalizeTrace(text: string) {
  const clean = text.replace(/^\uFEFF/, '').trim();
  let data;
  try { data = JSON.parse(clean); } catch { data = clean.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line)); }
  if (!Array.isArray(data) && Array.isArray(data?.spans)) return normalizeSpanTrace(data.spans);
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

type RawObject = Record<string, unknown>;

function object(value: unknown): RawObject {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as RawObject : {};
}

function flatMessages(attributes: RawObject, prefix: 'input' | 'output') {
  const messages = new Map<number, { role?: string; content?: unknown }>();
  const pattern = new RegExp(`^llm\\.${prefix}_messages\\.(\\d+)\\.message\\.(role|content)$`);
  for (const [key, value] of Object.entries(attributes)) {
    const match = key.match(pattern);
    if (!match) continue;
    const index = Number(match[1]);
    const message = messages.get(index) ?? {};
    if (match[2] === 'role') message.role = String(value);
    else message.content = value;
    messages.set(index, message);
  }
  return [...messages.entries()].sort(([a], [b]) => a - b).map(([, message]) => message);
}

function normalizeSpanTrace(rootSpans: unknown[]) {
  const flattened: { span: RawObject; depth: number; order: number }[] = [];
  const walk = (value: unknown, depth: number) => {
    const span = object(value);
    flattened.push({ span, depth, order: flattened.length });
    const children = Array.isArray(span.child_spans) ? span.child_spans : [];
    for (const child of children) walk(child, depth + 1);
  };
  for (const span of rootSpans) walk(span, 0);
  if (!flattened.length) throw Error('The raw span trace is empty.');
  flattened.sort((a, b) => {
    const byTime = String(a.span.timestamp ?? '').localeCompare(String(b.span.timestamp ?? ''));
    return byTime || a.order - b.order;
  });

  const seenMessages = new Set<string>();
  return flattened.map(({ span, depth }, index) => {
    const attributes = object(span.span_attributes);
    const inputs = flatMessages(attributes, 'input').filter((message) => {
      const key = JSON.stringify(message);
      if (seenMessages.has(key)) return false;
      seenMessages.add(key);
      return true;
    });
    const outputs = flatMessages(attributes, 'output');
    for (const message of outputs) seenMessages.add(JSON.stringify(message));
    const hasStructuredInput = Object.keys(attributes).some((key) => key.startsWith('llm.input_messages.'));
    const hasStructuredOutput = Object.keys(attributes).some((key) => key.startsWith('llm.output_messages.'));
    const retainedAttributes = Object.fromEntries(Object.entries(attributes).filter(([key]) =>
      !key.startsWith('pat.') &&
      !key.startsWith('llm.input_messages.') &&
      !key.startsWith('llm.output_messages.') &&
      !key.endsWith('.mime_type') &&
      !(key === 'input.value' && hasStructuredInput) &&
      !(key === 'output.value' && hasStructuredOutput)
    ));
    const payload = {
      span_name: span.span_name,
      span_id: span.span_id,
      parent_span_id: span.parent_span_id,
      timestamp: span.timestamp,
      duration: span.duration,
      status_code: span.status_code,
      status_message: span.status_message,
      depth,
      attributes: retainedAttributes,
      input_messages: inputs,
      output_messages: outputs,
      events: Array.isArray(span.events) ? span.events : [],
      logs: Array.isArray(span.logs) ? span.logs : [],
    };
    return {
      id: index + 1,
      agent: String(attributes['agent.name'] ?? attributes['llm.model_name'] ?? span.service_name ?? span.span_name ?? 'Span'),
      content: JSON.stringify(payload),
    };
  });
}
