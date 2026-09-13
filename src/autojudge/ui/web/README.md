# AutoJudge web UI

React/Vite interface for evaluating multi-agent traces with a fixed judge
pipeline.

## Workflow

1. Trace — import JSON or JSONL and inspect the live preview.
2. Design — set the objective, taxonomy, model, output schema, and optional few-shot examples.
3. Run — review the static pipeline and choose a dry or confirmed AI run.
4. Verdict — inspect the result and supporting node outputs.

The navigation contains Overview, New evaluation, Runs, Compare, Observability,
and Settings. Runs can be selected for export, cloning, archiving, deletion, and
comparison. The graph is a read-only visualization.

## Run

```powershell
npm install
npm run dev
```

Default address: `http://127.0.0.1:5173/`.

## Verify

```powershell
npm run build
node --test scripts/imports.test.mjs scripts/design-template.test.mjs
```
