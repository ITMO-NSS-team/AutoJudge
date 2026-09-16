# Local backend

FastAPI backend for validating traces and designs, generating AutoJudge judge
pipelines with the core meta-agents, running them, and storing runs in SQLite.

## Run

From the AutoJudge root:

```powershell
python -m pip install -r src/autojudge/ui/backend/requirements.txt
python -m uvicorn server:app --app-dir src/autojudge/ui/backend --host 127.0.0.1 --port 8000
```

Swagger is available at `http://127.0.0.1:8000/docs`.

## AI connection

The runner uses `OPENROUTER_API_KEY`, `ENDPOINT_API_URL`, `AGENT_NODE_MODEL`, and
`AGENT_NODE_TEMPERATURE`. `ENDPOINT_API_URL` defaults to
`https://openrouter.ai/api/v1` and may point to another OpenAI-compatible endpoint.
Model IDs must use printable ASCII without spaces. Saving settings never contacts
the provider.

Secrets use Windows DPAPI, macOS Keychain, or Linux Secret Service through
`keyring`. They are never returned by the API or stored in workspace snapshots.

For a remote deployment, set `AUTOJUDGE_ALLOWED_ORIGINS` to the exact external
HTTPS origin. Multiple origins are comma-separated. This extends origin checks
for settings and AI runs; never use a wildcard.

## Judge generation

`POST /api/design/generate` builds the pipeline with the core meta-agents; the UI
never writes judges itself.

```text
objective + taxonomy + schema + examples + trace
  -> PoolGenerator   (autojudge.meta_agents)  -> AgentPool
  -> GraphGenerator  (autojudge.meta_agents)  -> GraphDict
  -> validation + PipelineBuilder
  -> {nodes, edges, judge_instructions, judges, design_id, metadata}
```

Request: `{objective, taxonomy, schema, examples, steps}`. The response carries
`design_source: "generated"` and a `design_id` fingerprint of the inputs, nodes,
edges and instructions. `POST /api/runs` recomputes that fingerprint and refuses
(409) a design that no longer matches what was generated and accepted, so the
executed pipeline is always the one the user reviewed. `/api/runs` never
regenerates anything. Designs with `design_source` `manual` or `imported` and no
`design_id` run unchanged.

Generated instructions are the judge's own prompt in the runner, framed by a
shared security and context block; `FINAL_AGGREGATOR` additionally receives the
output contract. Nodes without generated instructions fall back to the shared
prompt.

`META_AGENT_MODEL` configures the Meta Agent that builds the judge pool and is
separate from `AGENT_NODE_MODEL`, which runs the judges. Generation requires an
approved origin and a configured key, and only one generation runs at a time.

## Evaluation contract

- Full trace only.
- Judge nodes and graph are generated, imported, or supplied manually by the UI.
- A non-empty Markdown taxonomy is passed to every judge.
- Output is checked against an object JSON Schema.
- Few-shot examples are an optional JSON array.
- Dry run validates structure without AI requests.
- AI run requires an allowed origin and configured provider credentials.
- One AI run at a time, with up to eight nodes. Trace size is not artificially limited by the application.

## API

- `GET /api/health`
- `GET/PUT /api/settings/env`
- `GET/PUT/DELETE /api/credentials/openrouter`
- `POST /api/design/validate`
- `POST /api/design/generate`
- `POST /api/graph/validate`
- `GET/POST /api/runs`
- `GET/PATCH/DELETE /api/runs/{id}`
- `POST /api/runs/{id}/cancel`
- `GET /api/runs/{id}/events`
- `GET/PUT /api/workspace`

`/api/workspace` persists only the current evaluation configuration and local
execution settings. Run records contain their own immutable configuration and
trace snapshots.

## Verify

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s src/autojudge/ui/backend -p 'test_*.py'
```
