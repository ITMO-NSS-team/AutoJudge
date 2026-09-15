# Local backend

FastAPI backend for validating traces and designs, running the fixed AutoJudge judge pipeline, and storing runs in SQLite.

## Run

From the AutoJudge root:

```powershell
python -m pip install -r src/autojudge/ui/backend/requirements.txt
python -m uvicorn server:app --app-dir src/autojudge/ui/backend --host 127.0.0.1 --port 8000
```

Swagger is available at `http://127.0.0.1:8000/docs`.

## AI connection

The runner uses `LLM_BASE_URL`, `LLM_API_KEY`, `AGENT_NODE_MODEL`, and
`AGENT_NODE_TEMPERATURE`. `OPENROUTER_API_KEY` is a compatibility fallback only
for the standard OpenRouter endpoint. Model IDs must use printable ASCII without
spaces. Saving settings never contacts the provider.

Secrets use Windows DPAPI, macOS Keychain, or Linux Secret Service through
`keyring`. They are never returned by the API or stored in workspace snapshots.

For a temporary authenticated tunnel, set `AUTOJUDGE_ALLOWED_ORIGINS` to the
exact external HTTPS origin. Multiple origins are comma-separated. This extends
origin checks for settings and AI runs; never use a wildcard.

## Evaluation contract

- Full trace only.
- Fixed judge nodes and graph supplied by the UI.
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
