# Hugging Face Deployment

## Target

Deploy AutoJudge as one private Hugging Face Docker Space. The container serves the
React build and FastAPI from the same origin on port `7860`, runs one Uvicorn worker,
and keeps provider credentials in Hugging Face Secrets.

## 1. Create the Space

Create a new Space with:

- SDK: `Docker`;
- visibility: `Private`;
- hardware: `CPU Basic`;
- repository contents: this reviewed project revision.

Docker Spaces require an eligible paid Hugging Face account even when CPU Basic has
no hourly hardware charge. Invite testers as Space collaborators after staging is
verified.

## 2. Configure persistence

Create or select a private Storage Bucket and attach it read-write at `/data`. Add
these Space Variables:

```text
AUTOJUDGE_DATA_DIR=/data/autojudge
AUTOJUDGE_SETTINGS_READ_ONLY=1
AUTOJUDGE_AI_ENABLED=1
AUTOJUDGE_ALLOWED_ORIGINS=https://OWNER-SPACE.hf.space
AGENT_NODE_MODEL=google/gemini-2.5-flash
AGENT_NODE_TEMPERATURE=0.1
```

Replace `OWNER-SPACE.hf.space` with the actual direct Space application origin. Keep
the origin exact and HTTPS. The provider endpoint is fixed in the backend to
`https://openrouter.ai/api/v1`.

The ordinary Space filesystem is ephemeral. Run history is durable only after the
bucket is attached and a restart test confirms that
`/data/autojudge/workspace.sqlite3` survives.

## 3. Configure secrets

Add this under Space Settings → Secrets:

```text
OPENROUTER_API_KEY=...
```

Do not commit the secret, add it as a Docker build argument, or enter it through the
deployed AutoJudge Settings page. Hosted environment settings are intentionally
read-only.

## 4. Deploy

Push the reviewed revision to the Space Git repository. Hugging Face reads the
metadata at the top of `README.md`, builds the root `Dockerfile`, and exposes port
`7860`.

Before inviting testers, verify:

1. `/api/health` returns HTTP 200.
2. Overview, New evaluation, Runs, Compare, and Settings load directly.
3. Settings reports deployment values as read-only and never returns a secret value.
4. Trace import, design validation, dry run, progress events, archive, and deletion work.
5. Only one AI evaluation can run at a time.
6. One owner-authorized AI smoke test completes with provider billing limits enabled.
7. Run history survives a Space restart or factory rebuild.

## Backup and rollback

Do not deploy while an AI evaluation is running. Before a new revision, copy
`workspace.sqlite3` to a timestamped backup in the private bucket and record the
current Space commit SHA. Validate the backup with SQLite before relying on it.

Roll back by pushing or selecting the last verified Space revision. Stop access and
roll back immediately if secrets appear in responses or logs, unauthorized users can
open the private Space, AI execution starts from an unapproved origin, stored runs are
lost, or the Trace → Design → Run → Verdict flow fails.

Set `AUTOJUDGE_AI_ENABLED=0` and restart the Space to disable new AI runs without
disabling dry runs or access to existing results.

## Local verification

```powershell
docker build -t autojudge-hf .
docker run --rm -p 7860:7860 `
  -e AUTOJUDGE_ALLOWED_ORIGINS=http://127.0.0.1:7860 `
  -e AGENT_NODE_MODEL=google/gemini-2.5-flash `
  -e AGENT_NODE_TEMPERATURE=0.1 `
  -v autojudge-data:/data `
  autojudge-hf
```

Omit `OPENROUTER_API_KEY` for dry-run verification. Add it only as an environment variable
for an explicitly authorized AI smoke test; never place its value in shell history or
documentation.
