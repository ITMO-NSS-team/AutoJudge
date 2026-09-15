# AutoJudge Hugging Face Deployment Plan

## Goal

Deploy AutoJudge as a small shared research application in a Hugging Face Docker
Space. The deployment must provide one stable HTTPS URL, controlled access,
persistent run history, and server-side provider credentials without turning the
project into a multi-tenant SaaS product.

## Recommended target

- Hugging Face Docker Space.
- CPU Basic hardware; model inference remains on the configured external provider.
- One container, one Uvicorn worker, and one shared research workspace.
- FastAPI serves both `/api/*` and the production React bundle.
- SQLite lives under `AUTOJUDGE_DATA_DIR` on an attached Storage Bucket.
- Provider configuration is injected through Hugging Face Secrets.

```text
Browser
   |
   | HTTPS
   v
Hugging Face Docker Space :7860
   `- FastAPI / Uvicorn (one worker)
      |- React static files and SPA fallback
      |- REST API and Server-Sent Events
      |- AutoJudge AI pipeline
      `- SQLite -> /data/autojudge/workspace.sqlite3
```

## Deliberate simplifications

- One shared workspace for a small trusted research group.
- One provider configuration managed by the project owner.
- One AI evaluation at a time.
- SQLite instead of Postgres.
- In-process run execution instead of Redis and a separate worker.
- No registration, organizations, roles, or horizontal scaling.

A container restart may interrupt an active AI evaluation. Recovery must mark the
run as interrupted instead of leaving it in a running state.

## Access model

Choose one of these before staging:

1. **Private Space:** only the owner and invited Hugging Face collaborators can
   open the application. This is the preferred starting point.
2. **Protected or public application with Basic Auth:** use this only when testers
   must access the app without Hugging Face accounts. Protect the complete UI and
   API, not only the Settings page.

If application-level Basic Auth is enabled, read its values only from
`AUTOJUDGE_USERNAME` and `AUTOJUDGE_PASSWORD` Secrets. Exempt `/api/health` only
when Hugging Face requires an unauthenticated liveness endpoint.

## Required implementation

### 1. Single-origin production server

- Build React during the Docker image build.
- Copy `src/autojudge/ui/web/dist` into the Python runtime image.
- Serve the frontend assets and SPA fallback from FastAPI.
- Keep `/api/*` and Server-Sent Events on the same origin.
- Do not run Vite or use its proxy and `allowedHosts` settings in production.
- Listen on `0.0.0.0:7860`, matching the Space `app_port` metadata.

### 2. Reproducible Docker image

- Add a multi-stage root `Dockerfile` with Node and Python build stages.
- Install only runtime Python packages in the final stage.
- Run the application as a non-root user.
- Start exactly one Uvicorn worker.
- Add a Docker health check for `/api/health`.
- Keep dependency lock files committed and never copy `.env`, databases, logs, or
  development caches into the image.

### 3. Persistent research data

- Add `AUTOJUDGE_DATA_DIR` support to the backend.
- Preserve the current local `.autojudge` directory as the default.
- In the Space, set `AUTOJUDGE_DATA_DIR=/data/autojudge`.
- Attach a writable Hugging Face Storage Bucket at `/data`.
- Store `workspace.sqlite3` and any future durable artifacts only below that path.
- Keep one container instance and one worker to avoid multiple SQLite writers.
- Document and test SQLite backup and restore.

The ordinary Space filesystem is ephemeral. Do not claim persistence until data
has survived an actual Space restart with the bucket attached.

### 4. Provider secrets

Create this Hugging Face Secret and these non-secret variables:

```text
OPENROUTER_API_KEY
AGENT_NODE_MODEL
AGENT_NODE_TEMPERATURE
```

The backend uses only the fixed `https://openrouter.ai/api/v1` endpoint; the
deployment cannot replace it with an arbitrary provider URL.

Do not use Windows DPAPI, macOS Keychain, or Linux Secret Service in the container.
Deployment mode must read provider values directly from environment variables.
Do not bake secrets into the image, repository, frontend bundle, logs, SQLite, or
run snapshots. Remote users must not be able to replace provider credentials from
Settings.

### 5. Research safeguards

- Keep Full trace validation, taxonomy/schema validation, DAG validation, provider
  timeouts, sanitized errors, and the eight-node pipeline limit.
- Do not impose an artificial application-level trace file-size limit. Surface a
  clear provider context-window error when a selected model cannot accept a trace.
- Keep one-AI-run concurrency, clear billing warnings, and provider-side spending controls.
- Use `AUTOJUDGE_AI_ENABLED=0` as the deployment-wide AI kill switch.
- Configure a spending ceiling at the provider when supported.
- State that cancellation may not refund an already-started provider request.

## Hugging Face repository metadata

Add this YAML block at the beginning of the root `README.md` used by the Space:

```yaml
---
title: AutoJudge
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---
```

Required deployment artifacts:

```text
Dockerfile
.dockerignore
README.md
DEPLOYMENT.md
```

`DEPLOYMENT.md` must describe Space creation, visibility, secrets, bucket mount,
health checks, backup, restore, rollback, and shutdown without real credentials.

## Implementation sequence

1. Review and commit the current UI/backend milestone.
2. Add `AUTOJUDGE_DATA_DIR` and verify the existing local default still works.
3. Add React static serving and same-origin API/SSE behavior to FastAPI.
4. Add deployment mode for environment-only provider secrets and lock remote
   credential editing.
5. Add optional Basic Auth if Private Space collaboration is insufficient.
6. Add the Dockerfile, Space metadata, and deployment runbook.
7. Build and test the image locally without provider network access.
8. Create a Private Docker Space and attach a writable bucket at `/data`.
9. Configure Secrets and deploy the staging revision.
10. Verify persistence, access control, offline workflow, and one owner-authorized
    paid AI smoke test.

## Local verification gates

- Production frontend build passes.
- Frontend and backend tests pass without provider network access.
- Docker image builds from a clean checkout.
- Container runs on `0.0.0.0:7860` as a non-root user.
- `/api/health` responds successfully.
- Missing or invalid authentication is rejected when Basic Auth is enabled.
- Trace import, design validation, offline run, SSE progress, and deletion work.
- A raw nested OpenTelemetry trace larger than the former 5 MB boundary reaches
  normalization without an application-size rejection.
- A mocked AI run completes without making a paid provider call.
- SQLite data survives a local container restart with a mounted volume.

## Staging verification gates

- HTTPS and the selected visibility/authentication model work externally.
- API endpoints cannot bypass the UI access restriction.
- Settings neither expose nor replace provider secrets.
- Storage Bucket data survives a Space restart and rebuild.
- One explicitly confirmed real AI smoke test completes.
- Token/cost reporting is recorded when returned by the provider.
- Restarting during a run leaves a visible interrupted state.
- Backup and restore are tested before inviting the full tester group.

## Deployment procedure

1. Create a new Space with SDK `Docker` and visibility `Private`.
2. Push the reviewed deployment branch to the Space Git repository.
3. Attach a writable Storage Bucket to `/data`.
4. Add provider configuration under Space Settings as Secrets.
5. Set `AUTOJUDGE_DATA_DIR=/data/autojudge` as a Space Variable.
6. Wait for the Docker build and verify `/api/health`.
7. Complete the staging verification gates.
8. Invite testers as collaborators only after access and persistence checks pass.

Each pushed commit causes the Space to rebuild and restart. Do not deploy while a
paid evaluation is running.

## Backup and rollback

Before each deployment revision:

- stop new AI runs;
- create a consistent SQLite backup from the mounted data directory;
- record the working Space commit SHA;
- verify that the backup can be opened or restored in a temporary environment.

Roll back by selecting or pushing the last verified Space commit, then confirm the
database and `/api/health` before reopening access.

Roll back immediately if:

- unauthorized users can access the UI, API, runs, or Settings;
- provider secrets appear in logs, responses, SQLite, or frontend assets;
- persisted data disappears or becomes corrupt after restart;
- AI calls start from an unapproved origin or without configured provider limits;
- cancellation or restart leaves a run silently active;
- the Trace → Design → Run → Verdict workflow fails in staging.

## Platform constraints and open decisions

- Docker Spaces currently require an eligible paid Hugging Face account even when
  CPU Basic has no hourly hardware charge.
- Free CPU hardware may sleep when unused; the first tester may see a cold start.
- Confirm Storage Bucket availability and price in the target account before work
  begins.
- Decide whether all testers can use Hugging Face collaborator accounts. If not,
  Basic Auth becomes required implementation rather than optional work.

## Deferred until justified

- Postgres and schema migrations.
- Redis and a separate worker.
- Per-user accounts and data ownership.
- Multiple application instances.
- Kubernetes.
- Organization-level billing and audit systems.

Introduce these only when the research workload exceeds the single-instance
constraints or the application becomes a broader external service.
