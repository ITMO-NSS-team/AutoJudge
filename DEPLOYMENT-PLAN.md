# AutoJudge Research Deployment Plan

## Goal

Deploy AutoJudge as a small shared research application with a stable HTTPS URL,
persistent run history, and controlled access, without turning it into a
multi-tenant SaaS product.

## Target architecture

Use one repository, one Docker image, one application instance, and one
persistent volume.

```text
Browser
   |
   | HTTPS + Basic Auth
   v
FastAPI
   |- React static files
   |- API and Server-Sent Events
   |- AutoJudge AI pipeline
   `- SQLite on a persistent volume
```

The React application is built during the Docker build and served by FastAPI.
This keeps the UI and API on one origin and removes the production dependency on
the Vite development server.

## Deliberate simplifications

- One shared research workspace.
- One shared provider configuration managed by the project owner.
- One AI evaluation at a time.
- SQLite instead of Postgres.
- In-process background execution instead of Redis and a separate worker.
- Basic Auth instead of user registration, roles, and organizations.
- One application instance with horizontal scaling disabled.

These constraints are acceptable for a small trusted research group. A server
restart may interrupt an active AI evaluation; the existing recovery behavior
must mark it as interrupted rather than silently leaving it running.

## Required implementation

### 1. Reproducible container

- Add a multi-stage `Dockerfile`.
- Build React with Node in the first stage.
- Install the minimal Python runtime dependencies in the second stage.
- Copy the frontend build into the runtime image.
- Run FastAPI with one Uvicorn worker as a non-root user.
- Bind to the platform-provided `PORT` on `0.0.0.0`.
- Add a container health check.
- Pin frontend dependencies and commit lock files.

### 2. Single-origin application

- Serve the React build and SPA fallback from FastAPI.
- Keep `/api` routes on the same origin.
- Preserve Server-Sent Events for run progress.
- Remove production reliance on Vite proxy and development `allowedHosts`.

### 3. Persistent research data

- Add `AUTOJUDGE_DATA_DIR`, defaulting to the existing local `.autojudge`
  directory.
- Store the deployment database at
  `${AUTOJUDGE_DATA_DIR}/workspace.sqlite3`.
- Mount the hosting volume at `/data` and set `AUTOJUDGE_DATA_DIR=/data`.
- Keep a single application instance to avoid concurrent SQLite writers across
  machines.
- Document database backup and restore commands.

### 4. Authentication and access

- Protect the complete UI and API with Basic Auth over HTTPS.
- Read credentials only from `AUTOJUDGE_USERNAME` and
  `AUTOJUDGE_PASSWORD` deployment secrets.
- Exempt only liveness checks if required by the hosting platform.
- Do not use wildcard origins.
- Rotate the shared password when a tester leaves the group.

### 5. Provider secrets

- Supply `LLM_API_KEY`, `LLM_BASE_URL`, `AGENT_NODE_MODEL`, and
  `AGENT_NODE_TEMPERATURE` through hosting secrets.
- Do not use DPAPI or desktop keyring inside the deployment container.
- Do not bake secrets into the Docker image or frontend bundle.
- Restrict remote Settings so testers cannot replace provider credentials.
- Keep explicit paid-run confirmation and the one-AI-run concurrency limit.

### 6. Research safeguards

- Preserve trace, taxonomy, schema, node-count, and output-size limits.
- Keep provider request timeouts and sanitized errors.
- Add a deployment-wide AI kill switch.
- Add a daily or monthly spending ceiling when the provider supports it.
- Show that cancellation may not refund an already-started provider request.

## Minimal repository artifacts

```text
Dockerfile
.dockerignore
.env.example
DEPLOYMENT.md
render.yaml or railway configuration
```

`DEPLOYMENT.md` must contain setup, secret names, volume mount, health checks,
backup, restore, rollback, and shutdown instructions. It must not contain real
credentials.

## Verification gates

### Local

- Production frontend build passes.
- Frontend tests pass.
- Backend tests pass without provider network access.
- Docker image builds from a clean checkout.
- Container starts with temporary test secrets.
- `/api/health` responds successfully.
- Basic Auth rejects missing and invalid credentials.
- Trace import, design validation, offline run, SSE progress, and deletion work.
- A mocked AI run completes without a paid provider call.
- SQLite data survives a container restart with the volume attached.

### Staging

- HTTPS and authentication work from an external browser.
- Settings do not expose or allow replacement of provider secrets.
- One explicitly confirmed real AI smoke test completes.
- Token and cost reporting are recorded when supplied by the provider.
- Restarting the service marks an active run as interrupted.
- Backup and restore are tested before production use.

## Deployment sequence

1. Review and commit the current UI/backend milestone.
2. Add single-origin static serving and `AUTOJUDGE_DATA_DIR`.
3. Add Basic Auth and deployment-safe secret behavior.
4. Add and locally verify the Docker image.
5. Choose Railway with a volume or Render with a persistent disk.
6. Deploy a staging instance and complete the verification gates.
7. Create a backup and document rollback.
8. Promote the verified image to the shared research deployment.

## Rollback triggers

Roll back immediately if any of the following occurs:

- unauthorized access to the UI, API, runs, or Settings;
- provider secrets appear in logs, responses, or stored run data;
- SQLite data disappears or becomes corrupt after restart;
- AI runs continue after cancellation or service restart without a visible state;
- the provider is called without explicit paid-run confirmation;
- the primary Trace to Design to Run to Verdict flow fails in staging.

## Deferred until justified

- Postgres and schema migrations.
- Redis and a separate worker.
- Per-user accounts and data ownership.
- Multiple application instances.
- Kubernetes.
- Organization-level billing and audit systems.

Introduce these only when the research workload exceeds the single-instance
constraints or the application becomes a broader external service.
