"""OpenRouter calls with durable checkpoints, per-attempt accounting and bounded concurrency."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

from .protocol import (
    ARMS,
    digest,
    encode,
    load_run,
    now,
    read_json,
    validate_output,
    write_json,
)

API = "https://openrouter.ai/api/v1"


class HaltRun(RuntimeError):
    """Stop new paid requests; let already-started requests save their responses."""


class StageFailure(RuntimeError):
    """A trace exhausted the declared validation/retry policy."""


def redact(value: Any, key: str) -> Any:
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {k: redact(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    return value


def call_records(path: Path, run_id: str) -> list[dict]:
    records = []
    for file in sorted((path / "calls").glob("*.json")):
        record = read_json(file)
        if record["run_id"] != run_id or file.stem != record["call_id"]:
            raise ValueError(f"Foreign or renamed call record: {file}")
        records.append(record)
    return records


def usage_fields(response: dict, generation: dict | None = None) -> dict:
    usage = response.get("usage") or {}
    generation = generation or {}
    cost = usage.get("cost")
    source = "openrouter_usage"
    if cost is None:
        cost = generation.get("total_cost")
        source = "openrouter_generation"
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        cost, source = None, None
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "cost_usd": cost,
        "cost_source": source,
        "prompt_tokens": usage.get("prompt_tokens", generation.get("native_tokens_prompt")),
        # completion_tokens already includes reasoning; never add it a second time.
        "completion_tokens": usage.get(
            "completion_tokens", generation.get("native_tokens_completion")
        ),
        "reasoning_tokens": details.get(
            "reasoning_tokens", generation.get("native_tokens_reasoning")
        ),
        "cached_tokens": prompt_details.get("cached_tokens"),
        "cache_write_tokens": prompt_details.get("cache_write_tokens"),
        "provider": response.get("provider") or generation.get("provider_name"),
        "response_model": response.get("model") or generation.get("model"),
        "generation_id": response.get("id") or generation.get("id"),
        "service_tier": response.get("service_tier") or generation.get("service_tier"),
    }


class OpenRouterClient:
    def __init__(
        self,
        path: Path,
        manifest: dict,
        protocol: dict,
        api_key: str,
        max_cost: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY before using run")
        if not math.isfinite(max_cost) or max_cost <= 0:
            raise ValueError("--max-cost-usd must be positive")
        self.path, self.manifest, self.config = path, manifest, protocol["config"]
        self.key, self.max_cost = api_key, max_cost
        self.http = httpx.AsyncClient(
            base_url=API + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.config["timeout_seconds"],
            transport=transport,
        )
        self.semaphore = asyncio.Semaphore(self.config["concurrency"])
        self.lock = asyncio.Lock()
        self.reserved = 0.0
        self.spent = 0.0
        self.stopping = False

    def save(self, record: dict) -> None:
        write_json(self.path / "calls" / f"{record['call_id']}.json", redact(record, self.key))

    async def close(self) -> None:
        await self.http.aclose()

    async def get_generation(self, generation_id: str) -> dict:
        for delay in (0, 1, 2):
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await self.http.get("generation", params={"id": generation_id})
                if response.is_success:
                    value = response.json().get("data")
                    if isinstance(value, dict) and value.get("total_cost") is not None:
                        return value
            except (httpx.HTTPError, ValueError):
                pass
        return {}

    async def preflight(self) -> None:
        # GETs retrieve billing/price metadata only; they do not invoke a model.
        for record in call_records(self.path, self.manifest["run_id"]):
            if record.get("cost_usd") is None and record.get("response", {}).get("id"):
                generation = await self.get_generation(record["response"]["id"])
                record.update(usage_fields(record["response"], generation))
                if generation:
                    record["generation"] = generation
                if record.get("cost_usd") is not None and record.get("response", {}).get("choices"):
                    record["state"] = "received"
                self.save(record)
            if record.get("cost_usd") is None or record["state"] in {"in_flight", "unknown"}:
                raise HaltRun(
                    f"Unreconciled call {record['call_id']}. Resolve its billing before resume; "
                    "it will not be automatically resubmitted."
                )
            self.spent += record["cost_usd"]
            self.check_response_route(record)
        response = await self.http.get(f"models/{self.config['model']}/endpoints")
        response.raise_for_status()
        endpoints = response.json()["data"]["endpoints"]
        endpoint = next((e for e in endpoints if e.get("tag") == self.config["provider"]), None)
        if endpoint is None:
            raise HaltRun("The frozen Standard provider endpoint is unavailable")
        prices = endpoint["pricing"]
        for actual, setting in (
            ("prompt", "input_usd_per_million"),
            ("completion", "output_usd_per_million"),
        ):
            if float(prices[actual]) * 1e6 > self.config[setting] + 1e-9:
                raise HaltRun("Current provider prices exceed frozen budget reservation prices")
        write_json(self.path / "pricing.json", {"checked_at": now(), "endpoint": endpoint})

    def check_response_route(self, record: dict) -> None:
        if record.get("response", {}).get("choices"):
            if record.get("response_model") != self.config["model"]:
                raise HaltRun(f"Unexpected response model for {record['call_id']}")
            if record.get("provider") != "Google AI Studio":
                raise HaltRun(f"Unexpected or missing provider for {record['call_id']}")
            if record.get("service_tier") not in {None, "default", "standard"}:
                raise HaltRun(f"Unexpected service tier for {record['call_id']}")

    def payload(self, messages: list[dict], schema: dict, stage: str, task: str, arm: str) -> dict:
        category = (
            "prepare"
            if stage.startswith("prepare_")
            else ("judge" if stage.startswith("judge_") else "overall")
        )
        reasoning = (
            {"max_tokens": self.config["reasoning_tokens"], "exclude": False}
            if self.config["reasoning_tokens"]
            else {"enabled": False}
        )
        return {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config[f"{category}_temperature"],
            "max_tokens": self.config[f"{category}_max_tokens"],
            "reasoning": reasoning,
            "seed": int(digest([self.config["seed"], task, arm, stage])[:8], 16) % (2**31),
            "service_tier": "default",
            "provider": {
                "only": [self.config["provider"]],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": category, "strict": True, "schema": schema},
            },
            "stream": True,
        }

    def reservation(self, payload: dict) -> float:
        # UTF-8 byte count is deliberately conservative versus text-token estimates.
        # Include schema and generous chat framing, and reserve the full output allowance.
        input_bound = (
            len(encode(payload["messages"]).encode())
            + len(encode(payload["response_format"]).encode())
            + 4096
        )
        return (
            input_bound * self.config["input_usd_per_million"]
            + payload["max_tokens"] * self.config["output_usd_per_million"]
        ) / 1e6

    async def request(self, payload: dict, task: str, arm: str, stage: str, attempt: int) -> dict:
        call_id = digest([self.manifest["run_id"], task, arm, stage, attempt])
        file = self.path / "calls" / f"{call_id}.json"
        request_hash = digest(payload)
        if file.exists():
            record = read_json(file)
            if record["request_sha256"] != request_hash:
                raise HaltRun("Cached request differs from frozen checkpoint")
            if record["state"] in {"unknown", "in_flight"} or record.get("cost_usd") is None:
                raise HaltRun(f"Unreconciled request {call_id}")
            return record
        async with self.semaphore:
            reserve = self.reservation(payload)
            async with self.lock:
                if self.stopping:
                    raise HaltRun("New requests stopped; already-started requests were saved")
                if self.spent + self.reserved + reserve > self.max_cost:
                    self.stopping = True
                    raise HaltRun(
                        "Budget reached: insufficient balance for conservative reservation"
                    )
                self.reserved += reserve
            record = {
                "call_id": call_id,
                "run_id": self.manifest["run_id"],
                "task_id": task,
                "arm": arm,
                "stage": stage,
                "attempt": attempt,
                "state": "in_flight",
                "started_at": now(),
                "request_sha256": request_hash,
                "request": payload,
                "reserved_usd": reserve,
                "cost_usd": None,
                "cost_source": None,
            }
            self.save(record)  # Durable intent before any paid request.
            start = time.monotonic()
            try:
                from .streaming import receive

                status, data = await receive(self, payload, record)
                record.update(
                    response=redact(data, self.key),
                    state="received",
                    duration_seconds=time.monotonic() - start,
                    completed_at=now(),
                )
                self.save(record)  # Save response before parsing/validation/billing lookup.
                fields = usage_fields(data)
                if fields["cost_usd"] is None or not fields["provider"]:
                    if data.get("id"):
                        generation = await self.get_generation(data["id"])
                        record["generation"] = generation
                        fields = usage_fields(data, generation)
                record.update(fields)
                # Explicit pre-generation rejections contain no generation ID or completion.
                # Keep their non-provider-reported accounting source visible in reports.
                if (
                    status in {400, 401, 402, 403, 422, 429}
                    and data.get("error")
                    and not data.get("id")
                    and not data.get("choices")
                    and record["cost_usd"] is None
                ):
                    record.update(
                        state="rejected", cost_usd=0.0, cost_source="pre_generation_http_rejection"
                    )
                if record["cost_usd"] is None:
                    self.stopping = True
                    record["state"] = "unknown"
                    raise HaltRun(f"Missing billing information for {call_id}; stopping paid calls")
                if not 200 <= status < 300:
                    record["state"] = "rejected" if record["state"] == "rejected" else "api_error"
                    if status != 429:
                        self.stopping = True
                        raise HaltRun(
                            f"OpenRouter HTTP {status}; inspect call {call_id}"
                        )
                if 200 <= status < 300:
                    try:
                        self.check_response_route(record)
                    except HaltRun:
                        self.stopping = True
                        raise
                return record
            except (httpx.HTTPError, OSError) as exc:
                self.stopping = True
                record.update(
                    state="unknown",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    duration_seconds=time.monotonic() - start,
                )
                raise HaltRun(
                    f"Network error; billing uncertain for {call_id}; automatic replay disabled"
                ) from None
            finally:
                self.save(record)
                async with self.lock:
                    self.reserved -= reserve
                    if record.get("cost_usd") is not None:
                        self.spent += record["cost_usd"]
                        if self.spent > self.max_cost:
                            self.stopping = True

    async def complete(
        self,
        task: str,
        arm: str,
        stage: str,
        system: str,
        user: dict | str,
        schema: dict,
        inp: dict,
    ) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user if isinstance(user, str) else encode(user)},
        ]
        for attempt in range(1, self.config["max_attempts"] + 1):
            payload = self.payload(messages, schema, stage, task, arm)
            record = await self.request(payload, task, arm, stage, attempt)
            if record["state"] == "abandoned":
                error = "Previous response was lost; billing was explicitly reconciled"
            elif record["state"] == "rejected":
                error = f"OpenRouter rejected request (HTTP {record['http_status']})"
                if record["http_status"] != 429:
                    raise HaltRun(error)
                await asyncio.sleep(min(2**attempt, 15))
            else:
                try:
                    choice = record["response"]["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError(f"Non-complete response: {choice.get('finish_reason')}")
                    value = json.loads(choice["message"]["content"])
                    validate_output(
                        "judge" if stage.startswith("judge_") else stage, value, schema, inp
                    )
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    error = str(exc)[:500]
                    record.update(state="invalid", validation_error=error)
                else:
                    record.update(state="validated", output=value)
                    self.save(record)
                    return value
                self.save(record)
            # Same retry policy in both arms; feedback never includes reference labels.
            messages = messages[:2] + [
                {
                    "role": "user",
                    "content": (
                        "Your previous attempt did not satisfy the required output contract: "
                        + error
                        + ". Return a complete valid JSON object satisfying the schema."
                    ),
                }
            ]
        raise StageFailure(f"{stage}: exhausted {self.config['max_attempts']} attempts")


async def evaluate_arm(client: OpenRouterClient, entry: dict, arm: str, prompts: dict) -> dict:
    from .aj_bridge import evaluate_using_aj

    return await evaluate_using_aj(client, entry, arm, prompts)


def check_pilot(path: Path, protocol: dict, pilot_path: Path | None) -> None:
    if pilot_path is None:
        raise HaltRun("Main execution requires --pilot-dir pointing to a completed technical pilot")
    pilot, pilot_protocol, _ = load_run(pilot_path)
    if pilot["phase"] != "pilot" or len(pilot["entries"]) != 6:
        raise HaltRun("Expected the six-trace technical pilot")
    if digest(pilot_protocol) != digest(protocol):
        raise HaltRun(
            "Pilot prompts/code/settings differ; repeat the technical pilot with frozen settings"
        )
    for entry in pilot["entries"]:
        for arm in ARMS:
            result_path = pilot_path / "results" / arm / f"{entry['task_id']}.json"
            if not result_path.exists():
                raise HaltRun("Technical pilot is incomplete")
            result = read_json(result_path)
            if (
                result.get("run_id") != pilot["run_id"]
                or result.get("status") != "success"
                or result.get("input_sha256") != entry["input_sha256"]
            ):
                raise HaltRun("Technical pilot contains failed or mismatched results")
    records = call_records(pilot_path, pilot["run_id"])
    if not records or any(
        r.get("cost_usd") is None or r["state"] in {"in_flight", "unknown"} for r in records
    ):
        raise HaltRun("Resolve technical-pilot accounting before main execution")
    write_json(
        path / "pilot_link.json",
        {
            "path": str(pilot_path.resolve()),
            "run_id": pilot["run_id"],
            "known_cost_usd": sum(r["cost_usd"] for r in records),
        },
    )


async def run(
    path: Path,
    api_key: str,
    max_cost: float,
    transport: httpx.AsyncBaseTransport | None = None,
    pilot_path: Path | None = None,
) -> None:
    import fcntl

    manifest, protocol, prompts = load_run(path)
    if manifest["phase"] == "main":
        check_pilot(path, protocol, pilot_path)
    with (path / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HaltRun("Another process is already running this experiment directory") from None
        client = OpenRouterClient(path, manifest, protocol, api_key, max_cost, transport)
        try:
            await client.preflight()
            write_json(path / "last_session.json", {"started_at": now(), "max_cost_usd": max_cost})
            for i, entry in enumerate(manifest["entries"], 1):
                task = entry["task_id"]
                # Balance arm order deterministically to reduce systematic order/cache effects.
                arms = (
                    ARMS
                    if int(digest([protocol["config"]["seed"], task])[:8], 16) % 2
                    else ARMS[::-1]
                )
                for arm in arms:
                    output = path / "results" / arm / f"{task}.json"
                    if output.exists():
                        result = read_json(output)
                        if (
                            result["run_id"] != manifest["run_id"]
                            or result["input_sha256"] != entry["input_sha256"]
                        ):
                            raise HaltRun("Result identity differs from the frozen manifest")
                        continue
                    try:
                        result = await evaluate_arm(client, entry, arm, prompts)
                    except StageFailure as exc:
                        result = {"status": "failed", "error": str(exc)}
                    result.update(
                        run_id=manifest["run_id"],
                        input_sha256=entry["input_sha256"],
                        task_id=task,
                        arm=arm,
                        completed_at=now(),
                    )
                    write_json(output, result)
                    print(
                        f"[{i}/{len(manifest['entries'])}] {arm} {task}: {result['status']}; "
                        f"cumulative ${client.spent:.4f}",
                        flush=True,
                    )
        finally:
            await client.close()


def resolve_lost_call(path: Path, call_id: str, cost: float, note: str) -> None:
    """Explicit manual reconciliation; never silently assumes that a lost response was free."""
    import fcntl

    if not math.isfinite(cost) or cost < 0 or not note.strip():
        raise ValueError("Supply a nonnegative verified charge and an audit note")
    manifest, _, _ = load_run(path)
    with (path / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = next(
            (r for r in call_records(path, manifest["run_id"]) if r["call_id"] == call_id), None
        )
        if record is None or record["state"] not in {"in_flight", "unknown"}:
            raise ValueError("Only unresolved/lost calls can be manually reconciled")
        record.update(
            cost_usd=cost,
            cost_source="manual_reconciliation",
            reconciled_at=now(),
            reconciliation_note=note,
            state="received" if record.get("response", {}).get("choices") else "abandoned",
        )
        write_json(path / "calls" / f"{call_id}.json", record)
