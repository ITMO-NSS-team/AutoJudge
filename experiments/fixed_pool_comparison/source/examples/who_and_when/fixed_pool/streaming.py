"""Durably retain SSE wire events and assemble the JSON consumed by AJ."""

from __future__ import annotations

import json
import os
import time

import httpx


def merge_chunk(result: dict, chunk: dict) -> None:
    for name, value in chunk.items():
        if name != "choices":
            result[name] = value
    choices = result.setdefault("choices", [])
    for choice in chunk.get("choices", []):
        index = choice.get("index", 0)
        target = next((c for c in choices if c.get("index") == index), None)
        if target is None:
            target = {"index": index, "message": {}}
            choices.append(target)
        for name, value in choice.items():
            if name not in {"delta", "message"} and value is not None:
                target[name] = value
        for name, value in (choice.get("delta") or choice.get("message") or {}).items():
            message = target["message"]
            if name in {"content", "reasoning", "refusal"} and isinstance(value, str):
                message[name] = (message.get(name) or "") + value
            elif name == "reasoning_details" and isinstance(value, list):
                message.setdefault(name, []).extend(value)
            elif value is not None:
                message[name] = value


async def receive(client, payload: dict, record: dict) -> tuple[int, dict]:
    # Import here to avoid a module-level runner/streaming cycle.
    from .runner import redact

    async with client.http.stream("POST", "chat/completions", json=payload) as response:
        record["http_status"] = response.status_code
        generation = response.headers.get("x-generation-id")
        if generation:
            record["response"] = {"id": generation}
        client.save(record)  # Preserve the generation ID before waiting for the body.
        if "text/event-stream" not in response.headers.get("content-type", ""):
            raw = await response.aread()
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    data = {"unexpected_response": data}
            except ValueError:
                data = {"unparsed_response": raw.decode("utf-8", errors="replace")}
            return response.status_code, data

        file = client.path / "streams" / f"{record['call_id']}.sse"
        file.parent.mkdir(parents=True, exist_ok=True)
        record.update(
            stream_file=str(file.relative_to(client.path)),
            stream_done=False,
            response_assembled_from_stream=True,
        )
        result = record.setdefault("response", {})
        last_save = time.monotonic()
        fields = []

        def consume():
            nonlocal last_save
            if not fields:
                return
            data = "\n".join(fields)
            fields.clear()
            if data == "[DONE]":
                record["stream_done"] = True
            else:
                try:
                    chunk = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise ValueError("SSE event is not a JSON object")
                except ValueError as exc:
                    record["stream_parse_error"] = str(exc)
                    client.save(record)
                    raise httpx.RemoteProtocolError(
                        "Invalid SSE event; raw stream retained"
                    ) from exc
                merge_chunk(result, redact(chunk, client.key))
            if time.monotonic() - last_save > 0.5 or record["stream_done"]:
                client.save(record)
                last_save = time.monotonic()

        try:
            with file.open("w", encoding="utf-8", buffering=1) as wire:
                async for line in response.aiter_lines():
                    wire.write(redact(line, client.key) + "\n")
                    if not line:
                        consume()
                        wire.flush()
                        os.fsync(wire.fileno())
                    elif line.startswith("data:"):
                        fields.append(line[5:].removeprefix(" "))
                consume()
            if not record["stream_done"]:
                raise httpx.RemoteProtocolError(
                    "SSE ended before [DONE]; partial response retained"
                )
            return response.status_code, result
        finally:
            client.save(record)
