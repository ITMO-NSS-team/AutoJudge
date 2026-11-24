"""Transport and response validation for OpenRouter Decisions calls.

The evaluator passes only typed questions and evidence to this module. Every
HTTP attempt is recorded because a retry can affect latency and cost.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from autojudge.model_connection import decision_connection

JEV_MODEL = "typesafe/jev-1.14"


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]


class DecisionUsage(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class DecisionEnvelope(BaseModel):
    model: str
    answers: dict[str, ChoiceAnswer]
    usage: DecisionUsage | None = None
    id: str | None = None


def _receipt_cost(payload: Any) -> float | None:
    """Recover a billing receipt even when the answer itself is malformed."""
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None
    raw_cost = payload["usage"].get("cost")
    if isinstance(raw_cost, bool):
        return None
    try:
        cost = float(raw_cost)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


class JevClient:
    """A small Decisions adapter with bounded retries and an attempt ledger."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str = JEV_MODEL,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
        max_retry_delay_seconds: float = 120.0,
        max_concurrent_requests: int = 4,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be finite and nonnegative")
        if not math.isfinite(max_retry_delay_seconds) or max_retry_delay_seconds < 0:
            raise ValueError("max_retry_delay_seconds must be finite and nonnegative")
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        self.client = client
        self.model = model
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.request_slots = asyncio.Semaphore(max_concurrent_requests)
        self._cooldown_until = 0.0
        self.endpoint, self.api_key = decision_connection()

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            try:
                date = parsedate_to_datetime(raw)
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        return seconds if math.isfinite(seconds) and seconds >= 0 else None

    def _retry_delay(self, attempt_number: int, response: httpx.Response | None) -> float:
        try:
            delay = math.ldexp(self.retry_delay_seconds, min(attempt_number - 1, 60))
        except OverflowError:
            delay = self.max_retry_delay_seconds
        if response is not None and response.status_code in {429, 503}:
            retry_after = self._retry_after_seconds(response)
            if retry_after is not None:
                delay = max(delay, retry_after)
        return min(delay, self.max_retry_delay_seconds)

    async def _wait_for_cooldown(self) -> None:
        while remaining := self._cooldown_until - perf_counter():
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    @staticmethod
    def _validate_questions(questions: dict[str, dict[str, Any]]) -> None:
        if not questions:
            raise ValueError("At least one decision question is required")
        for question_id, question in questions.items():
            criteria = question.get("criteria")
            if question.get("type") != "choice" or not isinstance(criteria, dict):
                raise ValueError(f"{question_id}: only choice questions are supported")
            if not 2 <= len(criteria) <= 255:
                raise ValueError(f"{question_id}: Jev requires 2 to 255 choices")
            if not question.get("instructions"):
                raise ValueError(f"{question_id}: instructions are empty")

    def _validate_response(
        self, payload: Any, questions: dict[str, dict[str, Any]]
    ) -> DecisionEnvelope:
        envelope = DecisionEnvelope.model_validate(payload)
        if envelope.model != self.model and not envelope.model.startswith(self.model + "-"):
            raise ValueError(f"Unexpected Decisions model: {envelope.model}")
        if set(envelope.answers) != set(questions):
            raise ValueError("Decisions answer keys do not match submitted question keys")
        for question_id, answer in envelope.answers.items():
            choices = set(questions[question_id]["criteria"])
            if answer.choice not in choices or set(answer.probabilities) != choices:
                raise ValueError(f"{question_id}: response choices do not match the request")
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in answer.probabilities.values()):
                raise ValueError(f"{question_id}: invalid probability")
        return envelope

    async def decide(
        self,
        *,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        stage: str,
    ) -> dict[str, Any]:
        """Return answers and a ledger, including failures and unknown costs."""
        self._validate_questions(questions)
        attempts: list[dict[str, Any]] = []
        started = perf_counter()
        envelope: DecisionEnvelope | None = None
        final_error: str | None = None

        for attempt_number in range(1, self.max_attempts + 1):
            attempt_started = perf_counter()
            response: httpx.Response | None = None
            payload: Any = None
            try:
                async with self.request_slots:
                    await self._wait_for_cooldown()
                    response = await self.client.post(
                        self.endpoint,
                        json={"model": self.model, "state": state, "questions": questions},
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    if response.status_code in {429, 503}:
                        self._cooldown_until = max(
                            self._cooldown_until,
                            perf_counter() + self._retry_delay(attempt_number, response),
                        )
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                response.raise_for_status()
                envelope = self._validate_response(payload, questions)
                attempts.append(
                    {
                        "number": attempt_number,
                        "status_code": response.status_code,
                        "elapsed_seconds": perf_counter() - attempt_started,
                        "cost_usd": _receipt_cost(payload),
                        "error": None,
                    }
                )
                break
            except Exception as exc:
                # Keep a receipt for unexpected transport/serialization errors
                # too. Cancellation and process interrupts still propagate.
                final_error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {
                        "number": attempt_number,
                        "status_code": response.status_code if response is not None else None,
                        "elapsed_seconds": perf_counter() - attempt_started,
                        "cost_usd": _receipt_cost(payload),
                        "error": final_error,
                    }
                )
                status = response.status_code if response is not None else None
                retryable = (
                    status is None or status in {200, 408, 425, 429} or status >= 500
                )
                if not retryable or attempt_number == self.max_attempts:
                    break
                delay = self._retry_delay(attempt_number, response)
                attempts[-1]["retry_delay_seconds"] = delay
                await asyncio.sleep(delay)

        # An HTTP error or timeout without a receipt may still have incurred a
        # charge. Represent that uncertainty explicitly instead of reporting $0.
        cost_complete = all(attempt["cost_usd"] is not None for attempt in attempts)
        recorded_cost = sum(attempt["cost_usd"] or 0.0 for attempt in attempts)
        return {
            "stage": stage,
            "question_ids": list(questions),
            "status": "ok" if envelope is not None else "error",
            "answers": (
                {key: value.model_dump(mode="json") for key, value in envelope.answers.items()}
                if envelope is not None
                else {}
            ),
            "model": envelope.model if envelope is not None else self.model,
            "response_id": envelope.id if envelope is not None else None,
            "input_tokens": (
                envelope.usage.input_tokens if envelope and envelope.usage else None
            ),
            "output_tokens": (
                envelope.usage.output_tokens if envelope and envelope.usage else None
            ),
            "attempts": attempts,
            "recorded_cost_usd": recorded_cost,
            "cost_usd": recorded_cost if cost_complete else None,
            "cost_complete": cost_complete,
            "elapsed_seconds": perf_counter() - started,
            "error": None if envelope is not None else final_error,
        }
