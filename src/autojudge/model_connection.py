"""Resolve model endpoints without sharing credentials across services."""

import os
import ssl
from urllib.parse import urlsplit

import httpx
import truststore
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


def omniroute_base_url() -> str:
    base = os.getenv("OMNIROUTE_BASE_URL", "").strip().rstrip("/")
    if base and not urlsplit(base).path.endswith("/v1"):
        return base + "/v1"
    return base


def chat_api_key() -> str:
    return required_env("OMNIROUTE_API_KEY" if omniroute_base_url() else "OPENROUTER_API_KEY")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before running inference")
    return value


def model_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        timeout=httpx.Timeout(120, connect=15),
    )


def chat_provider(api_key: str) -> OpenAIProvider | OpenRouterProvider:
    base = omniroute_base_url()
    if base:
        return OpenAIProvider(base_url=base, api_key=api_key, http_client=model_http_client())
    return OpenRouterProvider(api_key=api_key)


def decision_connection() -> tuple[str, str]:
    direct = os.getenv("OPENROUTER_DECISIONS_URL", "").strip()
    if direct:
        if direct != OPENROUTER_DECISIONS_URL:
            raise ValueError("OPENROUTER_DECISIONS_URL must use the official OpenRouter endpoint")
        if os.getenv("OMNIROUTE_DECISIONS_URL", "").strip():
            raise ValueError("Set only one of OPENROUTER_DECISIONS_URL and OMNIROUTE_DECISIONS_URL")
        return direct, required_env("OPENROUTER_API_KEY")

    base = omniroute_base_url()
    if not base:
        return OPENROUTER_DECISIONS_URL, required_env("OPENROUTER_API_KEY")

    url = required_env("OMNIROUTE_DECISIONS_URL")
    gateway, endpoint = urlsplit(base), urlsplit(url)
    if (
        (endpoint.scheme, endpoint.hostname, endpoint.port)
        != (gateway.scheme, gateway.hostname, gateway.port)
        or endpoint.username is not None
        or endpoint.password is not None
    ):
        raise ValueError("OMNIROUTE_DECISIONS_URL must use the same origin as OMNIROUTE_BASE_URL")
    return url, required_env("OMNIROUTE_API_KEY")
