"""Allowlisted environment settings; never serialize secret values."""
import math
import os
from dotenv import dotenv_values

# Template defaults are suggestions; runtime defaults are documented explicitly.
FIELDS = [
    ('LLM_BASE_URL', 'AI connection', 'url', 'https://openrouter.ai/api/v1'),
    ('LLM_API_KEY', 'AI connection', 'secret', ''),
    ('OPENROUTER_API_KEY', 'API keys', 'secret', ''),
    ('HF_TOKEN', 'API keys', 'secret', ''),
    ('GITHUB_TOKEN', 'API keys', 'secret', ''),
    ('E2B_API_KEY', 'API keys', 'secret', ''),
    ('DEFAULT_META_MODEL', 'Models', 'text', 'anthropic/claude-sonnet-4'),
    ('AGENT_NODE_MODEL', 'Models', 'text', 'google/gemini-2.5-flash'),
    ('AGENT_NODE_TEMPERATURE', 'Models', 'temperature', '0.1'),
    ('POOL_GEN_MODEL', 'Models', 'text', 'deepseek/deepseek-v4-pro'),
    ('POOL_GEN_TEMPERATURE', 'Models', 'temperature', '0.3'),
    ('GRAPH_GEN_MODEL', 'Models', 'text', 'google/gemini-2.5-flash'),
    ('AUDIO_MODEL', 'MCP models', 'text', 'google/gemini-2.5-flash'),
    ('IMAGE_MODEL', 'MCP models', 'text', 'google/gemini-2.5-flash'),
    ('VIDEO_MODEL', 'MCP models', 'text', 'google/gemini-2.5-flash'),
    ('BROWSERUSE_MODEL', 'MCP models', 'text', 'google/gemini-2.5-flash'),
    ('MASEVAL_DEFAULT_MODEL', 'MCP models', 'text', 'google/gemini-2.5-flash'),
    ('LANGFUSE_PUBLIC_KEY', 'Langfuse', 'secret', ''),
    ('LANGFUSE_SECRET_KEY', 'Langfuse', 'secret', ''),
    ('LANGFUSE_HOST', 'Langfuse', 'url', ''),
    ('DB_NAME', 'PostgreSQL', 'text', 'maseval'),
    ('DB_USER', 'PostgreSQL', 'text', 'postgres'),
    ('DB_PASSWORD', 'PostgreSQL', 'secret', ''),
    ('DB_HOST', 'PostgreSQL', 'text', 'localhost'),
    ('DB_PORT', 'PostgreSQL', 'port', '5432'),
]


def resolve(path, stored, legacy):
    values = dotenv_values(path, interpolate=False) if path.exists() else {}
    result = []
    for name, group, kind, default in FIELDS:
        value = os.environ.get(name, values.get(name) or default)
        source = 'process' if name in os.environ else '.env' if values.get(name) else 'default'
        if name in stored:
            source = 'saved'
            value = stored[name].get('value', '') if kind != 'secret' else stored[name].get('ciphertext', '')
        elif name == 'OPENROUTER_API_KEY' and legacy:
            source, value = 'saved', 'encrypted'
        result.append({'name':name, 'group':group, 'kind':kind, 'source':source,
                       'configured':bool(value), 'value':None if kind=='secret' else value,
                       'overridden':name in stored})
    return result


def validate(name, value):
    kind = next(f[2] for f in FIELDS if f[0] == name)
    if not isinstance(value, str) or len(value) > 4096 or '\n' in value or '\r' in value:
        raise ValueError('Invalid setting')
    if kind == 'temperature' and (not math.isfinite(float(value)) or not 0 <= float(value) <= 2):
        raise ValueError('Temperature must be between 0 and 2')
    if kind == 'port' and (not value.isdecimal() or not 1 <= int(value) <= 65535):
        raise ValueError('Port must be between 1 and 65535')
    if kind == 'url' and value:
        from urllib.parse import urlparse
        parsed = urlparse(value)
        if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use HTTP(S) URL without credentials or query')
    if name == 'LLM_BASE_URL':
        from urllib.parse import urlparse
        parsed = urlparse(value)
        if not value or (parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1')):
            raise ValueError('Use HTTPS, or HTTP on localhost only')
    return kind
