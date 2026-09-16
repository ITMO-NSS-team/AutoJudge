"""Allowlisted environment settings; never serialize secret values."""
import math
import os
import re
from dotenv import dotenv_values

# Template defaults are suggestions; runtime defaults are documented explicitly.
FIELDS = [
    ('OPENROUTER_API_KEY', 'API keys', 'secret', ''),
    ('AGENT_NODE_MODEL', 'Models', 'text', 'google/gemini-2.5-flash'),
    ('AGENT_NODE_TEMPERATURE', 'Models', 'temperature', '0'),
    # Meta Agent that generates the judge pool; kept separate from the judge
    # execution model above.
    ('META_AGENT_MODEL', 'Models', 'text', 'google/gemini-2.5-flash'),
    ('ENDPOINT_API_URL', 'Endpoint', 'url', 'https://openrouter.ai/api/v1'),
]

MODEL_ID = re.compile(r'^[\x21-\x7e]{1,200}$')


def valid_model_id(value):
    return isinstance(value, str) and bool(MODEL_ID.fullmatch(value))


def resolve(path, stored, legacy):
    values = dotenv_values(path, interpolate=False) if path.exists() else {}
    result = []
    for name, group, kind, default in FIELDS:
        value = os.environ.get(name, values.get(name) or default)
        source = 'process' if name in os.environ else '.env' if values.get(name) else 'default'
        if name in stored:
            source = 'saved'
            value = stored[name].get('value', '') if kind != 'secret' else (
                stored[name].get('ciphertext') or stored[name].get('keyring') or '')
        elif name == 'OPENROUTER_API_KEY' and legacy:
            source, value = 'saved', 'encrypted'
        if name.endswith('_MODEL') and not valid_model_id(value):
            value = values.get(name) or default
            source = '.env' if values.get(name) else 'default'
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
    if name.endswith('_MODEL') and not valid_model_id(value):
        raise ValueError('Model ID must contain printable ASCII characters without spaces')
    if kind == 'port' and (not value.isdecimal() or not 1 <= int(value) <= 65535):
        raise ValueError('Port must be between 1 and 65535')
    if kind == 'url' and value:
        from urllib.parse import urlparse
        parsed = urlparse(value)
        if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use HTTP(S) URL without credentials or query')
    return kind
