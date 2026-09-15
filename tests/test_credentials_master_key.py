"""Offline check: DPAPI on Windows unchanged; master-key mode works in simulated container."""
import json
import sys

sys.path.insert(0, r'C:\Users\r-ber\OneDrive\Documents\AutoJudge\src\autojudge\ui\backend')
import credentials

info = credentials.storage_info()
print('win storage:', info['name'])
assert info['name'] == 'Windows DPAPI'

SRC = credentials.__file__
with open(SRC, encoding='utf-8') as f:
    src = f.read()
# Simulate linux container: flip win32 branches.
simulated = src.replace("sys.platform == 'win32'", 'False').replace("sys.platform != 'win32'", 'True')
ns = {'__name__': 'container_credentials'}
exec(compile(simulated, 'cc', 'exec'), ns)

ns['os'].environ['AUTOJUDGE_CREDENTIALS_KEY'] = 'test-master-secret-123'
print('container storage:', ns['storage_info']())
assert ns['storage_info']()['available']

secret = 'sk-or-v1-testvalue-000000'
payload = ns['store']('OPENROUTER_API_KEY', secret)
print('payload:', json.dumps(payload))
assert payload['storage'] == 'master-key'
assert secret not in json.dumps(payload), 'plaintext leaked!'

loaded = ns['load']('OPENROUTER_API_KEY', payload)
print('roundtrip ok:', loaded == secret)
assert loaded == secret

ns['os'].environ['AUTOJUDGE_CREDENTIALS_KEY'] = 'wrong-key'
try:
    ns['load']('OPENROUTER_API_KEY', payload)
    print('wrong key: DECRYPTED (BAD)')
    raise SystemExit(1)
except RuntimeError:
    print('wrong key: rejected (ok)')

# no master key and no OS keyring -> honest refusal (dev Windows has a keyring,
# so simulate the bare container by removing that backend too)
del ns['os'].environ['AUTOJUDGE_CREDENTIALS_KEY']
ns['storage_info'] = lambda: {'name': 'Environment only', 'available': False, 'persistent': False}
try:
    ns['store']('OPENROUTER_API_KEY', secret)
    print('no key: stored plaintext (BAD)')
    raise SystemExit(1)
except RuntimeError:
    print('no master key: refused (ok)')

print('ALL CHECKS PASSED')
