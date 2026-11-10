"""Credential-free launcher source; the API key is accepted through hidden stdin only."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import termios
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.who_and_when.fixed_pool.archive import export_archive
from examples.who_and_when.fixed_pool.cli import main
from examples.who_and_when.fixed_pool.protocol import load_run, write_json

RUN_ROOT = ROOT / 'examples/who_and_when/results/fixed_pool_comparison'
CAMPAIGN = RUN_ROOT / 'campaign_20261109_01'
PILOT = RUN_ROOT / 'pilot_20261109_01'
MAIN = RUN_ROOT / 'main_20261109_01'
ARTIFACTS = ROOT / 'artifacts/fixed_pool_comparison'
GLOBAL_CAP = Decimal('25')
KEY_PATTERN = re.compile(rb'sk-or-v1-[0-9a-fA-F]{64}')


def now():
    return datetime.now(timezone.utc).isoformat()


def ledger(path):
    return [json.loads(f.read_text()) for f in (path / 'calls').glob('*.json')]


def known_cost(records):
    return sum((Decimal(str(r['cost_usd'])) for r in records if r.get('cost_usd') is not None), Decimal(0))


def read_key():
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    quiet = termios.tcgetattr(fd)
    quiet[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, quiet)
    try:
        print('KEY_INPUT_READY', flush=True)
        return sys.stdin.readline().strip()
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, previous)


class RedactingWriter:
    def __init__(self, stream, key):
        self.stream, self.key = stream, key
    def write(self, value):
        return self.stream.write(value.replace(self.key, '[REDACTED]'))
    def flush(self):
        return self.stream.flush()
    def isatty(self):
        return False


def run_stage(path, cap, key, pilot=None):
    arguments = ['run', '--output-dir', str(path), '--max-cost-usd', str(cap)]
    if pilot is not None:
        arguments.extend(['--pilot-dir', str(pilot)])
    with (path / 'console.log').open('a', buffering=1) as stream:
        writer = RedactingWriter(stream, key)
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            try:
                code = main(arguments)
            except BaseException:
                traceback.print_exc()
                code = 1
    archive_path = ARTIFACTS / f'{path.name}.zip'
    audit = export_archive(path, archive_path)
    print(json.dumps({'stage': path.name, 'exit_code': code, 'audit': audit}), flush=True)
    return code, audit


def bundle(state):
    target = ARTIFACTS / 'fixed_pool_comparison.zip'
    sums = {}
    archives = [ARTIFACTS / 'smoke_20261109_complete_with_history.zip']
    archives += [ARTIFACTS / f'{p.name}.zip' for p in (PILOT, MAIN) if (ARTIFACTS / f'{p.name}.zip').exists()]
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for source in archives:
            with zipfile.ZipFile(source) as original:
                assert original.testzip() is None
                for name in original.namelist():
                    data = original.read(name)
                    if KEY_PATTERN.search(data):
                        raise ValueError(f'Credential-like content in {source.name}/{name}')
                    prefix = 'smoke_history' if source.name.startswith('smoke_') else source.stem
                    filename = prefix + '/' + name
                    out.writestr(filename, data)
                    sums[filename] = hashlib.sha256(data).hexdigest()
        for file in (CAMPAIGN / 'launcher.py', CAMPAIGN / 'status.json'):
            data = file.read_bytes()
            if KEY_PATTERN.search(data):
                raise ValueError(f'Credential-like content in {file.name}')
            filename = 'campaign/' + file.name
            out.writestr(filename, data)
            sums[filename] = hashlib.sha256(data).hexdigest()
        readme = ('# Controlled fixed-pool comparison\n\n'
                  'Start with main_20261109_01/experiment/report.md and artifact_audit.json.\n'
                  'pilot_20261109_01 contains the independent technical pilot.\n'
                  'smoke_history preserves the initial two-trace test and transport failures.\n'
                  'campaign/status.json records launch status, actual known expense and conservative reservations for earlier unresolved calls.\n'
                  'Unknown charges are not zero. Output token counts include reasoning. Equal call counts do not establish strict budget matching.\n'
                  'Each run contains original requests, SSE events, assembled responses, intermediate findings, native AJ histories, frozen data/prompts/source, and SHA256 checksums. Credentials and .env are excluded.\n').encode()
        out.writestr('README.md', readme)
        sums['README.md'] = hashlib.sha256(readme).hexdigest()
        out.writestr('SHA256SUMS.json', json.dumps(sums, indent=2))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Bundle CRC verification failed')
        for name, expected in sums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Bundle checksum mismatch: {name}')
    return target


def execute(key):
    os.environ['OPENROUTER_API_KEY'] = key
    old = [r for i in range(1, 5) for r in ledger(RUN_ROOT / f'smoke_20261109_0{i}')]
    legacy_known = known_cost(old)
    legacy_reserve = sum((Decimal(str(r['reserved_usd'])) for r in old if r.get('cost_usd') is None), Decimal(0))
    state = {'started_at': now(), 'pid': os.getpid(), 'state': 'pilot_running',
             'pilot_dir': str(PILOT), 'main_dir': str(MAIN), 'global_cap_usd': float(GLOBAL_CAP),
             'legacy_known_cost_usd': float(legacy_known), 'legacy_unknown_reservation_usd': float(legacy_reserve),
             'legacy_unknown_call_ids': [r['call_id'] for r in old if r.get('cost_usd') is None],
             'pilot_cap_usd': 4.0}
    def save():
        state['updated_at'] = now()
        write_json(CAMPAIGN / 'status.json', state)
    save()
    print('PILOT_STARTED cap=4.00; global experiment cap=25.00', flush=True)
    code, audit = run_stage(PILOT, Decimal('4'), key)
    state['pilot_exit_code'] = code
    state['pilot_known_cost_usd'] = float(known_cost(ledger(PILOT)))
    if code or not audit['storage_ok'] or not audit['execution_complete'] or any(t['status'] != 'success' for t in audit['traces']):
        state['state'] = 'stopped_at_pilot'
        save()
        state['archive_path'] = str(bundle(state))
        save()
        return 2
    pilot_cost = known_cost(ledger(PILOT))
    available = (GLOBAL_CAP - legacy_known - legacy_reserve - pilot_cost).quantize(Decimal('0.000001'), rounding=ROUND_DOWN)
    if available <= 0:
        raise ValueError('Global expense cap leaves no main-run allowance')
    state.update(state='main_running', main_cap_usd=float(available), main_started_at=now())
    save()
    print(f'MAIN_STARTED traces=126; cap={available}', flush=True)
    code, audit = run_stage(MAIN, available, key, PILOT)
    state.update(main_exit_code=code, main_known_cost_usd=float(known_cost(ledger(MAIN))),
                 state='completed' if not code and audit['storage_ok'] and audit['execution_complete'] else 'stopped_at_main',
                 finished_at=now())
    state['known_cost_usd_all_attempts'] = float(legacy_known + pilot_cost + known_cost(ledger(MAIN)))
    state['archive_path'] = str(ARTIFACTS / 'fixed_pool_comparison.zip')
    save()
    bundle(state)
    print(json.dumps(state), flush=True)
    return code


if __name__ == '__main__':
    key = read_key()
    if not key:
        raise SystemExit('Missing credential')
    try:
        result = execute(key)
    except BaseException as exc:
        message = f'{type(exc).__name__}: {exc}'.replace(key, '[REDACTED]')
        print('CAMPAIGN_ERROR ' + message, flush=True)
        raise SystemExit(1)
    print(f'CAMPAIGN_COMPLETE exit={result}', flush=True)
    raise SystemExit(result)
