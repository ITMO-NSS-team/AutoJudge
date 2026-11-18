"""Read-only monitoring of Table 3 progress and transport errors."""
import collections
import datetime
import json
import time
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'results/table3-reconstruction-20261118-olama-v3'
while True:
    counts = collections.Counter()
    states = collections.Counter()
    known = reserve = 0.0
    latest = 0.0
    for file in root.glob('*/results/autojudge/*.json'):
        data = json.loads(file.read_text())
        counts[data['status']] += 1
        latest = max(latest, file.stat().st_mtime)
    for file in root.glob('*/calls/*.json'):
        data = json.loads(file.read_text())
        states[data.get('state')] += 1
        cost = data.get('cost_usd')
        if cost is not None:
            if data.get('cost_source') == 'reserved_upper_bound_not_actual':
                reserve += cost
            else:
                known += cost
    snapshot = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), results=dict(counts), calls=dict(states), known_cost_usd=known, uncertain_reserve_usd=reserve, budget_debit_usd=known+reserve, seconds_since_last_result=round(time.time()-latest) if latest else None)
    with (root / 'monitor.jsonl').open('a') as stream:
        stream.write(json.dumps(snapshot)+'\n')
    temp = root / 'monitor-status.tmp'
    temp.write_text(json.dumps(snapshot, indent=2)+'\n')
    temp.replace(root / 'monitor-status.json')
    if sum(counts.values()) == 348:
        break
    time.sleep(30)
