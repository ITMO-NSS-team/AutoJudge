"""Read-only checks every five minutes for forty minutes."""
import collections
import datetime
import json
import time
from pathlib import Path

root=Path(__file__).resolve().parents[2]/'results/table3-reconstruction-20261118-olama-v3'
start=time.monotonic()
log=root/('watch40-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.jsonl')
for minute in range(0,41,5):
    time.sleep(max(0,start+minute*60-time.monotonic()))
    now=time.time();results=collections.Counter();recent_results=collections.Counter();calls=collections.Counter();errors=collections.Counter();known=reserve=0.;latest=0
    for file in root.glob('*/results/autojudge/*.json'):
        data=json.loads(file.read_text());results[data['status']]+=1
        completed=datetime.datetime.fromisoformat(data['completed_at'].replace('Z','+00:00')).timestamp();latest=max(latest,completed)
        if completed>now-300:recent_results[data['status']]+=1
    for file in root.glob('*/calls/*.json'):
        data=json.loads(file.read_text())
        if file.stat().st_mtime>now-300:
            calls[data.get('state')]+=1
            if data.get('error_type'):errors[data['error_type']]+=1
        cost=data.get('cost_usd')
        if cost is not None:
            if data.get('cost_source')=='reserved_upper_bound_not_actual':reserve+=cost
            else:known+=cost
    processes=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if command.startswith('/home/alina/miniconda3/envs/olama/bin/python ') and 'table3_parallel_conditions.py run' in command:processes.append(int(proc.name))
        except OSError:pass
    status=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).isoformat(),minute=minute,results=dict(results),last5min_results=dict(recent_results),last5min_calls=dict(calls),last5min_errors=dict(errors),known_cost_usd=round(known,6),uncertain_reserve_usd=round(reserve,6),runner_pids=processes,seconds_since_last_result=round(now-latest))
    with log.open('a') as stream:stream.write(json.dumps(status)+'\n')
    print(json.dumps(status),flush=True)
print('Forty-minute monitoring complete. Log: '+str(log),flush=True)
