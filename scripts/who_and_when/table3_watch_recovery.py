"""Read-only recovery monitoring every three minutes until completion or runner exit."""
import collections, datetime, json, time
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'results/table3-reconstruction-20261118-recovery-r1'
log=root/'watch-until-complete.jsonl'
while True:
    now=time.time();counts=collections.Counter();recent=collections.Counter();states=collections.Counter();errors=collections.Counter();known=reserve=0
    for f in root.glob('*/results/autojudge/*.json'):
        d=json.loads(f.read_text());counts[d['status']]+=1
    for f in root.glob('*/calls/*.json'):
        d=json.loads(f.read_text());states[d.get('state')]+=1
        if f.stat().st_mtime>now-180:
            recent[d.get('state')]+=1
            if d.get('error_type'):errors[d['error_type']]+=1
        cost=d.get('cost_usd')
        if cost is not None:
            if d.get('cost_source')=='reserved_upper_bound_not_actual':reserve+=cost
            else:known+=cost
    pids=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if cmd.startswith('/home/alina/miniconda3/envs/olama/bin/python ') and 'table3_recovery.py run' in cmd:pids.append(int(proc.name))
    row={'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).isoformat(),'recovered':counts['success']-286,'failed':counts['failed'],'pending':348-sum(counts.values()),'last3min_calls':dict(recent),'last3min_errors':dict(errors),'new_known_cost_usd':round(known,6),'new_reserved_cost_usd':round(reserve,6),'runner_pids':pids}
    with log.open('a') as out:out.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if sum(counts.values())==348 or not pids:break
    time.sleep(180)
