"""Independent offline scorer and provenance audit for the reconstructed few-shot study.

Run from AutoJudge: python rebuttal/table3_reconstruction/verify.py
Standard library only; never makes model calls.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = Path(__file__).resolve().parent/'runs/first_pass'
RECOVERY = Path(__file__).resolve().parent/'runs/recovered'
OUT = Path(__file__).resolve().parent
CONDITIONS = ['six','four','completion_complexity','complexity_role_coverage','three','completion_guidelines']


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def score(prediction,reference):
    agent = type(prediction.get('agent')) is str and prediction['agent'] == reference['agent']
    step = type(prediction.get('step')) is int and prediction['step'] == reference['step_zero_based'] + 1
    return int(agent),int(step),int(agent and step)


def main():
    report=read(RECOVERY/'report.json');plan=read(RECOVERY/'recovery-plan.json')
    campaign=read(ORIGINAL/'campaign.json')
    for relative,expected in campaign['experiment_hashes'].items():
        assert sha(ROOT/relative)==expected, 'Frozen execution source differs: '+relative
    assert sha(ORIGINAL/'campaign.json')==plan['original_campaign_sha256']
    assert sha(ORIGINAL/'report.json')==plan['original_report_sha256']
    rows={};common_ids=None;common_refs=None;retained=recovered=0
    targets={(t['condition'],t['task_id']):t for t in plan['targets']}
    for condition in CONDITIONS:
        sub=RECOVERY/condition;oldsub=ORIGINAL/condition
        manifest=read(sub/'manifest.json');oldmanifest=read(oldsub/'manifest.json')
        ids=[e['task_id'] for e in manifest['entries']]
        assert len(ids)==len(set(ids))==58 and ids==[e['task_id'] for e in oldmanifest['entries']]
        refs=read(sub/'references.json');assert refs==read(oldsub/'references.json')
        if common_ids is None:common_ids,common_refs=ids,refs
        assert ids==common_ids and refs==common_refs
        totals=[0,0,0];first=[0,0,0];failures=0
        for entry in manifest['entries']:
            task=entry['task_id'];inp=read(sub/'inputs'/(task+'.json'))
            assert inp==read(oldsub/'inputs'/(task+'.json'))
            # The protocol hashes canonical JSON with sorted keys and compact separators.
            digest=hashlib.sha256(json.dumps(inp,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert digest==entry['input_sha256']
            original_file=oldsub/'results/autojudge'/(task+'.json')
            old=read(original_file);new=read(sub/'results/autojudge'/(task+'.json'))
            assert new['status']=='success' and new['task_id']==task
            assert new['run_id']==manifest['run_id'] and new['input_sha256']==entry['input_sha256']
            if old['status']=='success':
                assert new['prediction']==old['prediction'] and new['judges']==old['judges'] and new['findings']==old['findings']
                assert new['reused_original_result_sha256']==sha(original_file)
                retained+=1
            else:
                assert old['status']=='failed' and (condition,task) in targets
                assert targets[(condition,task)]['result_sha256']==sha(original_file)
                failures+=1;recovered+=1
            values=score(new['prediction'],refs[task])
            first_values=score(old['prediction'],refs[task]) if old['status']=='success' else (0,0,0)
            totals=[a+b for a,b in zip(totals,values)]
            first=[a+b for a,b in zip(first,first_values)]
        metrics=dict(zip(['agent','step','joint'],totals))
        assert all(metrics[k]==report['conditions'][condition][k] for k in metrics)
        rows[condition]={'n':58,'correct':metrics,'accuracy_percent':{k:v/58*100 for k,v in metrics.items()},'original_failures':failures,'first_pass_correct':dict(zip(['agent','step','joint'],first))}
    assert retained==286 and recovered==62 and len(targets)==62
    accounting=read(RECOVERY/'combined-accounting.json')
    summary={'conditions':rows,'retained_original_successes':retained,'recovered_failures':recovered,
             'paired_exploratory_unadjusted':report['paired_exploratory_unadjusted'],
             'combined_accounting':accounting,'scoring':'Case-sensitive canonical agent equality; independent exact step: integer prediction == zero-based reference + 1; joint is conjunction; all 58 traces per condition.',
             'sources':{'original_campaign_sha256':sha(ORIGINAL/'campaign.json'),'recovery_plan_sha256':sha(RECOVERY/'recovery-plan.json'),'report_sha256':sha(RECOVERY/'report.json'),'checker_sha256':sha(Path(__file__))}}
    (OUT/'verified-results.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Verified reconstructed few-shot ablation','','All 58 Hand development traces in each condition; one generation repeat; two reconstructed few-shot examples.','','| Condition | Agent correct / 58 (%) | Exact step correct / 58 (%) | First-pass technical failures |','| --- | ---: | ---: | ---: |']
    for name,r in rows.items():
        lines.append(f"| {name} | {r['correct']['agent']}/58 ({r['accuracy_percent']['agent']:.2f}) | {r['correct']['step']}/58 ({r['accuracy_percent']['step']:.2f}) | {r['original_failures']} |")
    lines+=['',f"Confirmed unchanged original successes: {retained}; recovered technical failures: {recovered}.",'',f"Combined known charges: ${accounting['combined_known_cost_usd']:.8f}; conservative budget debit including lost-response reservations: ${accounting['combined_budget_debit_usd']:.8f}. Actual total remains unknown."]
    (OUT/'verified-results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__': main()
