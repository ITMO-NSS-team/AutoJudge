"""One uniform technical recovery pass; preserve original campaign and valid stages."""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import table3_parallel_conditions as parallel
experiment = parallel.experiment
ROOT = experiment.ROOT
ORIGINAL = ROOT / 'results/table3-reconstruction-20261118-olama-v3'
OUTPUT = ROOT / 'results/table3-reconstruction-20261118-recovery-r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    experiment.validate_campaign(ORIGINAL)
    if OUTPUT.exists():
        raise ValueError('Recovery directory already exists')
    report = experiment.p.read_json(ORIGINAL / 'report.json')
    assert report['complete']
    baseline = report['accounting']['budget_debit_usd']
    assert 0 < baseline < 100
    targets = []
    for name in experiment.CONDITIONS:
        for file in sorted((ORIGINAL/name/'results/autojudge').glob('*.json')):
            result = experiment.p.read_json(file)
            if result['status'] == 'failed':
                targets.append({'condition': name, 'task_id': result['task_id'],
                                'result_sha256': sha(file), 'error': result['error']})
            elif result['status'] != 'success':
                raise ValueError('Unexpected original status')
    assert len(targets) == 62
    plan = {
        'created_at': experiment.p.now(), 'round': 1, 'original': str(ORIGINAL),
        'original_campaign_sha256': sha(ORIGINAL/'campaign.json'),
        'original_report_sha256': sha(ORIGINAL/'report.json'),
        'script_sha256': sha(Path(__file__)), 'targets': targets,
        'selection': 'Every failed trace-condition result, selected solely by technical status; no accuracy filtering.',
        'policy': 'One recovery pass; up to three fresh attempts per unrecovered stage. Reuse validated original stages only when initial request and schema match. No prompt, seed, model or reference-dependent changes.',
        'original_budget_debit_usd': baseline,
        'recovery_budget_limit_usd': 100-baseline,
        'combined_budget_limit_usd': 100,
        'original_accounting': report['accounting'],
    }
    OUTPUT.mkdir(parents=True)
    experiment.p.write_json(OUTPUT/'recovery-plan.json', plan)
    shutil.copy2(ORIGINAL/'campaign.json', OUTPUT/'campaign.json')
    for name in experiment.CONDITIONS:
        source, dest = ORIGINAL/name, OUTPUT/name
        dest.mkdir()
        for filename in ['protocol.json','prompts.json','references.json']:
            shutil.copy2(source/filename, dest/filename)
        shutil.copytree(source/'inputs',dest/'inputs')
        manifest = experiment.p.read_json(source/'manifest.json')
        original_id = manifest.pop('run_id')
        manifest['recovery_source_run_id'] = original_id
        manifest['recovery_plan_sha256'] = sha(OUTPUT/'recovery-plan.json')
        manifest['run_id'] = experiment.p.digest(manifest)
        experiment.p.write_json(dest/'manifest.json',manifest)
        for file in (source/'results/autojudge').glob('*.json'):
            result = experiment.p.read_json(file)
            if result['status'] != 'success':
                continue
            result.update(run_id=manifest['run_id'], original_run_id=original_id,
                          reused_original_result_sha256=sha(file), recovery_action='original_success_unchanged')
            experiment.p.write_json(dest/'results/autojudge'/file.name,result)
    (OUTPUT/'README.md').write_text(
        '# Uniform technical recovery, round 1\n\n'
        'All 62 original technical failures are included. The original campaign is unchanged. '
        'The 286 original successful results are retained, with original hashes and run IDs. '
        'Validated intermediate stages are reused only after checking request identity. '
        'Every unrecovered stage receives up to three fresh attempts in a new receipt namespace. '
        'The same models, prompts, schemas, seeds, generation limits and full 58-trace denominator apply. '
        'This is a separate, declared recovery pass, not the original three-attempt protocol. '
        'Report both the first-pass results and this recovery policy in the paper.\n\n'
        'The report.json accounting covers NEW calls only. combined-accounting.json adds the original '
        'campaign exactly once, including uncertain-cost reserves. The aggregate ceiling is $100.\n')
    print(json.dumps({'targets':len(targets),'retained_successes':286,'new_budget_limit':100-baseline}))


class RecoveryClient(parallel.SharedClient):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        plan = experiment.p.read_json(self.path.parent/'recovery-plan.json')
        if sha(Path(__file__)) != plan['script_sha256']:
            raise ValueError('Recovery code differs from frozen plan')
        if self.max_cost > plan['recovery_budget_limit_usd'] + 1e-8:
            raise ValueError('Recovery budget exceeds combined $100 ceiling')
        if sha(ORIGINAL/'campaign.json') != plan['original_campaign_sha256']:
            raise ValueError('Original campaign changed')
        source = ORIGINAL/self.path.name
        source_manifest = experiment.p.read_json(source/'manifest.json')
        if self.manifest['recovery_source_run_id'] != source_manifest['run_id']:
            raise ValueError('Source run identity mismatch')
        self.cached_stages = {}
        for file in (source/'calls').glob('*.json'):
            record = experiment.p.read_json(file)
            if record.get('state') == 'validated':
                key = (record['task_id'],record['arm'],record['stage'])
                if key in self.cached_stages:
                    raise ValueError('Multiple validated original stage receipts')
                self.cached_stages[key] = (record, file)

    async def complete(self,task,arm,stage,system,user,schema,inp):
        cached = self.cached_stages.get((task,arm,stage))
        if cached:
            record, source = cached
            messages=[{'role':'system','content':system}, {'role':'user','content':user if isinstance(user,str) else experiment.p.encode(user)}]
            payload = self.payload(messages,schema,stage,task,arm)
            original_payload = copy.deepcopy(record['request'])
            original_payload['messages'] = original_payload['messages'][:2]
            if experiment.p.digest(payload) != experiment.p.digest(original_payload):
                raise experiment.HaltRun('Cached stage differs from recovery request; no silent regeneration')
            if record['run_id'] != self.manifest['recovery_source_run_id']:
                raise experiment.HaltRun('Cached source run mismatch')
            self.check_response_route(record)
            experiment.p.validate_output('judge' if stage.startswith('judge_') else stage,record['output'],schema,inp)
            experiment.p.write_json(self.path/'reused_stages'/task/(stage+'.json'), {
                'source':str(source),'source_sha256':sha(source),'original_call_id':record['call_id'],
                'request_initial_sha256':experiment.p.digest(payload),
                'output_sha256':experiment.p.digest(record['output']),
                'additional_cost_usd':0,'note':'Original cost counted once in original accounting.'})
            return copy.deepcopy(record['output'])
        return await super().complete(task,arm,stage,system,user,schema,inp)


def combined_accounting():
    if not (OUTPUT/'report.json').exists():
        return
    original = experiment.p.read_json(OUTPUT/'recovery-plan.json')['original_accounting']
    recovery = experiment.p.read_json(OUTPUT/'report.json')['accounting']
    experiment.p.write_json(OUTPUT/'combined-accounting.json', {
        'original':original, 'recovery_new_calls_only':recovery,
        'combined_known_cost_usd':original['known_cost_usd']+recovery['known_cost_usd'],
        'combined_budget_debit_usd':original['budget_debit_usd']+recovery['budget_debit_usd'],
        'combined_actual_cost_usd': None,
        'note':'Lost-response actual charges remain unknown; original and recovery calls counted once.'})

experiment.ConservativeClient = RecoveryClient
if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare-recovery':
        prepare()
    else:
        try:
            experiment.main()
        finally:
            if len(sys.argv)>1 and sys.argv[1]=='report':
                combined_accounting()
