"""Validate cache reuse and reject changed requests without any network calls."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/who_and_when'))
recovery=importlib.import_module('table3_recovery')


def cached_client(tmp_path):
    client=object.__new__(recovery.RecoveryClient)
    client.path=tmp_path
    client.manifest={'recovery_source_run_id':'original'}
    inp={'query':'Find an item','messages':[{'step':1,'agent':'A','message':{'content':'Wrong item'}}]}
    schema=recovery.experiment.p.schemas(inp)['overall']
    client.payload=lambda messages,schema,stage,task,arm:{'messages':messages,'model':'same','schema':schema}
    client.check_response_route=lambda record:None
    payload=client.payload([{'role':'system','content':'judge'},{'role':'user','content':'trace'}],schema,'overall','task','autojudge')
    payload['messages'].append({'role':'user','content':'Previous format error; try again'})
    record={'request':payload,'run_id':'original','call_id':'receipt','output':{'agent':'A','step':1,'reason':'Wrong target'}}
    source=tmp_path/'original.json';recovery.experiment.p.write_json(source,record)
    client.cached_stages={('task','autojudge','overall'):(record,source)}
    return client,inp,schema


@pytest.mark.asyncio
async def test_reuse_validated_stage_preserves_output_and_records_zero_new_cost(tmp_path):
    client,inp,schema=cached_client(tmp_path)
    output=await client.complete('task','autojudge','overall','judge','trace',schema,inp)
    assert output=={'agent':'A','step':1,'reason':'Wrong target'}
    receipt=recovery.experiment.p.read_json(tmp_path/'reused_stages/task/overall.json')
    assert receipt['additional_cost_usd']==0
    output['step']=2
    assert client.cached_stages[('task','autojudge','overall')][0]['output']['step']==1


@pytest.mark.asyncio
async def test_changed_judge_instructions_cannot_reuse_original_stage(tmp_path):
    client,inp,schema=cached_client(tmp_path)
    with pytest.raises(recovery.experiment.HaltRun,match='differs'):
        await client.complete('task','autojudge','overall','different instructions','trace',schema,inp)
    assert not (tmp_path/'reused_stages').exists()
