import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai import models
import server
import pool_runner
import ai_runner
from test_ai_runner import CONFIG, STEPS

POOL={'judges':[{'name':'Evidence','instructions':'Check evidence.'},{'name':'FINAL_AGGREGATOR','instructions':'Aggregate evidence.'}]}


class PoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_no_network(self):
        def answer(messages, info):
            return ModelResponse(parts=[TextPart(json.dumps(POOL))])
        with models.override_allow_model_requests(False), patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            result=await pool_runner.generate(CONFIG,'fake','https://custom.example/v1','custom',0.3,FunctionModel(answer))
        self.assertEqual(result['judges'],POOL['judges'])
        self.assertEqual(result['usage']['calls'],1)

    async def test_instructions_reach_pipeline(self):
        seen=[]
        def answer(messages, info):
            seen.append(str(messages)+str(info))
            return ModelResponse(parts=[TextPart('{"verdict":"ok"}')])
        with models.override_allow_model_requests(False), patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            await ai_runner.run({**CONFIG,'judge_instructions':{'Judge':'UNIQUE_JUDGE_INSTRUCTION'}},STEPS,'fake',0.1,lambda n,o:None,FunctionModel(answer))
        self.assertIn('UNIQUE_JUDGE_INSTRUCTION',''.join(seen))

    def test_invalid_pools(self):
        for value in [{'judges':[]},{'judges':[{'name':'x','instructions':'ok'},{'name':'x','instructions':'ok'}]}]:
            with self.assertRaises(ValueError): pool_runner.validate_pool(value)

    def test_api_confirmation_and_history(self):
        body={'objective':'Test','taxonomy':'Evidence','schema_text':'{"type":"object"}'}
        with tempfile.TemporaryDirectory() as directory, patch.object(server,'DB_PATH',Path(directory)/'test.sqlite'), patch.object(server,'ai_settings',return_value=('fake',0.1,'judge','https://custom.example/v1')), patch.object(pool_runner,'generate',new_callable=AsyncMock,return_value={**POOL,'usage':{'calls':1,'tokens':2,'cost':None}}) as generate:
            with TestClient(server.app) as client:
                headers={'origin':'http://127.0.0.1:5173'}
                self.assertEqual(client.post('/api/judge-pools/generate',json=body,headers=headers).status_code,403)
                generate.assert_not_called()
                response=client.post('/api/judge-pools/generate',json={**body,'confirm_paid':True},headers=headers)
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(client.get('/api/judge-pools').json()[0]['judges'],POOL['judges'])
                generate.assert_awaited_once()
