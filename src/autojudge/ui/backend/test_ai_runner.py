"""Exercise the real Pipeline with FunctionModel. No provider requests allowed."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_runner
import server
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai import models

CONFIG={'nodes':['Judge','FINAL_AGGREGATOR'],'edges':[['Judge','FINAL_AGGREGATOR']],
        'schema':'{"type":"object","required":["verdict"],"properties":{"verdict":{"type":"string"}}}',
        'examples':'[]','objective':'Evaluate the trace','taxonomy':'Unsupported claim',
        'mode':'Full trace','model':'test/model'}
STEPS=[{'id':1,'agent':'Test agent','content':'A test claim'}]


def answer(messages, info):
    return ModelResponse(parts=[TextPart('{"verdict":"test judgment"}')])


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_endpoint(self):
        import httpx
        def handler(request):
            self.assertEqual(str(request.url), 'https://custom.example/v1/chat/completions')
            self.assertEqual(json.loads(request.content)['model'], 'custom-model')
            self.assertEqual(request.headers['authorization'], 'Bearer custom-key')
            return httpx.Response(200,json={'id':'test','object':'chat.completion','created':0,'model':'custom-model',
                'choices':[{'index':0,'message':{'role':'assistant','content':'{"verdict":"ok"}'},'finish_reason':'stop'}]})
        original_client=httpx.AsyncClient
        with patch.object(ai_runner,'AsyncClient',side_effect=lambda **kw: original_client(transport=httpx.MockTransport(handler),**kw)), patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            result=await ai_runner.run({**CONFIG,'model':'custom-model','applied_base_url':'https://custom.example/v1'},STEPS,'custom-key',0.1,lambda n,o:None)
        self.assertEqual(result['final_output']['verdict'],'ok')

    async def test_openrouter_sdk_with_mock_http(self):
        import httpx
        requests=[]
        def handler(request):
            requests.append(request)
            self.assertEqual(request.url.host,'openrouter.ai')
            payload=json.loads(request.content)
            self.assertEqual(payload['model'],'test/model')
            return httpx.Response(200,json={
                'id':'test','object':'chat.completion','created':0,'model':'test/model',
                'choices':[{'index':0,'message':{'role':'assistant','content':'{"verdict":"ok"}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
        transport=httpx.MockTransport(handler)
        original_client=httpx.AsyncClient
        with patch.object(ai_runner,'AsyncClient',side_effect=lambda **kw: original_client(transport=transport,**kw)), patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            result=await ai_runner.run(CONFIG,STEPS,'fake-key',0.1,lambda n,o:None)
        self.assertEqual(result['final_output']['verdict'],'ok')
        self.assertEqual(len(requests),2)

    async def test_actual_pipeline_no_network(self):
        events=[]
        with models.override_allow_model_requests(False), patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            result=await ai_runner.run(CONFIG,STEPS,'fake-key',0.1,
                lambda n,o:events.append((n,o)),FunctionModel(answer))
        self.assertEqual(result['final_output']['verdict'],'test judgment')
        self.assertEqual(len(result['trace']['node_traces']),2)
        self.assertEqual([n for n,o in events if o['status']=='Completed'],CONFIG['nodes'])
        self.assertNotIn('fake-key',json.dumps(result))

    async def test_failure_is_sanitized(self):
        def fail(messages, info): raise RuntimeError('private-key-must-not-leak')
        events=[]
        with models.override_allow_model_requests(False):
            with self.assertRaises(RuntimeError) as error:
                await ai_runner.run(CONFIG,STEPS,'fake-key',0.1,lambda n,o:events.append(o),FunctionModel(fail))
        self.assertNotIn('private-key',str(error.exception))
        self.assertEqual(events[-1]['status'],'Failed')

    async def test_cancel_real_pipeline(self):
        async def slow(messages, info):
            await asyncio.sleep(10)
            return answer(messages,info)
        started=asyncio.Event()
        events=[]
        def emit(n,o):
            events.append(o)
            if o['status']=='Running': started.set()
        with models.override_allow_model_requests(False):
            task=asyncio.create_task(ai_runner.run(CONFIG,STEPS,'fake-key',0.1,emit,FunctionModel(slow)))
            await asyncio.wait_for(started.wait(),5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(events[-1]['status'],'Cancelled')


class IntegrationTests(unittest.TestCase):
    def test_api_real_pipeline_and_confirmation(self):
        real_run=ai_runner.run
        async def local_run(config,steps,key,temp,emit):
            return await real_run(config,steps,key,temp,emit,FunctionModel(answer))
        with tempfile.TemporaryDirectory() as directory, patch.object(server,'DB_PATH',Path(directory)/'test.sqlite'), patch.object(server,'ai_settings',return_value=('test-secret',0.1,'test/model','https://openrouter.ai/api/v1')), patch.object(ai_runner,'run',side_effect=local_run), models.override_allow_model_requests(False):
            with TestClient(server.app) as client:
                data={'config':CONFIG,'steps':STEPS,'execution':'ai'}
                headers={'Origin':'http://127.0.0.1:5173'}
                self.assertEqual(client.post('/api/runs',json=data,headers=headers).status_code,403)
                data['confirm_paid']=True
                self.assertEqual(client.post('/api/runs',json=data).status_code,403)
                result=client.post('/api/runs',json=data,headers=headers)
                self.assertEqual(result.status_code,201,result.text)
                key=result.json()['id']
                events=client.get('/api/runs/'+key+'/events').text
                result=client.get('/api/runs/'+key).json()
                self.assertEqual(result['status'],'Completed',result)
                self.assertEqual(result['execution'],'ai')
                self.assertEqual(result['usage']['calls'],2)
                self.assertIsNone(result['usage']['cost'])
                self.assertNotIn('test-secret',events)
                self.assertEqual(result['final_output']['verdict'],'test judgment')


if __name__=='__main__': unittest.main()
