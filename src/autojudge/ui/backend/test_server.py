import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import server


class ApiTests(unittest.TestCase):
    def test_env_settings(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'DB_PATH', Path(directory)/'test.sqlite'), patch.object(server, 'ENV_PATH', Path(directory)/'.env'), patch.dict('os.environ', {}, clear=True):
            with TestClient(server.app) as client:
                headers={'Origin':'http://127.0.0.1:5173'}
                path='/api/settings/env'
                fields=client.get(path).json()['fields']
                self.assertEqual(len(fields),25)
                values={'HF_TOKEN':'fake-token-for-offline-test','DB_PORT':'5433','AGENT_NODE_TEMPERATURE':'0.7'}
                response=client.put(path,json={'values':values},headers=headers)
                self.assertEqual(response.status_code,200)
                self.assertNotIn(values['HF_TOKEN'],response.text)
                self.assertNotIn(values['HF_TOKEN'].encode(),server.DB_PATH.read_bytes())
                self.assertNotIn('HF_TOKEN', client.get('/api/workspace').text)
                self.assertEqual(client.put(path,json={'values':{'DB_PORT':'99999'}},headers=headers).status_code,422)
                self.assertEqual(client.put(path,json={'values':{'UNKNOWN':'x'}},headers=headers).status_code,422)
                self.assertEqual(client.put(path,json={'values':values}).status_code,403)
                client.put(path,json={'reset':['DB_PORT']},headers=headers)
                current={f['name']:f for f in client.get(path).json()['fields']}
                self.assertEqual(current['DB_PORT']['value'],'5432')
                self.assertTrue(current['HF_TOKEN']['configured'])
                client.put(path,json={'values':{'HF_TOKEN':''}},headers=headers)
                self.assertFalse(next(f for f in client.get(path).json()['fields'] if f['name']=='HF_TOKEN')['configured'])

    def test_credentials(self):
        secret = 'test-only-not-a-real-provider-key-12345'
        headers = {'Origin': 'http://127.0.0.1:5173'}
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'DB_PATH', Path(directory)/'test.sqlite'), patch.object(server, 'ENV_PATH', Path(directory)/'.env'), patch.dict('os.environ', {}, clear=True):
            with TestClient(server.app) as client:
                path = '/api/credentials/openrouter'
                self.assertFalse(client.get(path).json()['configured'])
                self.assertEqual(client.put(path, json={'key':secret}).status_code, 403)
                invalid = client.put(path, json={'key':'short'}, headers=headers)
                self.assertEqual(invalid.status_code, 422)
                self.assertNotIn('short', invalid.text)
                response = client.put(path, json={'key':secret}, headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()['configured'])
                self.assertNotIn(secret, response.text)
                ciphertext = server.records('credentials')[0]['ciphertext']
                self.assertEqual(server.credentials.decrypt(ciphertext), secret)
                self.assertNotIn(secret.encode(), server.DB_PATH.read_bytes())
                for endpoint in (path, '/api/workspace', '/api/runs', '/api/health'):
                    self.assertNotIn(secret, client.get(endpoint).text)
                self.assertFalse(client.get('/api/health').json()['paid_calls_enabled'])
            with TestClient(server.app) as client:
                self.assertTrue(client.get(path).json()['configured'])
                self.assertEqual(client.delete(path, headers={'Origin':'https://example.com'}).status_code, 403)
                self.assertFalse(client.delete(path, headers=headers).json()['configured'])
                self.assertEqual(server.records('credentials'), [])

    def test_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'DB_PATH', Path(directory)/'test.sqlite'):
            request={'config':{'nodes':['Judge','FINAL_AGGREGATOR'],'edges':[['Judge','FINAL_AGGREGATOR']],'schema':'{"type":"object"}','examples':'[]'},'steps':[{'id':1,'content':'test'}]}
            with TestClient(server.app) as client:
                self.assertFalse(client.get('/api/health').json()['paid_calls_enabled'])
                self.assertEqual(client.post('/api/runs',json={**request,'execution':'llm'}).status_code,403)
                broken={**request,'config':{**request['config'],'edges':[['Judge','Judge']]}}
                self.assertEqual(client.post('/api/runs',json=broken).status_code,422)
                run=client.post('/api/runs',json=request).json()
                with client.stream('GET',f"/api/runs/{run['id']}/events") as response:
                    events=[json.loads(line[6:]) for line in response.iter_lines() if line.startswith('data: ')]
                self.assertEqual(events[-1]['status'],'Completed')
                self.assertEqual(events[-1]['usage']['calls'],0)
                self.assertEqual(len(events[-1]['outputs']),2)
                client.patch(f"/api/runs/{run['id']}",json={'archived':True})
                second=client.post('/api/runs',json=request).json()
                cancelled=client.post(f"/api/runs/{second['id']}/cancel").json()
                self.assertEqual(cancelled['status'],'Cancelled')
                self.assertEqual(client.put('/api/workspace',json={'traces':[{'name':'test'}]}).status_code,200)
            with TestClient(server.app) as client:
                self.assertTrue(client.get(f"/api/runs/{run['id']}").json()['archived'])
                self.assertEqual(len(client.get('/api/runs').json()),2)
                self.assertEqual(client.get('/api/workspace').json()['traces'][0]['name'],'test')


if __name__=='__main__': unittest.main()
