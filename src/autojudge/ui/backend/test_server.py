import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import server


class ApiTests(unittest.TestCase):
    def test_hosted_environment_settings_are_read_only(self):
        hosted = {
            'AUTOJUDGE_SETTINGS_READ_ONLY': '1',
            'OPENROUTER_API_KEY': 'hosted-test-secret',
            'AGENT_NODE_MODEL': 'test/hosted-model',
            'AGENT_NODE_TEMPERATURE': '0.4',
            'AUTOJUDGE_AI_ENABLED': '0',
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, 'DB_PATH', Path(directory) / 'test.sqlite'
        ), patch.object(server, 'ENV_PATH', Path(directory) / '.env'), patch.dict(
            'os.environ', hosted, clear=True
        ):
            with TestClient(server.app) as client:
                response = client.get('/api/settings/env')
                self.assertTrue(response.json()['read_only'])
                self.assertEqual(response.json()['secret_storage']['name'], 'Hosting environment')
                self.assertNotIn(hosted['OPENROUTER_API_KEY'], response.text)
                self.assertFalse(client.get('/api/health').json()['ai_available'])
                self.assertEqual(
                    client.put('/api/settings/env', json={'values': {}}, headers={
                        'Origin': 'http://127.0.0.1:8000'
                    }).status_code,
                    403,
                )
                self.assertEqual(
                    client.put('/api/credentials/openrouter', json={'key': 'x' * 20}).status_code,
                    403,
                )
                ai_request = {
                    'config': {
                        'nodes': ['Judge', 'FINAL_AGGREGATOR'],
                        'edges': [['Judge', 'FINAL_AGGREGATOR']],
                        'schema': '{"type":"object"}',
                        'examples': '[]',
                        'objective': 'Check',
                        'taxonomy': 'Test taxonomy',
                        'mode': 'Full trace',
                        'model': 'test/hosted-model',
                    },
                    'steps': [{'id': 1, 'content': 'test'}],
                    'execution': 'ai',
                }
                self.assertEqual(client.post('/api/runs', json=ai_request).status_code, 503)
            self.assertEqual(
                server.ai_settings(),
                ('hosted-test-secret', 0.4, 'test/hosted-model'),
            )

    def test_master_key_storage_encrypts_at_rest(self):
        secret='master-key-storage-test-secret'
        headers={'Origin':'http://127.0.0.1:5173'}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, 'DB_PATH', Path(directory)/'test.sqlite'
        ), patch.object(server, 'ENV_PATH', Path(directory)/'.env'), patch.dict(
            'os.environ', {'AUTOJUDGE_CREDENTIALS_KEY':'unit-test-master-key'}, clear=True
        ), patch.object(server.credentials.sys, 'platform', 'linux'):
            with TestClient(server.app) as client:
                storage=client.get('/api/settings/env').json()['secret_storage']
                self.assertEqual(storage['name'], 'Encrypted container storage')
                self.assertTrue(storage['available'])
                path='/api/credentials/openrouter'
                response=client.put(path, json={'key':secret}, headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()['configured'])
                self.assertNotIn(secret, response.text)
                payload=server.records('credentials')[0]
                self.assertEqual(payload.get('storage'), 'master-key')
                self.assertNotIn(secret.encode(), server.DB_PATH.read_bytes())
                self.assertEqual(server.credentials.load('OPENROUTER_API_KEY', payload), secret)

    def test_frontend_build_and_spa_fallback_are_served(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, 'WEB_DIST', Path(directory)
        ):
            web = Path(directory)
            (web / 'index.html').write_text('<main>AutoJudge production</main>', encoding='utf-8')
            (web / 'asset.txt').write_text('asset', encoding='utf-8')
            with TestClient(server.app) as client:
                self.assertIn('AutoJudge production', client.get('/').text)
                self.assertIn('AutoJudge production', client.get('/nested/route').text)
                self.assertEqual(client.get('/asset.txt').text, 'asset')
                self.assertEqual(client.get('/api/not-a-real-route').status_code, 404)

    def test_non_windows_keyring_payload_contains_no_secret(self):
        secret='cross-platform-test-secret'
        with patch.object(server.credentials.sys,'platform','linux'), patch.object(
            server.credentials,'storage_info',return_value={'name':'Linux Secret Service','available':True,'persistent':True}
        ), patch('keyring.set_password') as save, patch('keyring.get_password',return_value=secret) as load, patch('keyring.delete_password') as remove:
            payload=server.credentials.store('OPENROUTER_API_KEY',secret)
            self.assertEqual(payload,{'keyring':True,'storage':'keyring'})
            self.assertNotIn(secret,json.dumps(payload))
            self.assertEqual(server.credentials.load('OPENROUTER_API_KEY',payload),secret)
            server.credentials.remove('OPENROUTER_API_KEY',payload)
            save.assert_called_once_with('AutoJudge','OPENROUTER_API_KEY',secret)
            load.assert_called_once_with('AutoJudge','OPENROUTER_API_KEY')
            remove.assert_called_once_with('AutoJudge','OPENROUTER_API_KEY')

    def test_env_settings(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'DB_PATH', Path(directory)/'test.sqlite'), patch.object(server, 'ENV_PATH', Path(directory)/'.env'), patch.dict('os.environ', {}, clear=True):
            with TestClient(server.app) as client:
                headers={'Origin':'http://127.0.0.1:5173'}
                path='/api/settings/env'
                fields=client.get(path).json()['fields']
                self.assertEqual(len(fields),14)
                self.assertNotIn('MCP models',{field['group'] for field in fields})
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

    def test_remote_origin_is_explicitly_configurable(self):
        allowed='https://autojudge-preview.example.test'
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'DB_PATH', Path(directory)/'test.sqlite'), patch.object(server, 'ENV_PATH', Path(directory)/'.env'), patch.dict('os.environ', {'AUTOJUDGE_ALLOWED_ORIGINS':allowed}, clear=True):
            with TestClient(server.app) as client:
                path='/api/settings/env'
                self.assertEqual(client.put(path,json={'values':{'DB_PORT':'5433'}},headers={'Origin':allowed}).status_code,200)
                self.assertEqual(client.put(path,json={'values':{'DB_PORT':'5434'}},headers={'Origin':'https://other.example.test'}).status_code,403)

    def test_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'DB_PATH', Path(directory)/'test.sqlite'):
            request={'config':{'nodes':['Judge','FINAL_AGGREGATOR'],'edges':[['Judge','FINAL_AGGREGATOR']],'schema':'{"type":"object"}','examples':'[]','taxonomy':'Test taxonomy'},'steps':[{'id':1,'content':'test'}],'execution':'offline'}
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
                self.assertEqual(client.put('/api/workspace',json={'config':request['config']}).status_code,200)
            with TestClient(server.app) as client:
                self.assertTrue(client.get(f"/api/runs/{run['id']}").json()['archived'])
                self.assertEqual(len(client.get('/api/runs').json()),2)
                self.assertEqual(client.get('/api/workspace').json()['config']['nodes'][0],'Judge')

    def test_sequential_judge_chain_levels(self):
        config={
            'nodes':['Evidence','Critic','FINAL_AGGREGATOR'],
            'edges':[['Evidence','Critic'],['Critic','FINAL_AGGREGATOR']],
            'schema':'{"type":"object"}',
            'examples':'[]',
            'taxonomy':'Test taxonomy',
        }
        with TestClient(server.app) as client:
            response=client.post('/api/graph/validate',json=config)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['levels'],[['Evidence'],['Critic'],['FINAL_AGGREGATOR']])

    def test_graph_rejects_edge_from_missing_node(self):
        config={
            'nodes':['Judge','FINAL_AGGREGATOR'],
            'edges':[['Removed judge','FINAL_AGGREGATOR']],
            'schema':'{"type":"object"}',
            'examples':'[]',
            'taxonomy':'Test taxonomy',
        }
        with TestClient(server.app) as client:
            self.assertEqual(client.post('/api/graph/validate',json=config).status_code,422)

    def test_design_validation(self):
        design={'objective':'Check the trace','taxonomy':'Unsupported claim','model':'z-ai/glm-5.3-flash','schema':'{"type":"object","properties":{"verdict":{"type":"string"}}}','examples':'[]'}
        with TestClient(server.app) as client:
            response=client.post('/api/design/validate',json=design)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['properties'],1)
            self.assertEqual(response.json()['taxonomy_chars'],17)

    def test_model_id_validation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'DB_PATH', Path(directory)/'test.sqlite'), patch.object(server, 'ENV_PATH', Path(directory)/'.env'):
            with TestClient(server.app) as client:
                headers={'Origin':'http://127.0.0.1:5173'}
                self.assertEqual(client.put('/api/settings/env',json={'values':{'AGENT_NODE_MODEL':'z-ai/glm-5.3-flash'}},headers=headers).status_code,200)
                self.assertEqual(client.put('/api/settings/env',json={'values':{'AGENT_NODE_MODEL':'z-ai модель'}},headers=headers).status_code,422)

    def test_ai_validation_does_not_limit_normalized_raw_trace_size(self):
        request=server.RunRequest(
            config={
                'nodes':['Judge','FINAL_AGGREGATOR'],
                'edges':[['Judge','FINAL_AGGREGATOR']],
                'mode':'Full trace',
                'objective':'Evaluate raw spans',
                'taxonomy':'Test taxonomy',
                'schema':'{"type":"object"}',
                'model':'test/model',
            },
            steps=[{'id':1,'content':'x'*5_100_000}],
            execution='ai',
        )
        server.validate_ai(request)

    def test_run_can_be_permanently_deleted(self):
        request={'config':{'nodes':['Judge','FINAL_AGGREGATOR'],'edges':[['Judge','FINAL_AGGREGATOR']],'schema':'{"type":"object"}','examples':'[]','objective':'Check','taxonomy':'Test taxonomy'},'steps':[{'id':1,'content':'test'}],'execution':'offline'}
        with tempfile.TemporaryDirectory() as directory, patch.object(server,'DB_PATH',Path(directory)/'test.sqlite'):
            with TestClient(server.app) as client:
                run=client.post('/api/runs',json=request).json()
                with client.stream('GET',f"/api/runs/{run['id']}/events") as response:
                    list(response.iter_lines())
                self.assertEqual(client.delete(f"/api/runs/{run['id']}").status_code,204)
                self.assertEqual(client.get(f"/api/runs/{run['id']}").status_code,404)


if __name__=='__main__': unittest.main()
