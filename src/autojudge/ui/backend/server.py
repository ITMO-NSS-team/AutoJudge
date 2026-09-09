"""Local API with free offline validation and explicitly confirmed AI runs."""
import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import credentials
import env_settings

DB_PATH = Path(__file__).resolve().parents[4] / '.autojudge' / 'workspace.sqlite3'
ENV_PATH = Path(__file__).resolve().parents[4] / '.env'


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))')
    try:
        with db:
            yield db
    finally:
        db.close()


def put(kind, key, value):
    with connect() as db:
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)', (kind, key, json.dumps(value)))


def records(kind):
    with connect() as db:
        return [json.loads(r[0]) for r in db.execute('SELECT payload FROM records WHERE kind=? ORDER BY rowid DESC', (kind,))]


def get_run(key):
    with connect() as db:
        row = db.execute('SELECT payload FROM records WHERE kind=? AND id=?', ('runs',key)).fetchone()
    if not row:
        raise HTTPException(404, 'Run not found')
    return json.loads(row[0])


tasks = {}


@asynccontextmanager
async def lifespan(app):
    for run in records('runs'):
        if run['status'] == 'Running':
            run.update(status='Interrupted', verdict='Server restarted; inspect partial outputs; AI requests may be billed')
            put('runs', run['id'], run)
    yield
    for task in list(tasks.values()):
        task.cancel()
    await asyncio.gather(*list(tasks.values()), return_exceptions=True)


app = FastAPI(title='AutoJudge local API', lifespan=lifespan)


def stored_env():
    with connect() as db:
        return {name:json.loads(data) for name,data in db.execute(
            'SELECT id,payload FROM records WHERE kind=?', ('env',))}


@app.get('/api/settings/env')
def get_env_settings():
    return {'fields':env_settings.resolve(ENV_PATH, stored_env(), records('credentials')),
            'execution':'offline by default', 'applied_to_runner':True,
            'runner_fields':['LLM_BASE_URL','LLM_API_KEY','OPENROUTER_API_KEY','AGENT_NODE_MODEL','AGENT_NODE_TEMPERATURE']}


@app.put('/api/settings/env')
async def save_env_settings(request: Request):
    check_credential_origin(request)
    if request.headers.get('content-type','').split(';')[0] != 'application/json':
        raise HTTPException(415,'JSON required')
    raw = await request.body()
    if len(raw)>131072: raise HTTPException(413,'Settings request too large')
    try:
        body=json.loads(raw)
        if not isinstance(body,dict) or set(body)-{'values','reset'}: raise ValueError()
        values=body.get('values',{})
        reset=body.get('reset',[])
        names={f[0] for f in env_settings.FIELDS}
        if not isinstance(values,dict) or not isinstance(reset,list) or any(not isinstance(n,str) for n in reset): raise ValueError()
        if (set(values)|set(reset))-names or set(values)&set(reset): raise ValueError()
        updates={}
        for name,value in values.items():
            kind=env_settings.validate(name,value)
            updates[name]={'ciphertext':credentials.encrypt(value) if value else ''} if kind=='secret' else {'value':value}
    except (ValueError,TypeError,StopIteration):
        raise HTTPException(422,'Invalid settings: check names, temperature (0–2), port (1–65535), and URL')
    except RuntimeError:
        raise HTTPException(503,'Encrypted credential storage unavailable')
    with connect() as db:
        for name in reset:
            db.execute('DELETE FROM records WHERE kind=? AND id=?',('env',name))
            if name=='OPENROUTER_API_KEY':
                db.execute('DELETE FROM records WHERE kind=? AND id=?',('credentials','openrouter'))
        for name,value in updates.items():
            db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)',('env',name,json.dumps(value)))
    return get_env_settings()


def credential_status():
    field = next(f for f in get_env_settings()['fields'] if f['name']=='OPENROUTER_API_KEY')
    return {'provider': 'openrouter', 'configured': field['configured'],
            'storage': 'Windows DPAPI', 'verified': False,
            'paid_calls_enabled': False}


@app.get('/api/credentials/openrouter')
def read_credential_status():
    return credential_status()


def check_credential_origin(request: Request):
    allowed = {'http://127.0.0.1:5173', 'http://localhost:5173',
               'http://127.0.0.1:8000', 'http://localhost:8000'}
    if request.headers.get('origin') not in allowed:
        raise HTTPException(403, 'Credential changes require a local application origin')


@app.put('/api/credentials/openrouter')
async def set_credential(request: Request):
    check_credential_origin(request)
    if request.headers.get('content-type', '').split(';')[0] != 'application/json':
        raise HTTPException(415, 'JSON required')
    # Manual validation avoids framework errors reflecting the secret input.
    raw = await request.body()
    if len(raw) > 8192:
        raise HTTPException(413, 'Credential request too large')
    try:
        body = json.loads(raw)
        secret = body.get('key') if isinstance(body, dict) else None
        if not isinstance(secret, str) or not 16 <= len(secret.strip()) <= 4096:
            raise ValueError()
        secret = secret.strip()
        if any(c.isspace() for c in secret):
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(422, 'Enter a key of 16–4096 characters without whitespace')
    try:
        encrypted = credentials.encrypt(secret)
    except RuntimeError:
        raise HTTPException(503, 'Encrypted credential storage unavailable')
    put('credentials', 'openrouter', {'ciphertext': encrypted})
    put('env', 'OPENROUTER_API_KEY', {'ciphertext': encrypted})
    return credential_status()


@app.delete('/api/credentials/openrouter')
def delete_credential(request: Request):
    check_credential_origin(request)
    with connect() as db:
        db.execute('DELETE FROM records WHERE kind=? AND id=?', ('credentials', 'openrouter'))
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)', ('env','OPENROUTER_API_KEY',json.dumps({'ciphertext':''})))
    return credential_status()


class RunRequest(BaseModel):
    config: dict
    steps: list[dict] = Field(min_length=1, max_length=10000)
    execution: str = 'offline'
    confirm_paid: bool = False


def ai_settings():
    import os
    from dotenv import dotenv_values
    stored=stored_env()
    env=dotenv_values(ENV_PATH,interpolate=False) if ENV_PATH.exists() else {}
    if 'OPENROUTER_API_KEY' in stored:
        ciphertext=stored['OPENROUTER_API_KEY'].get('ciphertext','')
        key=credentials.decrypt(ciphertext) if ciphertext else ''
    elif records('credentials'):
        key=credentials.decrypt(records('credentials')[0]['ciphertext'])
    else:
        key=os.environ.get('OPENROUTER_API_KEY',env.get('OPENROUTER_API_KEY') or '')
    fields={f['name']:f['value'] for f in get_env_settings()['fields']}
    temp=fields['AGENT_NODE_TEMPERATURE']
    env_settings.validate('AGENT_NODE_TEMPERATURE',temp)
    endpoint = fields['LLM_BASE_URL'].rstrip('/')
    env_settings.validate('LLM_BASE_URL', endpoint)
    if 'LLM_API_KEY' in stored:
        ciphertext = stored['LLM_API_KEY'].get('ciphertext', '')
        key = credentials.decrypt(ciphertext) if ciphertext else ''
    elif 'LLM_API_KEY' in os.environ or env.get('LLM_API_KEY'):
        key = os.environ.get('LLM_API_KEY', env.get('LLM_API_KEY') or '')
    elif endpoint != 'https://openrouter.ai/api/v1':
        key = ''  # Never send the legacy OpenRouter credential to another provider.
    return key,float(temp),fields['AGENT_NODE_MODEL'],endpoint


def validate_ai(request):
    config=request.config
    if config.get('mode')!='Full trace':
        raise HTTPException(422,'AI execution currently requires Full trace')
    if len(config['nodes'])>8 or len(json.dumps(request.steps))>100000:
        raise HTTPException(422,'First AI runner supports at most 8 nodes and 100 KB of trace JSON')
    if not config.get('objective','').strip() or not config.get('taxonomy','').strip():
        raise HTTPException(422,'Objective and taxonomy are required')
    schema=json.loads(config['schema'])
    def has_reference(obj):
        if isinstance(obj,dict): return any(k in ('$ref','$dynamicRef','$recursiveRef') or has_reference(v) for k,v in obj.items())
        return isinstance(obj,list) and any(has_reference(v) for v in obj)
    if has_reference(schema): raise HTTPException(422,'Schema references are not supported by the first AI runner')
    from jsonschema import Draft202012Validator
    try: Draft202012Validator.check_schema(schema)
    except Exception: raise HTTPException(422,'Invalid output JSON schema')
    model=config.get('model','').strip()
    if len(model) > 256:
        raise HTTPException(422,'Model ID is too long')


async def execute_ai(key, secret, temperature):
    def redact(value):
        if isinstance(value,str): return value.replace(secret,'[REDACTED]') if secret else value
        if isinstance(value,dict): return {k:redact(v) for k,v in value.items()}
        if isinstance(value,list): return [redact(v) for v in value]
        return value
    def emit(node, output):
        run=get_run(key)
        run['outputs'][node]=redact(output)
        outputs=list(run['outputs'].values())
        run['usage']={'calls':sum(o.get('calls',0) for o in outputs),
                      'tokens':sum(o.get('input_tokens',0)+o.get('output_tokens',0) for o in outputs),
                      'cost':None,'partial':any(o.get('usage_unknown',False) for o in outputs)}
        run['phase']=2 if node=='FINAL_AGGREGATOR' else 1
        put('runs',key,run)
    try:
        import ai_runner
        run=get_run(key)
        result=await ai_runner.run(run['config'],run['steps'],secret,temperature,emit)
        run=get_run(key)
        run.update(status='Completed',phase=3,**redact(result))
        run['verdict']=str(result['final_output'].get('verdict','Completed; inspect final JSON'))
        put('runs',key,redact(run))
    except asyncio.CancelledError:
        run=get_run(key)
        run.update(status='Cancelled',verdict='AI run cancelled; provider requests may already be billed')
        put('runs',key,run)
        raise
    except Exception:
        run=get_run(key)
        run.update(status='Failed',verdict='AI run failed or output did not match schema; inspect node outputs')
        put('runs',key,run)
    finally:
        tasks.pop(key,None)


def validate(config):
    nodes = config.get('nodes', [])
    edges = config.get('edges', [])
    if not nodes or len(nodes) > 100 or any(not isinstance(n,str) or not n.strip() for n in nodes) or len(set(nodes)) != len(nodes) or 'FINAL_AGGREGATOR' not in nodes:
        raise HTTPException(422, 'Unique nodes and FINAL_AGGREGATOR required (maximum 100)')
    graph = {n: [] for n in nodes}
    incoming = dict.fromkeys(nodes, 0)
    for edge in edges:
        if not isinstance(edge,list) or len(edge)!=2 or any(n not in graph for n in edge) or edge[0]=='FINAL_AGGREGATOR':
            raise HTTPException(422, 'Invalid edge')
        a,b = edge
        if b in graph[a]:
            raise HTTPException(422, 'Duplicate edge')
        graph[a].append(b)
        incoming[b]+=1
    levels=[]
    ready=[n for n in nodes if incoming[n]==0]
    while ready:
        levels.append(ready)
        following=[]
        for n in ready:
            for child in graph[n]:
                incoming[child]-=1
                if incoming[child]==0: following.append(child)
        ready=following
    if sum(map(len,levels))!=len(nodes) or any(not graph[n] for n in nodes if n!='FINAL_AGGREGATOR'):
        raise HTTPException(422, 'Graph must be acyclic and end at FINAL_AGGREGATOR')
    try:
        schema=json.loads(config.get('schema',''))
        assert isinstance(schema,dict) and schema.get('type')=='object'
        assert isinstance(json.loads(config.get('examples','[]')),list)
    except (ValueError,TypeError,AssertionError):
        raise HTTPException(422,'Invalid schema or examples')
    return levels


@app.get('/api/health')
def health():
    return {'status':'ok','execution':'offline','paid_calls_enabled':False,
            'ai_available':True,'ai_requires_explicit_confirmation':True}


@app.get('/api/runs')
def list_runs():
    return records('runs')


@app.get('/api/runs/{key}')
def run_detail(key: str):
    return get_run(key)


@app.post('/api/graph/validate')
def graph_validate(config: dict):
    return {'valid':True,'levels':validate(config)}


async def execute(key, levels):
    try:
        for index, level in enumerate(levels):
            run=get_run(key)
            for n in level: run['outputs'][n]={'status':'Running'}
            run['phase']=min(2,index+1)
            put('runs',key,run)
            await asyncio.sleep(0.6)
            run=get_run(key)
            for n in level:
                run['outputs'][n]={'status':'Completed','execution':'offline','message':'Input and dependencies accepted; no AI judgment','steps':len(run['steps'])}
            put('runs',key,run)
        run.update(status='Completed',verdict='Offline validation completed — no AI judgment',phase=3)
        put('runs',key,run)
    except asyncio.CancelledError:
        run=get_run(key)
        run.update(status='Cancelled',verdict='Cancelled — no AI judgment')
        put('runs',key,run)
        raise
    except Exception:
        run=get_run(key)
        run.update(status='Failed',verdict='Offline executor failed')
        put('runs',key,run)
    finally:
        tasks.pop(key,None)


@app.post('/api/runs', status_code=201)
async def create_run(request: RunRequest, http_request: Request):
    if request.execution not in ('offline','ai'):
        raise HTTPException(403,'Unsupported execution mode')
    if request.execution == 'ai':
        check_credential_origin(http_request)
        if not request.confirm_paid:
            raise HTTPException(403,'Explicit paid-run confirmation required')
        if any(r['status']=='Running' and r.get('execution')=='ai' for r in records('runs')):
            raise HTTPException(409,'Another AI run is already active')
    levels=validate(request.config)
    ids=[s.get('id') for s in request.steps]
    if any(not isinstance(i,int) for i in ids) or len(set(ids))!=len(ids):
        raise HTTPException(422,'Step IDs must be unique integers')
    secret=''
    temperature=0.1
    if request.execution=='ai':
        validate_ai(request)
        try: secret,temperature,default_model,endpoint=ai_settings()
        except Exception: raise HTTPException(503,'Cannot read AI settings; check credential storage and temperature')
        from urllib.parse import urlparse
        if not secret and urlparse(endpoint).hostname not in ('localhost','127.0.0.1','::1'):
            raise HTTPException(422,'Set LLM_API_KEY in Settings (OPENROUTER_API_KEY is only used for OpenRouter)')
        if not request.config.get('model','').strip() or request.config['model']=='openrouter/auto':
            request.config['model']=default_model
        if not request.config['model']:
            raise HTTPException(422,'Set the provider model ID in AGENT_NODE_MODEL')
        request.config['applied_base_url']=endpoint
        request.config['applied_temperature']=temperature
        request.config['max_output_tokens']=1024
        request.config['dollar_budget_enforced']=False
        try: import ai_runner
        except ImportError: raise HTTPException(503,'Install backend AI dependencies before running')
    key=uuid.uuid4().hex
    run={'id':key,'date':datetime.now(timezone.utc).isoformat(),'status':'Running','config':request.config,'steps':request.steps,'verdict':'Pending offline validation','phase':0,'execution':'offline','usage':{'calls':0,'tokens':0,'cost':0},'outputs':{}}
    put('runs',key,run)
    if request.execution=='ai':
        run.update(execution='ai',verdict='Pending AI evaluation',usage={'calls':0,'tokens':0,'cost':None})
        put('runs',key,run)
    tasks[key]=asyncio.create_task(execute_ai(key,secret,temperature) if request.execution=='ai' else execute(key,levels))
    return run


@app.post('/api/runs/{key}/cancel')
async def cancel(key: str):
    get_run(key)
    task=tasks.get(key)
    if task:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        run=get_run(key)
        if run['status']=='Running':
            run.update(status='Cancelled',verdict='Cancelled before execution')
            put('runs',key,run)
    return get_run(key)


@app.patch('/api/runs/{key}')
def archive(key: str, body: dict):
    run=get_run(key)
    if not isinstance(body.get('archived'),bool): raise HTTPException(422,'archived must be boolean')
    run['archived']=body['archived']
    put('runs',key,run)
    return run


@app.get('/api/runs/{key}/events')
async def events(key: str):
    get_run(key)
    async def stream():
        previous=None
        while True:
            run=get_run(key)
            data=json.dumps(run)
            if data!=previous:
                yield f'data: {data}\n\n'
                previous=data
            if run['status']!='Running': break
            await asyncio.sleep(.15)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})


@app.get('/api/workspace')
def workspace():
    return records('workspace')[0] if records('workspace') else {}


@app.put('/api/workspace')
def save_workspace(body: dict):
    allowed={k:body[k] for k in ('config','versions','traces','settings') if k in body}
    put('workspace','current',allowed)
    return {'saved':True}
