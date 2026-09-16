"""Local API with AI execution by default and optional offline validation."""
import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import credentials
import env_settings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(os.environ.get('AUTOJUDGE_DATA_DIR') or PROJECT_ROOT / '.autojudge')
DB_PATH = DATA_DIR / 'workspace.sqlite3'
ENV_PATH = PROJECT_ROOT / '.env'
WEB_DIST = Path(os.environ.get('AUTOJUDGE_WEB_DIST') or PROJECT_ROOT / 'src' / 'autojudge' / 'ui' / 'web' / 'dist')


def settings_read_only():
    return os.environ.get('AUTOJUDGE_SETTINGS_READ_ONLY', '').strip().lower() in {
        '1', 'true', 'yes', 'on'
    }


def ai_enabled():
    return os.environ.get('AUTOJUDGE_AI_ENABLED', '1').strip().lower() not in {
        '0', 'false', 'no', 'off'
    }


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
        stored = {name:json.loads(data) for name,data in db.execute(
            'SELECT id,payload FROM records WHERE kind=?', ('env',))}
    # Preserve endpoints saved before LLM_BASE_URL was renamed.
    if 'ENDPOINT_API_URL' not in stored and 'LLM_BASE_URL' in stored:
        stored['ENDPOINT_API_URL'] = stored['LLM_BASE_URL']
    return stored


@app.get('/api/settings/env')
def get_env_settings():
    stored = {} if settings_read_only() else stored_env()
    legacy = [] if settings_read_only() else records('credentials')
    storage = ({'name':'Hosting environment','available':False,'persistent':False}
               if settings_read_only() else credentials.storage_info())
    return {'fields':env_settings.resolve(ENV_PATH, stored, legacy),
            'secret_storage':storage,
            'read_only':settings_read_only(),
            'execution':'AI by default', 'applied_to_runner':True,
            'runner_fields':['OPENROUTER_API_KEY','ENDPOINT_API_URL','AGENT_NODE_MODEL','AGENT_NODE_TEMPERATURE']}


@app.put('/api/settings/env')
async def save_env_settings(request: Request):
    if settings_read_only():
        raise HTTPException(403, 'Deployment settings are read-only; use the hosting environment')
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
            if kind=='secret':
                value=value.strip()
            updates[name]=credentials.store(name,value) if kind=='secret' and value else {'disabled':True} if kind=='secret' else {'value':value}
    except (ValueError,TypeError,StopIteration):
        raise HTTPException(422,'Invalid settings: check names, temperature (0–2), port (1–65535), and URL')
    except RuntimeError:
        raise HTTPException(503,'Encrypted credential storage unavailable')
    with connect() as db:
        for name in reset:
            credentials.remove(name,stored_env().get(name))
            db.execute('DELETE FROM records WHERE kind=? AND id=?',('env',name))
            if name == 'ENDPOINT_API_URL':
                db.execute('DELETE FROM records WHERE kind=? AND id=?',('env','LLM_BASE_URL'))
            if name=='OPENROUTER_API_KEY':
                db.execute('DELETE FROM records WHERE kind=? AND id=?',('credentials','openrouter'))
        for name,value in updates.items():
            if value.get('disabled'):
                credentials.remove(name,stored_env().get(name))
            db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)',('env',name,json.dumps(value)))
            if name == 'ENDPOINT_API_URL':
                db.execute('DELETE FROM records WHERE kind=? AND id=?',('env','LLM_BASE_URL'))
    return get_env_settings()


def credential_status():
    field = next(f for f in get_env_settings()['fields'] if f['name']=='OPENROUTER_API_KEY')
    storage=credentials.storage_info()
    return {'provider': 'openrouter', 'configured': field['configured'],
            'storage': storage['name'], 'storage_available':storage['available'],
            'persistent_storage':storage['persistent'],
            'paid_calls_enabled': False}


@app.get('/api/credentials/openrouter')
def read_credential_status():
    return credential_status()


def check_credential_origin(request: Request):
    allowed = {'http://127.0.0.1:5173', 'http://localhost:5173',
               'http://127.0.0.1:8000', 'http://localhost:8000'}
    allowed.update(
        origin.strip().rstrip('/')
        for origin in os.environ.get('AUTOJUDGE_ALLOWED_ORIGINS', '').split(',')
        if origin.strip().startswith('https://') and len(origin.strip()) <= 2048
    )
    if request.headers.get('origin') not in allowed:
        raise HTTPException(403, 'This action requires an approved application origin')


@app.put('/api/credentials/openrouter')
async def set_credential(request: Request):
    if settings_read_only():
        raise HTTPException(403, 'Deployment settings are read-only; use the hosting environment')
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
    except (ValueError, TypeError):
        raise HTTPException(422, 'Enter a key of 16–4096 characters without whitespace')
    try:
        payload = credentials.store('OPENROUTER_API_KEY',secret)
    except RuntimeError:
        raise HTTPException(503, 'Encrypted credential storage unavailable')
    put('credentials', 'openrouter', payload)
    put('env', 'OPENROUTER_API_KEY', payload)
    return credential_status()


@app.delete('/api/credentials/openrouter')
def delete_credential(request: Request):
    if settings_read_only():
        raise HTTPException(403, 'Deployment settings are read-only; use the hosting environment')
    check_credential_origin(request)
    credentials.remove('OPENROUTER_API_KEY',stored_env().get('OPENROUTER_API_KEY'))
    with connect() as db:
        db.execute('DELETE FROM records WHERE kind=? AND id=?', ('credentials', 'openrouter'))
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)', ('env','OPENROUTER_API_KEY',json.dumps({'disabled':True})))
    return credential_status()


class RunRequest(BaseModel):
    config: dict
    steps: list[dict] = Field(min_length=1, max_length=10000)
    execution: str = 'ai'


def ai_settings():
    if settings_read_only():
        model = os.environ.get('AGENT_NODE_MODEL', 'google/gemini-2.5-flash')
        temp = os.environ.get('AGENT_NODE_TEMPERATURE', '0.1')
        base_url = os.environ.get(
            'ENDPOINT_API_URL',
            os.environ.get('LLM_BASE_URL', 'https://openrouter.ai/api/v1'),
        )
        env_settings.validate('ENDPOINT_API_URL', base_url)
        env_settings.validate('AGENT_NODE_MODEL', model)
        env_settings.validate('AGENT_NODE_TEMPERATURE', temp)
        key = os.environ.get('OPENROUTER_API_KEY', '')
        return key, float(temp), model, base_url
    from dotenv import dotenv_values
    stored=stored_env()
    env=dotenv_values(ENV_PATH,interpolate=False) if ENV_PATH.exists() else {}
    if 'OPENROUTER_API_KEY' in stored:
        key=credentials.load('OPENROUTER_API_KEY',stored['OPENROUTER_API_KEY'])
    elif records('credentials'):
        key=credentials.load('OPENROUTER_API_KEY',records('credentials')[0])
    else:
        key=os.environ.get('OPENROUTER_API_KEY',env.get('OPENROUTER_API_KEY') or '')
    fields={f['name']:f['value'] for f in get_env_settings()['fields']}
    temp=fields['AGENT_NODE_TEMPERATURE']
    env_settings.validate('AGENT_NODE_TEMPERATURE',temp)
    base_url=fields.get('ENDPOINT_API_URL') or 'https://openrouter.ai/api/v1'
    return key,float(temp),fields['AGENT_NODE_MODEL'],base_url


def validate_ai(request):
    config=request.config
    if config.get('mode')!='Full trace':
        raise HTTPException(422,'AI execution currently requires Full trace')
    if len(config['nodes'])>8:
        raise HTTPException(422,'AI runner supports at most 8 nodes')
    if not config.get('objective','').strip():
        raise HTTPException(422,'Objective is required')
    if not isinstance(config.get('taxonomy'),str) or not config['taxonomy'].strip():
        raise HTTPException(422,'Taxonomy is required')
    if not isinstance(config.get('schema'),str) or not config['schema'].strip():
        raise HTTPException(422,'An output schema or format description is required')
    try:
        schema=json.loads(config['schema'])
        if not isinstance(schema,dict) or schema.get('type')!='object':
            raise ValueError
        def has_reference(obj):
            if isinstance(obj,dict): return any(k in ('$ref','$dynamicRef','$recursiveRef') or has_reference(v) for k,v in obj.items())
            return isinstance(obj,list) and any(has_reference(v) for v in obj)
        if has_reference(schema): raise HTTPException(422,'Schema references are not supported by the first AI runner')
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
    except (ValueError,TypeError):
        pass  # Free-form format instruction (e.g. a prompt template); validated as text only.
    model=config.get('model','').strip()
    if model and not env_settings.valid_model_id(model):
        raise HTTPException(422,'Model ID must contain printable ASCII characters without spaces')


async def execute_ai(key, secret, temperature, base_url='https://openrouter.ai/api/v1'):
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
        result=await ai_runner.run(run['config'],run['steps'],secret,temperature,emit,base_url=base_url)
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
    if not isinstance(config.get('taxonomy'),str) or not config['taxonomy'].strip():
        raise HTTPException(422,'Taxonomy is required')
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
        if not str(config.get('schema','')).strip():
            raise HTTPException(422,'Invalid schema or examples')
    return levels


@app.get('/api/health')
def health():
    return {'status':'ok','execution':'offline','paid_calls_enabled':False,
            'ai_available':ai_enabled(),'ai_requires_explicit_confirmation':False}


@app.get('/api/runs')
def list_runs():
    return records('runs')


@app.get('/api/runs/{key}')
def run_detail(key: str):
    return get_run(key)


@app.post('/api/graph/validate')
def graph_validate(config: dict):
    return {'valid':True,'levels':validate(config)}


@app.post('/api/design/validate')
def design_validate(config: dict):
    objective=config.get('objective','')
    if not isinstance(objective,str) or not objective.strip():
        raise HTTPException(422,'Objective is required')
    taxonomy=config.get('taxonomy','')
    if not isinstance(taxonomy,str) or not taxonomy.strip():
        raise HTTPException(422,'Taxonomy is required')
    if not env_settings.valid_model_id(config.get('model','')):
        raise HTTPException(422,'Model ID must contain printable ASCII characters without spaces')
    if not isinstance(config.get('schema'),str) or not config['schema'].strip():
        raise HTTPException(422,'An output schema or format description is required')
    examples_ok=False
    schema_props=0
    try:
        examples=json.loads(config.get('examples','[]'))
        if not isinstance(examples,list): raise ValueError()
        examples_ok=True
    except Exception:
        pass
    try:
        schema=json.loads(config.get('schema',''))
        if not isinstance(schema,dict) or schema.get('type')!='object':
            raise ValueError()
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
        def has_reference(obj):
            if isinstance(obj,dict): return any(k in ('$ref','$dynamicRef','$recursiveRef') or has_reference(v) for k,v in obj.items())
            return isinstance(obj,list) and any(has_reference(v) for v in obj)
        if has_reference(schema):
            raise HTTPException(422,'Schema references are not supported by the runner')
        schema_props=len(schema.get('properties',{}))
    except Exception:
        pass  # Free-form format instruction; accepted as text.
    if not examples_ok:
        raise HTTPException(422,'Use a JSON array of examples')
    return {'valid':True,'taxonomy_chars':len(taxonomy.strip()),'properties':schema_props,'examples':len(examples)}


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
        run.update(status='Completed',verdict='Dry run completed — structure valid, no AI judgment',phase=3)
        put('runs',key,run)
    except asyncio.CancelledError:
        run=get_run(key)
        run.update(status='Cancelled',verdict='Cancelled — no AI judgment')
        put('runs',key,run)
        raise
    except Exception:
        run=get_run(key)
        run.update(status='Failed',verdict='Dry run executor failed')
        put('runs',key,run)
    finally:
        tasks.pop(key,None)


@app.post('/api/runs', status_code=201)
async def create_run(request: RunRequest, http_request: Request):
    if request.execution not in ('offline','ai'):
        raise HTTPException(403,'Unsupported execution mode')
    if request.execution == 'ai':
        if not ai_enabled():
            raise HTTPException(503, 'AI execution is disabled by the deployment owner')
        check_credential_origin(http_request)
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
        try: secret,temperature,default_model,base_url=ai_settings()
        except Exception: raise HTTPException(503,'Cannot read AI settings; check credential storage and temperature')
        if not secret:
            raise HTTPException(422,'Set OPENROUTER_API_KEY in Settings')
        if not request.config.get('model','').strip() or request.config['model']=='openrouter/auto':
            request.config['model']=default_model
        if not request.config['model']:
            raise HTTPException(422,'Set the provider model ID in AGENT_NODE_MODEL')
        if not env_settings.valid_model_id(request.config['model']):
            raise HTTPException(422,'Model ID must contain printable ASCII characters without spaces')
        request.config['applied_temperature']=temperature
        request.config['max_output_tokens']=1024
        try: import ai_runner
        except ImportError: raise HTTPException(503,'Install backend AI dependencies before running')
    key=uuid.uuid4().hex
    run={'id':key,'date':datetime.now(timezone.utc).isoformat(),'status':'Running','config':request.config,'steps':request.steps,'verdict':'Pending dry run','phase':0,'execution':'offline','usage':{'calls':0,'tokens':0,'cost':0},'outputs':{}}
    put('runs',key,run)
    if request.execution=='ai':
        run.update(execution='ai',verdict='Pending AI evaluation',usage={'calls':0,'tokens':0,'cost':None})
        put('runs',key,run)
    tasks[key]=asyncio.create_task(execute_ai(key,secret,temperature,base_url) if request.execution=='ai' else execute(key,levels))
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


@app.delete('/api/runs/{key}', status_code=204)
def delete_run(key: str):
    run=get_run(key)
    if run['status']=='Running' or key in tasks:
        raise HTTPException(409,'Cancel the running evaluation before deleting it')
    with connect() as db:
        db.execute('DELETE FROM records WHERE kind=? AND id=?',('runs',key))


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
    allowed={k:body[k] for k in ('config','settings') if k in body}
    put('workspace','current',allowed)
    return {'saved':True}


@app.get('/{path:path}', include_in_schema=False)
def frontend(path: str):
    if path.startswith('api/') or not WEB_DIST.is_dir():
        raise HTTPException(404, 'Not found')
    root = WEB_DIST.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(404, 'Not found')
    if path and candidate.is_file():
        return FileResponse(candidate)
    index = root / 'index.html'
    if not index.is_file():
        raise HTTPException(404, 'Frontend build not found')
    return FileResponse(index)
