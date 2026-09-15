import { useEffect, useState } from 'react';

type Setting = {name:string;group:string;kind:string;value:string|null;source:string;configured:boolean;overridden:boolean};
type Response = {fields:Setting[];secret_storage:{name:string;available:boolean;persistent:boolean};read_only?:boolean};
export default function EnvSettings({onSettingsSaved}:{onSettingsSaved?:(fields:Setting[])=>void}) {
  const [fields,setFields]=useState<Setting[]>([]);
  const [draft,setDraft]=useState<Record<string,string>>({});
  const [message,setMessage]=useState('');
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  const [readOnly,setReadOnly]=useState(false);
  const [secretStorage,setSecretStorage]=useState<Response['secret_storage']|null>(null);
  async function request(body?:unknown):Promise<Response> {
    const res=await fetch('/api/settings/env',{method:body===undefined?'GET':'PUT',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
    if(!res.ok)throw Error(res.status===422?'Check the temperature (0–2), port (1–65535), and URL without secrets.':'Settings are unavailable. Check the backend.');
    return res.json();
  }
  useEffect(()=>{let live=true;request().then(r=>{if(live){setFields(r.fields);setSecretStorage(r.secret_storage);setReadOnly(Boolean(r.read_only));}}).catch(e=>{if(live)setError(e.message);});return()=>{live=false;};},[]);
  async function commit(values:Record<string,string>,reset:string[]=[]) {
    setBusy(true);setError('');setMessage('');
    // Clear secret inputs on submission, including failed saves.
    setDraft(d=>Object.fromEntries(Object.entries(d).filter(([name])=>!fields.some(f=>f.name===name&&f.kind==='secret'))));
    try { const r=await request({values,reset});setFields(r.fields);setSecretStorage(r.secret_storage);setDraft({});onSettingsSaved?.(r.fields);setMessage('Settings were saved on the server. No AI requests were made.'); }
    catch(e){setError(String(e));}finally{setBusy(false);}
  }
  return <section><h2>Environment settings</h2><p>Settings from .env.template and additional settings used by the Python code.</p><p>{readOnly?'Deployment mode: values are read from the hosting environment and cannot be changed here.':'Precedence: saved settings → process environment → .env → default value. The .env file is not modified.'}</p><p>Secret storage: {secretStorage?.name??'detecting…'}{secretStorage&&!secretStorage.available?' — persistent saving is unavailable; use environment variables or a container secret mount.':''}</p><p>AI runs use the fixed OpenRouter API endpoint with OPENROUTER_API_KEY, AGENT_NODE_MODEL, and AGENT_NODE_TEMPERATURE.</p>
    {error&&<p role="alert">{error}</p>}{message&&<p role="status">{message}</p>}
    <form onSubmit={e=>{e.preventDefault();void commit(draft);}}>
    {Array.from(new Set(fields.map(f=>f.group))).map(group=><fieldset key={group} disabled={busy||readOnly} style={{border:'1px solid #dde4ef',borderRadius:8,margin:'16px 0',padding:16}}><legend>{group}</legend><div className="two-col">{fields.filter(f=>f.group===group).map(f=><div key={f.name}><label className="field"><span>{f.name} <span className="tag">{f.source}{f.kind==='secret'?f.configured?' · configured':' · not configured':''}</span></span><input type={f.kind==='secret'?'password':f.kind==='temperature'||f.kind==='port'?'number':'text'} autoComplete="off" spellCheck={false} min={f.kind==='port'?1:f.kind==='temperature'?0:undefined} max={f.kind==='port'?65535:f.kind==='temperature'?2:undefined} step={f.kind==='temperature'?'any':undefined} value={draft[f.name]??(f.kind==='secret'?'':f.value??'')} placeholder={f.kind==='secret'?(f.configured?'••••••••••••':'Enter a new secret to replace it'):''} onChange={e=>setDraft(d=>({...d,[f.name]:e.target.value}))}/></label>
    {!readOnly&&f.kind==='secret'&&<button type="button" disabled={!f.configured} onClick={()=>void commit({[f.name]:''})}>Disable secret</button>}
    {!readOnly&&<button type="button" disabled={!f.overridden} onClick={()=>void commit({},[f.name])}>Use env / default</button>}</div>)}</div></fieldset>)}
    {!readOnly&&<><button type="submit" disabled={busy||Object.keys(draft).length===0}>{busy?'Saving…':'Save changes'}</button>
    <button type="button" disabled={busy} onClick={()=>{setDraft({});setMessage('Form changes were discarded.');}}>Discard changes</button></>}
    </form></section>;
}
