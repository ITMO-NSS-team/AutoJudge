import { useEffect, useState } from 'react';

type Setting = {name:string;group:string;kind:string;value:string|null;source:string;configured:boolean;overridden:boolean};
type Response = {fields:Setting[]};
export default function EnvSettings() {
  const [fields,setFields]=useState<Setting[]>([]);
  const [draft,setDraft]=useState<Record<string,string>>({});
  const [message,setMessage]=useState('');
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  async function request(body?:unknown):Promise<Response> {
    const res=await fetch('/api/settings/env',{method:body===undefined?'GET':'PUT',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
    if(!res.ok)throw Error(res.status===422?'Проверьте температуру (0–2), порт (1–65535) и URL без секретов.':'Настройки недоступны. Проверьте backend.');
    return res.json();
  }
  useEffect(()=>{let live=true;request().then(r=>{if(live)setFields(r.fields);}).catch(e=>{if(live)setError(e.message);});return()=>{live=false;};},[]);
  async function commit(values:Record<string,string>,reset:string[]=[]) {
    setBusy(true);setError('');setMessage('');
    // Clear secret inputs on submission, including failed saves.
    setDraft(d=>Object.fromEntries(Object.entries(d).filter(([name])=>!fields.some(f=>f.name===name&&f.kind==='secret'))));
    try { const r=await request({values,reset});setFields(r.fields);setDraft({});setMessage('Настройки сохранены на сервере. Запросов к ИИ не было.'); }
    catch(e){setError(String(e));}finally{setBusy(false);}
  }
  return <section><h2>Environment settings</h2><p>Параметры из .env.template и дополнительные параметры, используемые Python-кодом.</p><p>Приоритет: сохранённые настройки → окружение процесса → .env → значение по умолчанию. Файл .env не изменяется.</p><p>Секреты шифруются Windows DPAPI. AI-run использует LLM_BASE_URL (адрес API, включая /v1), LLM_API_KEY и AGENT_NODE_TEMPERATURE. Поддерживаются OpenAI-совместимые Chat Completions endpoints. HTTPS обязателен, кроме HTTP на localhost. Локальному серверу ключ не обязателен. OPENROUTER_API_KEY используется только для стандартного адреса OpenRouter, если LLM_API_KEY не задан. При смене провайдера замените или отключите LLM_API_KEY и задайте его model ID. AGENT_NODE_MODEL применяется при выборе openrouter/auto в мастере; явно выбранная модель имеет приоритет. Генерация пула использует POOL_GEN_MODEL и POOL_GEN_TEMPERATURE с тем же подключением. Остальные настройки пока сохраняются без применения в этом исполнителе.</p>
    {error&&<p role="alert">{error}</p>}{message&&<p role="status">{message}</p>}
    <form onSubmit={e=>{e.preventDefault();void commit(draft);}}>
    {Array.from(new Set(fields.map(f=>f.group))).map(group=><fieldset key={group} disabled={busy} style={{border:'1px solid #dde4ef',borderRadius:8,margin:'16px 0',padding:16}}><legend>{group}</legend><div className="two-col">{fields.filter(f=>f.group===group).map(f=><div key={f.name}><label className="field"><span>{f.name} <span className="tag">{f.source}{f.kind==='secret'?f.configured?' · задан, не проверен':' · не задан':''}</span></span><input type={f.kind==='secret'?'password':f.kind==='temperature'||f.kind==='port'?'number':'text'} autoComplete="off" spellCheck={false} min={f.kind==='port'?1:f.kind==='temperature'?0:undefined} max={f.kind==='port'?65535:f.kind==='temperature'?2:undefined} step={f.kind==='temperature'?'any':undefined} value={draft[f.name]??(f.kind==='secret'?'':f.value??'')} placeholder={f.kind==='secret'?'Введите новый секрет, чтобы заменить':''} onChange={e=>setDraft(d=>({...d,[f.name]:e.target.value}))}/></label>
    {f.kind==='secret'&&<button type="button" disabled={!f.configured} onClick={()=>void commit({[f.name]:''})}>Отключить секрет</button>}
    <button type="button" disabled={!f.overridden} onClick={()=>void commit({},[f.name])}>Использовать env / default</button></div>)}</div></fieldset>)}
    <button type="submit" disabled={busy||Object.keys(draft).length===0}>{busy?'Сохранение…':'Сохранить изменения'}</button>
    <button type="button" disabled={busy} onClick={()=>{setDraft({});setMessage('Изменения формы отменены.');}}>Отменить изменения</button>
    </form></section>;
}
