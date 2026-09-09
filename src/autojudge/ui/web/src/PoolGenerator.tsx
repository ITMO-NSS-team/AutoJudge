import { useEffect, useState } from 'react';

type Judge = {name:string;instructions:string};
type Pool = {id:string;status:string;model:string;judges?:Judge[];usage?:{calls:number;tokens:number}};
type Design = {objective:string;taxonomy:string;schema:string;examples:string};
export default function PoolGenerator({design,apply}:{design:Design;apply:(judges:Judge[])=>void}) {
  const [history,setHistory]=useState<Pool[]>([]);
  const [draft,setDraft]=useState<Judge[]>([]);
  const [confirmed,setConfirmed]=useState(false);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const [connection,setConnection]=useState('');
  useEffect(()=>{let alive=true;void Promise.all([fetch('/api/judge-pools').then(r=>r.json()),fetch('/api/settings/env').then(r=>r.json())]).then(([p,s])=>{if(alive){setHistory(p);setConnection(s.fields.filter((f:{name:string})=>['LLM_BASE_URL','POOL_GEN_MODEL'].includes(f.name)).map((f:{value:string})=>f.value).join(' · '));}}).catch(()=>{if(alive)setError('Не удалось загрузить историю или настройки.');});return()=>{alive=false;};},[]);
  useEffect(()=>setConfirmed(false),[design.objective,design.taxonomy,design.schema,design.examples]);
  async function generate() {
    setBusy(true);setError('');setConfirmed(false);
    try {
      const response=await fetch('/api/judge-pools/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({objective:design.objective,taxonomy:design.taxonomy,schema_text:design.schema,examples:design.examples,confirm_paid:true})});
      if(!response.ok)throw Error(await response.text());
      const pool:Pool=await response.json();setHistory(h=>[pool,...h]);setDraft(pool.judges??[]);
    }catch(e){setError(String(e));}finally{setBusy(false);}
  }
  const valid=draft.length>=2&&draft.length<=8&&new Set(draft.map(j=>j.name)).size===draft.length&&draft.filter(j=>j.name==='FINAL_AGGREGATOR').length===1&&draft.every(j=>j.name.trim()&&j.instructions.trim());
  return <section><h3>Generate judge pool</h3><p>{connection || 'Подключение из Settings'}</p><p>Один AI-запрос, до 4000 выходных токенов, без повторов и tools. Передаются objective, taxonomy, schema и examples; сама трасса не отправляется. Возможна тарификация провайдером. Результат сохраняется отдельно и не заменяет текущий пул автоматически.</p>
    <label><input type="checkbox" checked={confirmed} disabled={busy} onChange={e=>setConfirmed(e.target.checked)}/> Подтверждаю передачу design провайдеру и возможные расходы на генерацию</label>
    <button disabled={!confirmed||busy||!connection} onClick={()=>void generate()}>{busy?'Генерация… не запускайте повторно':'Generate judges'}</button>
    {error&&<p role="alert">{error}</p>}
    {draft.length>0&&<><h3>Черновик — проверьте инструкции</h3>{draft.map((j,i)=><div key={i}><label className="field">Name<input value={j.name} onChange={e=>setDraft(d=>d.map((v,k)=>k===i?{...v,name:e.target.value}:v))}/></label><label className="field">Instructions<textarea value={j.instructions} onChange={e=>setDraft(d=>d.map((v,k)=>k===i?{...v,instructions:e.target.value}:v))}/></label></div>)}<p>Применение заменит текущий пул и граф на связи специалистов с FINAL_AGGREGATOR.</p><button disabled={!valid||busy} onClick={()=>{apply(draft);setDraft([]);}}>Применить проверенный пул</button><button onClick={()=>setDraft([])}>Закрыть черновик</button></>}
    <h3>Сохранённые генерации</h3>{history.map(p=><div className="edge-row" key={p.id}><span>{p.id.slice(0,8)} · {p.model} · {p.status}{p.usage?` · ${p.usage.calls} calls / ${p.usage.tokens} tokens`:''}</span>{p.judges&&<button disabled={busy} onClick={()=>setDraft(structuredClone(p.judges!))}>Открыть без нового запроса</button>}</div>)}
  </section>;
}
