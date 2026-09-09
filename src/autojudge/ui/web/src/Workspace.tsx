import { useEffect, useState } from 'react';
import { Scale, LayoutDashboard, Play, History, GitFork, Layers, Database, Activity, Settings, FlaskConical, Download, Plus, Check, ArrowRight, Search } from 'lucide-react';
import './workspace.css';
import './wizard.css';
import EnvSettings from './EnvSettings';
import PoolGenerator from './PoolGenerator';
import { exampleTaxonomy, exampleOutputSchema } from './designTemplate';
import { normalizeTrace, parseDesignFile } from './imports';

type Step = { id: number; agent: string; content: string };
type Config = { judge_instructions?: Record<string,string>; name: string; objective: string; taxonomy: string; schema: string; examples: string; model: string; mode: string; budget: number; nodes: string[]; edges: string[][] };
async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> { const res = await fetch('/api'+path, {method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)}); if(!res.ok) throw Error(await res.text()); return res.json(); }

type Run = { execution?: string; final_output?: unknown; usage?: {calls:number;tokens:number;cost:number|null;partial?:boolean}; outputs?: Record<string, unknown>; phase?: number; id: string; date: string; status: string; config: Config; steps: Step[]; verdict: string; archived?: boolean };
type Batch = { id: string; total: number; done: number; status: string };
const sections = ['Overview', 'New evaluation', 'Runs', 'Compare', 'Judge Studio', 'Batches', 'Data', 'Optimizer Lab', 'Observability', 'Settings'];
const icons = [LayoutDashboard, Play, History, GitFork, Scale, Layers, Database, FlaskConical, Activity, Settings];
const roles = ['Evidence Grounding', 'Causal Attribution', 'Tool-Use Compliance', 'Logical Consistency', 'FINAL_AGGREGATOR'];
const sample: Step[] = [{ id: 12, agent: 'Researcher', content: 'The source requires evidence of intent.' }, { id: 13, agent: 'Retriever', content: 'Retrieved the supporting source.' }, { id: 17, agent: 'Researcher', content: 'Intent is not required.' }, { id: 24, agent: 'Reviewer', content: 'Final answer submitted.' }];
const initial: Config = { name: 'Trace #042', objective: 'Find unsupported claims and attribute failures to agent decisions.', taxonomy: 'Unsupported claim\nContradiction\nIncorrect tool use', schema: '{"type":"object","properties":{"verdict":{"type":"string"},"evidence":{"type":"array"}}}', examples: '[]', model: 'openrouter/auto', mode: 'Full trace', budget: 1, nodes: roles, edges: roles.slice(0, -1).map(n => [n, 'FINAL_AGGREGATOR']) };
function read<T,>(key: string, fallback: T): T { try { return JSON.parse(localStorage.getItem(key) || 'null') ?? fallback; } catch { return fallback; } }
function saveFile(name: string, data: unknown) { const url = URL.createObjectURL(new Blob([typeof data === 'string' ? data : JSON.stringify(data, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 500); }
function validate(c: Config) {
  if (!c.nodes.includes('FINAL_AGGREGATOR')) return 'Добавьте FINAL_AGGREGATOR.';
  if (new Set(c.nodes).size !== c.nodes.length || c.nodes.some(n => !n.trim())) return 'Имена узлов должны быть непустыми и уникальными.';
  if (c.edges.some(([a,b]) => !c.nodes.includes(a) || !c.nodes.includes(b) || a === 'FINAL_AGGREGATOR')) return 'Недопустимое ребро или исходящая связь агрегатора.';
  const visit = (n: string, trail: string[]): boolean => trail.includes(n) || c.edges.filter(e => e[0] === n).some(e => visit(e[1], [...trail, n]));
  if (c.nodes.some(n => visit(n, []))) return 'Граф содержит цикл.';
  const reaches = (n: string): boolean => n === 'FINAL_AGGREGATOR' || c.edges.filter(e => e[0] === n).some(e => reaches(e[1]));
  if (c.nodes.some(n => !reaches(n))) return 'Каждый судья должен передавать результат агрегатору.';
  return '';
}
function Field({ label, value, change, area = false, type = 'text' }: { label: string; value: string | number; change: (v: string) => void; area?: boolean; type?: string }) { return <label className="field"><span>{label}</span>{area ? <textarea value={value} onChange={e => change(e.target.value)} /> : <input type={type} value={value} onChange={e => change(e.target.value)} />}</label>; }
function Graph({ config }: { config: Config }) { return <div className="dag"><div className="dag-row">{config.nodes.filter(n => n !== 'FINAL_AGGREGATOR').map(n => <div key={n}><GitFork size={18}/><strong>{n}</strong></div>)}</div><div className="dag-links">{config.edges.map(([a,b], i) => <span key={i}>{a} → {b}</span>)}</div><div className="aggregator"><Layers size={18}/> FINAL_AGGREGATOR</div></div>; }

export default function Workspace() {
  const [previousDesign,setPreviousDesign]=useState<{taxonomy:string;schema:string}|null>(null);
  const [connected, setConnected] = useState(false);
  const [execution,setExecution]=useState('offline');
  const [confirmPaid,setConfirmPaid]=useState(false);
  const [launching,setLaunching]=useState(false);
  const [page, setPage] = useState(() => sections.includes(decodeURIComponent(location.hash.slice(1))) ? decodeURIComponent(location.hash.slice(1)) : 'Overview');
  useEffect(()=>{setConfirmPaid(false);},[page]);
  const [config, setConfig] = useState<Config>(() => read('aj-config', initial));
  const [steps, setSteps] = useState<Step[]>(sample);
  const [raw, setRaw] = useState(JSON.stringify(sample, null, 2));
  const [runs, setRuns] = useState<Run[]>(() => read('aj-runs', []));
  const [versions, setVersions] = useState<Config[]>(() => read('aj-versions', []));
  const [traces, setTraces] = useState<{name: string; steps: Step[]}[]>(() => read('aj-traces', []));
  const [batches, setBatches] = useState<Batch[]>(() => read('aj-batches', []));
  const [wizard, setWizard] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('All');
  const [compare, setCompare] = useState<string[]>([]);
  const [detail, setDetail] = useState<Run | null>(null);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [tab, setTab] = useState('Taxonomies');
  const [phase, setPhase] = useState(-1);
  const [active, setActive] = useState<Run | null>(null);
  const [edgeFrom, setEdgeFrom] = useState(roles[0]);
  const [edgeTo, setEdgeTo] = useState('FINAL_AGGREGATOR');
  const [batchSize, setBatchSize] = useState(10);
  const [draftPrompt, setDraftPrompt] = useState('Inspect every claim and cite the supporting trace step.');
  const [candidate, setCandidate] = useState('');
  const [settings, setSettings] = useState(() => read('aj-settings', { timeout: 120, retries: 2, concurrency: 4, budget: 10, langfuse: '' }));
  const navigate = (p: string) => { setPage(p); location.hash = encodeURIComponent(p); setError(''); setQuery(''); };
  useEffect(() => { const fn = () => { const p = decodeURIComponent(location.hash.slice(1)); if (sections.includes(p)) setPage(p); }; window.addEventListener('hashchange', fn); return () => window.removeEventListener('hashchange', fn); }, []);
  useEffect(() => { try { localStorage.setItem('aj-runs', JSON.stringify(runs)); localStorage.setItem('aj-config', JSON.stringify(config)); localStorage.setItem('aj-versions', JSON.stringify(versions)); localStorage.setItem('aj-traces', JSON.stringify(traces)); localStorage.setItem('aj-batches', JSON.stringify(batches)); localStorage.setItem('aj-settings', JSON.stringify(settings)); } catch { setError('Локальное хранилище заполнено или недоступно. Экспортируйте результаты.'); } }, [runs, config, versions, traces, batches, settings]);
  useEffect(() => {
    let mounted = true;
    Promise.all([api<Run[]>('/runs'),api<{config?:Config;versions?:Config[];traces?:{name:string;steps:Step[]}[]}>('/workspace')]).then(([rs, ws]) => {
      if(!mounted)return;
      setRuns(rs); if(ws.config)setConfig(ws.config); if(ws.versions)setVersions(ws.versions); if(ws.traces)setTraces(ws.traces);
      setActive(rs.find(r=>r.status==='Running')??null);setConnected(true);
    }).catch(e=>setError('Backend offline: '+e.message));
    return ()=>{mounted=false;};
  }, []);
  useEffect(() => {
    if(!connected)return;
    const timer=setTimeout(()=>api('/workspace','PUT',{config,versions,traces,settings}).catch(e=>setError(e.message)),500);
    return ()=>clearTimeout(timer);
  },[connected,config,versions,traces,settings]);
  useEffect(() => {
    if(!active)return;
    const source=new EventSource('/api/runs/'+active.id+'/events');
    source.onmessage=e=>{const r=JSON.parse(e.data) as Run;setPhase(r.phase??0);setRuns(rs=>[r,...rs.filter(x=>x.id!==r.id)]);
      if(r.status!=='Running'){setDetail(r);setActive(null);setPhase(-1);setWizard(5);source.close();}
    };
    source.onerror=()=>setError('Поток событий отключён. Браузер переподключается; результат сохраняется на сервере.');
    return ()=>source.close();
  },[active?.id]);
  useEffect(() => { if (!batches.some(b => b.status === 'Running')) return; const timer = setTimeout(() => setBatches(bs => bs.map(b => b.status !== 'Running' ? b : { ...b, done: b.done + 1, status: b.done + 1 >= b.total ? 'Completed' : 'Running' })), 800); return () => clearTimeout(timer); }, [batches]);
  const update = (key: keyof Config, value: Config[keyof Config]) => setConfig(c => ({ ...c, [key]: value }));
  function parseTrace(text: string, persist = false) {
    try { const normalized = normalizeTrace(text);
      if (persist) setTraces(t => [...t, { name: config.name, steps: normalized }]);
      setSteps(normalized); setError(''); setNotice(`Проверено ${normalized.length} шагов`); return true;
    } catch (e) { setError(e instanceof Error ? e.message : 'Некорректный JSON'); return false; }
  }
  const templateButtons = <div className="toolbar"><button type="button" onClick={()=>{
    setPreviousDesign({taxonomy:config.taxonomy,schema:config.schema});
    setConfig(c=>({...c,taxonomy:exampleTaxonomy,schema:JSON.stringify(exampleOutputSchema,null,2)}));
    setConfirmPaid(false);setError('');setNotice('Шаблон из example.md применён. Taxonomy и schema заменены; остальные поля сохранены.');
  }}>Применить taxonomy + output schema из example.md</button>{previousDesign&&<button type="button" onClick={()=>{
    setConfig(c=>({...c,...previousDesign}));setPreviousDesign(null);setConfirmPaid(false);setNotice('Предыдущие taxonomy и schema восстановлены.');
  }}>Отменить применение шаблона</button>}</div>;
  async function loadTestTrace(id: number) {
    try {
      const response = await fetch('/test-data/trace-0'+id+'.json');
      if (!response.ok) throw Error('Не удалось загрузить тестовую трассу.');
      const text = await response.text();
      if (parseTrace(text)) { setRaw(text); update('name','GAIA test '+id); update('mode','Full trace'); setConfirmPaid(false); }
    } catch (e) { setError(String(e)); }
  }
  async function importDesign(kind: 'schema' | 'taxonomy', file?: File) {
    if (!file) return;
    try {
      if (file.size > 1_000_000) throw Error('Лимит файла: 1 MB.');
      const value = parseDesignFile(kind, await file.text());
      update(kind, value); setError(''); setNotice('Загружен ' + file.name);
    } catch (e) { setError(String(e)); }
  }
  const designUpload = (kind: 'schema' | 'taxonomy') => <label className="file-input">
    {kind === 'schema' ? 'Загрузить output schema (.json)' : 'Загрузить taxonomy (.txt, .md, .json)'}
    <input aria-label={kind === 'schema' ? 'Upload output schema' : 'Upload taxonomy'} type="file"
      accept={kind === 'schema' ? '.json' : '.txt,.md,.json'} onChange={e=>{
        void importDesign(kind,e.target.files?.[0]); e.target.value='';
      }}/>
  </label>;
  function designValid() { try { const schema = JSON.parse(config.schema); if (!schema || schema.type !== 'object') throw Error('Schema должна описывать object.'); if (!Array.isArray(JSON.parse(config.examples))) throw Error('Examples должны быть массивом.'); if (!config.objective.trim() || !config.taxonomy.trim() || !config.model.trim() || !Number.isFinite(config.budget) || config.budget <= 0) throw Error('Заполните цель, taxonomy, модель и положительный budget.'); setError(''); return true; } catch (e) { setError(String(e)); return false; } }
  async function launch() { if (!parseTrace(raw)) { setWizard(0); return; } const issue=validate(config);if(issue||!designValid()){if(issue)setError(issue);return;}try{
    if(execution==='ai'&&!confirmPaid){setError('Подтвердите отправку трассы провайдеру и платный запуск.');return;}
    setLaunching(true);
    const run=await api<Run>('/runs','POST',{config,steps:normalizeTrace(raw),execution,confirm_paid:confirmPaid});setConfirmPaid(false);setDetail(null);setActive(run);setPhase(0);setWizard(4);
  }catch(e){setError(String(e));}finally{setLaunching(false);}}
  async function cancel(){if(!active)return;try{await api('/runs/'+active.id+'/cancel','POST');}catch(e){setError(String(e));}}
  async function archiveRun(r:Run){try{const updated=await api<Run>('/runs/'+r.id,'PATCH',{archived:!r.archived});setRuns(rs=>rs.map(x=>x.id===r.id?updated:x));}catch(e){setError(String(e));}}
  function next() { if (wizard === 0 && !parseTrace(raw)) return; if (wizard === 1 && !designValid()) return; if (wizard === 3 && validate(config)) { setError(validate(config)); return; } setWizard(Math.min(5, wizard + 1)); }
  const filteredRuns = runs.filter(r => (filter === 'Archived' ? r.archived : !r.archived && (filter === 'All' || r.status === filter)) && `${r.id} ${r.config.name} ${r.status}`.toLowerCase().includes(query.toLowerCase()));
  const table = (list: Run[]) => <div className="table-wrap"><table><thead><tr><th>Compare</th><th>Run / Trace</th><th>Pipeline</th><th>Verdict</th><th>Model / Context</th><th>Actions</th></tr></thead><tbody>{list.map(r => <tr key={r.id}><td><input aria-label={`Compare ${r.id}`} type="checkbox" checked={compare.includes(r.id)} onChange={e => setCompare(e.target.checked ? [...compare,r.id] : compare.filter(id => id !== r.id))}/></td><td><button className="link" onClick={() => {setDetail(r); navigate('Runs');}}>{r.config.name}</button><small>{r.id} · {new Date(r.date).toLocaleString()}</small></td><td><span className="tag">{r.status}</span></td><td>{r.verdict}</td><td>{r.config.model}<small>{r.config.mode}</small></td><td><button onClick={() => saveFile(`run-${r.id}.json`, r)}>Export</button><button onClick={() => {setConfig(structuredClone(r.config)); setSteps(r.steps); setRaw(JSON.stringify(r.steps,null,2)); setWizard(0); navigate('New evaluation');}}>Clone</button><button onClick={() => archiveRun(r)}>{r.archived ? 'Restore' : 'Archive'}</button></td></tr>)}</tbody></table>{!list.length && <div className="empty">Запусков пока нет. Создайте демонстрационную оценку.</div>}</div>;
  const traceView = (data: Step[]) => <div className="step-list">{data.filter(s => `${s.agent} ${s.content} ${s.id}`.toLowerCase().includes(query.toLowerCase())).map(s => <button key={s.id} className={selectedStep === s.id ? 'picked' : ''} onClick={() => setSelectedStep(s.id)}><span>#{s.id} · {s.agent}</span><p>{s.content}</p></button>)}</div>;
  const resultView = (r: Run) => <div className="two-col"><section className="panel"><h2>Final verdict <span className="tag">{r.status}</span></h2><div className="finding">{r.verdict}</div><p>{r.execution==='ai' ? 'Реальная оценка AutoJudge. Проверьте вывод и доказательства.' : 'Offline-проверка: без ИИ-вызовов.'}</p>{r.usage&&<p>Calls: {r.usage.calls} · Tokens: {r.usage.tokens}{r.usage.partial?' (неполные данные)':''} · Cost: {r.usage.cost===null?'не получена от провайдера':r.usage.cost+' USD'}</p>}{r.final_output!==undefined&&<pre>{JSON.stringify(r.final_output,null,2)}</pre>}<h3>Source steps</h3>{r.steps.slice(0,2).map(s => <button key={s.id} onClick={() => {setSelectedStep(s.id);setQuery('');}}>Открыть шаг #{s.id}</button>)}<h3>Node outputs</h3>{r.config.nodes.map(n => <details key={n}><summary>{n}</summary><pre>{JSON.stringify(r.outputs?.[n] ?? {status:'Not executed'},null,2)}</pre></details>)}<button onClick={() => saveFile('run-'+r.id+'.json',r)}><Download size={15}/> Export snapshot</button></section><section className="panel"><h2>Source trace</h2>{traceView(r.steps)}</section></div>;

  return <div className="workspace-app"><aside><div className="brand"><Scale/> AutoJudge</div><small className="nav-caption">RESEARCH WORKSPACE</small><nav>{sections.map((s,i) => {const Icon = icons[i]; return <button key={s} className={page === s ? 'active' : ''} onClick={() => navigate(s)}><Icon size={17}/>{s}</button>;})}</nav><div className="local-note">Локальный прототип<br/>Запуски и конфигурации сохраняются через API</div></aside><div className="work-body"><header><span>Workspace / <strong>{page}</strong></span><span className="tag demo">{connected ? 'Backend connected · AI available' : 'Backend offline'}</span></header><main><div className="work-title"><div><span className="eyebrow">AUTOJUDGE / WORKBENCH</span><h1>{page}</h1><p>От конфигурации судей до проверяемого результата.</p></div><button className="primary-button" onClick={() => {setWizard(0);navigate('New evaluation');}}><Plus size={16}/> New evaluation</button></div>{error && <div role="alert" className="error-banner">{error}<button onClick={() => setError('')}>Закрыть</button></div>}{notice && <div role="status" className="notice">{notice}<button onClick={() => setNotice('')}>×</button></div>}

  {page === 'Overview' && <><div className="kpis">{[['Runs',runs.length],['Completed pipelines',runs.filter(r=>r.status==='Completed').length],['Active batches',batches.filter(b=>b.status==='Running').length],['Actual LLM cost','—']].map(([label,value])=><section className="panel" key={label}><small>{label}</small><strong>{value}</strong></section>)}</div><section className="panel"><h2>Последние запуски</h2>{table(runs.filter(r=>!r.archived).slice(0,5))}</section><div className="two-col"><section className="panel"><h2>Service health</h2>{['OpenRouter','Langfuse','PostgreSQL'].map(s=><p key={s}>{s}<span className="tag right">Не подключён</span></p>)}</section><section className="panel"><h2>Начните исследование</h2><p>Загрузите trace, настройте судей, проверьте граф и пройдите демонстрационный запуск.</p><button onClick={()=>navigate('New evaluation')}>Открыть мастер <ArrowRight size={16}/></button><button onClick={()=>navigate('Batches')}>Создать batch</button></section></div></>}

  {page === 'New evaluation' && <><div className="wizard-tabs">{['Trace','Design','Judge pool','Graph','Run','Verdict'].map((s,i)=><button type="button" key={s} aria-current={wizard===i?'step':undefined} className={wizard===i?'current':''} onClick={()=>{setWizard(i);setError('');}}>{i+1}. {s}</button>)}</div><section className="panel">
    {wizard===0 && <div className="two-col"><div><h2>Загрузить trace</h2><p>Тестовые трассы из архива GAIA, без эталонной разметки. Вложения не включены.</p><div className="toolbar">{[1,2,3].map(id=><button key={id} onClick={()=>void loadTestTrace(id)}>GAIA test {id}</button>)}</div><Field label="Trace name" value={config.name} change={v=>update('name',v)}/><label className="file-input">JSON / JSONL<input type="file" accept=".json,.jsonl" onChange={async e=>{const f=e.target.files?.[0]; if(f){if(f.size>5_000_000){setError('Лимит локального прототипа: 5 MB.');return;}const text=await f.text();setRaw(text);update('name',f.name);parseTrace(text);}}}/></label><Field label="Вставить JSON / JSONL" area value={raw} change={setRaw}/><button onClick={()=>parseTrace(raw)}>Проверить trace</button><button onClick={()=>{if(parseTrace(raw, true)) setNotice('Трасса сохранена в Data.');}}>Сохранить в Data</button><Field label="Context mode" value={config.mode} change={v=>update('mode',v)}/><select aria-label="Context mode" value={config.mode} onChange={e=>update('mode',e.target.value)}>{['Full trace','Summary only','Summary + retrieval','Auto'].map(m=><option key={m}>{m}</option>)}</select>{config.mode!=='Full trace'&&<p className="hint">Настройка сохраняется; summarizer и retrieval требуют backend.</p>}</div><div><h2>Trace preview · {steps.length} steps</h2>{traceView(steps)}</div></div>}
    {wizard===1 && <>{templateButtons}<p>Шаблон предназначен для атрибуции ошибки; для успешных и неопределённых случаев адаптируйте schema.</p><p>Тестовые файлы: <a href="/test-data/output-schema.json" download>output schema</a> · <a href="/test-data/TAXONOMY.md" download>taxonomy</a>. Скачайте и загрузите нужный файл ниже.</p><div className="two-col"><div><Field label="Evaluation objective" area value={config.objective} change={v=>update('objective',v)}/>{designUpload('taxonomy')}<Field label="Taxonomy" area value={config.taxonomy} change={v=>update('taxonomy',v)}/><Field label="Judge model" value={config.model} change={v=>update('model',v)}/><Field label="Budget target ($, не жёсткий лимит)" type="number" value={config.budget} change={v=>update('budget',Number(v))}/></div><div>{designUpload('schema')}<Field label="Output schema (JSON)" area value={config.schema} change={v=>update('schema',v)}/><Field label="Few-shot examples (JSON array)" area value={config.examples} change={v=>update('examples',v)}/><button onClick={()=>{if(designValid())setNotice('Синтаксис schema и examples проверен.');}}>Validate design</button></div></div></>}
    {wizard===2 && <><h2>Review judge pool <span className="tag">Редактируемый пул</span></h2><p>Проверьте роли и инструкции до запуска. Можно использовать шаблон, ручной пул или AI-генерацию.</p><PoolGenerator design={config} apply={judges=>{const nodes=judges.map(j=>j.name);setConfig(c=>({...c,nodes,judge_instructions:Object.fromEntries(judges.map(j=>[j.name,j.instructions])),edges:nodes.filter(n=>n!=='FINAL_AGGREGATOR').map(n=>[n,'FINAL_AGGREGATOR'])}));setNotice('Пул применён и сохранится с конфигурацией. Проверьте Graph.');}}/><div className="role-grid">{config.nodes.map((n,i)=><div className="role-card" key={i}><Scale size={22}/><Field label={`Judge ${i+1}`} value={n} change={v=>{const old=config.nodes[i];setConfig(c=>({...c,judge_instructions:{...c.judge_instructions,[v]:c.judge_instructions?.[old]??''},nodes:c.nodes.map((x,k)=>k===i?v:x),edges:c.edges.map(e=>e.map(x=>x===old?v:x))}));}}/><Field label="Judge instructions" area value={config.judge_instructions?.[n]??''} change={v=>update('judge_instructions',{...config.judge_instructions,[n]:v})}/><span className="tag">{config.model}</span><button onClick={()=>setConfig(c=>({...c,nodes:c.nodes.filter((_,k)=>k!==i),edges:c.edges.filter(e=>!e.includes(n))}))}>Remove</button></div>)}</div><button onClick={()=>update('nodes',[...config.nodes,`Judge ${config.nodes.length+1}`])}>Add judge</button><button onClick={()=>{update('nodes',roles);update('edges',initial.edges);update('judge_instructions',{});}}>Restore template</button></>}
    {wizard===3 && <><h2>Graph editor</h2><Graph config={config}/><div className="toolbar"><select aria-label="Source node" value={edgeFrom} onChange={e=>setEdgeFrom(e.target.value)}>{config.nodes.map(n=><option key={n}>{n}</option>)}</select><ArrowRight size={18}/><select aria-label="Target node" value={edgeTo} onChange={e=>setEdgeTo(e.target.value)}>{config.nodes.map(n=><option key={n}>{n}</option>)}</select><button onClick={()=>{if(!config.edges.some(e=>e[0]===edgeFrom&&e[1]===edgeTo))update('edges',[...config.edges,[edgeFrom,edgeTo]]);}}>Add edge</button><button onClick={()=>{const issue=validate(config);setError(issue);if(!issue)setNotice('DAG валиден: циклов нет, все узлы ведут к terminal aggregator.');}}>Validate DAG</button></div>{config.edges.map((e,i)=><div className="edge-row" key={i}>{e.join(' → ')}<button onClick={()=>update('edges',config.edges.filter((_,k)=>k!==i))}>Remove</button></div>)}</>}
    {wizard===4 && <><h2>{active?'Сервер выполняет '+(active.execution==='ai'?'AI pipeline':'offline-проверку'):'Ready to run'}</h2>{!active&&<><label className="field"><span>Execution mode</span><select disabled={launching} value={execution} onChange={e=>{setExecution(e.target.value);setConfirmPaid(false);}}><option value="offline">Offline — бесплатно, проверка структуры</option><option value="ai">AI — реальный AutoJudge Pipeline</option></select></label>{execution==='ai'&&<div className="finding"><p>Endpoint из Settings · {config.model==='openrouter/auto'?'модель из AGENT_NODE_MODEL':config.model} · {config.nodes.length} узлов. До одного запроса и 1024 выходных токенов на узел, без автоматических повторов. Только Full trace, без инструментов.</p><p>Трасса будет передана на endpoint из Settings. Budget target не является жёстким денежным лимитом. Стоимость проверяйте у провайдера; используйте лимит ключа у провайдера. Отмена не возвращает уже списанные средства.</p><label><input type="checkbox" checked={confirmPaid} onChange={e=>setConfirmPaid(e.target.checked)}/> Подтверждаю отправку данных и платный запуск</label></div>}</>}<Graph config={active?.config ?? config}/>{phase>=0&&<><progress max={4} value={phase+1}/><div aria-live="polite">{['Preparing trace','Executing judges','Aggregation','Saving result'][phase]}</div><button onClick={cancel}>Cancel run</button></>}{!active&&<button className="primary-button" disabled={launching||!connected||(execution==='ai'&&!confirmPaid)} onClick={launch}><Play size={16}/>{launching?'Создание запуска…':execution==='ai'?'Run AI evaluation':'Run offline validation'}</button>}</>}
    {wizard===5 && (detail?resultView(detail):<p>Результата пока нет. Откройте Run и выполните offline-проверку или подтверждённую AI-оценку.</p>)}
    <div className="wizard-footer"><button disabled={wizard===0||!!active} onClick={()=>setWizard(wizard-1)}>Back</button>{wizard<4&&<button className="primary-button" onClick={next}>Продолжить <ArrowRight size={15}/></button>}</div></section></>}

  {page==='Runs'&&<><section className="panel"><div className="toolbar"><Search size={17}/><input aria-label="Search runs" placeholder="Поиск по trace, ID, статусу" value={query} onChange={e=>setQuery(e.target.value)}/><select aria-label="Run status" value={filter} onChange={e=>setFilter(e.target.value)}>{['All','Completed','Cancelled','Archived'].map(s=><option key={s}>{s}</option>)}</select><button disabled={compare.length<2} onClick={()=>navigate('Compare')}>Compare ({compare.length})</button></div>{table(filteredRuns)}</section>{detail&&resultView(detail)}</>}
  {page==='Compare'&&<section className="panel"><h2>Compare runs</h2><p>Выберите минимум два запуска в Runs. Стоимость и quality metrics недоступны без реальных оценок.</p>{compare.length<2?<button onClick={()=>navigate('Runs')}>Выбрать запуски</button>:<div className="comparison">{runs.filter(r=>compare.includes(r.id)).map(r=><article key={r.id}><h3>{r.config.name}</h3><span className="tag">{r.id}</span><p>{r.verdict}</p><p>Model: {r.config.model}</p><p>Context: {r.config.mode}</p><p>Judges: {r.config.nodes.length}</p><details><summary>Configuration snapshot</summary><pre>{JSON.stringify(r.config,null,2)}</pre></details></article>)}</div>}</section>}
  {page==='Judge Studio'&&<section className="panel">{templateButtons}<div className="tabs">{['Taxonomies','Output schemas','Examples','Judge pools','Graphs'].map(t=><button className={tab===t?'active':''} onClick={()=>setTab(t)} key={t}>{t}</button>)}</div>{tab==='Taxonomies'&&<>{designUpload('taxonomy')}<Field label="Taxonomy draft" area value={config.taxonomy} change={v=>update('taxonomy',v)}/ ></>}{tab==='Output schemas'&&<>{designUpload('schema')}<Field label="Schema draft" area value={config.schema} change={v=>update('schema',v)}/ ></>}{tab==='Examples'&&<Field label="Example set" area value={config.examples} change={v=>update('examples',v)}/ >}{tab==='Judge pools'&&<><h2>{config.nodes.length} judges</h2>{config.nodes.map(n=><p key={n}>{n} <span className="tag">{config.model}</span></p>)}<button onClick={()=>{setWizard(2);navigate('New evaluation');}}>Edit pool</button></>}{tab==='Graphs'&&<><Graph config={config}/><button onClick={()=>{setWizard(3);navigate('New evaluation');}}>Edit graph</button></>}<div className="toolbar"><button onClick={()=>{if(designValid()&&!validate(config)){setVersions(v=>[...v,structuredClone(config)]);setNotice('Новая версия конфигурации сохранена.');}else if(validate(config))setError(validate(config));}}>Save new version</button><button onClick={()=>saveFile('judge-design.json',config)}>Export design</button></div><h3>Version history</h3>{versions.map((v,i)=><div className="edge-row" key={i}><span>v{i+1} · {v.name} · {v.nodes.length} judges</span><button onClick={()=>{setConfig(structuredClone(v));setNotice(`Загружена копия v${i+1}`);}}>Use copy</button><button onClick={()=>saveFile(`design-v${i+1}.json`,v)}>Export</button></div>)}</section>}
  {page==='Batches'&&<><section className="panel"><h2>Batch simulation</h2><p>Проверка интерфейса calibration, pause и resume. Здесь не обрабатывается dataset и не расходуется API budget.</p><Field label="Количество демонстрационных элементов (1–100)" type="number" value={batchSize} change={v=>setBatchSize(Number(v))}/><button disabled={!Number.isInteger(batchSize)||batchSize<1||batchSize>100} onClick={()=>setBatches(b=>[...b,{id:crypto.randomUUID().slice(0,8),total:batchSize,done:0,status:'Calibration review'}])}>Create calibration draft</button></section>{batches.map(b=><section className="panel" key={b.id}><h2>Batch {b.id}<span className="tag right">{b.status}</span></h2><progress max={b.total} value={b.done}/><p>{b.done} completed · {b.total-b.done} pending</p>{b.status==='Calibration review'&&<><p>Проверьте демонстрационную конфигурацию: {config.nodes.length} judges, {config.mode}. Реальная калибровка требует API.</p><button onClick={()=>setBatches(bs=>bs.map(x=>x.id===b.id?{...x,status:'Running'}:x))}>Approve demo simulation</button></>}{['Running','Paused'].includes(b.status)&&<><button onClick={()=>setBatches(bs=>bs.map(x=>x.id===b.id?{...x,status:x.status==='Running'?'Paused':'Running'}:x))}>{b.status==='Running'?'Pause':'Resume'}</button><button onClick={()=>setBatches(bs=>bs.map(x=>x.id===b.id?{...x,status:'Stopped'}:x))}>Stop</button></>}<button onClick={()=>saveFile(`batch-${b.id}.json`,b)}>Export state</button></section>)}</>}
  {page==='Data'&&<section className="panel"><h2>Trace library</h2>{!traces.length&&<p>Сохраните проверенный trace на первом шаге мастера.</p>}{traces.map((t,i)=><div className="edge-row" key={i}><span>{t.name} · {t.steps.length} steps</span><button onClick={()=>{setSteps(t.steps);setRaw(JSON.stringify(t.steps,null,2));update('name',t.name);setWizard(0);navigate('New evaluation');}}>Use trace</button><button onClick={()=>saveFile(`${t.name}.json`,t.steps)}>Download</button></div>)}<h3>Summaries</h3><p>Не созданы. Summarizer API не подключён.</p><h3>Retrieval store <span className="tag">Offline</span></h3><p>PostgreSQL не подключён. States, coverage и raw content недоступны.</p></section>}
  {page==='Optimizer Lab'&&<section className="panel"><h2>Prompt optimizer <span className="tag demo">Experimental</span></h2><p>Подготовьте и сохраните вариант prompt вручную. Автоматическая оптимизация и scoring требуют backend Judge API.</p><div className="two-col"><Field label="Original prompt" area value={draftPrompt} change={setDraftPrompt}/><Field label="Candidate prompt" area value={candidate} change={setCandidate}/></div><button disabled={!candidate.trim()} onClick={()=>{setDraftPrompt(candidate);setCandidate('');setNotice('Кандидат принят в локальный редактор.');}}>Accept candidate</button><button onClick={()=>setCandidate('')}>Reject</button><button onClick={()=>saveFile('prompt-experiment.json',{original:draftPrompt,candidate})}>Export experiment</button><h3>Graph evolution</h3><p>Blocked: восстановление Judge/scoring API необходимо для расчёта fitness и запуска поколений.</p><button disabled>Run evolution</button></section>}
  {page==='Observability'&&<section className="panel"><h2>Usage & cost</h2><p>Подтверждённых AI-запросов: {runs.filter(r=>r.execution==='ai').reduce((n,r)=>n+(r.usage?.calls??0),0)}. Стоимость пока проверяйте у провайдера; ошибки и отмены могут иметь неполный usage.</p><h3>Artifacts</h3>{runs.map(r=><div className="edge-row" key={r.id}><span>{r.id} · result + configuration + trace</span><button onClick={()=>saveFile(`run-${r.id}.json`,r)}>Download JSON</button></div>)}<h3>Langfuse</h3><p>Не подключён. Демонстрационные запуски не имеют внешних trace ID.</p></section>}
  {page==='Settings'&&<section className="panel"><h2>Execution defaults</h2><div className="two-col">{(['timeout','retries','concurrency','budget'] as const).map(k=><Field key={k} label={k} type="number" value={settings[k]} change={v=>setSettings(s=>({...s,[k]:Math.max(k==='retries'?0:1,Number(v)||0)}))}/>)}</div><p>Сохраняются локально. Применение к реальному runner появится вместе с API.</p><EnvSettings/><button onClick={()=>saveFile('execution-defaults.json',settings)}>Export defaults</button></section>}
  </main></div></div>;
}
