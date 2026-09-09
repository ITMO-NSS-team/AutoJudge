import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Braces,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Clock3,
  Database,
  FileSearch,
  FlaskConical,
  GitFork,
  History,
  Layers3,
  Menu,
  PanelLeftClose,
  Play,
  RotateCcw,
  Scale,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Upload,
  Users,
  Wrench,
  X,
  Zap,
} from "lucide-react";

type JudgeState = "ready" | "queued" | "running" | "complete";

type Judge = {
  id: string;
  title: string;
  short: string;
  description: string;
  accent: string;
  icon: typeof FileSearch;
};

const judges: Judge[] = [
  {
    id: "grounding",
    title: "Evidence Grounding",
    short: "Grounding",
    description: "Checks claims against retrieved evidence and source context.",
    accent: "blue",
    icon: FileSearch,
  },
  {
    id: "causal",
    title: "Causal Attribution",
    short: "Causality",
    description: "Identifies the responsible agent and causal trace step.",
    accent: "violet",
    icon: GitFork,
  },
  {
    id: "tools",
    title: "Tool-Use Compliance",
    short: "Tool use",
    description: "Inspects tool choice, parameters, and workflow compliance.",
    accent: "cyan",
    icon: Wrench,
  },
  {
    id: "logic",
    title: "Logical Consistency",
    short: "Logic",
    description: "Detects contradictions and unsupported inference.",
    accent: "orange",
    icon: Scale,
  },
];

const traceSteps = [
  { id: 12, agent: "Researcher Agent", tool: "Search", kind: "blue" },
  { id: 13, agent: "Retrieval Agent", tool: "Retriever", kind: "green" },
  { id: 14, agent: "Analyst Agent", tool: "Summarizer", kind: "violet" },
  { id: 15, agent: "Researcher Agent", tool: "Search", kind: "blue" },
  { id: 16, agent: "Reviewer Agent", tool: "Critique", kind: "amber" },
  { id: 17, agent: "Researcher Agent", tool: "Search", kind: "blue" },
  { id: 18, agent: "Retrieval Agent", tool: "Retriever", kind: "green" },
  { id: 19, agent: "Analyst Agent", tool: "Summarizer", kind: "violet" },
  { id: 20, agent: "Researcher Agent", tool: "Search", kind: "blue" },
  { id: 24, agent: "Reviewer Agent", tool: "Finalize", kind: "violet" },
];

const navItems = [
  { label: "Evaluate", icon: Sparkles },
  { label: "Runs", icon: History },
  { label: "Judge Studio", icon: Users },
  { label: "Batches", icon: Layers3 },
  { label: "Optimizer", icon: FlaskConical },
  { label: "Observability", icon: Activity },
];

const runStages = ["Preparing trace", ...judges.map((judge) => judge.short), "Aggregation"];

function Logo() {
  return (
    <div className="brand" aria-label="AutoJudge">
      <div className="brand-mark"><Scale size={19} strokeWidth={2.3} /></div>
      <span>AutoJudge</span>
    </div>
  );
}

function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
      <div className="sidebar-head">
        <Logo />
        <button className="icon-button sidebar-close" onClick={onClose} aria-label="Close navigation">
          <PanelLeftClose size={18} />
        </button>
      </div>
      <nav className="main-nav" aria-label="Primary navigation">
        <span className="nav-label">Workspace</span>
        {navItems.map(({ label, icon: Icon }, index) => (
          <button className={`nav-item ${index === 0 ? "active" : ""}`} key={label}>
            <Icon size={18} />
            <span>{label}</span>
            {label === "Batches" && <span className="nav-count">3</span>}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <button className="nav-item"><Settings2 size={18} /><span>Settings</span></button>
        <div className="user-card">
          <div className="avatar">RB</div>
          <div><strong>Research workspace</strong><span>Local prototype</span></div>
        </div>
      </div>
    </aside>
  );
}

function Header({ onMenu }: { onMenu: () => void }) {
  return (
    <header className="topbar">
      <button className="icon-button menu-button" onClick={onMenu} aria-label="Open navigation"><Menu size={20} /></button>
      <div className="breadcrumbs">
        <span>Evaluations</span><span>/</span><strong>Trace #042</strong>
      </div>
      <div className="topbar-actions">
        <span className="environment"><span className="live-dot" />Local environment</span>
        <button className="icon-button" aria-label="Open settings"><Settings2 size={19} /></button>
      </div>
    </header>
  );
}

function TracePanel({ selectedStep, setSelectedStep }: { selectedStep: number; setSelectedStep: (id: number) => void }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"full" | "summary">("summary");
  const filtered = traceSteps.filter((step) => `${step.id} ${step.agent} ${step.tool}`.toLowerCase().includes(query.toLowerCase()));

  return (
    <section className="surface trace-panel" aria-label="Trace configuration">
      <div className="section-heading">
        <div><span className="eyebrow">Input</span><h2>Trace & configuration</h2></div>
        <button className="icon-button" aria-label="Configuration"><Settings2 size={18} /></button>
      </div>

      <button className="upload-zone">
        <span className="upload-icon"><Upload size={18} /></span>
        <span><strong>Trace #042</strong><small>JSONL · 24 steps · 86 KB</small></span>
        <CheckCircle2 size={18} className="success-icon" />
      </button>

      <div className="subhead"><span>Trace steps</span><span className="count-pill">24</span></div>
      <label className="search-field">
        <Search size={16} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search steps" aria-label="Search trace steps" />
        {query && <button onClick={() => setQuery("")} aria-label="Clear search"><X size={14} /></button>}
      </label>

      <div className="trace-list">
        {filtered.map((step) => (
          <button key={step.id} className={`trace-row ${selectedStep === step.id ? "selected" : ""}`} onClick={() => setSelectedStep(step.id)}>
            <span className="step-index">{step.id}</span>
            <span className="trace-agent">{step.agent}</span>
            <span className={`tool-tag ${step.kind}`}>{step.tool}</span>
            {selectedStep === step.id && <ArrowRight size={15} />}
          </button>
        ))}
      </div>

      <div className="configuration">
        <button className="select-row"><span className="select-icon"><GitFork size={18} /></span><span><small>Taxonomy</small><strong>Agent failure analysis · v1.0</strong></span><ChevronDown size={17} /></button>
        <button className="select-row"><span className="select-icon"><Braces size={18} /></span><span><small>Output schema</small><strong>AutoJudge verdict · v1.0</strong></span><ChevronDown size={17} /></button>
        <div className="mode-block">
          <span className="mode-title">Context mode</span>
          <div className="segmented">
            <button className={mode === "full" ? "active" : ""} onClick={() => setMode("full")}>Full trace</button>
            <button className={mode === "summary" ? "active" : ""} onClick={() => setMode("summary")}>Summary + retrieval</button>
          </div>
        </div>
      </div>
    </section>
  );
}

function JudgeCard({ judge, state, selected, onSelect }: { judge: Judge; state: JudgeState; selected: boolean; onSelect: () => void }) {
  const Icon = judge.icon;
  return (
    <button className={`judge-card ${selected ? "selected" : ""}`} onClick={onSelect}>
      <div className="judge-top">
        <span className={`judge-icon ${judge.accent}`}><Icon size={20} /></span>
        <span className={`judge-state state-${state}`}>
          {state === "complete" ? <Check size={13} /> : state === "running" ? <Zap size={13} /> : null}
          {state}
        </span>
      </div>
      <h3>{judge.title}</h3>
      <p>{judge.description}</p>
      <div className="judge-meta"><span>openrouter/auto</span><ArrowRight size={15} /></div>
    </button>
  );
}

function Graph({ states }: { states: Record<string, JudgeState> }) {
  return (
    <div className="graph" aria-label="Judge pipeline graph">
      <div className="graph-nodes">
        {judges.map((judge) => {
          const Icon = judge.icon;
          return <div className={`mini-node ${states[judge.id]}`} key={judge.id}><span className={`judge-icon ${judge.accent}`}><Icon size={17} /></span><span>{judge.short}</span></div>;
        })}
      </div>
      <div className="graph-lines" aria-hidden="true"><i /><i /><i /><i /></div>
      <div className={`aggregator ${states.aggregator}`}><Layers3 size={18} /><span>Final aggregator</span></div>
    </div>
  );
}

function RunPanel({ judgeStates, selectedJudge, setSelectedJudge }: {
  judgeStates: Record<string, JudgeState>;
  selectedJudge: string;
  setSelectedJudge: (id: string) => void;
}) {
  return (
    <section className="surface run-panel">
      <div className="section-heading run-heading">
        <div><span className="eyebrow">Generated pool</span><h2>Judge pipeline</h2></div>
        <div className="pool-status"><Users size={16} /><strong>4 judges</strong><span>Parallel</span></div>
      </div>
      <div className="judge-grid">
        {judges.map((judge) => (
          <JudgeCard key={judge.id} judge={judge} state={judgeStates[judge.id]} selected={selectedJudge === judge.id} onSelect={() => setSelectedJudge(judge.id)} />
        ))}
      </div>
      <Graph states={judgeStates} />
    </section>
  );
}

function VerdictPanel({ runState, activeStage, selectedStep }: { runState: "idle" | "running" | "complete"; activeStage: number; selectedStep: number }) {
  const progress = runState === "idle" ? 0 : runState === "complete" ? 100 : Math.round(((activeStage + 1) / runStages.length) * 100);
  return (
    <section className="surface verdict-panel" aria-live="polite">
      <div className="section-heading">
        <div><span className="eyebrow">Evaluation result</span><h2>Final verdict</h2></div>
        <span className={`run-badge ${runState}`}><span />{runState === "idle" ? "Ready" : runState === "running" ? "Running" : "Complete"}</span>
      </div>

      {runState === "running" ? (
        <div className="running-state">
          <div className="pulse-orbit"><Sparkles size={25} /></div>
          <h3>{runStages[activeStage]}</h3>
          <p>Evaluating trace evidence and agent decisions.</p>
          <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
          <div className="progress-label"><span>{progress}%</span><span>{activeStage + 1} of {runStages.length} stages</span></div>
          <div className="live-log">
            {runStages.slice(0, activeStage + 1).reverse().slice(0, 4).map((stage, index) => (
              <div key={stage}><span className={index === 0 ? "log-live" : "log-done"}>{index === 0 ? <Zap size={12} /> : <Check size={12} />}</span><span>{stage}</span><time>{index === 0 ? "now" : `${index * 3 + 2}s`}</time></div>
            ))}
          </div>
        </div>
      ) : runState === "complete" ? (
        <>
          <div className="verdict-alert"><span className="alert-icon"><AlertTriangle size={24} /></span><div><small>Finding detected</small><strong>Unsupported claim</strong></div><span className="confidence">91%</span></div>
          <div className="verdict-facts">
            <div><span>Culprit</span><strong>Researcher Agent</strong></div>
            <div><span>Critical step</span><strong>#{selectedStep}</strong></div>
            <div><span>Severity</span><strong className="severity">High</strong></div>
          </div>
          <div className="evidence-section">
            <div className="subhead"><span>Evidence chain</span><span className="count-pill">2</span></div>
            <button className="evidence-card source"><div><span>Step 12</span><strong>Source evidence</strong></div><p>“The court held that the statute requires a showing of intent to defraud...”</p></button>
            <div className="evidence-link"><span /><AlertTriangle size={14} /><span /></div>
            <button className="evidence-card conflict"><div><span>Step 17</span><strong>Unsupported conclusion</strong></div><p>“Therefore, the statute imposes strict liability regardless of intent.”</p></button>
          </div>
          <div className="explanation"><strong>Why this failed</strong><p>Step 17 contradicts the binding evidence in Step 12. The conclusion removes the required intent element without supporting evidence.</p></div>
          <div className="metrics">
            <div><Clock3 size={17} /><span><strong>18.4 s</strong><small>Total time</small></span></div>
            <div><Zap size={17} /><span><strong>14</strong><small>LLM calls</small></span></div>
            <div><CircleDollarSign size={17} /><span><strong>$0.08</strong><small>Total cost</small></span></div>
          </div>
        </>
      ) : (
        <div className="empty-verdict">
          <div className="empty-graphic"><ShieldCheck size={34} /></div>
          <h3>Ready to evaluate</h3>
          <p>Four specialized judges will inspect the trace in parallel and synthesize one verdict.</p>
          <div className="readiness-list">
            <span><CheckCircle2 size={15} />Trace validated</span>
            <span><CheckCircle2 size={15} />Judge pool generated</span>
            <span><CheckCircle2 size={15} />Output schema ready</span>
          </div>
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedStep, setSelectedStep] = useState(17);
  const [selectedJudge, setSelectedJudge] = useState("grounding");
  const [runState, setRunState] = useState<"idle" | "running" | "complete">("idle");
  const [activeStage, setActiveStage] = useState(0);

  const judgeStates = useMemo(() => {
    const result: Record<string, JudgeState> = { aggregator: "ready" };
    judges.forEach((judge, index) => {
      if (runState === "idle") result[judge.id] = "ready";
      else if (runState === "complete") result[judge.id] = "complete";
      else if (activeStage === 0) result[judge.id] = "queued";
      else if (activeStage >= 1 && activeStage <= 4) {
        result[judge.id] = index + 1 < activeStage ? "complete" : index + 1 === activeStage ? "running" : "queued";
      } else result[judge.id] = "complete";
    });
    result.aggregator = runState === "complete" ? "complete" : activeStage === 5 ? "running" : "ready";
    return result;
  }, [runState, activeStage]);

  function startRun() {
    if (runState === "running") return;
    setRunState("running");
    setActiveStage(0);
    runStages.forEach((_, index) => {
      window.setTimeout(() => {
        setActiveStage(index);
        if (index === runStages.length - 1) window.setTimeout(() => setRunState("complete"), 850);
      }, index * 850);
    });
  }

  function resetRun() {
    setRunState("idle");
    setActiveStage(0);
  }

  return (
    <div className="app-shell">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      {sidebarOpen && <button className="backdrop" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />}
      <div className="app-content">
        <Header onMenu={() => setSidebarOpen(true)} />
        <main>
          <div className="page-intro">
            <div>
              <div className="title-row"><h1>Evaluate trace</h1><span className="ready-pill"><CheckCircle2 size={15} />Ready</span></div>
              <p>Inspect the generated judge pool, run the pipeline, and trace every finding back to evidence.</p>
            </div>
            <div className="intro-actions">
              {runState === "complete" && <button className="secondary-button" onClick={resetRun}><RotateCcw size={17} />Reset</button>}
              <button className="primary-button" onClick={startRun} disabled={runState === "running"}>
                {runState === "running" ? <><span className="button-spinner" />Evaluating…</> : <><Play size={17} fill="currentColor" />{runState === "complete" ? "Run again" : "Run evaluation"}</>}
              </button>
            </div>
          </div>

          <div className="workflow-bar">
            {["Trace", "Design", "Judge pool", "Graph", "Run", "Verdict"].map((step, index) => {
              const isDone = index < 4 || (index >= 4 && runState === "complete");
              const isCurrent = index === 4 && runState !== "complete";
              return (
                <div className={isDone ? "done" : isCurrent ? "current" : "pending"} key={step}>
                  <span>{isDone ? <Check size={12} /> : index + 1}</span><strong>{step}</strong>
                </div>
              );
            })}
          </div>

          <div className="workspace-grid">
            <TracePanel selectedStep={selectedStep} setSelectedStep={setSelectedStep} />
            <RunPanel judgeStates={judgeStates} selectedJudge={selectedJudge} setSelectedJudge={setSelectedJudge} />
            <VerdictPanel runState={runState} activeStage={activeStage} selectedStep={selectedStep} />
          </div>
        </main>
      </div>
    </div>
  );
}
