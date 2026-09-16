import { useCallback, useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Scale,
  LayoutDashboard,
  Play,
  History,
  GitFork,
  Activity,
  Settings,
  Download,
  Plus,
  Check,
  ArrowRight,
  Search,
  Server,
  Cpu,
  AlertTriangle,
  Sparkles,
  RotateCcw,
} from "lucide-react";
import "./tokens.css";
import "./workspace.css";
import "./wizard.css";
import SemanticBadge from "./SemanticBadge";
import PipelineGraph from "./PipelineGraph";
import EnvSettings from "./EnvSettings";
import { api } from "./apiClient";
import { exampleOutputSchema, exampleTaxonomy } from "./designTemplate";
import { fewShotPresets } from "./fewShotPresets";
import { normalizeTrace, parseDesignFile } from "./imports";
import {
  designKey,
  designSourceHint as sourceHintFor,
  designStateLabel as stateLabelFor,
  nextDesignState,
  runBlockedReason as blockedReasonFor,
} from "./designState";
import type {
  Design,
  DesignMetadata,
  DesignSource,
  DesignState,
  JudgeMetadata,
} from "./designState";

type Step = { id: number; agent: string; content: string };
type Config = {
  judge_instructions?: Record<string, string>;
  judges?: JudgeMetadata[];
  design_source?: DesignSource;
  design_id?: string;
  name: string;
  objective: string;
  taxonomy: string;
  schema: string;
  examples: string;
  model: string;
  mode: string;
  nodes: string[];
  edges: string[][];
};
type Run = {
  execution?: string;
  final_output?: unknown;
  usage?: {
    calls: number;
    tokens: number;
    cost: number | null;
    partial?: boolean;
  };
  outputs?: Record<string, unknown>;
  phase?: number;
  id: string;
  date: string;
  status: string;
  config: Config;
  steps: Step[];
  verdict: string;
  archived?: boolean;
};
type Toast = { id: number; kind: "error" | "notice"; text: string };
const sections = [
  "Overview",
  "New evaluation",
  "Runs",
  "Compare",
  "Observability",
  "Settings",
];
const icons = [
  LayoutDashboard,
  Play,
  History,
  GitFork,
  Activity,
  Settings,
];
const wizardSteps = ["Trace", "Design", "Meta Agent", "Run", "Verdict"];
const STEP = { trace: 0, design: 1, judges: 2, run: 3, verdict: 4 };
const sample: Step[] = [
  {
    id: 12,
    agent: "Researcher",
    content: "The source requires evidence of intent.",
  },
  { id: 13, agent: "Retriever", content: "Retrieved the supporting source." },
  { id: 17, agent: "Researcher", content: "Intent is not required." },
  { id: 24, agent: "Reviewer", content: "Final answer submitted." },
];
const initial: Config = {
  name: "Trace #042",
  objective:
    "Find unsupported claims and attribute failures to agent decisions.",
  taxonomy: exampleTaxonomy,
  schema: exampleOutputSchema,
  examples: "[]",
  model: "",
  mode: "Full trace",
  nodes: [],
  edges: [],
  judge_instructions: {},
  judges: [],
};
function read<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) || "null") ?? fallback;
  } catch {
    return fallback;
  }
}
function saveFile(name: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob(
      [typeof data === "string" ? data : JSON.stringify(data, null, 2)],
      { type: "application/json" },
    ),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}
function validate(c: Config) {
  if (!c.nodes.includes("FINAL_AGGREGATOR")) return "Add FINAL_AGGREGATOR.";
  if (
    new Set(c.nodes).size !== c.nodes.length ||
    c.nodes.some((n) => !n.trim())
  )
    return "Node names must be non-empty and unique.";
  if (
    c.edges.some(
      ([a, b]) =>
        !c.nodes.includes(a) ||
        !c.nodes.includes(b) ||
        a === "FINAL_AGGREGATOR",
    )
  )
    return "Invalid edge or outgoing aggregator connection.";
  const visit = (n: string, trail: string[]): boolean =>
    trail.includes(n) ||
    c.edges.filter((e) => e[0] === n).some((e) => visit(e[1], [...trail, n]));
  if (c.nodes.some((n) => visit(n, []))) return "The graph contains a cycle.";
  const reaches = (n: string): boolean =>
    n === "FINAL_AGGREGATOR" ||
    c.edges.filter((e) => e[0] === n).some((e) => reaches(e[1]));
  if (c.nodes.some((n) => !reaches(n)))
    return "Every judge must pass its result to the aggregator.";
  return "";
}
function Field({
  label,
  value,
  change,
  area = false,
  type = "text",
}: {
  label: string;
  value: string | number;
  change: (v: string) => void;
  area?: boolean;
  type?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {area ? (
        <textarea value={value} onChange={(e) => change(e.target.value)} />
      ) : (
        <input
          type={type}
          value={value}
          onChange={(e) => change(e.target.value)}
        />
      )}
    </label>
  );
}
export default function Workspace() {
  const [previousDesign, setPreviousDesign] = useState<{ taxonomy: string; schema: string } | null>(null);
  const [connected, setConnected] = useState(false);
  const [execution, setExecution] = useState("ai");
  const [selectedExample, setSelectedExample] = useState("");
  const [launching, setLaunching] = useState(false);
  const [page, setPage] = useState(() =>
    sections.includes(decodeURIComponent(location.hash.slice(1)))
      ? decodeURIComponent(location.hash.slice(1))
      : "Overview",
  );
  const [config, setConfig] = useState<Config>(() => ({
    ...initial,
    ...read("aj-config", initial),
    model: "",
  }));
  const [designState, setDesignState] = useState<DesignState>("not_generated");
  const [designAccepted, setDesignAccepted] = useState(false);
  const [designError, setDesignError] = useState("");
  const [designMeta, setDesignMeta] = useState<DesignMetadata | null>(null);
  const [designSnapshot, setDesignSnapshot] = useState<string | null>(null);
  const [steps, setSteps] = useState<Step[]>(sample);
  const [raw, setRaw] = useState(JSON.stringify(sample, null, 2));
  const [traceIssue, setTraceIssue] = useState("");
  const [runs, setRuns] = useState<Run[]>(() => read("aj-runs", []));
  const [wizard, setWizard] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastId = useRef(0);
  const pushToast = useCallback((kind: Toast["kind"], text: string) => {
    const id = ++toastId.current;
    setToasts((list) => [...list.slice(-4), { id, kind, text }]);
    window.setTimeout(
      () => setToasts((list) => list.filter((t) => t.id !== id)),
      6_000,
    );
  }, []);
  const closeToast = (id: number) =>
    setToasts((list) => list.filter((t) => t.id !== id));
  const setError = useCallback(
    (text: string) => {
      if (text) pushToast("error", text);
    },
    [pushToast],
  );
  const setNotice = useCallback(
    (text: string) => {
      if (text) pushToast("notice", text);
    },
    [pushToast],
  );
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All");
  const [compare, setCompare] = useState<string[]>([]);
  const [detail, setDetail] = useState<Run | null>(null);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [phase, setPhase] = useState(-1);
  const [active, setActive] = useState<Run | null>(null);
  const [settings, setSettings] = useState(() =>
    read("aj-settings", {
      timeout: 120,
      retries: 2,
      concurrency: 4,
      langfuse: "",
    }),
  );
  const navigate = (p: string) => {
    setPage(p);
    location.hash = encodeURIComponent(p);
    setError("");
    setQuery("");
  };
  useEffect(() => {
    const fn = () => {
      const p = decodeURIComponent(location.hash.slice(1));
      if (sections.includes(p)) setPage(p);
    };
    window.addEventListener("hashchange", fn);
    return () => window.removeEventListener("hashchange", fn);
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        setSteps(normalizeTrace(raw));
        setTraceIssue("");
      } catch (e) {
        setTraceIssue(e instanceof Error ? e.message : "Invalid trace");
      }
    }, 250);
    return () => window.clearTimeout(timer);
  }, [raw]);
  useEffect(() => {
    try {
      localStorage.setItem("aj-runs", JSON.stringify(runs));
      localStorage.setItem("aj-config", JSON.stringify(config));
      localStorage.setItem("aj-settings", JSON.stringify(settings));
    } catch {
      setError("Local storage is full or unavailable. Export your results.");
    }
  }, [runs, config, settings]);
  useEffect(() => {
    const nodeSet = new Set(config.nodes);
    const validEdges = config.edges.filter(
      ([from, to], index, edges) =>
        nodeSet.has(from) &&
        nodeSet.has(to) &&
        from !== "FINAL_AGGREGATOR" &&
        from !== to &&
        edges.findIndex(([a, b]) => a === from && b === to) === index,
    );
    if (validEdges.length !== config.edges.length) {
      setConfig((current) => ({ ...current, edges: validEdges }));
      setNotice("Removed stale or invalid graph connections. Review the DAG before running.");
    }
  }, [config.nodes, config.edges]);
  useEffect(() => {
    if (designSnapshot === null) return;
    const current = designKey(config, steps);
    setDesignState((state) => nextDesignState(state, current, designSnapshot));
  }, [config.objective, config.taxonomy, config.schema, config.examples, steps, designSnapshot]);
  useEffect(() => {
    if (designState === "stale") setDesignAccepted(false);
  }, [designState]);
  useEffect(() => {
    let mounted = true;
    Promise.all([
      api<Run[]>("/runs"),
      api<{
        config?: Config;
      }>("/workspace"),
      api<{fields:{name:string;value:string|null}[]}>("/settings/env"),
    ])
      .then(([rs, ws, env]) => {
        if (!mounted) return;
        setRuns(rs);
        const selectedModel=env.fields.find((field)=>field.name==="AGENT_NODE_MODEL")?.value?.trim();
        if (ws.config) {
          const restored = ws.config;
          const nodes = restored.nodes?.length ? restored.nodes : initial.nodes;
          setConfig({
            ...restored,
            taxonomy: restored.taxonomy?.trim() ? restored.taxonomy : exampleTaxonomy,
            schema: restored.schema?.trim() ? restored.schema : exampleOutputSchema,
            mode: "Full trace",
            model: selectedModel || "",
            nodes,
            edges: restored.nodes?.length ? restored.edges ?? [] : initial.edges,
            judge_instructions: restored.judge_instructions ?? {},
            judges: restored.judges ?? [],
            design_source: restored.design_source ?? "manual",
            design_id: restored.design_id,
          });
          // A restored design is not re-verified here; the server checks its
          // identity again, and the user must regenerate or accept it.
          if (restored.design_source === "generated") {
            setDesignState("stale");
            setDesignAccepted(false);
          }
        }
        else setConfig((current)=>({...current,model:selectedModel || ""}));
        setActive(rs.find((r) => r.status === "Running") ?? null);
        setConnected(true);
      })
      .catch((e) => setError("Backend offline: " + e.message));
    return () => {
      mounted = false;
    };
  }, []);
  useEffect(() => {
    if (!connected) return;
    const timer = setTimeout(
      () =>
        api("/workspace", "PUT", { config, settings }).catch(
          (e) => setError(e.message),
        ),
      500,
    );
    return () => clearTimeout(timer);
  }, [connected, config, settings]);
  useEffect(() => {
    if (!active) return;
    const source = new EventSource("/api/runs/" + active.id + "/events");
    source.onmessage = (e) => {
      const r = JSON.parse(e.data) as Run;
      setPhase(r.phase ?? 0);
      setRuns((rs) => [r, ...rs.filter((x) => x.id !== r.id)]);
      if (r.status !== "Running") {
        setDetail(r);
        setActive(null);
        setPhase(-1);
        setWizard(STEP.verdict);
        // The consumed Meta Agent output is cleared; the next evaluation needs a fresh one.
        resetJudgePool();
        // Taxonomy and output schema fall back to the example.md defaults for the next evaluation.
        setConfig((c) => ({
          ...c,
          taxonomy: exampleTaxonomy,
          schema: exampleOutputSchema,
        }));
        source.close();
      }
    };
    source.onerror = () =>
      setError(
        "The event stream disconnected. The browser is reconnecting; the result is retained on the server.",
      );
    return () => source.close();
  }, [active?.id]);
  const update = (key: keyof Config, value: Config[keyof Config]) =>
    setConfig((c) => ({ ...c, [key]: value }));
  function parseTrace(text: string) {
    try {
      const normalized = normalizeTrace(text);
      setSteps(normalized);
      setTraceIssue("");
      setError("");
      setNotice(`${normalized.length} steps validated`);
      return true;
    } catch (e) {
      setTraceIssue(e instanceof Error ? e.message : "Invalid JSON");
      setError(e instanceof Error ? e.message : "Invalid JSON");
      return false;
    }
  }
  const templateButtons = (
    <div className="toolbar">
      <button
        type="button"
        onClick={() => {
          setPreviousDesign({
            taxonomy: config.taxonomy,
            schema: config.schema,
          });
          setConfig((c) => ({
            ...c,
            taxonomy: exampleTaxonomy,
            schema: exampleOutputSchema,
          }));
          setError("");
          setNotice(
            "The example.md taxonomy and output schema were applied; other fields were retained.",
          );
        }}
      >
        Apply taxonomy and output schema from example.md
      </button>
      {previousDesign && (
        <button
          type="button"
          onClick={() => {
            setConfig((c) => ({ ...c, ...previousDesign }));
            setPreviousDesign(null);
            setNotice("The previous taxonomy and output schema were restored.");
          }}
        >
          Undo template application
        </button>
      )}
    </div>
  );
  async function loadTestTrace(id: number) {
    try {
      const response = await fetch("/test-data/trace-0" + id + ".json");
      if (!response.ok) throw Error("Could not load the test trace.");
      const text = await response.text();
      if (parseTrace(text)) {
        setRaw(text);
        update("name", "Example " + id);
        update("mode", "Full trace");
      }
    } catch (e) {
      setError(String(e));
    }
  }
  async function importDesign(kind: "taxonomy" | "schema", file?: File) {
    if (!file) return;
    try {
      if (file.size > 1_000_000) throw Error("File limit: 1 MB.");
      const value = parseDesignFile(kind, await file.text());
      update(kind, value);
      setError("");
      setNotice("Loaded " + file.name);
    } catch (e) {
      setError(String(e));
    }
  }
  const designUpload = (kind: "taxonomy" | "schema") => (
    <label className="file-input">
      {kind === "taxonomy" ? "Upload taxonomy (.md, .txt, .json)" : "Upload output schema (.json, .md, .txt)"}
      <input
        aria-label={kind === "taxonomy" ? "Upload taxonomy" : "Upload output schema"}
        type="file"
        accept={kind === "taxonomy" ? ".md,.txt,.json,text/markdown,text/plain,application/json" : ".json,.md,.txt,application/json,text/markdown,text/plain"}
        onChange={(e) => {
          void importDesign(kind, e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </label>
  );
  function designValid(requireModel = true) {
    try {
      const schema = parseDesignFile("schema", config.schema);
      const taxonomy = parseDesignFile("taxonomy", config.taxonomy);
      if (!Array.isArray(JSON.parse(config.examples)))
        throw Error("Examples must be an array.");
      if (!config.objective.trim())
        throw Error("Enter an evaluation objective.");
      if (requireModel && !config.model.trim())
        throw Error("Set the judge model in Settings (AGENT_NODE_MODEL).");
      if (config.model.trim() && !/^[\x21-\x7e]{1,200}$/.test(config.model.trim()))
        throw Error("Model ID must use printable ASCII without spaces.");
      setError("");
      if (schema !== config.schema || taxonomy !== config.taxonomy)
        setConfig((c) => ({ ...c, schema, taxonomy }));
      return { ...config, schema, taxonomy };
    } catch (e) {
      setError(String(e));
      return null;
    }
  }
  async function validateDesign() {
    const normalized = designValid();
    if (!normalized) return;
    try {
      const result = await api<{ valid: boolean; taxonomy_chars: number; properties: number; examples: number }>("/design/validate", "POST", {
        objective: normalized.objective,
        taxonomy: normalized.taxonomy,
        model: normalized.model,
        schema: normalized.schema,
        examples: normalized.examples,
      });
      setNotice(`Design valid · ${result.taxonomy_chars} taxonomy characters · ${result.properties} schema properties · ${result.examples} examples`);
    } catch (e) {
      setError(String(e));
    }
  }
  async function generateDesign() {
    if (!parseTrace(raw)) {
      setWizard(STEP.trace);
      return;
    }
    const normalized = designValid(false);
    if (!normalized) return;
    const parsedSteps = normalizeTrace(raw);
    setDesignState("generating");
    setDesignError("");
    setDesignAccepted(false);
    try {
      const design = await api<Design>("/design/generate", "POST", {
        objective: normalized.objective,
        taxonomy: normalized.taxonomy,
        schema: normalized.schema,
        examples: normalized.examples,
        steps: parsedSteps,
      });
      setConfig((c) => ({
        ...c,
        objective: normalized.objective,
        taxonomy: normalized.taxonomy,
        schema: normalized.schema,
        examples: normalized.examples,
        nodes: design.nodes,
        edges: design.edges,
        judge_instructions: design.judge_instructions,
        judges: design.judges ?? [],
        design_source: "generated",
        design_id: design.design_id,
      }));
      setDesignMeta(design.metadata ?? null);
      setDesignSnapshot(designKey(normalized, parsedSteps));
      setDesignState("generated");
      setNotice(
        `Generated ${design.nodes.length} judges and ${design.edges.length} connections. Review and accept the pool.`,
      );
      if (design.metadata?.unused_judges?.length)
        setNotice(
          `Left out of the DAG: ${design.metadata.unused_judges.join(", ")}.`,
        );
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setDesignState("failed");
      setDesignError(message);
      setError(message);
    }
  }
  function acceptDesign() {
    setDesignAccepted(true);
    setNotice("Meta Agent accepted. This exact pool and DAG will be executed.");
  }
  function resetJudgePool() {
    setConfig((c) => ({
      ...c,
      nodes: [],
      edges: [],
      judge_instructions: {},
      judges: [],
      design_source: undefined,
      design_id: undefined,
    }));
    setDesignState("not_generated");
    setDesignAccepted(false);
    setDesignSnapshot(null);
    setDesignMeta(null);
    setDesignError("");
  }
  const generatedDesign = config.design_source === "generated";
  // While a fresh pool waits for approval, accepting it is the primary action.
  const awaitingAcceptance =
    generatedDesign && designState === "generated" && !designAccepted;
  const runBlockedReason = blockedReasonFor(
    config.design_source,
    designState,
    designAccepted,
  );
  const runBlocked = runBlockedReason !== "";
  async function launch() {
    if (runBlocked) {
      setError(runBlockedReason);
      setWizard(STEP.judges);
      return;
    }
    if (!parseTrace(raw)) {
      setWizard(STEP.trace);
      return;
    }
    const issue = validate(config);
    const normalized = issue ? null : designValid();
    if (issue || !normalized) {
      if (issue) setError(issue);
      return;
    }
    try {
      setLaunching(true);
      const run = await api<Run>("/runs", "POST", {
        config: normalized,
        steps: normalizeTrace(raw),
        execution,
      });
      setDetail(null);
      setActive(run);
      setPhase(0);
      setWizard(STEP.run);
    } catch (e) {
      setError(String(e));
    } finally {
      setLaunching(false);
    }
  }
  async function cancel() {
    if (!active) return;
    try {
      await api("/runs/" + active.id + "/cancel", "POST");
    } catch (e) {
      setError(String(e));
    }
  }
  function cloneRun(r: Run) {
    const cloned: Config = {
      name: r.config.name,
      objective: r.config.objective,
      taxonomy: r.config.taxonomy?.trim() ? r.config.taxonomy : exampleTaxonomy,
      schema: r.config.schema?.trim()
        ? structuredClone(r.config.schema)
        : exampleOutputSchema,
      examples: structuredClone(r.config.examples),
      model: config.model,
      mode: "Full trace",
      nodes: structuredClone(r.config.nodes),
      edges: structuredClone(r.config.edges),
      judge_instructions: structuredClone(r.config.judge_instructions ?? {}),
      judges: structuredClone(r.config.judges ?? []),
      design_source: r.config.design_source ?? "manual",
      design_id: r.config.design_id,
    };
    setConfig(cloned);
    setSteps(r.steps);
    setRaw(JSON.stringify(r.steps, null, 2));
    // The cloned design keeps its identity, so it only needs to be accepted again.
    setDesignSnapshot(cloned.design_id ? designKey(cloned, r.steps) : null);
    setDesignState(cloned.design_id ? "generated" : "not_generated");
    setDesignAccepted(false);
    setDesignMeta(null);
    setDesignError("");
    setWizard(STEP.trace);
    navigate("New evaluation");
  }
  function next() {
    if (wizard === STEP.trace && !parseTrace(raw)) return;
    if (wizard === STEP.design && !designValid()) return;
    if (wizard === STEP.judges && validate(config)) {
      setError(validate(config));
      return;
    }
    setWizard(Math.min(STEP.verdict, wizard + 1));
  }
  const designStateLabel = stateLabelFor(designState, designAccepted);
  const designSourceHint = sourceHintFor(config.design_source);
  const filteredRuns = runs.filter(
    (r) =>
      (filter === "Archived"
        ? r.archived
        : !r.archived && (filter === "All" || r.status === filter)) &&
      `${r.id} ${r.config.name} ${r.status}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const selectedRuns = runs.filter((run) => compare.includes(run.id));
  const activeModelLabel = config.model || "Not configured";
  async function archiveSelectedRuns() {
    if (!selectedRuns.length) return;
    try {
      const shouldArchive = selectedRuns.some((run) => !run.archived);
      const updated = await Promise.all(
        selectedRuns.map((run) =>
          api<Run>("/runs/" + run.id, "PATCH", { archived: shouldArchive }),
        ),
      );
      const byId = new Map(updated.map((run) => [run.id, run]));
      setRuns((current) => current.map((run) => byId.get(run.id) ?? run));
      setCompare([]);
      setNotice(`${updated.length} run${updated.length === 1 ? "" : "s"} ${shouldArchive ? "archived" : "restored"}.`);
    } catch (e) {
      setError(String(e));
    }
  }
  async function deleteSelectedRuns() {
    const deletable = selectedRuns.filter((run) => run.status !== "Running");
    if (!deletable.length) return;
    if (!window.confirm(`Permanently delete ${deletable.length} selected run${deletable.length === 1 ? "" : "s"}? This cannot be undone.`)) return;
    try {
      await Promise.all(deletable.map((run) => api<void>("/runs/" + run.id, "DELETE")));
      const deletedIds = new Set(deletable.map((run) => run.id));
      setRuns((current) => current.filter((run) => !deletedIds.has(run.id)));
      setCompare((current) => current.filter((id) => !deletedIds.has(id)));
      if (detail && deletedIds.has(detail.id)) setDetail(null);
      setNotice(`${deletable.length} run${deletable.length === 1 ? "" : "s"} permanently deleted.`);
    } catch (e) {
      setError(String(e));
    }
  }
  const overviewMetrics: {
    label: string;
    value: number;
    icon: LucideIcon;
    accent: "indigo" | "emerald" | "blue" | "amber";
  }[] = [
    { label: "Runs", value: runs.length, icon: History, accent: "indigo" },
    {
      label: "Completed pipelines",
      value: runs.filter((r) => r.status === "Completed").length,
      icon: Check,
      accent: "emerald",
    },
    {
      label: "Active runs",
      value: runs.filter((r) => r.status === "Running").length,
      icon: Activity,
      accent: "blue",
    },
    {
      label: "Requires attention",
      value: runs.filter((r) => ["Failed", "Partial", "Stale"].includes(r.status)).length,
      icon: AlertTriangle,
      accent: "amber",
    },
  ];
  const table = (list: Run[]) => (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Select</th>
            <th>Run / Trace</th>
            <th>Pipeline</th>
            <th>Verdict</th>
            <th>Model / Context</th>
          </tr>
        </thead>
        <tbody>
          {list.map((r) => (
            <tr key={r.id}>
              <td data-label="Select">
                <input
                  aria-label={`Select ${r.id}`}
                  type="checkbox"
                  checked={compare.includes(r.id)}
                  onChange={(e) =>
                    setCompare(
                      e.target.checked
                        ? [...compare, r.id]
                        : compare.filter((id) => id !== r.id),
                    )
                  }
                />
              </td>
              <td data-label="Run / Trace">
                <button
                  className="link"
                  onClick={() => {
                    setDetail(r);
                    navigate("Runs");
                  }}
                >
                  {r.config.name}
                </button>
                <small>
                  {r.id} · {new Date(r.date).toLocaleString()}
                </small>
              </td>
              <td data-label="Pipeline">
                <SemanticBadge status={r.status} />
              </td>
              <td data-label="Verdict">{r.verdict}</td>
              <td data-label="Model / Context">
                {r.config.model}
                <small>{r.config.mode}</small>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!list.length && (
        <div className="empty">
          No runs yet. Create a demo evaluation.
        </div>
      )}
    </div>
  );
  const traceView = (data: Step[]) => (
    <div className="step-list">
      {data
        .filter((s) =>
          `${s.agent} ${s.content} ${s.id}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        )
        .map((s) => (
          <button
            key={s.id}
            className={selectedStep === s.id ? "picked" : ""}
            onClick={() => setSelectedStep(s.id)}
          >
            <span>
              #{s.id} · {s.agent}
            </span>
            <p>{s.content}</p>
          </button>
        ))}
    </div>
  );
  const resultView = (r: Run) => (
    <div className="two-col">
      <section className="panel">
        <h2>
          Final verdict <span className="tag">{r.status}</span>
        </h2>
        <div className="finding">{r.verdict}</div>
        <p>
          {r.execution === "ai"
            ? "Live AutoJudge evaluation. Review the output and evidence."
            : "Dry run: structure checked, no AI judgment."}
        </p>
        {r.usage && (
          <p>
            Calls: {r.usage.calls} · Tokens: {r.usage.tokens}
            {r.usage.partial ? " (incomplete data)" : ""} · Cost:{" "}
            {r.usage.cost === null
              ? "not returned by provider"
              : r.usage.cost + " USD"}
          </p>
        )}
        {r.final_output !== undefined && (
          <pre>{JSON.stringify(r.final_output, null, 2)}</pre>
        )}
        <h3>
          Node outputs <span className="tag">{r.config.design_source ?? "manual"}</span>
        </h3>
        {r.config.nodes.map((n) => (
          <details key={n}>
            <summary>{n}</summary>
            <pre>
              {JSON.stringify(
                r.outputs?.[n] ?? { status: "Not executed" },
                null,
                2,
              )}
            </pre>
            {r.config.judge_instructions?.[n] && (
              <details>
                <summary>Executed instructions</summary>
                <pre>{r.config.judge_instructions[n]}</pre>
              </details>
            )}
          </details>
        ))}
        <button onClick={() => saveFile("run-" + r.id + ".json", r)}>
          <Download size={15} /> Export snapshot
        </button>
      </section>
      <section className="panel">
        <h2>Source trace</h2>
        {traceView(r.steps)}
      </section>
    </div>
  );

  return (
    <div className={`workspace-app ${page === "New evaluation" ? "evaluation-page" : ""}`}>
      <aside>
        <div className="brand workspace-brand">
          <span className="workspace-brand-mark"><Scale size={19} /></span>
          <span>AutoJudge</span>
        </div>
        <small className="nav-caption">WORKSPACE</small>
        <nav>
          {sections.map((s, i) => {
            const Icon = icons[i];
            return (
              <button
                key={s}
                className={page === s ? "active" : ""}
                onClick={() => navigate(s)}
              >
                <Icon size={17} />
                {s}
              </button>
            );
          })}
        </nav>
        <div className="local-note">
          <div className="sidebar-status-row">
            <Server size={15} />
            <span>Backend</span>
            <SemanticBadge status={connected ? "Completed" : "Offline"} label={connected ? "Online" : "Offline"} />
          </div>
          <div className="sidebar-model">
            <Cpu size={15} />
            <span><small>Judge model</small><strong>{activeModelLabel}</strong></span>
          </div>
        </div>
      </aside>
      <div className={`work-body ${page === "New evaluation" ? "evaluation-active" : ""}`}>
        <header>
          <span>
            Workspace / <strong>{page}</strong>
          </span>
          <div className="header-context">
            <span className="header-model"><Cpu size={14} /> {activeModelLabel}</span>
            <SemanticBadge status={connected ? "Completed" : "Offline"} label={connected ? "Backend online" : "Backend offline"} />
          </div>
        </header>
        <main className={page === "New evaluation" ? "evaluation-main" : undefined}>
          <div className="work-title">
            <div>
              <span className="eyebrow">AUTOJUDGE / WORKBENCH</span>
              <h1>{page}</h1>
              <p>From judge configuration to a verifiable result.</p>
            </div>
            <button
              className="primary-button"
              onClick={() => {
                setWizard(STEP.trace);
                navigate("New evaluation");
              }}
            >
              <Plus size={16} /> New evaluation
            </button>
          </div>
          <div className="toast-stack" aria-live="polite">
            {toasts.map((toast) => (
              <div
                key={toast.id}
                role={toast.kind === "error" ? "alert" : "status"}
                className={`toast toast--${toast.kind}`}
              >
                {toast.text}
                <button aria-label="Close notification" onClick={() => closeToast(toast.id)}>×</button>
              </div>
            ))}
          </div>

          {page === "Overview" && (
            <>
              <section className="cockpit-hero">
                <div>
                  <h2>Evaluation workspace is {connected ? "ready" : "offline"}</h2>
                  <p>
                    Review the active model and pipeline state before starting an evaluation.
                  </p>
                </div>
                <div className="cockpit-hero-meta">
                  <span><Cpu size={16} /><small>Judge model</small><strong>{activeModelLabel}</strong></span>
                  <span><Server size={16} /><small>Backend</small><SemanticBadge status={connected ? "Completed" : "Offline"} label={connected ? "Connected" : "Offline"} /></span>
                </div>
              </section>
              <div className="kpis">
                {overviewMetrics.map(({ label, value, icon: MetricIcon, accent }) => (
                  <section className={`panel metric-card metric-card--${accent}`} key={label}>
                    <span className="metric-icon"><MetricIcon size={18} /></span>
                    <div><small>{label}</small><strong>{value}</strong></div>
                  </section>
                ))}
              </div>
              <section className="panel recent-runs-panel">
                <div className="panel-heading-row">
                  <div><span className="eyebrow">ACTIVITY</span><h2>Recent runs</h2></div>
                  <button onClick={() => navigate("Runs")}>View all <ArrowRight size={15} /></button>
                </div>
                {table(runs.filter((r) => !r.archived).slice(0, 5))}
              </section>
              <div className="two-col">
                <section className="panel attention-panel">
                  <div className="panel-heading-row"><div><span className="eyebrow">SIGNALS</span><h2>Requires attention</h2></div><AlertTriangle size={19} /></div>
                  {runs.filter((r) => ["Failed", "Partial", "Stale"].includes(r.status)).length ? (
                    runs.filter((r) => ["Failed", "Partial", "Stale"].includes(r.status)).slice(0, 3).map((r) => (
                      <button className="attention-run" key={r.id} onClick={() => { setDetail(r); navigate("Runs"); }}>
                        <span><strong>{r.config.name}</strong><small>{new Date(r.date).toLocaleString()}</small></span>
                        <SemanticBadge status={r.status} />
                      </button>
                    ))
                  ) : (
                    <div className="clear-state"><Check size={18} /><span><strong>No runs need attention</strong><small>Failed, partial, and stale runs will appear here.</small></span></div>
                  )}
                </section>
                <section className="panel launch-panel">
                  <span className="eyebrow">NEXT ACTION</span>
                  <h2>Start an evaluation</h2>
                  <p>
                    Upload a trace, validate the graph, and run the AI pipeline.
                  </p>
                  <button className="primary-button" onClick={() => navigate("New evaluation")}>
                    Open evaluation wizard <ArrowRight size={16} />
                  </button>
                </section>
              </div>
            </>
          )}

          {page === "New evaluation" && (
            <>
              <div className="wizard-tabs">
                {wizardSteps.map((s, i) => (
                  <button
                    type="button"
                    key={s}
                    aria-current={wizard === i ? "step" : undefined}
                    className={wizard === i ? "current" : ""}
                    onClick={() => {
                      setWizard(i);
                      setError("");
                    }}
                  >
                    {i + 1}. {s}
                  </button>
                ))}
              </div>
              <section className="panel evaluation-panel">
                {wizard === STEP.trace && (
                  <div className="evaluation-split">
                    <div className="evaluation-pane evaluation-pane--controls">
                      <div className="evaluation-pane-heading">
                        <span className="eyebrow">SOURCE</span>
                        <h2>Trace setup</h2>
                      </div>
                      <p>
                        Examples from the GAIA archive, without ground-truth
                        annotations. Attachments are not included.
                      </p>
                      <label className="field">
                        <span>Load example</span>
                        <select
                          aria-label="Load example trace"
                          value={selectedExample}
                          onChange={(e) => {
                            const value = e.target.value;
                            setSelectedExample(value);
                            if (value) void loadTestTrace(Number(value));
                          }}
                        >
                          <option value="">Choose an example…</option>
                          {[1, 2, 3].map((id) => (
                            <option key={id} value={id}>Example {id}</option>
                          ))}
                        </select>
                      </label>
                      <Field
                        label="Trace name"
                        value={config.name}
                        change={(v) => update("name", v)}
                      />
                      <label className="file-input">
                        JSON / JSONL / raw OpenTelemetry spans
                        <input
                          type="file"
                          accept=".json,.jsonl"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              const text = await f.text();
                              setRaw(text);
                              update("name", f.name);
                              parseTrace(text);
                            }
                          }}
                        />
                      </label>
                      <div className="compact-source-editor">
                        <Field
                          label="Paste JSON / JSONL"
                          area
                          value={raw}
                          change={setRaw}
                        />
                      </div>
                      <div className="evaluation-inline-actions">
                        <button onClick={() => parseTrace(raw)}>
                          Parse &amp; validate trace
                        </button>
                        <span className="hint">Context: Full trace</span>
                      </div>
                    </div>
                    <div className="evaluation-divider" aria-hidden="true" />
                    <div className="evaluation-pane evaluation-pane--preview">
                      <div className="evaluation-pane-heading evaluation-pane-heading--row">
                        <div>
                          <span className="eyebrow">DATA PREVIEW</span>
                          <h2>Normalized trace</h2>
                        </div>
                        <span className="preview-count">{steps.length} steps</span>
                      </div>
                      {traceIssue && <p role="alert" className="hint">Preview paused: {traceIssue}</p>}
                      {traceView(steps)}
                    </div>
                  </div>
                )}
                {wizard === STEP.design && (
                  <div className="evaluation-split design-fields">
                    <div className="evaluation-pane evaluation-pane--controls">
                      <div className="evaluation-pane-heading">
                        <span className="eyebrow">SETUP</span>
                        <h2>Evaluation design</h2>
                      </div>
                      {templateButtons}
                      <p className="evaluation-copy">
                        Apply the example or provide your own taxonomy and output schema.
                        Taxonomy accepts Markdown or JSON (an array of categories or section
                        object); the schema accepts JSON or Markdown with a fenced JSON block.
                        Test files: <a href="/test-data/output-schema.json" download>output schema</a>
                        {" · "}<a href="/test-data/TAXONOMY.md" download>taxonomy</a>
                        {" · "}<a href="/test-data/trail-output-schema.json" download>TRAIL output schema</a>
                        {" · "}<a href="/test-data/trail-taxonomy.md" download>TRAIL taxonomy</a>.
                      </p>
                      <div>
                        <Field
                          label="Evaluation objective"
                          area
                          value={config.objective}
                          change={(v) => update("objective", v)}
                        />
                      </div>
                      <div className="design-file-stack">
                        {designUpload("taxonomy")}
                        {designUpload("schema")}
                      </div>
                    </div>
                    <div className="evaluation-divider" aria-hidden="true" />
                    <div className="evaluation-pane evaluation-pane--preview">
                      <div className="evaluation-pane-heading evaluation-pane-heading--row">
                        <div>
                          <span className="eyebrow">DATA PREVIEW</span>
                          <h2>Taxonomy &amp; output</h2>
                        </div>
                        <button onClick={() => void validateDesign()}>
                          Validate design
                        </button>
                      </div>
                      <div className="design-preview-grid">
                        <Field
                          label="Taxonomy (Markdown or JSON)"
                          area
                          value={config.taxonomy}
                          change={(v) => update("taxonomy", v)}
                        />
                        <Field
                          label="Output schema (JSON or Markdown with fenced JSON)"
                          area
                          value={config.schema}
                          change={(v) => update("schema", v)}
                        />
                      </div>
                      <details className="optional-field few-shot-field">
                          <summary>Optional few-shot examples</summary>
                          <div className="toolbar">
                            {fewShotPresets.map((preset) => (
                              <button
                                key={preset.id}
                                type="button"
                                title={preset.description}
                                onClick={() => {
                                  setConfig((c) => ({
                                    ...c,
                                    examples: JSON.stringify(preset.examples, null, 2),
                                  }));
                                  setError("");
                                  setNotice(`Few-shot preset "${preset.name}" applied (${preset.examples.length} examples).`);
                                }}
                              >
                                {preset.name}
                              </button>
                            ))}
                            {config.examples !== "[]" && config.examples.trim() !== "" && (
                              <button
                                type="button"
                                onClick={() => {
                                  setConfig((c) => ({ ...c, examples: "[]" }));
                                  setNotice("Few-shot examples cleared.");
                                }}
                              >
                                Clear examples
                              </button>
                            )}
                          </div>
                          <p className="hint">
                            Presets fill the JSON array below; each example pairs a trace
                            excerpt with the expected judge output. Edit the array freely
                            after applying a preset.
                          </p>
                          <Field
                            label="Examples (JSON array)"
                            area
                            value={config.examples}
                            change={(v) => update("examples", v)}
                          />
                      </details>
                    </div>
                  </div>
                )}
                {wizard === STEP.judges && (
                  <section className="judge-pool judge-pool-step">
                    <div className="judge-pool-head">
                      <div>
                        <span className="eyebrow">META AGENT</span>
                        <h2>
                          {config.nodes.length} judges ·{" "}
                          {config.edges.length} connections
                        </h2>
                        <p className="hint">{designSourceHint}</p>
                      </div>
                      <div className="judge-pool-actions">
                        <span className={`design-state design-state--${designState}`}>
                          {designStateLabel}
                        </span>
                        <button
                          className={awaitingAcceptance ? "" : "primary-button"}
                          disabled={designState === "generating" || !connected}
                          onClick={() => void generateDesign()}
                        >
                          <Sparkles size={15} />
                          {designState === "generating"
                            ? "Generating…"
                            : generatedDesign
                              ? "Regenerate Meta Agent"
                              : "Run Meta Agent"}
                        </button>
                        {awaitingAcceptance && (
                          <button className="primary-button" onClick={acceptDesign}>
                            <Check size={15} /> Use this pool
                          </button>
                        )}
                        {config.nodes.length > 0 && (
                          <button onClick={resetJudgePool}>
                            <RotateCcw size={15} /> Reset Meta Agent
                          </button>
                        )}
                      </div>
                    </div>
                    {designState === "generating" && (
                      <p className="hint" role="status">
                        The meta-agent is designing the judges and how they
                        connect for this trace.
                      </p>
                    )}
                    {designState === "failed" && (
                      <div className="finding" role="alert">
                        <p>{designError}</p>
                        <button onClick={() => void generateDesign()}>
                          Retry generation
                        </button>
                      </div>
                    )}
                    {designState === "stale" && (
                      <p className="hint" role="alert">
                        {runBlockedReason} The evaluation stays blocked until the
                        pool is regenerated.
                      </p>
                    )}
                    {generatedDesign && designState === "generated" && (
                      <p className="hint" role="status">
                        {designAccepted
                          ? "Accepted. This exact pool and DAG are what the run executes."
                          : "Review the generated judges, then accept the pool to unlock the run."}
                      </p>
                    )}
                    {designMeta && (
                      <p className="hint">
                        Meta Agent model: {designMeta.pool_model || "n/a"}
                        {designMeta.unused_judges?.length
                          ? ` · Left out of the pipeline: ${designMeta.unused_judges.join(", ")}`
                          : ""}
                      </p>
                    )}
                    {config.nodes.length > 0 ? (
                      <>
                        <div className="judge-cards">
                          {config.nodes.map((name) => {
                            const judge = config.judges?.find((j) => j.name === name);
                            const instructions =
                              config.judge_instructions?.[name] ?? judge?.instructions ?? "";
                            return (
                              <details key={name} className="judge-card">
                                <summary>
                                  <strong>{name}</strong>
                                  <small>
                                    {judge?.model || config.model || "model from settings"}
                                  </small>
                                </summary>
                                {instructions ? (
                                  <pre>{instructions}</pre>
                                ) : (
                                  <p className="hint">
                                    No judge-specific instructions; the runner uses the
                                    shared evaluation prompt for this node.
                                  </p>
                                )}
                              </details>
                            );
                          })}
                        </div>
                        <PipelineGraph config={config} />
                      </>
                    ) : (
                      designState !== "generating" && (
                        <p className="hint" role="status">
                          No judges yet. Run Meta Agent to continue.
                        </p>
                      )
                    )}
                  </section>
                )}
                {wizard === STEP.run && (
                  <div className="evaluation-split evaluation-run-layout">
                    <div className="evaluation-pane evaluation-pane--controls">
                      <div className="evaluation-pane-heading">
                        <span className="eyebrow">EXECUTION</span>
                        <h2>
                          {active
                            ? "The server is running " +
                              (active.execution === "ai" ? "AI pipeline" : "dry run")
                            : "Ready to run"}
                        </h2>
                      </div>
                      {!active && (
                        <>
                        <label className="field">
                          <span>Execution mode</span>
                          <select
                            disabled={launching}
                            value={execution}
                            onChange={(e) => setExecution(e.target.value)}
                          >
                            <option value="ai">
                              LLM Judge
                            </option>
                            <option value="offline">
                              Dry run
                            </option>
                          </select>
                        </label>
                        {execution === "ai" && (
                          <div className="finding">
                            <p>
                              OpenRouter API ·{" "}
                              {activeModelLabel}
                            </p>
                          </div>
                        )}
                        </>
                      )}
                      {phase >= 0 && (
                        <div className="run-progress-block">
                        <progress max={4} value={phase + 1} />
                        <div aria-live="polite">
                          {
                            [
                              "Preparing trace",
                              "Executing judges",
                              "Aggregation",
                              "Saving result",
                            ][phase]
                          }
                        </div>
                        <button onClick={cancel}>Cancel run</button>
                        </div>
                      )}
                      {!active && runBlocked && (
                        <p className="hint" role="alert">
                          {runBlockedReason}{" "}
                          <button className="link" onClick={() => setWizard(STEP.judges)}>
                            Open Meta Agent
                          </button>
                        </p>
                      )}
                      {!active && (
                        <button
                          className="primary-button"
                          disabled={launching || !connected || runBlocked}
                          onClick={launch}
                        >
                          <Play size={16} />
                          {launching
                            ? "Creating run…"
                            : execution === "ai"
                              ? "Run AI evaluation"
                              : "Run dry check"}
                        </button>
                      )}
                    </div>
                    <div className="evaluation-divider" aria-hidden="true" />
                    <div className="evaluation-pane evaluation-pane--preview">
                      <div className="evaluation-pane-heading evaluation-pane-heading--row">
                        <div>
                          <span className="eyebrow">PIPELINE PREVIEW</span>
                          <h2>{config.nodes.length} configured judges</h2>
                        </div>
                        <span className="preview-count">{steps.length} trace steps</span>
                      </div>
                      <PipelineGraph config={active?.config ?? config} />
                    </div>
                  </div>
                )}
                {wizard === STEP.verdict &&
                  (detail ? (
                    <div className="evaluation-result">{resultView(detail)}</div>
                  ) : (
                    <p className="evaluation-empty-result">
                      No result yet. Open Run and start an AI evaluation or dry check.
                    </p>
                  ))}
                <div className="wizard-footer">
                  <button
                    disabled={wizard === STEP.trace || !!active}
                    onClick={() => setWizard(wizard - 1)}
                  >
                    Back
                  </button>
                  {wizard < STEP.run && (
                    <button className="primary-button" onClick={next}>
                      Continue <ArrowRight size={15} />
                    </button>
                  )}
                </div>
              </section>
            </>
          )}

          {page === "Runs" && (
            <>
              <section className="panel runs-panel">
                <div className="toolbar">
                  <Search size={17} />
                  <input
                    aria-label="Search runs"
                    placeholder="Search by trace, ID, or status"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <select
                    aria-label="Run status"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                  >
                    {["All", "Completed", "Failed", "Cancelled", "Archived"].map((s) => (
                      <option key={s}>{s}</option>
                    ))}
                  </select>
                </div>
                <div className="run-actions" role="toolbar" aria-label="Selected run actions">
                  <span>{selectedRuns.length ? `${selectedRuns.length} selected` : "Select runs below"}</span>
                  <button
                    disabled={selectedRuns.length !== 1}
                    onClick={() => {
                      const run = selectedRuns[0];
                      if (run) saveFile(`run-${run.id}.json`, run);
                    }}
                  >
                    Export
                  </button>
                  <button disabled={selectedRuns.length !== 1} onClick={() => selectedRuns[0] && cloneRun(selectedRuns[0])}>
                    Clone
                  </button>
                  <button disabled={!selectedRuns.length} onClick={() => void archiveSelectedRuns()}>
                    {selectedRuns.length > 0 && selectedRuns.every((run) => run.archived) ? "Restore" : "Archive"}
                  </button>
                  <button
                    className="danger-button"
                    disabled={!selectedRuns.some((run) => run.status !== "Running")}
                    onClick={() => void deleteSelectedRuns()}
                  >
                    Delete
                  </button>
                </div>
                {table(filteredRuns)}
                {compare.length > 0 && (
                  <div className="compare-tray" role="region" aria-label="Selected runs for comparison">
                    <div>
                      <strong>{compare.length} selected</strong>
                      <span>{runs.filter((run) => compare.includes(run.id)).map((run) => run.config.name).join(" · ")}</span>
                    </div>
                    <button onClick={() => setCompare([])}>Clear</button>
                    <button className="primary-button" disabled={compare.length < 2} onClick={() => navigate("Compare")}>Compare selected</button>
                  </div>
                )}
              </section>
              {detail && resultView(detail)}
            </>
          )}
          {page === "Compare" && (
            <section className="panel">
              <h2>Compare runs</h2>
              <p>
                Select at least two runs in Runs. Cost and quality metrics are
                unavailable without live evaluations.
              </p>
              {compare.length < 2 ? (
                <button onClick={() => navigate("Runs")}>
                  Select runs
                </button>
              ) : (
                <div className="comparison">
                  {runs
                    .filter((r) => compare.includes(r.id))
                    .map((r) => (
                      <article key={r.id}>
                        <h3>{r.config.name}</h3>
                        <span className="tag">{r.id}</span>
                        <p>{r.verdict}</p>
                        <p>Model: {r.config.model}</p>
                        <p>Context: {r.config.mode}</p>
                        <p>Judges: {r.config.nodes.length}</p>
                      </article>
                    ))}
                </div>
              )}
            </section>
          )}

          {page === "Observability" && (
            <section className="panel">
              <h2>Usage & cost</h2>
              <p>
                AI requests:{" "}
                {runs
                  .filter((r) => r.execution === "ai")
                  .reduce((n, r) => n + (r.usage?.calls ?? 0), 0)}
                . Check costs with the provider for now; errors and cancellations
                may have incomplete usage data.
              </p>
              <h3>Artifacts</h3>
              {runs.map((r) => (
                <div className="edge-row" key={r.id}>
                  <span>{r.id} · result + configuration + trace</span>
                  <button onClick={() => saveFile(`run-${r.id}.json`, r)}>
                    Download JSON
                  </button>
                </div>
              ))}
              <h3>Langfuse</h3>
              <p>
                Not connected. Demo runs have no external trace IDs.
              </p>
            </section>
          )}
          {page === "Settings" && (
            <section className="panel">
              <h2>Execution defaults</h2>
              <div className="two-col">
                {(["timeout", "retries", "concurrency"] as const).map(
                  (k) => (
                    <Field
                      key={k}
                      label={k}
                      type="number"
                      value={settings[k]}
                      change={(v) =>
                        setSettings((s) => ({
                          ...s,
                          [k]: Math.max(
                            k === "retries" ? 0 : 1,
                            Number(v) || 0,
                          ),
                        }))
                      }
                    />
                  ),
                )}
              </div>
              <p>
                Saved locally. Application to the live runner will arrive with the
                API.
              </p>
              <EnvSettings
                onSettingsSaved={(fields) => {
                  const model = fields.find((field) => field.name === "AGENT_NODE_MODEL")?.value?.trim() || "";
                  setConfig((current) => ({ ...current, model }));
                }}
              />
              <button
                onClick={() => saveFile("execution-defaults.json", settings)}
              >
                Export defaults
              </button>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
