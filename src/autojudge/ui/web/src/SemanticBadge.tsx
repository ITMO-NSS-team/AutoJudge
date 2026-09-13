import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  Clock3,
  FlaskConical,
  LoaderCircle,
  MinusCircle,
  WifiOff,
  XCircle,
} from "lucide-react";

type BadgeTone =
  | "neutral"
  | "info"
  | "success"
  | "warning"
  | "error"
  | "violet";

const statusMeta: Record<string, { tone: BadgeTone; icon: typeof CheckCircle2 }> = {
  draft: { tone: "neutral", icon: CircleDashed },
  ready: { tone: "info", icon: CheckCircle2 },
  running: { tone: "info", icon: LoaderCircle },
  completed: { tone: "success", icon: CheckCircle2 },
  complete: { tone: "success", icon: CheckCircle2 },
  partial: { tone: "warning", icon: AlertCircle },
  failed: { tone: "error", icon: XCircle },
  cancelled: { tone: "neutral", icon: MinusCircle },
  stale: { tone: "warning", icon: Clock3 },
  offline: { tone: "error", icon: WifiOff },
  demo: { tone: "violet", icon: FlaskConical },
  "not connected": { tone: "neutral", icon: WifiOff },
};

export default function SemanticBadge({
  status,
  label,
  tone,
}: {
  status: string;
  label?: string;
  tone?: BadgeTone;
}) {
  const normalized = status.trim().toLowerCase();
  const meta = statusMeta[normalized] ?? { tone: "neutral" as BadgeTone, icon: CircleDashed };
  const Icon = meta.icon;
  return (
    <span className={`semantic-badge semantic-badge--${tone ?? meta.tone}`}>
      <Icon size={13} aria-hidden="true" className={normalized === "running" ? "is-spinning" : ""} />
      {label ?? status}
    </span>
  );
}
