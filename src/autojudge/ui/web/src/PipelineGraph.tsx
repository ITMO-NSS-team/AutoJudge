import { useEffect, useMemo, useRef, useState } from "react";
import { GitFork, Layers3 } from "lucide-react";

type GraphConfig = { nodes: string[]; edges: string[][] };
type Point = { x: number; y: number };

const NODE_WIDTH = 190;
const NODE_GAP = 16;

function executionLevels(config: GraphConfig) {
  const incoming = Object.fromEntries(config.nodes.map((node) => [node, 0]));
  const children = Object.fromEntries(config.nodes.map((node) => [node, [] as string[]]));
  config.edges.forEach(([from, to]) => {
    if (from in children && to in incoming) {
      children[from].push(to);
      incoming[to] += 1;
    }
  });
  const levels: string[][] = [];
  let ready = config.nodes.filter((node) => incoming[node] === 0);
  const seen = new Set<string>();
  while (ready.length) {
    levels.push(ready);
    ready.forEach((node) => seen.add(node));
    ready = ready.flatMap((node) => children[node]).filter((node) => {
      incoming[node] -= 1;
      return incoming[node] === 0;
    });
  }
  const unresolved = config.nodes.filter((node) => !seen.has(node));
  if (unresolved.length) levels.push(unresolved);
  return levels;
}

export default function PipelineGraph({ config }: { config: GraphConfig }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const levels = useMemo(() => executionLevels(config), [config.nodes, config.edges]);
  const compact = width < 640;
  const nodeWidth = compact ? Math.max(NODE_WIDTH, width - 32) : NODE_WIDTH;
  const rowHeight = compact ? 82 : 96;
  const orderedNodes = levels.flat();
  const height = compact
    ? Math.max(280, orderedNodes.length * rowHeight + 24)
    : Math.max(300, levels.length * rowHeight + 52);
  const widestLevel = Math.max(1, ...levels.map((level) => level.length));
  // Wide levels get a canvas wider than the visible pane instead of shrinking
  // or overlapping nodes; the pane scrolls horizontally to show the rest.
  const canvasWidth = compact
    ? width
    : Math.max(width, widestLevel * (nodeWidth + NODE_GAP) + NODE_GAP);
  const scrollable = !compact && canvasWidth > width;

  useEffect(() => {
    if (!rootRef.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(rootRef.current);
    return () => observer.disconnect();
  }, []);

  const defaults = useMemo(() => {
    const result: Record<string, Point> = {};
    if (compact) {
      orderedNodes.forEach((node, index) => {
        result[node] = { x: 16, y: 16 + index * rowHeight };
      });
      return result;
    }
    levels.forEach((level, levelIndex) => {
      level.forEach((node, nodeIndex) => {
        const center = (canvasWidth * (nodeIndex + 1)) / (level.length + 1);
        result[node] = {
          x: Math.max(16, Math.min(canvasWidth - nodeWidth - 16, center - nodeWidth / 2)),
          y: 18 + levelIndex * rowHeight,
        };
      });
    });
    return result;
  }, [canvasWidth, compact, levels, nodeWidth, orderedNodes, rowHeight]);

  const pointFor = (node: string) => defaults[node] ?? { x: 16, y: 16 };

  return (
    <div className="pipeline-graph-wrap">
      <div className="pipeline-graph-key">
        <span><i className="graph-key-dot graph-key-dot--judge" />Judge</span>
        <span><i className="graph-key-dot graph-key-dot--final" />Final judge</span>
        {scrollable && <span className="pipeline-graph-hint">Scroll to see every judge</span>}
      </div>
      <div
        ref={rootRef}
        className={`pipeline-graph ${scrollable ? "pipeline-graph--scrollable" : ""}`}
        style={{ height }}
        aria-label="Judge pipeline graph"
      >
        <div className="pipeline-graph-canvas" style={{ width: canvasWidth, height }}>
          <svg className="pipeline-edges" width={canvasWidth} height={height} aria-hidden="true">
            <defs>
              <marker id="pipeline-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" />
              </marker>
            </defs>
            {config.edges.map(([from, to], index) => {
              const start = pointFor(from);
              const end = pointFor(to);
              const x1 = start.x + nodeWidth / 2;
              const y1 = start.y + 58;
              const x2 = end.x + nodeWidth / 2;
              const y2 = end.y;
              const bend = Math.max(26, Math.abs(y2 - y1) / 2);
              return (
                <path
                  key={`${from}-${to}-${index}`}
                  d={`M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`}
                  markerEnd="url(#pipeline-arrow)"
                />
              );
            })}
          </svg>
          {config.nodes.map((node) => {
            const finalNode = node === "FINAL_AGGREGATOR";
            const point = pointFor(node);
            return (
              <div
                key={node}
                className={`pipeline-node ${finalNode ? "pipeline-node--final" : ""}`}
                title={node}
                style={{ width: nodeWidth, transform: `translate(${point.x}px, ${point.y}px)` }}
              >
                <span className="pipeline-node-icon">{finalNode ? <Layers3 size={18} /> : <GitFork size={17} />}</span>
                <strong>{node}</strong>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
