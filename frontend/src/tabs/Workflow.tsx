import { useEffect, useMemo, useState } from "react";
import { ReactFlow, Background, Controls, MarkerType, Position, Handle, type Node, type Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, AREAS, ENGINES, Area, ChatResult, Engine } from "../api";

function Step({ data }: { data: { label: string } }) {
  return (
    <div style={{ padding: "10px 8px", textAlign: "center", fontSize: 12 }}>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
      <Handle id="loop-out" type="source" position={Position.Top} style={{ left: "70%" }} />
      <Handle id="loop-in" type="target" position={Position.Top} style={{ left: "30%" }} />
      {data.label}
    </div>
  );
}
const nodeTypes = { step: Step };

export default function Workflow({ onChange }: { onChange: () => void }) {
  const [engine, setEngine] = useState<Engine>("graph");
  const [wf, setWf] = useState<Awaited<ReturnType<typeof api.workflow>> | null>(null);
  const [offN, setOffN] = useState<Set<string>>(new Set());
  const [offE, setOffE] = useState<Set<string>>(new Set());
  const [q, setQ] = useState("Which tax penalties for the parent of the company acquired by Orion Holdings are covered by the SPA indemnity?");
  const [area, setArea] = useState<Area>("deals");
  const [res, setRes] = useState<ChatResult | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => { setOffN(new Set()); setOffE(new Set()); setRes(null); api.workflow(engine).then(setWf).catch((e) => setErr(e.message)); }, [engine]);

  const nodes: Node[] = useMemo(() => (wf?.nodes ?? []).map((n, i) => ({
    id: n.id, type: "step", position: { x: i * 210, y: 130 }, data: { label: `${n.label}${offN.has(n.id) ? " (off)" : ""}` },
    style: { width: 170, opacity: offN.has(n.id) ? 0.4 : 1, border: `2px ${offN.has(n.id) ? "dashed" : "solid"} var(--accent)`,
      background: "var(--surface-1)", color: "var(--text-primary)", borderRadius: 8 },
  })), [wf, offN]);
  const edges: Edge[] = useMemo(() => (wf?.edges ?? []).map((e) => ({
    id: e.id, source: e.source, target: e.target, animated: !offE.has(e.id), label: e.cyclic ? "loop back if context is thin" : undefined, labelBgStyle: { fill: "var(--surface-2)" },
    type: "smoothstep", ...(e.cyclic ? { sourceHandle: "loop-out", targetHandle: "loop-in" } : {}), markerEnd: { type: MarkerType.ArrowClosed },
    style: { strokeDasharray: offE.has(e.id) ? "4 4" : undefined, opacity: offE.has(e.id) ? 0.3 : 1, stroke: "var(--text-secondary)" },
  })), [wf, offE]);

  const flip = (s: Set<string>, id: string, set: (x: Set<string>) => void) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); set(n); };
  const run = async () => {
    setBusy(true); setErr("");
    try { setRes(await api.chat({ query: q, engine, practice_area: area, disabled_nodes: [...offN], disabled_edges: [...offE] })); onChange(); }
    catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };

  return (
    <div>
      <div className="row">
        <select value={engine} onChange={(e) => setEngine(e.target.value as Engine)}>{ENGINES.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}</select>
        <select value={area} onChange={(e) => setArea(e.target.value as Area)}>{AREAS.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}</select>
        <span className="muted">Click a node to bypass it (state passes through) · click an edge to cut it (flow ends upstream)</span>
      </div>
      <div className="canvas-box" style={{ height: 300, background: "var(--surface-2)" }}>
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView nodesDraggable={false}
          onNodeClick={(_, n) => flip(offN, n.id, setOffN)} onEdgeClick={(_, e) => flip(offE, e.id, setOffE)}>
          <Background /><Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <div className="row" style={{ marginTop: 12 }}>
        <input style={{ flex: 1, minWidth: 280 }} value={q} onChange={(e) => setQ(e.target.value)} maxLength={2000} />
        <button className="btn" disabled={busy || !q.trim()} onClick={run}>{busy ? "Running…" : "Run variation"}</button>
      </div>
      {err && <div className="err">{err}</div>}
      {res && <div className="card">
        <div className="muted">Executed path: {res.trace.join(" → ")}</div>
        <div className="muted">Latency (ms): {Object.entries(res.timings_ms).map(([k, v]) => `${k} ${Math.round(v)}`).join(" · ")}</div>
        <div className="msg">{res.answer}</div>
        <div>{res.citations.map((c) => <span key={c.chunk_id} className="cite">{c.source} · {c.section}</span>)}</div>
      </div>}
    </div>
  );
}
