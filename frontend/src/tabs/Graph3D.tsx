import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import SpriteText from "three-spritetext";
import { api, GraphData } from "../api";

const R = 140;
// 3 categorical hues (validated for dark surfaces) + neutral grey; the type is also in the legend and the label text.
const CAT = {
  org: { label: "Company / party", color: "#3987e5" },
  authority: { label: "Regulator / authority", color: "#d95926" },
  legal: { label: "Statute, rule, control or role", color: "#199e70" },
  other: { label: "Other term", color: "#9aa3b5" },
} as const;
type CatKey = keyof typeof CAT;
function category(id: string): CatKey {
  if (/\b(Ltd|Pvt|Holdings)\b/.test(id)) return "org";
  if (/^(CBDT|CBIC|RBI|FIU|SEBI|GST Council)$/.test(id)) return "authority";
  if (/^(Section|Rule|Para|Clause)\b|\bAct\b|\bRules\b|^IA-\d+|Officer/.test(id)) return "legal";
  return "other";
}
function hasWebGL() {
  try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch { return false; }
}
/** Fibonacci sphere: pins each node to a point on a sphere of radius R. */
function onSphere(i: number, n: number) {
  const y = 1 - (2 * (i + 0.5)) / n, r = Math.sqrt(1 - y * y), th = Math.PI * (1 + Math.sqrt(5)) * i;
  const x = R * r * Math.cos(th), yy = R * y, z = R * r * Math.sin(th);
  // x/y/z are set explicitly as well: with the force engine frozen (cooldownTicks=0) fx/fy/fz alone are never applied.
  return { x, y: yy, z, fx: x, fy: yy, fz: z };
}

export default function Graph3D({ onChange }: { onChange: () => void }) {
  const [data, setData] = useState<GraphData | null>(null);
  const [err, setErr] = useState("");
  const [showRefs, setShowRefs] = useState(false);
  const [selN, setSelN] = useState<Set<string>>(new Set());
  const [selE, setSelE] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(900);
  const fg = useRef<any>(null);
  const webgl = useMemo(hasWebGL, []);

  const [busy, setBusy] = useState(false);
  const [allLabels, setAllLabels] = useState(false);
  const load = () => api.graph().then((g) => { setData(g); setSelN(new Set(g.selection.nodes)); setSelE(new Set(g.selection.edges)); setLoaded(true); setErr(""); })
    .catch((e) => setErr(e.message));
  const loadSample = async () => {
    setBusy(true);
    try { await api.ingestSample(); await load(); onChange(); } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };
  useEffect(() => { load(); }, []);
  useEffect(() => { if (box.current) setW(box.current.clientWidth); }, [data]);
  useEffect(() => {   // translucent wireframe shell so the spherical layout is visible
    const sc = fg.current?.scene?.(); if (!sc || !webgl) return;
    const shell = new THREE.Mesh(new THREE.SphereGeometry(R, 32, 20),
      new THREE.MeshBasicMaterial({ color: 0x3a5a99, wireframe: true, transparent: true, opacity: 0.12 }));
    sc.add(shell);
    return () => { sc.remove(shell); shell.geometry.dispose(); (shell.material as THREE.Material).dispose(); };
  }, [data, showRefs, webgl]);
  useEffect(() => { const t = setTimeout(() => fg.current?.cameraPosition({ x: 0, y: 0, z: 390 }), 150); return () => clearTimeout(t); }, [data, showRefs]);
  useEffect(() => { if (loaded) api.setFilter([...selN], [...selE]).then(onChange).catch((e) => setErr(e.message)); }, [selN, selE, loaded]);

  const graph = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    const keep = data.nodes.filter((n) => showRefs || n.kind === "entity");
    const ids = new Set(keep.map((n) => n.id));
    return {
      nodes: keep.map((n, i) => ({ ...n, ...onSphere(i, keep.length) })),
      links: data.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({ ...e })),
    };
  }, [data, showRefs]);

  const labelCut = useMemo(() => {   // label the ~28 best-connected entities by default; the rest on hover
    const d = graph.nodes.map((n: any) => n.degree).sort((a: number, b: number) => b - a);
    return d[Math.min(27, d.length - 1)] ?? 0;
  }, [graph]);
  const toggle = (s: Set<string>, id: string, set: (x: Set<string>) => void) => {
    const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); set(n);
  };
  const filtering = selN.size > 0 || selE.size > 0;

  return (
    <div>
      <div className="row">
        <label><input type="checkbox" checked={allLabels} onChange={(e) => setAllLabels(e.target.checked)} /> label all nodes</label>
        <label><input type="checkbox" checked={showRefs} onChange={(e) => setShowRefs(e.target.checked)} /> show document/section reference nodes</label>
        <button className="btn" onClick={loadSample} disabled={busy}>{busy ? "Indexing…" : "Load sample data"}</button>
        <button className="btn secondary" onClick={load}>Refresh</button>
        <button className="btn secondary" onClick={() => { setSelN(new Set()); setSelE(new Set()); }}>Clear selection</button>
        <span className="muted">{filtering ? `Retrieval restricted to ${selN.size} node(s) / ${selE.size} edge(s)` : "No selection — retrieval is unrestricted"}
          · click a node or edge to select / unselect · drag to rotate</span>
      </div>
      {err && <div className="err">{err}</div>}
      {data && !data.nodes.length && <div className="muted">The graph is empty — click “Load sample data” above.</div>}
      {!webgl && <div className="card" style={{ marginBottom: 12 }}>
        <div className="err">WebGL is not available in this browser, so the 3D canvas is disabled. Select entities from the list instead — the retrieval filter works the same way.</div>
        <div style={{ maxHeight: 420, overflow: "auto", marginTop: 8, columns: "3 220px" }}>
          {graph.nodes.map((n) => <label key={n.id} style={{ display: "block" }}>
            <input type="checkbox" checked={selN.has(n.id)} onChange={() => toggle(selN, n.id, setSelN)} /> {n.id} <span className="muted">({n.chunks})</span></label>)}
        </div></div>}
      <div className="canvas-box" ref={box} style={webgl ? undefined : { display: "none" }}>
        {webgl && graph.nodes.length > 0 && (
          <ForceGraph3D ref={fg} width={w} height={560} graphData={graph} backgroundColor="#0b0f1a" cooldownTicks={0}
            nodeLabel={(n: any) => `${n.id} (${n.chunks} chunks, degree ${n.degree})`}
            nodeColor={(n: any) => (selN.has(n.id) ? "#ffd166" : n.kind !== "entity" ? "#6b7280" : filtering ? "#4c5a73" : CAT[category(n.id)].color)}
            nodeThreeObjectExtend={true}
            nodeThreeObject={(n: any) => {
              if (!(allLabels || selN.has(n.id) || n.degree >= labelCut)) return false as any;
              const t = new SpriteText(n.id.length > 26 ? n.id.slice(0, 25) + "…" : n.id, 5.5, selN.has(n.id) ? "#ffd166" : "#e6ebf5");
              t.position.set(0, 6, 0); (t as any).material.depthWrite = false; return t;
            }}
            nodeVal={(n: any) => 1 + Math.min(n.degree, 20) / 4}
            linkColor={(l: any) => (selE.has(l.id) ? "#ffd166" : filtering ? "#2e384d" : "#5b6b8c")}
            linkOpacity={0.55} linkWidth={(l: any) => (selE.has(l.id) ? 2 : 0.4)}
            linkLabel={(l: any) => l.relation}
            onNodeClick={(n: any) => toggle(selN, n.id, setSelN)}
            onLinkClick={(l: any) => toggle(selE, l.id, setSelE)} />
        )}
      </div>
      <div className="legend" style={{ marginTop: 8 }}>{(Object.keys(CAT) as CatKey[]).map((k) =>
        <span key={k}><i style={{ background: CAT[k].color }} />{CAT[k].label}</span>)}<span><i style={{ background: "#ffd166" }} />Selected (retrieval filter)</span></div>
      {data && <div className="muted" style={{ marginTop: 6 }}>{data.nodes.length} nodes · {data.edges.length} edges (all nodes incl. references)</div>}
    </div>
  );
}
