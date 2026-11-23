import { useEffect, useRef, useState } from "react";
import { api, BenchResult, Cell, ENGINES, Engine } from "../api";

const TYPES = [
  { id: "simple", label: "Simple keyword" }, { id: "multi_hop", label: "Multi-hop relational" },
  { id: "vocab_mismatch", label: "Vocabulary mismatch" }, { id: "all", label: "All queries" },
];
const COLOR: Record<Engine, string> = { standard: "var(--series-standard)", hyde: "var(--series-hyde)", graph: "var(--series-graph)" };

// Expected behaviour per the architecture document (section 5) - targets, not measurements.
const EXPECT: Record<string, Record<Engine, string>> = {
  simple: { standard: "High precision / Low latency", hyde: "Moderate (overhead of HyDE)", graph: "Moderate (graph overhead)" },
  multi_hop: { standard: "Poor (misses indirect relationships)", hyde: "Moderate (semantic expansion)", graph: "Superior (explores k-hop edges)" },
  vocab_mismatch: { standard: "Poor (relies on exact vector distance)", hyde: "Superior (synthesizes domain terms)", graph: "High (entity resolution maps synonyms)" },
};

interface Tip { x: number; y: number; text: string }

function GroupedBars({ title, metric, fmt, max, matrix, setTip }: {
  title: string; metric: keyof Cell; fmt: (v: number) => string; max: number;
  matrix: Record<Engine, Record<string, Cell>>; setTip: (t: Tip | null) => void;
}) {
  const W = 420, H = 230, L = 34, B = 44, T = 16, bw = 18, gap = 2, gw = 3 * bw + 2 * gap, step = (W - L) / TYPES.length;
  const y = (v: number) => T + (H - T - B) * (1 - Math.min(v, max) / max);
  const ticks = [0, max / 2, max];
  return (
    <div className="card">
      <b>{title}</b>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${title} by query type and engine`}>
        {ticks.map((t) => <g key={t}>
          <line x1={L} x2={W} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={t === 0 ? 1 : 0.5} />
          <text x={L - 6} y={y(t) + 4} fontSize={10} textAnchor="end" fill="var(--text-muted)">{fmt(t)}</text></g>)}
        {TYPES.map((tp, gi) => {
          const x0 = L + gi * step + (step - gw) / 2;
          return <g key={tp.id}>
            {ENGINES.map((e, si) => {
              const v = matrix[e.id][tp.id][metric] as number, x = x0 + si * (bw + gap), top = y(v), h = Math.max(y(0) - top, 0), r = Math.min(4, h);
              return <g key={e.id}
                onMouseMove={(ev) => setTip({ x: ev.clientX + 12, y: ev.clientY + 12, text: `${e.label} · ${tp.label}: ${fmt(v)} (n=${matrix[e.id][tp.id].n})` })}
                onMouseLeave={() => setTip(null)}>
                <rect x={x - 2} y={T} width={bw + 4} height={H - T - B} fill="transparent" />{/* hit target wider than the mark */}
                <path d={`M${x},${y(0)} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${y(0)} Z`} fill={COLOR[e.id]} />
                <text x={x + bw / 2} y={top - 4} fontSize={9} textAnchor="middle" fill="var(--text-secondary)">{fmt(v)}</text>
              </g>;
            })}
            <text x={x0 + gw / 2} y={H - B + 16} fontSize={10} textAnchor="middle" fill="var(--text-secondary)">{tp.label.split(" ")[0]}</text>
            <text x={x0 + gw / 2} y={H - B + 28} fontSize={10} textAnchor="middle" fill="var(--text-secondary)">{tp.label.split(" ").slice(1).join(" ")}</text>
          </g>;
        })}
      </svg>
    </div>
  );
}

type Row = { label: string; measured: (e: Engine) => number; fmt: (v: number) => string; better: "high" | "low";
  expected: (e: Engine) => string };

function Matrix({ res, reference: ref }: { res: NonNullable<BenchResult["result"]>; reference: BenchResult["reference"] }) {
  const m = res.matrix;
  const rows: Row[] = [
    ...(["simple", "multi_hop", "vocab_mismatch"] as const).map((t, i): Row => ({
      label: ["Simple keyword queries", "Multi-hop relational queries", "Domain vocabulary mismatch"][i] + " — recall",
      measured: (e) => m[e][t].rcr, fmt: (v) => `${Math.round(v * 100)}%`, better: "high", expected: (e) => EXPECT[t][e] })),
    { label: "Average latency", measured: (e) => m[e].all.latency_ms, fmt: (v) => `${Math.round(v)} ms`, better: "low",
      expected: (e) => `~${ref.latency_ms[e].toLocaleString()} ms` },
    { label: "Hallucination risk (1 − faithfulness)", measured: (e) => m[e].all.hallucination_risk, fmt: (v) => v.toFixed(2), better: "low",
      expected: (e) => ref.hallucination_risk[e] },
  ];
  const best = (r: Row) => {
    // compare what is displayed (rounded), so "best" never appears on cells that look identical
    const shown = (e: Engine) => r.fmt(r.measured(e));
    const vals = ENGINES.map((e) => Number(shown(e.id).replace(/[^0-9.]/g, ""))), target = r.better === "high" ? Math.max(...vals) : Math.min(...vals);
    return vals.every((v) => v === target) ? new Set<Engine>() : new Set(ENGINES.filter((_, i) => vals[i] === target).map((e) => e.id));
  };
  return (
    <div className="card" style={{ overflowX: "auto" }}>
      <b>Comparative performance matrix</b>
      <div className="muted" style={{ marginBottom: 8 }}>Measured on this run (bold) vs. the architecture document's expectation (grey). “Best” marks the better measured value in each row; no mark means a tie.</div>
      <table className="matrix">
        <thead><tr><th>Evaluation metric</th>{ENGINES.map((e) => <th key={e.id}><i className="sw" style={{ background: COLOR[e.id] }} />{e.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => { const b = best(r); return (
            <tr key={r.label}><th scope="row">{r.label}</th>
              {ENGINES.map((e) => <td key={e.id}>
                <span className="mval">{r.fmt(r.measured(e.id))}</span>{b.has(e.id) && <span className="best">best</span>}
                <div className="muted">Expected: {r.expected(e.id)}</div></td>)}
            </tr>); })}
          <tr><th scope="row">Best enterprise use case</th>
            {ENGINES.map((e) => <td key={e.id}>{ref.best_use_case[e.id]}</td>)}</tr>
        </tbody>
      </table>
      <div className="muted" style={{ marginTop: 6 }}>Entity-relation resolution accuracy: {(res.er_accuracy * 100).toFixed(0)}% · n = {m.standard.all.n} gold questions per engine</div>
    </div>
  );
}

export default function Benchmark({ onChange }: { onChange: () => void }) {
  const [b, setB] = useState<BenchResult | null>(null);
  const [err, setErr] = useState("");
  const [table, setTable] = useState(false);
  const [tip, setTip] = useState<Tip | null>(null);
  const timer = useRef<number>();

  const poll = () => api.bench().then((r) => {
    setB(r);
    if (r.status === "running") timer.current = window.setTimeout(poll, 1500); else onChange();
  }).catch((e) => setErr(e.message));
  useEffect(() => { poll(); return () => clearTimeout(timer.current); }, []);
  const run = async () => { setErr(""); try { await api.runBench(); poll(); } catch (e) { setErr((e as Error).message); } };

  const res = b?.result;
  const maxLat = res ? Math.max(100, ...ENGINES.flatMap((e) => TYPES.map((t) => res.matrix[e.id][t.id].latency_ms))) : 100;
  const niceMax = Math.pow(10, Math.floor(Math.log10(maxLat))) * Math.ceil(maxLat / Math.pow(10, Math.floor(Math.log10(maxLat))));

  return (
    <div className="grid">
      <div className="row">
        <button className="btn" onClick={run} disabled={b?.status === "running"}>{b?.status === "running" ? "Running benchmark…" : "Run live benchmark"}</button>
        {res && <label><input type="checkbox" checked={table} onChange={(e) => setTable(e.target.checked)} /> detailed table (instead of charts)</label>}
        {res && <span className="badge">mode: {res.mode}</span>}
        <span className="muted">Runs all 3 engines over the gold QA set (9 questions · 3 query types).</span>
      </div>
      {err && <div className="err">{err}</div>}
      {b?.status === "error" && <div className="err">Benchmark failed: {b.error}</div>}
      {res?.mode === "offline-heuristic" && <div className="card muted">
        Offline heuristic mode (no ANTHROPIC_API_KEY): answers are extractive and query expansion / HyDE use a small built-in lexicon.
        Faithfulness is a lexical proxy over extractive text, so it is not comparable across engines. Treat RCR and latency as indicative only,
        and rerun with an API key for the real comparison.</div>}
      {!res && b?.status !== "running" && <div className="muted">No results yet — load the sample corpus in Tab 1, then run the benchmark.</div>}

      {res && <>
        <Matrix res={res} reference={b!.reference} />
        <div className="legend" aria-label="legend">{ENGINES.map((e) => <span key={e.id}><i style={{ background: COLOR[e.id] }} />{e.label}</span>)}</div>
        {!table ? <div className="grid c2">
          <GroupedBars title="Retrieval Context Recall (%)" metric="rcr" fmt={(v) => `${Math.round(v * 100)}`} max={1} matrix={res.matrix} setTip={setTip} />
          <GroupedBars title="Faithfulness (0–1)" metric="faithfulness" fmt={(v) => v.toFixed(2)} max={1} matrix={res.matrix} setTip={setTip} />
          <GroupedBars title="Average latency (ms)" metric="latency_ms" fmt={(v) => `${Math.round(v)}`} max={niceMax} matrix={res.matrix} setTip={setTip} />
        </div> : <div className="card"><table>
          <thead><tr><th>Engine</th><th>Query type</th><th>RCR</th><th>Faithfulness</th><th>Hallucination risk (1−F)</th><th>Latency ms</th><th>n</th></tr></thead>
          <tbody>{ENGINES.flatMap((e) => TYPES.map((t) => { const c = res.matrix[e.id][t.id];
            return <tr key={e.id + t.id}><td>{e.label}</td><td>{t.label}</td><td>{(c.rcr * 100).toFixed(0)}%</td><td>{c.faithfulness.toFixed(2)}</td>
              <td>{c.hallucination_risk.toFixed(2)}</td><td>{Math.round(c.latency_ms)}</td><td>{c.n}</td></tr>; }))}</tbody></table></div>}
      </>}

      {tip && <div className="tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </div>
  );
}
