import { useEffect, useState } from "react";
import { api, ENGINES } from "../api";

const HP: { key: string; label: string; hint: string }[] = [
  { key: "chunk_size", label: "Chunk size (tokens)", hint: "Spec 600 · applies to newly ingested docs" },
  { key: "chunk_overlap", label: "Chunk overlap (tokens)", hint: "Spec 60" },
  { key: "hyde_passages", label: "Multi-HyDE passages N", hint: "Spec 3" },
  { key: "hop_radius", label: "Graph hop radius k", hint: "Spec 2" },
  { key: "rrf_alpha", label: "RRF constant α", hint: "Spec 60" },
  { key: "top_k", label: "Context chunks (top-k)", hint: "Chunks passed to synthesis" },
];

export default function Telemetry({ onChange }: { onChange: () => void }) {
  const [hp, setHp] = useState<Record<string, number | boolean>>({});
  const [t, setT] = useState<Awaited<ReturnType<typeof api.telemetry>> | null>(null);
  const [err, setErr] = useState("");
  const load = () => { api.hp().then(setHp).catch((e) => setErr(e.message)); api.telemetry().then(setT).catch((e) => setErr(e.message)); };
  useEffect(load, []);
  const set = async (k: string, v: number) => {
    try { setHp(await api.putHp({ [k]: v })); setErr(""); onChange(); } catch (e) { setErr((e as Error).message); }
  };
  const pct = (x?: number) => (x == null ? "—" : `${Math.round(x * 100)}%`);
  const sum = t?.benchmark_summary;

  return (
    <div className="grid">
      {err && <div className="err">{err}</div>}
      <h3 style={{ margin: 0 }}>Hyperparameters</h3>
      {hp.reindex_required && <div className="muted">Chunking changed — re-ingest documents for it to take effect.</div>}
      <div className="grid c4">
        {HP.map((h) => (
          <label key={h.key} className="card">{h.label}<br />
            <input type="number" style={{ width: 90, margin: "6px 0" }} value={Number(hp[h.key] ?? 0)}
              onChange={(e) => e.target.value !== "" && set(h.key, Number(e.target.value))} /><br />
            <span className="muted">{h.hint}</span></label>))}
      </div>
      <h3 style={{ margin: 0 }}>KPIs {t && <span className="badge">{t.mode}</span>}</h3>
      {!sum && <div className="muted">Run the benchmark in Tab 5 to populate RCR and faithfulness.</div>}
      <div className="grid c4">
        <div className="card kpi"><div className="l">Entity-relation resolution accuracy</div><div className="v">{pct(t?.er_accuracy ?? undefined)}</div>
          <div className="muted">{t ? `${t.graph.nodes} nodes · ${t.graph.edges} edges` : ""}</div></div>
        {ENGINES.map((e) => (
          <div key={e.id} className="card kpi"><div className="l">{e.label} · retrieval context recall</div><div className="v">{pct(sum?.[e.id]?.rcr)}</div>
            <div className="muted">faithfulness {sum ? sum[e.id].faithfulness.toFixed(2) : "—"} · avg latency {sum ? Math.round(sum[e.id].latency_ms) + " ms" : "—"}</div></div>))}
      </div>
      <h3 style={{ margin: 0 }}>End-to-end latency breakdown (last query per engine, ms)</h3>
      <div className="card"><table>
        <thead><tr><th>Engine</th><th>Retrieval</th><th>Graph traversal</th><th>LLM / query processing</th><th>Other</th><th>Total</th></tr></thead>
        <tbody>{ENGINES.map((e) => { const l = t?.last_queries[e.id]?.timings_ms; const f = (k: string) => (l ? Math.round(l[k] ?? 0) : "—");
          return <tr key={e.id}><td>{e.label}</td><td>{f("retrieval")}</td><td>{f("graph")}</td><td>{f("llm")}</td><td>{f("other")}</td><td><b>{f("total")}</b></td></tr>; })}</tbody>
      </table><div className="muted">Run a query on each engine in Tab 1 or Tab 3 to fill this table.</div></div>
    </div>
  );
}
