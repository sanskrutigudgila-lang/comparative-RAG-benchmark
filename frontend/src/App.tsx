import { useEffect, useState } from "react";
import { api } from "./api";
import Chat from "./tabs/Chat";
import Graph3D from "./tabs/Graph3D";
import Workflow from "./tabs/Workflow";
import Telemetry from "./tabs/Telemetry";
import Benchmark from "./tabs/Benchmark";
import ErrorBoundary from "./ErrorBoundary";

const TABS = [
  { id: "chat", label: "1 · Ingestion & Chat", C: Chat },
  { id: "graph", label: "2 · Knowledge Graph (3D)", C: Graph3D },
  { id: "flow", label: "3 · LangGraph Workflow", C: Workflow },
  { id: "kpi", label: "4 · Hyperparameters & KPIs", C: Telemetry },
  { id: "bench", label: "5 · Comparative Benchmark", C: Benchmark },
] as const;

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>(() => (TABS.find((t) => t.id === location.hash.slice(1))?.id ?? "chat"));
  const [health, setHealth] = useState<{ mode: string; chunks: number; embed_model: string } | null>(null);
  const [down, setDown] = useState(false);
  const refresh = () => api.health().then((h) => { setHealth(h); setDown(false); }).catch(() => setDown(true));
  useEffect(() => { refresh(); }, [tab]);
  const Active = TABS.find((t) => t.id === tab)!.C;
  return (
    <>
      <header>
        <h1>Comparative RAG Benchmark — Standard vs HyDE vs Graph RAG</h1>
        <span className="sub">PwC India Advisory Practice · AI &amp; Advanced Analytics</span>
        {health && <><span className="badge">mode: {health.mode}</span><span className="badge">{health.chunks} chunks</span>
          <span className="badge">embeddings: {health.embed_model}</span></>}
        {down && <span className="badge err">backend unreachable (start the API; see README)</span>}
      </header>
      <nav role="tablist">
        {TABS.map((t) => <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>{t.label}</button>)}
      </nav>
      <main><ErrorBoundary key={tab}><Active onChange={refresh} /></ErrorBoundary></main>
    </>
  );
}
