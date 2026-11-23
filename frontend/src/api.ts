export type Engine = "standard" | "hyde" | "graph";
export type Area = "tax" | "grc" | "deals";
export const ENGINES: { id: Engine; label: string }[] = [
  { id: "standard", label: "Standard RAG" }, { id: "hyde", label: "HyDE RAG" }, { id: "graph", label: "Graph RAG" },
];
export const AREAS: { id: Area; label: string }[] = [
  { id: "tax", label: "Tax & Regulatory Services" }, { id: "grc", label: "Risk Advisory & Governance (GRC)" },
  { id: "deals", label: "Deal Advisory & Corporate Finance" },
];

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, init);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* non-JSON error body */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}
const json = (method: string, body: unknown): RequestInit => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export interface Citation { chunk_id: string; source: string; section: string; text: string }
export interface ChatResult {
  engine: Engine; answer: string; citations: Citation[]; timings_ms: Record<string, number>; trace: string[]; mode: string;
}
export interface GraphData {
  nodes: { id: string; kind: string; degree: number; chunks: number }[];
  edges: { id: string; source: string; target: string; relation: string }[];
  selection: { nodes: string[]; edges: string[] };
}
export interface Cell { rcr: number; faithfulness: number; latency_ms: number; hallucination_risk: number; n: number }
export interface BenchResult {
  status: "idle" | "running" | "done" | "error"; error?: string;
  result?: { mode: string; er_accuracy: number; matrix: Record<Engine, Record<string, Cell>> };
  reference: { latency_ms: Record<Engine, number>; hallucination_risk: Record<Engine, string>; best_use_case: Record<Engine, string> };
}

export const api = {
  health: () => req<{ ok: boolean; mode: string; embed_model: string; chunks: number }>("/health"),
  ingestSample: () => req<{ new_chunks: number; total_chunks: number }>("/ingest/sample", { method: "POST" }),
  ingest: (files: File[]) => {
    const fd = new FormData(); files.forEach((f) => fd.append("files", f));
    return req<{ files: string[]; new_chunks: number; total_chunks: number }>("/ingest", { method: "POST", body: fd });
  },
  chat: (b: { query: string; engine: Engine; practice_area: Area; disabled_nodes?: string[]; disabled_edges?: string[]; use_graph_selection?: boolean }) =>
    req<ChatResult>("/chat", json("POST", b)),
  graph: () => req<GraphData>("/graph"),
  setFilter: (nodes: string[], edges: string[]) => req("/graph/filter", json("POST", { nodes, edges })),
  workflow: (e: Engine) => req<{ nodes: { id: string; label: string }[]; edges: { id: string; source: string; target: string; cyclic?: boolean }[] }>(`/workflow/${e}`),
  hp: () => req<Record<string, number | boolean>>("/hyperparameters"),
  putHp: (b: Record<string, number>) => req<Record<string, number | boolean>>("/hyperparameters", json("PUT", b)),
  telemetry: () => req<{ last_queries: Record<string, { timings_ms: Record<string, number>; query: string }>; er_accuracy: number | null;
    benchmark_summary: Record<Engine, Cell> | null; graph: { nodes: number; edges: number }; mode: string }>("/telemetry"),
  runBench: () => req("/benchmark/run", { method: "POST" }),
  bench: () => req<BenchResult>("/benchmark/results"),
};
