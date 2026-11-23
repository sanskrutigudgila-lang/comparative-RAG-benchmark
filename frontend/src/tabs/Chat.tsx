import { useRef, useState } from "react";
import { api, AREAS, ENGINES, Area, ChatResult, Engine } from "../api";

interface Turn { q: string; r?: ChatResult; err?: string }

export default function Chat({ onChange }: { onChange: () => void }) {
  const [engine, setEngine] = useState<Engine>("graph");
  const [area, setArea] = useState<Area>("tax");
  const [q, setQ] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [over, setOver] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setStatus(`Uploading ${files.length} file(s)…`);
    try { const r = await api.ingest(files); setStatus(`Indexed ${r.new_chunks} new chunks (${r.total_chunks} total)`); onChange(); }
    catch (e) { setStatus(`Error: ${(e as Error).message}`); }
  };
  const sample = async () => {
    setStatus("Indexing sample corpus (first run downloads the embedding model)…");
    try { const r = await api.ingestSample(); setStatus(`Sample corpus ready: ${r.total_chunks} chunks`); onChange(); }
    catch (e) { setStatus(`Error: ${(e as Error).message}`); }
  };
  const ask = async () => {
    if (!q.trim() || busy) return;
    const query = q; setQ(""); setBusy(true);
    try { const r = await api.chat({ query, engine, practice_area: area }); setTurns((t) => [...t, { q: query, r }]); }
    catch (e) { setTurns((t) => [...t, { q: query, err: (e as Error).message }]); }
    setBusy(false);
  };

  return (
    <div className="grid c2" style={{ gridTemplateColumns: "minmax(260px,320px) 1fr" }}>
      <div className="grid" style={{ alignContent: "start" }}>
        <div className={`drop ${over ? "over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); upload(Array.from(e.dataTransfer.files)); }}>
          Drag &amp; drop PDF / DOCX / TXT / JSON / CSV here<br />
          <button className="btn secondary" style={{ marginTop: 8 }} onClick={() => fileRef.current?.click()}>Browse…</button>
          <input ref={fileRef} type="file" multiple hidden accept=".pdf,.docx,.txt,.md,.json,.csv"
            onChange={(e) => upload(Array.from(e.target.files ?? []))} />
        </div>
        <button className="btn" onClick={sample}>Load sample corpus (synthetic)</button>
        <div className="muted" role="status">{status}</div>
        <label>Practice area<br /><select value={area} onChange={(e) => setArea(e.target.value as Area)} style={{ width: "100%" }}>
          {AREAS.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}</select></label>
        <label>RAG engine<br /><select value={engine} onChange={(e) => setEngine(e.target.value as Engine)} style={{ width: "100%" }}>
          {ENGINES.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}</select></label>
        <div className="muted">Guardrails for the selected practice area are injected into the CO-STAR system prompt.</div>
      </div>
      <div>
        <div style={{ minHeight: 360 }}>
          {!turns.length && <div className="muted">Load the sample corpus (or upload documents), then ask a question.</div>}
          {turns.map((t, i) => (
            <div key={i}>
              <div className="msg user"><b>You:</b> {t.q}</div>
              {t.err && <div className="msg err">{t.err}</div>}
              {t.r && <div className="msg">
                <div className="muted">{t.r.engine} · {t.r.mode} · {Math.round(t.r.timings_ms.total)} ms · path: {t.r.trace.join(" → ")}</div>
                <div style={{ margin: "8px 0" }}>{t.r.answer}</div>
                <div>{t.r.citations.map((c) => (
                  <span key={c.chunk_id} className="cite" onClick={() => setOpen(open === c.chunk_id ? null : c.chunk_id)}>
                    {c.source} · {c.section}</span>))}</div>
                {t.r.citations.filter((c) => c.chunk_id === open).map((c) => <pre key={c.chunk_id} className="muted" style={{ whiteSpace: "pre-wrap" }}>{c.text}</pre>)}
              </div>}
            </div>))}
        </div>
        <div className="row">
          <textarea rows={2} style={{ flex: 1 }} value={q} placeholder="Ask a compliance question…" maxLength={2000}
            onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } }} />
          <button className="btn" disabled={busy || !q.trim()} onClick={ask}>{busy ? "Thinking…" : "Ask"}</button>
        </div>
      </div>
    </div>
  );
}
