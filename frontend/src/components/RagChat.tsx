import React, { useState } from 'react'

type Msg = { role: 'user' | 'assistant'; content: string; sources?: { label: string; url: string }[] }

export default function RagChat() {
  const [input, setInput] = useState('')
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [loading, setLoading] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [ingestLoading, setIngestLoading] = useState(false)
  const [ingestStatus, setIngestStatus] = useState<string | null>(null)
  const [indexedSources, setIndexedSources] = useState<string[]>([])

  const parseUrls = (raw: string) =>
    raw
      .split(/\s+/)
      .map((u) => u.trim())
      .filter(Boolean)

  const ingestUrls = async () => {
    const urls = parseUrls(urlInput)
    if (!urls.length) {
      setIngestStatus('Add at least one valid URL.')
      return
    }
    setIngestLoading(true)
    setIngestStatus(null)
    try {
      const res = await fetch('http://127.0.0.1:8000/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls }),
      })
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setIngestStatus(`Indexed ${data?.inserted ?? 0} chunk(s).`)
      setIndexedSources((prev) => {
        const next = new Set(prev)
        urls.forEach((u) => next.add(u))
        return Array.from(next)
      })
    } catch (e: any) {
      setIngestStatus(`Error: ${e?.message || e}`)
    } finally {
      setIngestLoading(false)
    }
  }

  const callQuery = async (question: string) => {
    setLoading(true)
    try {
      const res = await fetch('http://127.0.0.1:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: question,
          top_k: 5,
          with_answer: true,
          chat_history: msgs.map((m) => ({ role: m.role, content: m.content })),
          sources_filter: indexedSources.length ? indexedSources : undefined,
        }),
      })
      const data = await res.json()

      const sources =
        (data?.results || []).slice(0, 5).map((r: any, idx: number) => ({
          label: `[${idx + 1}] ${r?.metadata?.source || 'unknown'}`,
          url: r?.metadata?.source || '#',
        })) || []

      setMsgs((prev) => [
        ...prev,
        { role: 'user', content: question },
        { role: 'assistant', content: data?.answer || '(no answer)', sources },
      ])
    } catch (e: any) {
      setMsgs((prev) => [...prev, { role: 'assistant', content: `Error: ${e?.message || e}` }])
    } finally {
      setLoading(false)
    }
  }

  const onSend = () => {
    if (!input.trim()) return
    const q = input.trim()
    setInput('')
    callQuery(q)
  }

  return (
    <div style={{ maxWidth: 800, margin: '40px auto', fontFamily: 'ui-sans-serif, system-ui' }}>
      <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>MyRAG Chat (Ollama)</h2>
      <div style={{ border: '1px solid #e2e2e2', borderRadius: 12, padding: 16, marginBottom: 20 }}>
        <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Ingest URLs</h3>
        <p style={{ marginBottom: 12, color: '#555', fontSize: 14 }}>Paste one or more URLs (space or line separated) to index them.</p>
        <textarea
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          placeholder="https://example.com/docs\nhttps://example.com/blog"
          rows={3}
          style={{ width: '100%', padding: 12, borderRadius: 8, border: '1px solid #ddd', resize: 'vertical' }}
        />
        <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
          <button
            onClick={ingestUrls}
            disabled={ingestLoading}
            style={{ padding: '10px 16px', borderRadius: 8, minWidth: 120 }}
          >
            {ingestLoading ? 'Ingesting…' : 'Ingest URLs'}
          </button>
          {ingestStatus && (
            <span style={{ fontSize: 14, color: ingestStatus.startsWith('Error') ? '#b00020' : '#0a7d24' }}>
              {ingestStatus}
            </span>
          )}
        </div>
        {indexedSources.length > 0 && (
          <div style={{ marginTop: 10, fontSize: 13, color: '#444' }}>
            <div style={{ marginBottom: 4 }}>Active sources:</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {indexedSources.map((s, idx) => (
                <a key={s} href={s} target="_blank" rel="noreferrer">
                  [{idx + 1}] {s}
                </a>
              ))}
            </div>
            <button
              onClick={() => setIndexedSources([])}
              style={{ marginTop: 8, padding: '6px 12px', borderRadius: 6, fontSize: 12 }}
            >
              Clear sources filter
            </button>
          </div>
        )}
      </div>
      <div style={{ border: '1px solid #ddd', borderRadius: 12, padding: 16, minHeight: 400 }}>
        {msgs.length === 0 && <div style={{ color: '#777' }}>Ask me anything from your indexed docs…</div>}
        {msgs.map((m, i) => (
          <div key={i} style={{ marginBottom: 16 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{m.role === 'user' ? 'You' : 'Assistant'}</div>
            <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
            {m.sources && m.sources.length > 0 && (
              <div style={{ marginTop: 8, fontSize: 14, color: '#555' }}>
                Sources:&nbsp;
                {m.sources.map((s, j) => (
                  <a key={j} href={s.url} target="_blank" rel="noreferrer" style={{ marginRight: 8 }}>
                    {s.label}
                  </a>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div>Thinking…</div>}
      </div>

      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question…"
          style={{ flex: 1, padding: 12, borderRadius: 8, border: '1px solid #ddd' }}
          onKeyDown={(e) => e.key === 'Enter' && onSend()}
        />
        <button onClick={onSend} disabled={loading} style={{ padding: '12px 16px', borderRadius: 8 }}>
          Send
        </button>
      </div>
    </div>
  )
}
