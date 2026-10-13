import React, { useState } from 'react';
import { FileText, CheckCircle, XCircle, Search, ExternalLink } from 'lucide-react';

export default function EvidenceChunksView({ chunks = [], supportingChunks = [] }) {
  const [filterQuery, setFilterQuery] = useState('');
  const [expandedId, setExpandedId] = useState(null);

  if (!chunks || chunks.length === 0) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>📄</div>
        <h3 style={{ fontSize: '1.1rem', color: '#cbd5e1', marginBottom: '0.5rem' }}>No Evidence Chunks Retrieved</h3>
        <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
          Execute a query in the Legal Assistant tab to inspect retrieved chunks and similarity rankings.
        </p>
      </div>
    );
  }

  // Supporting chunk IDs or texts
  const supportingIds = new Set(
    supportingChunks.map((c) => c.chunk_id || c.text?.slice(0, 40))
  );

  const filteredChunks = chunks.filter((chunk) => {
    if (!filterQuery.trim()) return true;
    const q = filterQuery.toLowerCase();
    return (
      chunk.text?.toLowerCase().includes(q) ||
      chunk.document_name?.toLowerCase().includes(q) ||
      String(chunk.chunk_id).includes(q)
    );
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
            Retrieved Evidence Chunks ({chunks.length} Total)
          </h3>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Extracted from Bharatiya Nyaya Sanhita knowledge base via dense vector embeddings.
          </p>
        </div>

        <div style={{ position: 'relative', width: '260px' }}>
          <Search size={14} style={{ position: 'absolute', left: '10px', top: '10px', color: '#64748b' }} />
          <input
            type="text"
            placeholder="Search chunks..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            style={{
              width: '100%',
              background: 'rgba(12, 14, 34, 0.8)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '7px 12px 7px 32px',
              color: '#f8fafc',
              fontSize: '0.82rem',
              outline: 'none',
            }}
          />
        </div>
      </div>

      <div className="chunks-grid">
        {filteredChunks.map((chunk, index) => {
          const sim = chunk.similarity_score !== undefined ? (chunk.similarity_score * 100).toFixed(1) : 'N/A';
          const isSupporting = supportingIds.has(chunk.chunk_id || chunk.text?.slice(0, 40));
          const isExpanded = expandedId === index;

          return (
            <div key={index} className="chunk-card">
              <div className="chunk-header">
                <div className="chunk-meta">
                  <span
                    style={{
                      background: 'rgba(99, 102, 241, 0.2)',
                      color: '#a5b4fc',
                      padding: '3px 8px',
                      borderRadius: '4px',
                      fontWeight: '700',
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    #{chunk.chunk_id !== undefined ? chunk.chunk_id : index + 1}
                  </span>
                  <span style={{ fontWeight: '600', color: '#e2e8f0' }}>
                    {chunk.document_name || 'BNS Codex'}
                  </span>
                  {chunk.page_number && (
                    <span style={{ color: '#64748b' }}>· Page {chunk.page_number}</span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      padding: '3px 8px',
                      borderRadius: '4px',
                      fontSize: '0.72rem',
                      fontWeight: '700',
                      background: isSupporting
                        ? 'rgba(16, 185, 129, 0.15)'
                        : 'rgba(244, 63, 94, 0.15)',
                      color: isSupporting ? '#34d399' : '#fb7185',
                      border: `1px solid ${isSupporting ? 'rgba(16,185,129,0.3)' : 'rgba(244,63,94,0.3)'}`,
                    }}
                  >
                    {isSupporting ? <CheckCircle size={11} /> : <XCircle size={11} />}
                    {isSupporting ? 'Verified Evidence' : 'Filtered / Raw'}
                  </span>

                  <span
                    style={{
                      padding: '3px 8px',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      fontWeight: '600',
                      fontFamily: "'JetBrains Mono', monospace",
                      background: 'rgba(6, 182, 212, 0.15)',
                      color: '#22d3ee',
                      border: '1px solid rgba(6, 182, 212, 0.3)',
                    }}
                  >
                    {sim}% Match
                  </span>
                </div>
              </div>

              <div
                className="chunk-text"
                style={{
                  maxHeight: isExpanded ? 'none' : '110px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                {chunk.text}
              </div>

              {chunk.text && chunk.text.length > 200 && (
                <button
                  onClick={() => setExpandedId(isExpanded ? null : index)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#6366f1',
                    fontSize: '0.78rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    marginTop: '8px',
                    padding: '0',
                  }}
                >
                  {isExpanded ? '▲ Show less' : '▼ Read full chunk text'}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
