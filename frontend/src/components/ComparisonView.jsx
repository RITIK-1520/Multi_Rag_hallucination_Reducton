import React, { useState } from 'react';
import { GitCompare, ShieldCheck, Zap, AlertTriangle, Clock, ArrowRight } from 'lucide-react';
import StreamingAnswer from './StreamingAnswer';

export default function ComparisonView({
  onRunCompare,
  compareData = null,
  isLoading = false,
  activeQuery = '',
}) {
  const [queryInput, setQueryInput] = useState(activeQuery);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!queryInput.trim() || isLoading) return;
    onRunCompare(queryInput.trim());
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="glass-card">
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem' }}>
          Side-by-Side Comparison: Multi-Agent RAG vs. Basic RAG
        </h3>
        <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '1rem' }}>
          Evaluate how Multi-Agent RAG eliminates hallucinations, cross-verifies legal citations, and outputs structured, reliable bullet points compared to raw unverified retrieval.
        </p>

        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
          <input
            type="text"
            placeholder="Enter question to compare both pipelines (e.g., What are the penalties under Section 303?)"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            style={{
              flex: 1,
              background: 'rgba(10, 12, 28, 0.85)',
              border: '1px solid var(--border-medium)',
              borderRadius: '8px',
              padding: '10px 14px',
              color: '#f8fafc',
              fontSize: '0.9rem',
              outline: 'none',
            }}
          />
          <button
            type="submit"
            disabled={isLoading || !queryInput.trim()}
            className="btn-primary"
            style={{ flex: '0 0 auto', padding: '10px 20px' }}
          >
            {isLoading ? 'Comparing...' : 'Run Benchmark'}
          </button>
        </form>
      </div>

      {compareData && (
        <div className="comparison-grid">
          {/* Multi-Agent RAG Column */}
          <div className="comparison-column highlight">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '1.4rem' }}>🛡️</span>
                <div>
                  <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#34d399' }}>
                    Multi-Agent RAG (Proposed)
                  </h4>
                  <span style={{ fontSize: '0.72rem', color: '#a5b4fc', textTransform: 'uppercase' }}>
                    6-Agent Guard · Verified
                  </span>
                </div>
              </div>
              <span className="tag-badge emerald">
                {compareData.multi_agent?.verification_status || 'VERIFIED'}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '1rem' }}>
              <div style={{ background: 'rgba(6,8,20,0.5)', padding: '8px 10px', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Evidence Confidence</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8' }}>
                  {compareData.multi_agent?.evidence_confidence !== undefined
                    ? `${(compareData.multi_agent.evidence_confidence * 100).toFixed(1)}%`
                    : 'N/A'}
                </div>
              </div>
              <div style={{ background: 'rgba(6,8,20,0.5)', padding: '8px 10px', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Hallucination Risk</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#34d399' }}>
                  {compareData.multi_agent?.hallucination_risk !== null
                    ? `${(compareData.multi_agent.hallucination_risk * 100).toFixed(1)}%`
                    : '0.0%'}
                </div>
              </div>
            </div>

            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '6px', fontWeight: 600 }}>
              Generated Legal Answer:
            </div>
            <div style={{ background: 'rgba(6,8,20,0.6)', padding: '14px', borderRadius: '8px', minHeight: '180px' }}>
              <StreamingAnswer
                text={compareData.multi_agent?.answer || 'No response.'}
                speed={15}
                isVerified={true}
              />
            </div>

            <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748b' }}>
              <span>Time: {compareData.multi_agent?.processing_time}s</span>
              <span>Sources: {compareData.multi_agent?.sources?.join(', ') || 'BNS'}</span>
            </div>
          </div>

          {/* Basic RAG Column */}
          <div className="comparison-column">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '1.4rem' }}>⚡</span>
                <div>
                  <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#94a3b8' }}>
                    Basic RAG (Baseline)
                  </h4>
                  <span style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase' }}>
                    Retrieve & Generate · No Guard
                  </span>
                </div>
              </div>
              <span className="tag-badge">NOT VERIFIED</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '1rem' }}>
              <div style={{ background: 'rgba(6,8,20,0.5)', padding: '8px 10px', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Retrieval Sim</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#cbd5e1' }}>
                  {compareData.basic?.retrieval_confidence !== undefined
                    ? `${(compareData.basic.retrieval_confidence * 100).toFixed(1)}%`
                    : 'N/A'}
                </div>
              </div>
              <div style={{ background: 'rgba(6,8,20,0.5)', padding: '8px 10px', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Hallucination Guard</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f43f5e' }}>
                  Disabled
                </div>
              </div>
            </div>

            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '6px', fontWeight: 600 }}>
              Generated Legal Answer:
            </div>
            <div style={{ background: 'rgba(6,8,20,0.6)', padding: '14px', borderRadius: '8px', minHeight: '180px' }}>
              <StreamingAnswer
                text={compareData.basic?.answer || 'No response.'}
                speed={15}
                isVerified={false}
              />
            </div>

            <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748b' }}>
              <span>Time: {compareData.basic?.processing_time}s</span>
              <span>Sources: {compareData.basic?.sources?.join(', ') || 'BNS'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
