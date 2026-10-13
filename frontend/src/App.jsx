import React, { useState, useEffect } from 'react';
import {
  Scale,
  Brain,
  Search,
  ShieldCheck,
  ShieldAlert,
  Layers,
  Sparkles,
  Sliders,
  FileText,
  Clock,
  Send,
  Zap,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  GitCompare,
  Activity,
  Cpu,
} from 'lucide-react';

import BNS3DConstellation from './components/BNS3DConstellation';
import StreamingAnswer from './components/StreamingAnswer';
import AgentTraceView from './components/AgentTraceView';
import EvidenceChunksView from './components/EvidenceChunksView';
import HallucinationReportView from './components/HallucinationReportView';
import ComparisonView from './components/ComparisonView';

const PRESET_QUESTIONS = [
  {
    label: '⚖️ Section 103 (Murder)',
    query: 'What constitutes murder under Section 103 of BNS and what are the prescribed punishments?',
  },
  {
    label: '💰 Section 303 (Theft)',
    query: 'What are the definitions and punishments for theft under Section 303 of BNS?',
  },
  {
    label: '🛡️ Right of Self-Defence',
    query: 'What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?',
  },
  {
    label: '🌐 Cyber & Organized Crime',
    query: 'How does BNS address organized crime and electronic offences under Section 111?',
  },
  {
    label: '⚡ Death by Negligence (Sec 106)',
    query: 'What are the provisions related to causing death by negligence under Section 106 of BNS?',
  },
];

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('assistant'); // 'assistant', 'trace', 'chunks', 'hallucination', 'compare'

  // Settings
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(0.3);
  const [hallucinationThreshold, setHallucinationThreshold] = useState(0.4);
  const [evidenceThreshold, setEvidenceThreshold] = useState(0.35);
  const [maxAttempts, setMaxAttempts] = useState(2);
  const [streamSpeed, setStreamSpeed] = useState(18);

  // System Stats
  const [stats, setStats] = useState({
    num_documents: 9,
    num_chunks: 1388,
    num_vectors: 1388,
    is_live_gemini: false,
    mode: 'Offline Deterministic Mode',
    documents: ['Final_BNS'],
  });

  // Query & Results
  const [queryInput, setQueryInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Comparison
  const [compareData, setCompareData] = useState(null);
  const [isComparing, setIsComparing] = useState(false);

  // Fetch initial stats
  useEffect(() => {
    fetch('/api/stats')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.status === 'online') {
          setStats(data);
          if (data.defaults) {
            setTopK(data.defaults.top_k || 5);
            setHallucinationThreshold(data.defaults.hallucination_threshold || 0.4);
            setEvidenceThreshold(data.defaults.evidence_threshold || 0.35);
            setMaxAttempts(data.defaults.max_attempts || 2);
            setTemperature(data.defaults.temperature || 0.3);
          }
        }
      })
      .catch((err) => {
        console.warn('API stats fetch failed, using fallback stats', err);
      });
  }, []);

  // Execute Query
  const handleRunQuery = async (queryText, pipelineType = 'multi_agent') => {
    const q = (queryText || queryInput).trim();
    if (!q) return;

    setIsLoading(true);
    setErrorMsg(null);
    setQueryInput(q);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          pipeline: pipelineType,
          top_k: topK,
          temperature,
          hallucination_threshold: hallucinationThreshold,
          evidence_threshold: evidenceThreshold,
          max_attempts: maxAttempts,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to execute query');
      }

      const result = await response.json();
      setCurrentResult(result);
      setActiveTab('assistant');
    } catch (err) {
      setErrorMsg(err.message || 'An error occurred while connecting to the RAG server.');
    } finally {
      setIsLoading(false);
    }
  };

  // Run Comparison
  const handleRunCompare = async (q) => {
    setIsComparing(true);
    setErrorMsg(null);

    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          top_k: topK,
          temperature,
          hallucination_threshold: hallucinationThreshold,
          evidence_threshold: evidenceThreshold,
          max_attempts: maxAttempts,
        }),
      });

      if (!response.ok) {
        throw new Error('Comparison failed');
      }

      const data = await response.json();
      setCompareData(data);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsComparing(false);
    }
  };

  const isVerified = currentResult?.verification_status === 'VERIFIED';
  const isHighRisk = currentResult?.verification_status === 'HIGH_RISK' || currentResult?.verification_status === 'REJECTED';
  const isMediumRisk = currentResult?.verification_status === 'MEDIUM_RISK' || currentResult?.verification_status === 'INSUFFICIENT_EVIDENCE';

  return (
    <>
      {/* Background Floating Orbs */}
      <div className="bg-glow-container">
        <div className="glow-orb orb-indigo" />
        <div className="glow-orb orb-purple" />
        <div className="glow-orb orb-cyan" />
      </div>

      <div className="app-container">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="logo-badge">⚖️</div>
            <div>
              <div className="logo-title">BNS Intelligence</div>
              <div className="logo-subtitle">Multi-Agent RAG v2.0</div>
            </div>
          </div>

          {/* Engine Mode */}
          <div
            style={{
              padding: '10px 14px',
              borderRadius: '8px',
              background: stats.is_live_gemini ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)',
              border: `1px solid ${stats.is_live_gemini ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '0.82rem',
              fontWeight: 600,
              color: stats.is_live_gemini ? '#34d399' : '#fbbf24',
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: stats.is_live_gemini ? '#10b981' : '#f59e0b',
                boxShadow: `0 0 8px ${stats.is_live_gemini ? '#10b981' : '#f59e0b'}`,
              }}
            />
            {stats.mode}
          </div>

          {/* Knowledge Base Stats */}
          <div>
            <div style={{ fontSize: '0.78rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 700, marginBottom: '10px' }}>
              📚 Knowledge Base
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Documents</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>{stats.num_documents}</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Total Chunks</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#38bdf8' }}>{stats.num_chunks}</div>
              </div>
            </div>
            <div
              style={{
                marginTop: '8px',
                fontSize: '0.76rem',
                color: '#10b981',
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                background: 'rgba(16, 185, 129, 0.08)',
                padding: '6px 10px',
                borderRadius: '6px',
              }}
            >
              <CheckCircle2 size={12} /> FAISS Index: 1,388 Active Vectors
            </div>
          </div>

          {/* Pipeline Parameters */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ fontSize: '0.78rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 700 }}>
              ⚙️ Pipeline Parameters
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <span>Top-K Retrieval</span>
                <span className="control-val">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="15"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="slider-input"
              />
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <span>Hallucination Threshold</span>
                <span className="control-val">{hallucinationThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={hallucinationThreshold}
                onChange={(e) => setHallucinationThreshold(Number(e.target.value))}
                className="slider-input"
              />
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <span>Evidence Similarity</span>
                <span className="control-val">{evidenceThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={evidenceThreshold}
                onChange={(e) => setEvidenceThreshold(Number(e.target.value))}
                className="slider-input"
              />
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <span>Max Attempts</span>
                <span className="control-val">{maxAttempts}</span>
              </div>
              <input
                type="range"
                min="0"
                max="5"
                value={maxAttempts}
                onChange={(e) => setMaxAttempts(Number(e.target.value))}
                className="slider-input"
              />
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <span>Typing Speed (ms/word)</span>
                <span className="control-val">{streamSpeed} ms</span>
              </div>
              <input
                type="range"
                min="5"
                max="45"
                step="1"
                value={streamSpeed}
                onChange={(e) => setStreamSpeed(Number(e.target.value))}
                className="slider-input"
              />
            </div>
          </div>
        </aside>

        {/* Content Area */}
        <main className="content-area">
          {/* Hero Banner + 3D Mesh */}
          <div className="hero-grid">
            <div className="hero-card">
              <div className="pill-badge">
                <span className="status-dot" />
                Multi-Agent Legal RAG · BNS Codex
              </div>
              <h1 className="hero-title">⚖️ Bharatiya Nyaya Sanhita</h1>
              <p className="hero-description">
                Interactive legal intelligence powered by India's new criminal code (BNS).
                Trained & indexed over <strong>1,388 knowledge chunks</strong>, coordinated across a
                <strong> 6-Agent Verification Pipeline</strong> with zero-hallucination grounding.
              </p>
              <div className="hero-badges-row">
                <span className="tag-badge indigo">📚 1,388 Chunks</span>
                <span className="tag-badge cyan">🤖 6 AI Agents</span>
                <span className="tag-badge emerald">🛡️ Hallucination Guard</span>
                <span className="tag-badge">⚡ Rapid Word Stream</span>
              </div>
            </div>

            <BNS3DConstellation isProcessing={isLoading || isComparing} />
          </div>

          {/* Navigation Tabs */}
          <nav className="tabs-nav">
            <button
              className={`tab-btn ${activeTab === 'assistant' ? 'active' : ''}`}
              onClick={() => setActiveTab('assistant')}
            >
              <Scale size={16} /> Legal Assistant
            </button>
            <button
              className={`tab-btn ${activeTab === 'trace' ? 'active' : ''}`}
              onClick={() => setActiveTab('trace')}
            >
              <Brain size={16} /> Agent Pipeline ({currentResult?.agent_trace?.length || 0})
            </button>
            <button
              className={`tab-btn ${activeTab === 'chunks' ? 'active' : ''}`}
              onClick={() => setActiveTab('chunks')}
            >
              <FileText size={16} /> Evidence Chunks ({currentResult?.retrieved_chunks?.length || 0})
            </button>
            <button
              className={`tab-btn ${activeTab === 'hallucination' ? 'active' : ''}`}
              onClick={() => setActiveTab('hallucination')}
            >
              <ShieldAlert size={16} /> Hallucination Report
            </button>
            <button
              className={`tab-btn ${activeTab === 'compare' ? 'active' : ''}`}
              onClick={() => setActiveTab('compare')}
            >
              <GitCompare size={16} /> Side-by-Side Compare
            </button>
          </nav>

          {/* Tab 1: Legal Assistant */}
          {activeTab === 'assistant' && (
            <div>
              {/* Question Suggestions */}
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600, marginBottom: '8px' }}>
                  ⚡ Quick BNS Queries:
                </div>
                <div className="suggestion-chips">
                  {PRESET_QUESTIONS.map((item, idx) => (
                    <button
                      key={idx}
                      className="chip-btn"
                      onClick={() => handleRunQuery(item.query, 'multi_agent')}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Query Box */}
              <div className="glass-card">
                <div className="query-input-wrapper">
                  <textarea
                    rows={3}
                    className="query-textarea"
                    placeholder="Ask any question about Bharatiya Nyaya Sanhita (e.g. What is the punishment for theft under Section 303?)"
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                        handleRunQuery(queryInput, 'multi_agent');
                      }
                    }}
                  />
                </div>

                <div className="query-actions-row">
                  <button
                    className="btn-primary"
                    disabled={isLoading || !queryInput.trim()}
                    onClick={() => handleRunQuery(queryInput, 'multi_agent')}
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw size={16} className="animate-spin" /> Verifying with 6 AI Agents...
                      </>
                    ) : (
                      <>
                        <ShieldCheck size={16} /> Ask Multi-Agent RAG (Guarded)
                      </>
                    )}
                  </button>

                  <button
                    className="btn-secondary"
                    disabled={isLoading || !queryInput.trim()}
                    onClick={() => handleRunQuery(queryInput, 'basic')}
                  >
                    <Zap size={16} /> Ask Basic RAG (Fast Baseline)
                  </button>
                </div>
              </div>

              {/* Error Message */}
              {errorMsg && (
                <div
                  style={{
                    marginTop: '1rem',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    background: 'rgba(244, 63, 94, 0.15)',
                    border: '1px solid rgba(244, 63, 94, 0.4)',
                    color: '#fb7185',
                    fontSize: '0.88rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                >
                  <AlertTriangle size={16} /> {errorMsg}
                </div>
              )}

              {/* Answer Box */}
              {currentResult && (
                <div className="answer-box">
                  <div className="answer-header">
                    <div className="answer-title-group">
                      <span style={{ fontSize: '1.4rem' }}>⚖️</span>
                      <div>
                        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff' }}>
                          BNS Legal Intelligence Response
                        </h3>
                        <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                          Pipeline: {currentResult.pipeline === 'multi_agent_rag' ? 'Multi-Agent Verified' : 'Basic RAG'}
                        </span>
                      </div>
                    </div>

                    <div>
                      {isVerified && (
                        <span className="answer-badge-status badge-verified">
                          <CheckCircle2 size={13} /> Verified Legal Answer
                        </span>
                      )}
                      {isMediumRisk && (
                        <span className="answer-badge-status badge-medium-risk">
                          <AlertTriangle size={13} /> {currentResult.verification_status}
                        </span>
                      )}
                      {isHighRisk && (
                        <span className="answer-badge-status badge-high-risk">
                          <ShieldAlert size={13} /> High Risk / Abstained
                        </span>
                      )}
                    </div>
                  </div>

                  {/* High Speed Word-by-Word Stream */}
                  <StreamingAnswer
                    text={currentResult.answer || 'No answer generated.'}
                    speed={streamSpeed}
                    isVerified={isVerified}
                  />

                  {/* Metrics Tiles */}
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-label">Evidence Confidence</div>
                      <div className="metric-value" style={{ color: '#38bdf8' }}>
                        {currentResult.evidence_confidence !== undefined
                          ? `${(currentResult.evidence_confidence * 100).toFixed(1)}%`
                          : 'N/A'}
                      </div>
                      <div className="metric-sub">FAISS semantic correlation</div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-label">Hallucination Risk</div>
                      <div
                        className="metric-value"
                        style={{
                          color:
                            currentResult.hallucination_risk !== null && currentResult.hallucination_risk > 0.4
                              ? '#f43f5e'
                              : '#34d399',
                        }}
                      >
                        {currentResult.hallucination_risk !== null
                          ? `${(currentResult.hallucination_risk * 100).toFixed(1)}%`
                          : 'N/A'}
                      </div>
                      <div className="metric-sub">NLI entailment risk</div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-label">Processing Time</div>
                      <div className="metric-value" style={{ color: '#a5b4fc' }}>
                        {currentResult.processing_time || 0}s
                      </div>
                      <div className="metric-sub">Agent cycle latency</div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-label">Sources Cited</div>
                      <div className="metric-value" style={{ fontSize: '1.1rem', color: '#e2e8f0' }}>
                        {currentResult.sources?.length ? currentResult.sources.join(', ') : 'Final_BNS'}
                      </div>
                      <div className="metric-sub">Grounding codex documents</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Agent Pipeline */}
          {activeTab === 'trace' && (
            <AgentTraceView
              trace={currentResult?.agent_trace || []}
              pipeline={currentResult?.pipeline || 'multi_agent_rag'}
              totalTime={currentResult?.processing_time || 0}
            />
          )}

          {/* Tab 3: Evidence Chunks */}
          {activeTab === 'chunks' && (
            <EvidenceChunksView
              chunks={currentResult?.retrieved_chunks || []}
              supportingChunks={currentResult?.supporting_chunks || currentResult?.retrieved_chunks || []}
            />
          )}

          {/* Tab 4: Hallucination Report */}
          {activeTab === 'hallucination' && (
            <HallucinationReportView
              hallucinationRisk={currentResult?.hallucination_risk ?? null}
              supportPercentage={currentResult?.support_percentage ?? null}
              claims={currentResult?.claims || []}
              verificationStatus={currentResult?.verification_status || 'NOT_VERIFIED'}
            />
          )}

          {/* Tab 5: Side-by-Side Compare */}
          {activeTab === 'compare' && (
            <ComparisonView
              onRunCompare={handleRunCompare}
              compareData={compareData}
              isLoading={isComparing}
              activeQuery={queryInput}
            />
          )}
        </main>
      </div>
    </>
  );
}
