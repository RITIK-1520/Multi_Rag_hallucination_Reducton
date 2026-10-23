import React from 'react';

export default function AgentVisual({ isProcessing = false }) {
  const agents = [
    { icon: '🧠', label: 'Analyzer', color: '#6366f1' },
    { icon: '🔎', label: 'Retriever', color: '#06b6d4' },
    { icon: '🛡️', label: 'Verifier', color: '#10b981' },
    { icon: '✍️', label: 'Generator', color: '#f59e0b' },
    { icon: '⚖️', label: 'Critic', color: '#f43f5e' },
  ];

  return (
    <div className="agent-visual-card">
      <div className="agent-visual-hud">
        <span className="hud-pill">✦ 5-AGENT PIPELINE · BNS CODEX</span>
        <span className="hud-hint">{isProcessing ? '⚡ Processing...' : '● Ready'}</span>
      </div>
      <div className={`agent-flow-row ${isProcessing ? 'processing' : ''}`}>
        {agents.map((a, i) => (
          <React.Fragment key={i}>
            <div className="agent-dot" style={{ '--agent-color': a.color, animationDelay: `${i * 0.15}s` }}>
              <span className="agent-dot-icon">{a.icon}</span>
              <span className="agent-dot-label">{a.label}</span>
              <div className="agent-dot-ring" />
            </div>
            {i < agents.length - 1 && <div className="agent-connector"><div className="connector-pulse" /></div>}
          </React.Fragment>
        ))}
      </div>
      <div className="agent-visual-stats">
        <span>📚 Knowledge Chunks</span>
        <span>🤖 5 AI Agents</span>
        <span>🛡️ Evidence-first checks</span>
      </div>
    </div>
  );
}
