import React from 'react';

const agents = [
  { name: 'Query Analyzer', icon: '⌕' },
  { name: 'Retriever', icon: '⌘' },
  { name: 'Evidence Verifier', icon: '✓' },
  { name: 'Answer Generator', icon: '✧' },
  { name: 'Hallucination Critic', icon: '◎' },
];

export default function AgentVisual({ isProcessing = false }) {
  return (
    <section className={`agent-visual ${isProcessing ? 'is-processing' : ''}`} aria-label="Multi-agent pipeline">
      {agents.map((agent, index) => (
        <React.Fragment key={agent.name}>
          <div className="agent-visual-node">
            <span className="agent-visual-icon" aria-hidden="true">{agent.icon}</span>
            <span>{agent.name}</span>
          </div>
          {index < agents.length - 1 && <span className="agent-visual-link" aria-hidden="true">→</span>}
        </React.Fragment>
      ))}
    </section>
  );
}
