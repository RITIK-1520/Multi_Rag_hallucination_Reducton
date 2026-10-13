import React from 'react';
import { Search, ShieldAlert, Cpu, CheckCircle, FileText, ArrowRight, Clock, AlertTriangle } from 'lucide-react';

const AGENT_ICONS = {
  'Query Analyzer': '🧠',
  'Retriever': '🔎',
  'Evidence Verifier': '🛡️',
  'Answer Generator': '✍️',
  'Hallucination Critic': '⚖️',
  'Final Decision': '🎯',
};

export default function AgentTraceView({ trace = [], pipeline = 'multi_agent_rag', totalTime = 0 }) {
  if (!trace || trace.length === 0) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🔗</div>
        <h3 style={{ fontSize: '1.1rem', color: '#cbd5e1', marginBottom: '0.5rem' }}>No Agent Trace Yet</h3>
        <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
          Ask a question in the Legal Assistant tab to see the multi-agent pipeline execute in real time.
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
            Pipeline Execution Flow ({pipeline === 'multi_agent_rag' ? 'Multi-Agent Verified' : 'Basic RAG Baseline'})
          </h3>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Chronological trace of agent actions, verification thresholds, and reasoning.
          </p>
        </div>
        <div className="tag-badge cyan">
          <Clock size={12} /> Total: {totalTime}s
        </div>
      </div>

      <div className="pipeline-flow">
        {trace.map((step, idx) => {
          const icon = AGENT_ICONS[step.agent] || '🤖';
          const isError = step.status === 'error';
          const isAbstained = step.status === 'abstained';
          const isCompleted = step.status === 'completed' || step.status === 'success';

          return (
            <div key={idx} className="agent-node">
              <div className="agent-left">
                <div className="agent-icon-badge">
                  {icon}
                </div>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="agent-name">
                      Step {idx + 1}: {step.agent}
                    </span>
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: '700',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        textTransform: 'uppercase',
                        background: isError
                          ? 'rgba(244, 63, 94, 0.15)'
                          : isAbstained
                          ? 'rgba(245, 158, 11, 0.15)'
                          : 'rgba(16, 185, 129, 0.15)',
                        color: isError ? '#f43f5e' : isAbstained ? '#fbbf24' : '#34d399',
                        border: `1px solid ${isError ? 'rgba(244,63,94,0.3)' : isAbstained ? 'rgba(245,158,11,0.3)' : 'rgba(16,185,129,0.3)'}`,
                      }}
                    >
                      {step.status}
                    </span>
                  </div>
                  <div className="agent-summary">
                    {step.output_summary || step.action || 'Agent finished task successfully.'}
                  </div>
                </div>
              </div>

              <div className="agent-right">
                <span className="time-tag">
                  {step.time !== undefined ? `${step.time}s` : '0.00s'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
