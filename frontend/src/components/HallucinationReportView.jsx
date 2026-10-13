import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, CheckCircle2, XCircle, Info } from 'lucide-react';

export default function HallucinationReportView({
  hallucinationRisk = null,
  supportPercentage = null,
  claims = [],
  verificationStatus = 'UNKNOWN',
}) {
  if (hallucinationRisk === null && (!claims || claims.length === 0)) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🔍</div>
        <h3 style={{ fontSize: '1.1rem', color: '#cbd5e1', marginBottom: '0.5rem' }}>No Hallucination Data</h3>
        <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
          Run a query through the Multi-Agent pipeline to trigger the Natural Language Inference (NLI) Hallucination Critic.
        </p>
      </div>
    );
  }

  const riskPct = hallucinationRisk !== null ? (hallucinationRisk * 100).toFixed(1) : '0.0';
  const supPct = supportPercentage !== null ? (supportPercentage * 100).toFixed(1) : '100.0';

  const isSafe = hallucinationRisk < 0.35;
  const isModerate = hallucinationRisk >= 0.35 && hallucinationRisk < 0.6;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top summary cards */}
      <div className="metrics-grid">
        <div className={`metric-card ${isSafe ? 'risk-safe' : 'risk-alert'}`}>
          <div className="metric-label">Hallucination Risk</div>
          <div className="metric-value" style={{ color: isSafe ? '#34d399' : isModerate ? '#fbbf24' : '#f43f5e' }}>
            {riskPct}%
          </div>
          <div className="metric-sub">
            {isSafe ? 'Safe · Well within threshold' : isModerate ? 'Moderate risk' : 'Critical risk'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Support Ratio</div>
          <div className="metric-value" style={{ color: '#22d3ee' }}>
            {supPct}%
          </div>
          <div className="metric-sub">Evidence entailment score</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Claims Analyzed</div>
          <div className="metric-value" style={{ color: '#a5b4fc' }}>
            {claims.length}
          </div>
          <div className="metric-sub">Atomic legal facts checked</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Guard Status</div>
          <div className="metric-value" style={{ fontSize: '1.25rem', color: isSafe ? '#34d399' : '#fbbf24' }}>
            {verificationStatus}
          </div>
          <div className="metric-sub">Pipeline verdict</div>
        </div>
      </div>

      {/* Claims Breakdown Table */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem' }}>
          Atomic Claims & Grounding Evidence
        </h3>

        {claims.length === 0 ? (
          <div style={{ fontSize: '0.85rem', color: '#94a3b8', padding: '1rem 0' }}>
            The answer was verified at document-level without unsupported atomic claims.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {claims.map((claim, idx) => {
              const isClaimSupported =
                claim.status === 'supported' || claim.entailment_score >= 0.5;
              const isClaimPartial = claim.status === 'partially_supported';

              return (
                <div
                  key={idx}
                  style={{
                    background: 'rgba(10, 12, 28, 0.6)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '12px 16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.88rem', fontWeight: 600, color: '#f1f5f9' }}>
                      Claim #{idx + 1}: "{claim.claim || claim.text}"
                    </span>
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '0.72rem',
                        fontWeight: '700',
                        background: isClaimSupported
                          ? 'rgba(16, 185, 129, 0.15)'
                          : isClaimPartial
                          ? 'rgba(245, 158, 11, 0.15)'
                          : 'rgba(244, 63, 94, 0.15)',
                        color: isClaimSupported ? '#34d399' : isClaimPartial ? '#fbbf24' : '#f43f5e',
                      }}
                    >
                      {isClaimSupported ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
                      {claim.status || (isClaimSupported ? 'SUPPORTED' : 'UNSUPPORTED')}
                    </span>
                  </div>

                  {claim.evidence && (
                    <div
                      style={{
                        fontSize: '0.78rem',
                        color: '#94a3b8',
                        background: 'rgba(255, 255, 255, 0.02)',
                        padding: '6px 10px',
                        borderRadius: '6px',
                        borderLeft: '2px solid rgba(6, 182, 212, 0.4)',
                      }}
                    >
                      <strong style={{ color: '#22d3ee' }}>Grounding evidence:</strong> {claim.evidence}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
