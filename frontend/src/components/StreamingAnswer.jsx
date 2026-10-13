import React, { useState, useEffect, useRef } from 'react';
import { FastForward, CheckCircle2 } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function StreamingAnswer({ text = '', speed = 20, isVerified = false }) {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const wordsRef = useRef([]);
  const indexRef = useRef(0);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!text) {
      setDisplayedText('');
      setIsTyping(false);
      return;
    }

    // Split text into words while preserving line breaks
    // Replace newlines with a special marker to preserve formatting during word split
    const tokens = text.split(/(\s+)/);
    wordsRef.current = tokens;
    indexRef.current = 0;
    setDisplayedText('');
    setIsTyping(true);

    if (timerRef.current) clearInterval(timerRef.current);

    timerRef.current = setInterval(() => {
      if (indexRef.current < wordsRef.current.length) {
        indexRef.current += 1;
        setDisplayedText(wordsRef.current.slice(0, indexRef.current).join(''));
      } else {
        clearInterval(timerRef.current);
        setIsTyping(false);
        if (isVerified) {
          try {
            confetti({
              particleCount: 40,
              spread: 60,
              origin: { y: 0.8 },
              colors: ['#6366f1', '#06b6d4', '#10b981'],
            });
          } catch (e) {
            // ignore
          }
        }
      }
    }, speed);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [text, speed]);

  const handleSkip = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setDisplayedText(text);
    setIsTyping(false);
  };

  // Format markdown helper (bold, numbered points, sources)
  const renderFormattedText = (raw) => {
    const lines = raw.split('\n');
    return lines.map((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return <div key={idx} style={{ height: '0.6rem' }} />;
      }

      // Check if numbered or bullet point
      const isPoint = /^\d+\.\s/.test(trimmed) || trimmed.startsWith('•') || trimmed.startsWith('-');
      const isHeader = trimmed.startsWith('**') && trimmed.endsWith('**');

      // Highlight sources like (Source: Final_BNS)
      let content = line;
      const sourceMatch = content.match(/\(Source:\s*([^)]+)\)/i);
      let sourceTag = null;
      if (sourceMatch) {
        content = content.replace(sourceMatch[0], '');
        sourceTag = (
          <span
            style={{
              display: 'inline-block',
              marginLeft: '8px',
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '0.74rem',
              fontWeight: '600',
              fontFamily: "'JetBrains Mono', monospace",
              background: 'rgba(99, 102, 241, 0.15)',
              border: '1px solid rgba(99, 102, 241, 0.35)',
              color: '#a5b4fc',
            }}
          >
            § {sourceMatch[1]}
          </span>
        );
      }

      // Parse bold **text**
      const parts = content.split(/(\*\*[^*]+\*\*)/g);
      const renderedParts = parts.map((part, pIdx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <strong key={pIdx} style={{ color: '#ffffff', fontWeight: '700' }}>
              {part.slice(2, -2)}
            </strong>
          );
        }
        return part;
      });

      return (
        <div
          key={idx}
          style={{
            marginBottom: isPoint ? '0.65rem' : '0.45rem',
            paddingLeft: isPoint ? '0.5rem' : '0',
            borderLeft: isPoint ? '2px solid rgba(99, 102, 241, 0.4)' : 'none',
            color: isHeader ? '#38bdf8' : '#e2e8f0',
          }}
        >
          {renderedParts}
          {sourceTag}
        </div>
      );
    });
  };

  return (
    <div style={{ position: 'relative' }}>
      <div className="answer-content">
        {renderFormattedText(displayedText)}
        {isTyping && <span className="typing-cursor" />}
      </div>

      {isTyping && (
        <button
          onClick={handleSkip}
          style={{
            position: 'absolute',
            top: '-2.5rem',
            right: '0',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            background: 'rgba(99, 102, 241, 0.2)',
            border: '1px solid rgba(99, 102, 241, 0.4)',
            color: '#c7d2fe',
            borderRadius: '6px',
            padding: '4px 10px',
            fontSize: '0.75rem',
            cursor: 'pointer',
          }}
        >
          <FastForward size={12} />
          Skip typing
        </button>
      )}
    </div>
  );
}
