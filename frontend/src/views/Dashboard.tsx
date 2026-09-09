import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import clsx from 'clsx';
import { REGIME_COLORS } from '../lib/api';

interface Props {
  current: {
    current_regime: number;
    regime_label: string;
    regime_color: string;
    confidence: number;
    probabilities: Record<string, number>;
    date: string;
  } | null;
}

export default function Dashboard({ current }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const barsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !current) return;
    gsap.from(containerRef.current.children, {
      opacity: 0, y: 30, duration: 0.6, stagger: 0.1, ease: 'power3.out',
    });
  }, [current]);

  useEffect(() => {
    if (!barsRef.current || !current) return;
    const bars = barsRef.current.querySelectorAll('.prob-bar');
    gsap.fromTo(bars, { width: 0 }, {
      width: '100%', duration: 1.2, stagger: 0.08, ease: 'power3.out', delay: 0.3,
    });
  }, [current]);

  if (!current) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-text-muted text-sm font-mono">Loading regime data...</div>
      </div>
    );
  }

  const probEntries = Object.entries(current.probabilities);

  return (
    <div ref={containerRef} className="space-y-6">
      {/* Current Regime Hero */}
      <div className="bg-bg-card border border-border rounded-xl p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 opacity-5"
          style={{ background: `radial-gradient(circle, ${current.regime_color}, transparent)` }} />
        <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-2">
          Current Market Regime
        </div>
        <div className="flex items-baseline gap-4 mb-1">
          <span className="text-5xl font-bold tracking-tight"
            style={{ color: current.regime_color }}>
            {current.regime_label}
          </span>
        </div>
        <div className="flex items-center gap-6 mt-4 text-sm text-text-secondary">
          <span className="font-mono">
            Confidence: <span className="text-text-primary font-semibold">
              {(current.confidence * 100).toFixed(1)}%
            </span>
          </span>
          <span className="font-mono">
            Date: <span className="text-text-primary">{current.date}</span>
          </span>
        </div>
      </div>

      {/* Regime Probability Bars */}
      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-4">
          Regime Probability Distribution
        </div>
        <div ref={barsRef} className="space-y-3">
          {probEntries.map(([label, prob], i) => {
            const isActive = i === current.current_regime;
            return (
              <div key={label} className="flex items-center gap-4">
                <div className="w-28 text-sm font-mono text-right shrink-0"
                  style={{ color: REGIME_COLORS[i] }}>
                  {label}
                </div>
                <div className="flex-1 h-7 bg-bg-secondary rounded overflow-hidden relative">
                  <div className="prob-bar absolute inset-y-0 left-0 rounded"
                    style={{
                      width: `${prob * 100}%`,
                      background: isActive
                        ? `linear-gradient(90deg, ${REGIME_COLORS[i]}88, ${REGIME_COLORS[i]})`
                        : `${REGIME_COLORS[i]}33`,
                    }} />
                  <span className="absolute inset-0 flex items-center px-3 text-xs font-mono"
                    style={{ color: isActive ? '#fff' : REGIME_COLORS[i] }}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Nifty 50', value: '21,857', change: '+0.8%', positive: true },
          { label: 'India VIX', value: '13.2', change: '-2.1%', positive: true },
          { label: 'FII Net', value: '-1,234 Cr', change: 'Bearish', positive: false },
          { label: 'DII Net', value: '+2,567 Cr', change: 'Bullish', positive: true },
        ].map((m) => (
          <div key={m.label} className="bg-bg-card border border-border rounded-xl p-4">
            <div className="text-text-muted text-xs font-mono uppercase tracking-wider">{m.label}</div>
            <div className="text-2xl font-bold mt-1 font-mono">{m.value}</div>
            <div className={clsx('text-xs font-mono mt-1', m.positive ? 'text-green' : 'text-red')}>
              {m.change}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
