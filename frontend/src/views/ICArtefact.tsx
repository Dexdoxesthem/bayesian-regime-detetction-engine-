import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { type ICArtefact, REGIME_COLORS } from '../lib/api';

interface Props {
  data: ICArtefact | null;
}

export default function ICArtefact({ data }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || !data) return;
    gsap.from(ref.current, { opacity: 0, scale: 0.95, duration: 0.8, ease: 'power3.out' });
  }, [data]);

  if (!data) {
    return <div className="flex items-center justify-center h-96 text-text-muted font-mono text-sm">Loading IC artefact...</div>;
  }

  return (
    <div ref={ref} className="space-y-6">
      <div className="bg-bg-card border border-border rounded-xl p-8">
        <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-4">
          Investment Committee Artefact
        </div>

        {/* Regime Badge */}
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-bold"
            style={{ background: `${REGIME_COLORS[data.regime]}20`, color: REGIME_COLORS[data.regime] }}>
            {data.regime}
          </div>
          <div>
            <div className="text-3xl font-bold" style={{ color: REGIME_COLORS[data.regime] }}>
              {data.label}
            </div>
            <div className="text-sm font-mono text-text-secondary mt-1">
              Confidence: {(data.confidence * 100).toFixed(1)}% · As of {data.date}
            </div>
          </div>
        </div>

        {/* Recommendation */}
        <div className="bg-bg-secondary rounded-xl p-6 border border-border-light">
          <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-3">
            Conditional Statement
          </div>
          <p className="text-base leading-relaxed text-text-primary font-mono">
            {data.recommendation}
          </p>
        </div>

        {/* Model Lineage */}
        <div className="mt-6 pt-6 border-t border-border">
          <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-3">
            Model Lineage
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm font-mono">
            <div className="bg-bg-secondary rounded-lg p-3">
              <div className="text-text-muted text-xs">HMM State</div>
              <div className="text-text-primary font-semibold">5-class Viterbi</div>
            </div>
            <div className="bg-bg-secondary rounded-lg p-3">
              <div className="text-text-muted text-xs">Ensemble</div>
              <div className="text-text-primary font-semibold">BMA Weighted</div>
            </div>
            <div className="bg-bg-secondary rounded-lg p-3">
              <div className="text-text-muted text-xs">Conformal</div>
              <div className="text-text-primary font-semibold">Split + ACI</div>
            </div>
            <div className="bg-bg-secondary rounded-lg p-3">
              <div className="text-text-muted text-xs">Validation</div>
              <div className="text-green font-semibold">Crisis Replay ✓</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
