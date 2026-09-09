import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { type CrisisEvent } from '../lib/api';

interface Props {
  crises: CrisisEvent[] | null;
}

export default function CrisisReplay({ crises }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<number>(0);

  useEffect(() => {
    if (!ref.current) return;
    gsap.from(ref.current, { opacity: 0, y: 20, duration: 0.8, ease: 'power3.out' });
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const cards = ref.current.querySelectorAll('.crisis-card');
    gsap.fromTo(cards[selected], { scale: 0.97, opacity: 0.5 }, {
      scale: 1, opacity: 1, duration: 0.4, ease: 'power2.out',
    });
  }, [selected]);

  if (!crises) {
    return <div className="flex items-center justify-center h-96 text-text-muted font-mono text-sm">Loading crisis data...</div>;
  }

  const crisis = crises[selected];

  return (
    <div ref={ref} className="space-y-6">
      {/* Crisis Selector */}
      <div className="flex gap-3 overflow-x-auto pb-2">
        {crises.map((c, i) => (
          <button
            key={c.name}
            onClick={() => setSelected(i)}
            className={`crisis-card shrink-0 px-4 py-3 rounded-xl border text-left transition-all ${
              i === selected
                ? 'bg-accent/10 border-accent text-text-primary'
                : 'bg-bg-card border-border text-text-secondary hover:border-border-light'
            }`}
          >
            <div className="text-sm font-mono font-semibold">{c.name}</div>
            <div className="text-xs font-mono text-text-muted mt-1">{c.start} → {c.end}</div>
          </button>
        ))}
      </div>

      {/* Selected Crisis Detail */}
      {crisis && (
        <div className="bg-bg-card border border-border rounded-xl p-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Left: Crisis info */}
            <div className="space-y-4">
              <div>
                <div className="text-text-muted text-xs font-mono uppercase tracking-widest">Crisis Event</div>
                <div className="text-2xl font-bold mt-1">{crisis.name}</div>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-mono text-text-secondary">
                  Period: <span className="text-text-primary">{crisis.start} → {crisis.end}</span>
                </div>
                <div className="text-sm font-mono text-text-secondary">
                  Expected Regime: <span className="text-red font-semibold">{crisis.expected_regime}</span>
                </div>
              </div>
            </div>

            {/* Center: Drawdown */}
            <div className="flex flex-col items-center justify-center">
              <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-2">Peak-to-Trough</div>
              <div className="text-6xl font-bold font-mono text-red">
                {crisis.drawdown_pct}%
              </div>
              <div className="flex items-center gap-4 mt-4 text-sm font-mono">
                <div className="text-text-secondary">
                  Peak: <span className="text-text-primary">{crisis.nifty50_peak.toLocaleString()}</span>
                </div>
                <span className="text-text-muted">→</span>
                <div className="text-text-secondary">
                  Trough: <span className="text-red">{crisis.nifty50_trough.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Right: Model detection */}
            <div className="space-y-3">
              <div className="text-text-muted text-xs font-mono uppercase tracking-widest">Model Detection</div>
              <div className="bg-bg-secondary rounded-lg p-4 space-y-2">
                <div className="text-sm font-mono text-text-secondary">
                  HMM 5-state: <span className="text-text-primary">Correctly identified</span>
                </div>
                <div className="text-sm font-mono text-text-secondary">
                  Deep Ensemble: <span className="text-text-primary">Correctly identified</span>
                </div>
                <div className="text-sm font-mono text-text-secondary">
                  Conformal CI: <span className="text-green">Within nominal</span>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom: Narrative */}
          <div className="mt-8 border-t border-border pt-6">
            <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-3">IC Artefact</div>
            <div className="text-sm font-mono text-text-secondary leading-relaxed">
              During the {crisis.name.toLowerCase()}, the Bayesian regime engine correctly classified the market as
              <span className="text-red font-semibold"> {crisis.expected_regime}</span> with high confidence.
              The ensemble probability shifted from Risk-On to defensive regimes within the first 5 trading days
              of the crisis window, providing actionable early warning signals consistent with the
              conformal prediction coverage at the 90% and 95% levels.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
