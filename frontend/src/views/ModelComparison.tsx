import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { type ModelPerformance, REGIME_COLORS } from '../lib/api';

interface Props {
  data: ModelPerformance | null;
}

export default function ModelComparison({ data }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    gsap.from(ref.current.children, {
      opacity: 0, y: 20, duration: 0.6, stagger: 0.1, ease: 'power3.out',
    });
  }, [data]);

  if (!data) {
    return <div className="flex items-center justify-center h-96 text-text-muted font-mono text-sm">Loading model data...</div>;
  }

  const weightData = Object.entries(data.ensemble_weights).map(([name, weight]) => ({
    name: name.replace('_', ' ').toUpperCase(),
    weight,
  }));

  return (
    <div ref={ref} className="space-y-6">
      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {data.models.map((m) => (
          <div key={m.name} className="bg-bg-card border border-border rounded-xl p-5">
            <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-2">{m.type}</div>
            <div className="text-lg font-bold">{m.name}</div>
            <div className="mt-3 space-y-1.5 text-sm font-mono">
              {m.bic && (
                <div className="text-text-secondary">
                  BIC: <span className="text-text-primary">{m.bic.toLocaleString()}</span>
                </div>
              )}
              {m.aic && (
                <div className="text-text-secondary">
                  AIC: <span className="text-text-primary">{m.aic.toLocaleString()}</span>
                </div>
              )}
              {m.val_acc !== undefined && (
                <div className="text-text-secondary">
                  Val Acc: <span className="text-green">{(m.val_acc * 100).toFixed(1)}%</span>
                </div>
              )}
              {m.converged !== undefined && (
                <div className="text-text-secondary">
                  Converged: <span className={m.converged ? 'text-green' : 'text-red'}>
                    {m.converged ? 'Yes' : 'No'}
                  </span>
                </div>
              )}
              {m.framework && (
                <div className="text-text-secondary">
                  Framework: <span className="text-text-primary">{m.framework}</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Ensemble Weights */}
      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-4">
          BMA Ensemble Weights
        </div>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={weightData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
              <XAxis
                dataKey="name"
                tick={{ fill: '#8888a0', fontSize: 11, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={{ stroke: '#2a2a3a' }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#555570', fontSize: 10, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{ background: '#16161f', border: '1px solid #2a2a3a', borderRadius: 8 }}
                labelStyle={{ color: '#8888a0', fontFamily: 'monospace' }}
                formatter={(v) => [((v as number) * 100).toFixed(1) + '%', 'Weight']}
              />
              <Bar dataKey="weight" radius={[6, 6, 0, 0]}>
                {weightData.map((_, i) => (
                  <Cell key={i} fill={REGIME_COLORS[i]} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Conformal Calibration */}
      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-4">
          Conformal Prediction Calibration
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div className="bg-bg-secondary rounded-lg p-4 text-center">
            <div className="text-xs font-mono text-text-muted mb-1">90% Level Coverage</div>
            <div className="text-3xl font-bold font-mono text-cyan">
              {(data.conformal.level_90.coverage * 100).toFixed(1)}%
            </div>
            <div className="text-xs font-mono text-text-secondary mt-1">
              Target: 90.0% · Avg set size: {data.conformal.level_90.avg_set_size}
            </div>
          </div>
          <div className="bg-bg-secondary rounded-lg p-4 text-center">
            <div className="text-xs font-mono text-text-muted mb-1">95% Level Coverage</div>
            <div className="text-3xl font-bold font-mono text-accent">
              {(data.conformal.level_95.coverage * 100).toFixed(1)}%
            </div>
            <div className="text-xs font-mono text-text-secondary mt-1">
              Target: 95.0% · Avg set size: {data.conformal.level_95.avg_set_size}
            </div>
          </div>
          <div className="bg-bg-secondary rounded-lg p-4 text-center">
            <div className="text-xs font-mono text-text-muted mb-1">Expected Calibration Error</div>
            <div className="text-3xl font-bold font-mono text-amber">
              {data.conformal.ece.toFixed(4)}
            </div>
            <div className="text-xs font-mono text-text-secondary mt-1">
              Lower is better · Max tolerable: 0.05
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
