import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { REGIME_LABELS, REGIME_COLORS, type RegimeHistory } from '../lib/api';

interface Props {
  data: RegimeHistory | null;
}

export default function RegimeTimeline({ data }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    gsap.from(ref.current, { opacity: 0, y: 20, duration: 0.8, ease: 'power3.out' });
  }, []);

  if (!data) {
    return <div className="flex items-center justify-center h-96 text-text-muted font-mono text-sm">Loading timeline...</div>;
  }

  // Create chart data with nifty close
  const chartData = data.dates.map((d, i) => ({
    date: d,
    nifty: data.nifty50_index[i],
    regime: data.hmm_regimes[i],
    bdl: data.bdl_regimes[i],
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    const p = payload[0].payload;
    const r = p.regime;
    return (
      <div className="bg-bg-card border border-border rounded-lg p-3 shadow-xl">
        <div className="text-xs text-text-muted font-mono">{p.date}</div>
        <div className="text-lg font-bold font-mono" style={{ color: REGIME_COLORS[r] }}>
          {REGIME_LABELS[r]}
        </div>
        <div className="text-sm font-mono text-text-secondary">
          Nifty: {p.nifty?.toFixed(0) || 'N/A'}
        </div>
      </div>
    );
  };

  return (
    <div ref={ref} className="space-y-6">
      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-text-muted text-xs font-mono uppercase tracking-widest">
              Regime Classification Timeline
            </div>
            <div className="text-sm text-text-secondary mt-1 font-mono">
              Nifty 50 with HMM regime overlay · {data.dates.length} data points
            </div>
          </div>
          <div className="flex gap-3">
            {REGIME_LABELS.map((l, i) => (
              <div key={l} className="flex items-center gap-1.5 text-xs font-mono text-text-secondary">
                <div className="w-2.5 h-2.5 rounded-sm" style={{ background: REGIME_COLORS[i] }} />
                {l}
              </div>
            ))}
          </div>
        </div>

        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
              <defs>
                {REGIME_COLORS.map((color, i) => (
                  <linearGradient key={i} id={`regimeGrad${i}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                  </linearGradient>
                ))}
              </defs>
              <XAxis
                dataKey="date"
                tick={{ fill: '#555570', fontSize: 10, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={{ stroke: '#2a2a3a' }}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={['auto', 'auto']}
                tick={{ fill: '#555570', fontSize: 10, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={false}
                width={60}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="nifty"
                stroke="#6366f1"
                strokeWidth={2}
                fill="url(#regimeGrad0)"
                dot={false}
                activeDot={{ r: 4, fill: '#6366f1', stroke: '#0a0a0f', strokeWidth: 2 }}
              />
              {/* Regime color markers at bottom */}
              {chartData.map((d, i) => {
                if (i % Math.max(1, Math.floor(chartData.length / 100)) !== 0) return null;
                return (
                  <ReferenceLine
                    key={i}
                    x={d.date}
                    stroke={REGIME_COLORS[d.regime]}
                    strokeOpacity={0.15}
                    strokeDasharray="3 3"
                  />
                );
              })}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Regime Distribution Summary */}
      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="text-text-muted text-xs font-mono uppercase tracking-widest mb-4">
          Historical Regime Distribution
        </div>
        <div className="grid grid-cols-5 gap-3">
          {REGIME_LABELS.map((label, i) => {
            const count = data.hmm_regimes.filter(r => r === i).length;
            const pct = ((count / data.hmm_regimes.length) * 100).toFixed(1);
            return (
              <div key={label} className="text-center p-3 bg-bg-secondary rounded-lg">
                <div className="text-2xl font-bold font-mono" style={{ color: REGIME_COLORS[i] }}>
                  {pct}%
                </div>
                <div className="text-xs font-mono text-text-secondary mt-1">{label}</div>
                <div className="text-xs font-mono text-text-muted">{count} days</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
