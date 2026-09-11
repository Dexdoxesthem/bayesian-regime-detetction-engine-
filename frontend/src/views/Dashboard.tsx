import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import clsx from 'clsx';
import { REGIME_COLORS, type RegimeCurrent } from '../lib/api';

interface Props {
  current: RegimeCurrent | null;
}

export default function Dashboard({ current }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !current) return;
    gsap.from(containerRef.current.children, {
      opacity: 0, y: 30, duration: 0.6, stagger: 0.1, ease: 'power3.out',
    });
  }, [current]);

  if (!current) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-text-muted text-sm font-medium">Loading regime data...</div>
      </div>
    );
  }

  const probEntries = Object.entries(current.probabilities);
  
  // Create natural language insights based on data
  const isRiskOn = current.current_regime === 0 || current.current_regime === 1; // Assuming 0,1 are lower risk/bullish
  const insightText = isRiskOn 
    ? `Market conditions are favorable. Nifty is ${current.metrics.nifty.change > 0 ? 'up' : 'down'} ${Math.abs(current.metrics.nifty.change).toFixed(1)}%, supported by ${current.metrics.breadth.value.toFixed(0)}% breadth.`
    : `Defensive positioning recommended. VIX is ${current.metrics.vix.change > 0 ? 'up' : 'down'} ${Math.abs(current.metrics.vix.change).toFixed(1)}% and market breadth is narrow.`;

  return (
    <div ref={containerRef} className="space-y-6">
      
      {/* Page Title */}
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Overview</h1>
        <div className="flex gap-2">
          <div className="px-3 py-1.5 bg-bg-card rounded-md border border-border text-xs font-semibold text-text-secondary">
            {current.date}
          </div>
          <button className="px-4 py-1.5 bg-bg-card rounded-md border border-border text-xs font-semibold hover:bg-bg-secondary transition-colors">
            Daily ▾
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Main Regime Probabilities Card (Like "Payments" chart in Zentra) */}
        <div className="lg:col-span-2 bg-bg-card border border-border rounded-2xl p-8 shadow-[var(--shadow-card)] flex flex-col">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-xl font-bold">Regime Probabilities</h2>
            <button className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-text-muted hover:bg-bg-secondary">
              ⋯
            </button>
          </div>
          
          <div className="flex-1 flex flex-col justify-end gap-6">
            {probEntries.map(([label, prob], i) => {
              const isActive = i === current.current_regime;
              return (
                <div key={label} className="flex items-center gap-4">
                  <div className="w-32 text-sm font-semibold text-right shrink-0"
                    style={{ color: isActive ? '#111827' : '#9ca3af' }}>
                    {label}
                  </div>
                  <div className="flex-1 h-10 bg-bg-secondary rounded-lg overflow-hidden relative">
                    <div className="absolute inset-y-0 left-0 rounded-lg transition-all duration-1000"
                      style={{
                        width: `${Math.max(prob * 100, 2)}%`,
                        background: isActive
                          ? `linear-gradient(90deg, ${REGIME_COLORS[i]}cc, ${REGIME_COLORS[i]})`
                          : `#d1d5db`,
                      }} />
                  </div>
                  <div className="w-16 text-sm font-bold shrink-0 text-right"
                    style={{ color: isActive ? REGIME_COLORS[i] : '#9ca3af' }}>
                    {(prob * 100).toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Natural language query box */}
          <div className="mt-8 p-3 border border-border rounded-xl bg-bg-primary text-sm text-text-muted flex items-center gap-2">
            ✨ What would you like to explore next?
          </div>
        </div>

        {/* Right Sidebar Metrics */}
        <div className="space-y-6">
          {/* Nifty Metric Card */}
          <div className="bg-bg-card border border-border rounded-2xl p-6 shadow-[var(--shadow-card)]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold">Nifty 50</h3>
              <button className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-text-muted hover:bg-bg-secondary">⋯</button>
            </div>
            <div className="flex items-baseline gap-3 mb-6">
              <span className="text-4xl font-extrabold tracking-tight">
                {current.metrics.nifty.value.toLocaleString()}
              </span>
              <span className={clsx(
                "px-2 py-1 rounded-md text-xs font-bold flex items-center",
                current.metrics.nifty.change >= 0 ? "bg-green/10 text-green" : "bg-red/10 text-red"
              )}>
                {current.metrics.nifty.change >= 0 ? '▲' : '▼'} {Math.abs(current.metrics.nifty.change).toFixed(2)}%
              </span>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-text-secondary font-medium">Market Breadth</span>
                <span className="font-bold">{current.metrics.breadth.value.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2 bg-bg-secondary rounded-full overflow-hidden">
                <div className="h-full bg-green" style={{ width: `${current.metrics.breadth.value}%` }} />
              </div>
            </div>
          </div>

          {/* VIX Metric Card */}
          <div className="bg-bg-card border border-border rounded-2xl p-6 shadow-[var(--shadow-card)] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-bold">India VIX</h3>
                <button className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-text-muted hover:bg-bg-secondary">⋯</button>
              </div>
              <div className="text-3xl font-extrabold tracking-tight">
                {current.metrics.vix.value.toFixed(2)}
              </div>
            </div>
            <div className="mt-4 flex items-center justify-between text-sm font-medium">
              <span className="text-text-secondary">vs previous day</span>
              <span className={current.metrics.vix.change < 0 ? 'text-green' : 'text-red'}>
                {current.metrics.vix.change > 0 ? '+' : ''}{current.metrics.vix.change.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Flow Metrics */}
        <div className="bg-bg-card border border-border rounded-2xl p-6 shadow-[var(--shadow-card)]">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold">Institutional Flows</h3>
            <button className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-text-muted hover:bg-bg-secondary">⋯</button>
          </div>
          <div className="space-y-6">
            <div>
              <div className="text-sm font-medium text-text-secondary mb-1">FII Net</div>
              <div className={clsx("text-2xl font-bold", current.metrics.fii.value >= 0 ? 'text-green' : 'text-red')}>
                {current.metrics.fii.value > 0 ? '+' : ''}{Math.round(current.metrics.fii.value).toLocaleString()} Cr
              </div>
            </div>
            <div>
              <div className="text-sm font-medium text-text-secondary mb-1">DII Net</div>
              <div className={clsx("text-2xl font-bold", current.metrics.dii.value >= 0 ? 'text-green' : 'text-red')}>
                {current.metrics.dii.value > 0 ? '+' : ''}{Math.round(current.metrics.dii.value).toLocaleString()} Cr
              </div>
            </div>
          </div>
        </div>

        {/* Gradient Insights Card */}
        <div className="lg:col-span-2 rounded-2xl p-8 text-white relative overflow-hidden shadow-[var(--shadow-card)]"
             style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #a855f7 100%)' }}>
          {/* Decorative abstract shape */}
          <div className="absolute right-0 top-0 w-64 h-64 bg-white/10 rounded-full blur-3xl translate-x-1/2 -translate-y-1/2" />
          
          <div className="inline-flex items-center gap-1 px-3 py-1 bg-white/20 rounded-full text-xs font-semibold backdrop-blur-md mb-6">
            💡 Insights
          </div>
          
          <div className="text-5xl font-bold mb-4 tracking-tight">
            {(current.confidence * 100).toFixed(0)}%
          </div>
          <h3 className="text-xl font-bold mb-2">
            Model confidence in {current.regime_label} regime.
          </h3>
          <p className="text-white/80 max-w-lg leading-relaxed text-sm font-medium">
            {insightText} This classification is driven primarily by recent changes in {Math.abs(current.metrics.fii.value) > 1000 ? 'institutional flow dynamics' : 'price volatility'}.
          </p>
        </div>
      </div>
    </div>
  );
}
