import { useState, useEffect, useRef } from 'react';
import gsap from 'gsap';
import clsx from 'clsx';
import Dashboard from './views/Dashboard';
import RegimeTimeline from './views/RegimeTimeline';
import CrisisReplay from './views/CrisisReplay';
import ModelComparison from './views/ModelComparison';
import ICArtefact from './views/ICArtefact';
import {
  fetchJSON, type RegimeCurrent, type RegimeHistory,
  type CrisisEvent, type ModelPerformance, type ICArtefact as ICArtefactType,
} from './lib/api';

type View = 'dashboard' | 'timeline' | 'crisis' | 'models' | 'ic';

const NAV_ITEMS: { id: View; label: string; icon: string }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: '◉' },
  { id: 'timeline', label: 'Regime Timeline', icon: '━' },
  { id: 'crisis', label: 'Crisis Replay', icon: '⟐' },
  { id: 'models', label: 'Model Suite', icon: '⧫' },
  { id: 'ic', label: 'IC Artefact', icon: '⬡' },
];

export default function App() {
  const [view, setView] = useState<View>('dashboard');
  const [navOpen, setNavOpen] = useState(true);
  const mainRef = useRef<HTMLDivElement>(null);

  // Data
  const [current, setCurrent] = useState<RegimeCurrent | null>(null);
  const [history, setHistory] = useState<RegimeHistory | null>(null);
  const [crises, setCrises] = useState<CrisisEvent[] | null>(null);
  const [models, setModels] = useState<ModelPerformance | null>(null);
  const [icArtefact, setICArtefact] = useState<ICArtefactType | null>(null);

  useEffect(() => {
    fetchJSON<RegimeCurrent>('/api/regime/current').then(setCurrent).catch(console.error);
    fetchJSON<RegimeHistory>('/api/regime/history').then(setHistory).catch(console.error);
    fetchJSON<{ crises: CrisisEvent[] }>('/api/regime/crisis').then(d => setCrises(d.crises)).catch(console.error);
    fetchJSON<ModelPerformance>('/api/models/performance').then(setModels).catch(console.error);
    fetchJSON<ICArtefactType>('/api/regime/ic-artefact').then(setICArtefact).catch(console.error);
  }, []);

  useEffect(() => {
    if (!mainRef.current) return;
    gsap.fromTo(mainRef.current, { opacity: 0, x: 20 }, {
      opacity: 1, x: 0, duration: 0.5, ease: 'power2.out',
    });
  }, [view]);

  const switchView = (v: View) => {
    if (mainRef.current) {
      gsap.to(mainRef.current, {
        opacity: 0, x: -20, duration: 0.2, ease: 'power2.in',
        onComplete: () => setView(v),
      });
    } else {
      setView(v);
    }
  };

  const renderView = () => {
    switch (view) {
      case 'dashboard': return <Dashboard current={current} />;
      case 'timeline': return <RegimeTimeline data={history} />;
      case 'crisis': return <CrisisReplay crises={crises} />;
      case 'models': return <ModelComparison data={models} />;
      case 'ic': return <ICArtefact data={icArtefact} />;
    }
  };

  return (
    <div className="flex h-screen bg-bg-primary">
      {/* Sidebar */}
      <aside className={clsx(
        'flex flex-col border-r border-border bg-bg-secondary transition-all duration-300',
        navOpen ? 'w-64' : 'w-16',
      )}>
        {/* Logo */}
        <div className="h-16 flex items-center px-4 border-b border-border">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center text-white font-bold text-sm shrink-0">
              RE
            </div>
            {navOpen && (
              <span className="text-sm font-semibold whitespace-nowrap">Regime Engine</span>
            )}
          </div>
        </div>

        {/* Nav Items */}
        <nav className="flex-1 py-4 space-y-1 px-2">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              onClick={() => switchView(item.id)}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all',
                view === item.id
                  ? 'bg-accent/10 text-accent'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover',
              )}
            >
              <span className="text-lg shrink-0 w-6 text-center">{item.icon}</span>
              {navOpen && <span className="whitespace-nowrap">{item.label}</span>}
            </button>
          ))}
        </nav>

        {/* Status */}
        <div className="p-4 border-t border-border">
          {navOpen && current && (
            <div className="bg-bg-card rounded-lg p-3">
              <div className="text-xs font-mono text-text-muted uppercase tracking-wider">Live Regime</div>
              <div className="text-lg font-bold mt-1" style={{ color: current.regime_color }}>
                {current.regime_label}
              </div>
              <div className="text-xs font-mono text-text-secondary">
                {(current.confidence * 100).toFixed(1)}% · {current.date}
              </div>
            </div>
          )}
          <button
            onClick={() => setNavOpen(!navOpen)}
            className="mt-2 w-full text-center text-text-muted hover:text-text-secondary text-xs font-mono"
          >
            {navOpen ? '◂ collapse' : '▸'}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {/* Top Bar */}
        <header className="h-16 border-b border-border flex items-center justify-between px-8 bg-bg-secondary/50 backdrop-blur-sm sticky top-0 z-10">
          <div className="text-sm font-mono text-text-secondary">
            Bayesian Regime Detection · Indian Equity Markets
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-text-muted">
            <span>Python 3.14 · PyTorch 2.13 · PyMC 6.3</span>
            <div className="w-2 h-2 rounded-full bg-green animate-pulse" />
          </div>
        </header>

        {/* Content */}
        <div ref={mainRef} className="p-8">
          {renderView()}
        </div>
      </main>
    </div>
  );
}
