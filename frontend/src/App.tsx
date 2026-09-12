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
    <div className="min-h-screen flex flex-col font-sans bg-bg-primary text-text-primary">
      
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 bg-bg-card border-b border-border shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 mr-6">
              <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center text-white font-bold text-lg font-serif">
                Z
              </div>
              <span className="font-serif font-semibold text-lg tracking-tight text-text-primary">
                Zetheta Regime Engine
              </span>
            </div>

            {/* Nav Items */}
            <nav className="hidden md:flex items-center gap-2">
              {NAV_ITEMS.map(item => (
                <button
                  key={item.id}
                  onClick={() => switchView(item.id)}
                  className={clsx(
                    'px-4 py-1.5 rounded-md text-sm font-medium transition-all',
                    view === item.id
                      ? 'bg-text-primary text-white shadow-sm'
                      : 'text-text-secondary hover:text-text-primary hover:bg-bg-secondary',
                  )}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Right side status / user */}
          <div className="flex items-center gap-4">
            {current && (
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div className="text-xs font-semibold" style={{ color: current.regime_color }}>
                    {current.regime_label}
                  </div>
                  <div className="text-[10px] text-text-muted font-mono uppercase tracking-wider">
                    Live Status
                  </div>
                </div>
                <div 
                  className="w-8 h-8 rounded-full border-2 flex items-center justify-center"
                  style={{ borderColor: current.regime_color, backgroundColor: current.regime_color + '15' }}
                >
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: current.regime_color }} />
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-8">
        <div ref={mainRef}>
          {renderView()}
        </div>
      </main>
    </div>
  );
}
