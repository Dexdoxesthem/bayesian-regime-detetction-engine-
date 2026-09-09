export const API_BASE = '/api';

export async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

export interface RegimeCurrent {
  current_regime: number;
  regime_label: string;
  regime_color: string;
  confidence: number;
  probabilities: Record<string, number>;
  date: string;
}

export interface RegimeHistory {
  dates: string[];
  nifty50_index: number[];
  hmm_regimes: number[];
  bdl_regimes: number[];
  hmm_probs: number[][];
  bdl_probs: number[][];
}

export interface CrisisEvent {
  name: string;
  start: string;
  end: string;
  expected_regime: string;
  nifty50_peak: number;
  nifty50_trough: number;
  drawdown_pct: number;
}

export interface ModelPerformance {
  models: Array<{
    name: string;
    type: string;
    bic?: number;
    aic?: number;
    val_acc?: number;
    converged?: boolean;
    n_regimes?: number;
    framework?: string;
  }>;
  ensemble_weights: Record<string, number>;
  conformal: {
    level_90: { coverage: number; avg_set_size: number };
    level_95: { coverage: number; avg_set_size: number };
    ece: number;
  };
}

export interface ICArtefact {
  date: string;
  regime: number;
  label: string;
  confidence: number;
  recommendation: string;
}

export const REGIME_LABELS = ['Risk-On', 'Risk-Off', 'Transitional', 'Late-Cycle', 'Post-Shock'];
export const REGIME_COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#a855f7', '#06b6d4'];
