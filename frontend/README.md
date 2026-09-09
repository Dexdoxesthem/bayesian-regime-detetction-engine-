# Regime Engine Frontend

Institutional-grade React dashboard for the Bayesian Regime Detection Engine.

## Stack

- **React 19 + TypeScript + Vite**
- **GSAP** — page transitions, scroll animations, bar/meter reveals
- **Recharts** — interactive regime timeline and weight charts
- **Tailwind CSS v4** — dark institutional theme
- **FastAPI** backend serving live model inference

## Quick start

From the project root:

```bash
python start_dashboard.py
```

This launches both the FastAPI backend (`:8000`) and the Vite dev server (`:5173`).

- Dashboard: http://localhost:5173
- API docs:  http://localhost:8000/docs

## Manual start

Backend:

```bash
python api/run.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Views

| Route | Description |
|-------|-------------|
| Dashboard | Current regime, probability distribution, key market metrics |
| Regime Timeline | Historical Nifty 50 with HMM regime overlay, distribution summary |
| Crisis Replay | Interactive 2008/2013/2018/2020/2024 crisis case studies |
| Model Suite | Model cards, BMA ensembles weights, conformal calibration |
| IC Artefact | Plain-language investment committee statement with model lineage |

## API endpoints

- `GET /api/regime/current` — current regime + probabilities
- `GET /api/regime/history` — time series + regime classifications
- `GET /api/regime/crisis` — crisis case study metadata
- `GET /api/models/performance` — model cards, weights, conformal stats
- `GET /api/regime/ic-artefact` — IC statement

## Build

```bash
cd frontend
npm run build   # outputs static bundle to dist/
```
