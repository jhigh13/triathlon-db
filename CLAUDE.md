# triathlon-db

## Overview
World Triathlon race database, analytics, and ML prediction system. Scrapes race results from the World Triathlon API, stores them in PostgreSQL, and trains gradient boosting models to predict race outcomes with Monte Carlo simulation.

## Tech Stack
- **Database**: PostgreSQL (Supabase) via SQLAlchemy
- **ML**: scikit-learn HistGradientBoostingRegressor (LightGBM fallback)
- **Visualization**: Streamlit dashboard (`streamlit_app.py`), Plotly, Matplotlib
- **Data**: Pandas, NumPy, Parquet files
- **Experiment tracking**: MLflow (`mlruns/`)
- **Python**: 3.13+, virtual env at `.venv/`

## Project Structure
```
tri_analysis/              # Main Python package
├── prediction/            # ML prediction subsystem
│   ├── features.py        # Feature engineering (EWMA, tiers, pack metrics)
│   ├── train.py           # Model training + ModelBundle persistence
│   ├── simulate.py        # Monte Carlo simulation (pack dynamics, drafting)
│   ├── predict.py         # Deterministic predictions
│   ├── evaluate.py        # Backtesting and accuracy metrics
│   ├── sql.py             # DB queries for prediction data
│   └── utils_time.py      # Time parsing utilities
├── api_handling.py        # World Triathlon API client
├── database.py            # DB connection and queries
├── build_database.py      # ETL pipeline
├── metrics.py             # Performance metric calculations
├── h2h_analysis.py        # Head-to-head comparisons
├── elo_ratings.py         # Elo rating system
├── wtcs_pack_metrics.py   # Pack dynamics analysis
├── wtcs_performance.py    # WTCS event analytics
└── wtcs_radar.py          # Radar chart visualizations

scripts/                   # Executable entry points
├── train_models.py        # Train and save model bundles
├── predict_program.py     # Run predictions on upcoming events
├── analyze_predictions.py # Backtest accuracy analysis
└── run_backtest.py        # Backtesting runner

models/                    # Trained model bundles (joblib)
data/                      # Datasets (Excel, Parquet)
outputs/                   # Prediction CSV exports
docs/                      # Documentation, Power BI reports
```

## Key Prediction Pipeline
1. **Features** (`features.py`): EWMA form over last 5 races, event tier classification, pack membership history, field depth metrics
2. **Training** (`train.py`): Multi-target regression (swim/bike/run/total), persisted as `ModelBundle` via joblib
3. **Simulation** (`simulate.py`): Monte Carlo with causal chain (Swim → Pack Formation → Bike with drafting → Run → Total), shared form factor for correlation
4. **Evaluation** (`evaluate.py`): Spearman correlation, MAE, precision@k

## Running Scripts
```powershell
python scripts/train_models.py
python scripts/predict_program.py
python scripts/run_backtest.py
```

## Key Conventions
- Gender-specific models: `bundle_*_men.joblib` / `bundle_*_women.joblib`
- Model versions: latest is v21
- Event tiers: 1 (Olympics/WTCS Finals) through 4 (Continental Cups)
- Pack formation uses chain-rule algorithm (consecutive gap > 2s = new pack)
- All DB queries go through SQLAlchemy sessions

## Environment Variables
Defined in `.env` (see `.env.example`):
- `DATABASE_URL` — Supabase Postgres (triathlon race data)
- `TRI_API_KEY` — World Triathlon API key

## Cross-Repo Integration: PodiumDashboard
The PodiumDashboard repo (sibling at `../PodiumDashboard/PodiumDashboard/`) is a FastAPI+HTMX coaching dashboard that will integrate the prediction/simulation capabilities from this repo to provide a UI-friendly interface for race predictions.

## Shell
Use PowerShell for all terminal commands on this Windows machine.
