# triathlon-db

## Overview
World Triathlon race database, analytics, and ML prediction system. Scrapes race results from the World Triathlon API, stores them in PostgreSQL, and trains gradient boosting models to predict race outcomes with Monte Carlo simulation. Current champion model: v45 (P@10=74.2%, P@3=57.0%, Spearman=0.796).

## Tech Stack
- **Database**: PostgreSQL (Supabase) via SQLAlchemy 2.0
- **ML**: 50/50 ensemble of LGBMRanker + scikit-learn HistGradientBoostingRegressor (finish percentile target)
- **Simulation**: Monte Carlo with MVN noise from learned residual covariance
- **Visualization**: Streamlit dashboard (`streamlit_app.py`), Plotly, Matplotlib
- **Data**: Pandas, NumPy, Parquet files
- **Experiment tracking**: MLflow (`mlruns/`)
- **Python**: 3.13+, virtual env at `.venv/`

## Project Structure
```
tri_analysis/                   # Main Python package
├── prediction/                 # ML prediction subsystem
│   ├── features.py             # Feature engineering (EWMA form, tier percentiles, pack metrics, 80+ features)
│   ├── train.py                # Model training, ModelBundle persistence, CV, hyperparameter tuning
│   ├── predict.py              # Deterministic predictions (50/50 ranker+percentile ensemble)
│   ├── simulate.py             # Monte Carlo simulation (causal chain, MVN noise, pack dynamics)
│   ├── evaluate.py             # Backtesting (Spearman, precision@K, MAE, tier-specific)
│   ├── sql.py                  # Parameterized DB queries for prediction data
│   └── utils_time.py           # Time parsing utilities
├── api_handling.py             # World Triathlon API client
├── database.py                 # DB connection and queries (get_engine())
├── build_database.py           # ETL pipeline (API -> Postgres)
├── metrics.py                  # Position metric calculations
├── h2h_analysis.py             # Head-to-head comparisons
├── elo_ratings.py              # Elo rating system (tier-weighted K factors)
├── wtcs_pack_metrics.py        # Pack dynamics (two-threshold detection)
├── wtcs_performance.py         # WTCS event analytics
├── wtcs_radar.py               # Radar chart visualizations
├── relative_bike_metrics.py    # Bike performance relative to pack
└── weather.py                  # Weather data integration

scripts/                        # Executable entry points
├── train_models.py             # Train and save model bundles
├── predict_program.py          # Run predictions on upcoming events
├── run_backtest.py             # Backtesting runner
├── debug_diagnostics.py        # Unified diagnostic tool (--section overview|importances|field|accounting|simulation|athlete)
├── analyze_predictions.py      # Backtest accuracy analysis
├── error_analysis.py           # Residual analysis by athlete/event
├── residual_analysis.py        # Statistical analysis of prediction errors
├── sweep_ensemble_weight.py    # Ensemble weight tuning
├── sweep_3way_weights.py       # 3-model ensemble weighting
├── sweep_fieldsize_weights.py  # Field-size weight tuning
├── ablation_pack_features.py   # Pack feature ablation study
├── find_events.py              # Find upcoming events
├── backfill_weather.py         # Populate weather data
└── check_tiers.py              # Validate event tier classifications

para_triathlon_analysis/        # Active: para standards reporting (has own README)
models/                         # Trained model bundles (joblib)
data/                           # Datasets (Excel detailed results)
outputs/                        # Prediction CSVs, training logs, charts
docs/                           # Documentation
├── prediction_status.md        # Current model state, metrics, next steps
├── experiment-log.md           # Experiment tracking table
├── model_improvement_brainstorm.md  # Idea backlog with status markers
├── memory.md                   # Legacy change log
└── archive/                    # Archived: old roadmap, scaffolding, full improvement history
!archive/                       # Archived subprojects (Mixed_Relay, proj_pod)
```

## Prediction Pipeline
1. **features.py**: EWMA form (3/5/12 race windows), tier-conditioned split percentiles, pack access rates, field depth/seed rank, distance category, 80+ total features
2. **train.py**: Multi-target regression (swim/t1/bike/t2/run/total) + LGBMRanker + percentile model. Tier sample weights {1: 8.0, 2: 3.0, 3: 1.5, 4: 1.0}. Time-decay weighting. Persisted as `ModelBundle` via joblib
3. **predict.py**: 50/50 ensemble (ranker ranks + percentile finish position). Prediction anchoring caps at 110% of athlete's EMA. Deterministic ranking output
4. **simulate.py**: Monte Carlo (10k sims) with MVN noise from learned residual covariance. Causal chain: Swim -> T1 -> Pack Formation -> Bike (with drafting) -> T2 -> Run. Pack effects learned from data per distance
5. **evaluate.py**: Spearman rank correlation, precision@K (3/5/10/20), MAE per split, tier-specific backtesting, start-number baseline comparison

## Model Versioning
- **Current champion**: `models/bundle_elite_v45.joblib`
- **Naming convention**: `bundle_elite_v{N}.joblib` (v28+). Older: `bundle_claudev{N}_{men|women}.joblib` (archived)
- **Rule**: Always increment version number. Never overwrite a model file
- **Training cutoff**: 2025-06-30 (H2 2025 is held out for backtesting)
- **Backtest period**: H2 2025 (~90 events)

## Key Conventions
- **Event tiers**: 1a (WTCS/Championship Series), 1b (T100), 2 (World Cup), 3 (Continental Cup), 4 (Other)
- **Tier sample weights**: {1: 8.0, 2: 3.0, 3: 1.5, 4: 1.0}
- **Pack formation**: Chain-rule algorithm, two-threshold (initial gap=2s, continuation=1s)
- **Training data**: Never train past 2025-06-30 to avoid backtest leakage
- **DB queries**: All through SQLAlchemy sessions via `get_engine()` from `tri_analysis.database`
- **MC sim limitation**: Deterministic model always beats MC for ranking accuracy (19pt P@3 gap). MC value is in probability outputs only

## Workflow Commands
```powershell
# Train new model (always increment version)
python scripts/train_models.py --output models/bundle_elite_v{N}.joblib [--tune] [--cv] [--cv_splits N]

# Backtest (deterministic only -- faster, primary metric)
python scripts/run_backtest.py --model models/bundle_elite_v{N}.joblib --no_sim

# Backtest (with MC simulation)
python scripts/run_backtest.py --model models/bundle_elite_v{N}.joblib

# Predict upcoming event
python scripts/predict_program.py --event_id {ID} --prog_id {ID} --model_path models/bundle_elite_v{N}.joblib

# Find upcoming events
python scripts/find_events.py

# Diagnostics for a specific event
python scripts/debug_diagnostics.py --event_id {ID} --prog_id {ID} --model_path models/bundle_elite_v{N}.joblib --section overview
```

## Key Findings (from 45+ model iterations)
- **Pack features are high-leverage**: +25% P@10 vs no-pack baseline
- **MC can't match deterministic for ranking**: Structural 19pt P@3 gap. Use deterministic for rankings, MC for probabilities
- **Large-field accuracy drops**: 50+ athletes -> P@10=64% vs 82% for small fields. Cold-start athletes are the main cause
- **Tier weighting trades off**: 8x WTCS weight improved WTCS P@3 by +10% but cost Tier 2 -4.7%
- **Ensemble plateau**: ~74% P@10 is the ceiling for current architecture. Further gains need architectural changes (group-first bike model, hierarchical ability model, better cold-start handling)

## Slash Commands
- `/train` -- Train a new model version with auto-versioning and validation
- `/backtest` -- Run backtest and compare against champion with PROMOTE/REJECT verdict
- `/predict` -- Predict an upcoming event with deterministic + MC simulation
- `/experiment` -- Full autonomous cycle: implement change -> train -> backtest -> evaluate -> log
- `/diagnostics` -- Run diagnostic analysis on predictions for a specific event

## Environment
- `.env`: `DATABASE_URL` (Supabase Postgres), `TRI_API_KEY` (World Triathlon API)
- Experiment log: `docs/experiment-log.md`
- Prediction status: `docs/prediction_status.md`
- Idea backlog: `docs/model_improvement_brainstorm.md`

## Cross-Repo Integration
- **PodiumDashboard**: `../PodiumDashboard/PodiumDashboard/` -- FastAPI+HTMX coaching dashboard
- **Para Standards**: `para_triathlon_analysis/` -- active, has own README and scripts

## Shell
Use PowerShell for all terminal commands on this Windows machine.
