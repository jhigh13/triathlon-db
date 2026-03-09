# Triathlon Race Prediction — Implementation Scaffolding (WTCS Draft-Legal, Pre‑Race)

This is the *implementation companion* to `triathlon_race_prediction_roadmap.md`.  
Goal: take your existing Postgres schema (events / race_results / position_metrics / wtcs_pack_membership / program_entries) and produce:

- Predicted splits + total time (seconds)
- Predicted finishing order (sorted by predicted total)
- Probabilities for podium / top5 / top10 / top20 (via Monte Carlo)

---

## 1) Training “view” (modeling dataset) — exact columns

**Granularity:** one row per `(event_id, prog_id, athlete_id)` for races that have results.

### 1.1 Keys and metadata
| Column | Type | Source | Notes |
|---|---:|---|---|
| event_id | int | `events`, `race_results` | join key |
| prog_id | int | `events`, `race_results` | join key |
| athlete_id | int | `race_results` | join key |
| event_date | date | `events.event_date` | used for time-based features / leakage guard |
| prog_name | text | `events.prog_name` | optional (WTCS Men/Women etc.) |
| prog_distance_category | text | `events.prog_distance_category` | e.g., “Olympic” |
| event_country | text | `events.event_country` | coarse geography |
| event_venue | text | `events.event_venue` | optional |
| event_latitude | float | `events.event_latitude` | for weather later |
| event_longitude | float | `events.event_longitude` | for weather later |
| wetsuit | text | `events.wetsuit` | categorical if available |

### 1.2 Labels (what you predict)
All labels are seconds. Store as integers.

| Column | Type | Source | Transform |
|---|---:|---|---|
| swim_sec | int | `race_results.swimtime` | parse “hh:mm:ss” / “mm:ss” |
| t1_sec | int | `race_results.t1time` | parse |
| bike_sec | int | `race_results.biketime` | parse |
| t2_sec | int | `race_results.t2time` | parse |
| run_sec | int | `race_results.runtime` | parse |
| total_sec | int | `race_results.total_time` | parse |
| finish_status | text | `race_results.finish_status` | FINISH/DNF/DNS/DSQ/LAP |
| finish_position | int | `race_results.finish_position` | numeric place (finishers only) |
| position_sort | int | `race_results.position_sort` | numeric sort (finishers first) |

### 1.3 Derived label targets (classification)
| Column | Type | Derived from | Notes |
|---|---:|---|---|
| is_finisher | int | `finish_status == "FINISH"` | 1/0 |
| is_podium | int | `finish_position <= 3` | finisher only |
| is_top5 | int | `finish_position <= 5` | finisher only |
| is_top10 | int | `finish_position <= 10` | finisher only |
| is_top20 | int | `finish_position <= 20` | finisher only |

### 1.4 Pre-race athlete form features (NO leakage)
These are computed from races **strictly before** `event_date` for that athlete, optionally filtered to WTCS/World Cup.

**Recommended base set (start here):**
| Feature | Type | Definition |
|---|---:|---|
| races_12m | int | # of prior races in last 365d |
| days_since_last_race | int | `event_date - max(prior_event_date)` |
| ema_total_sec_5 | float | EWMA of total_sec over last 5 prior races |
| ema_swim_sec_5 / ema_bike_sec_5 / ema_run_sec_5 | float | EWMAs by segment |
| last_total_sec | int | most recent total_sec |
| best_total_sec_24m | int | best total_sec in last 730d |
| std_total_sec_24m | float | std dev of total_sec in last 730d |
| dnf_rate_24m | float | DNFs / starts in last 730d |
| trend_total_sec_5 | float | slope (sec/race) over last 5 races |

**Segment strength / relative features:**
| Feature | Type | Definition |
|---|---:|---|
| rel_total_z_24m | float | z-score of athlete total vs event medians of their prior races |
| rel_swim_rank_pct_24m | float | avg swimrank percentile across prior races (from `position_metrics`) |
| rel_run_rank_pct_24m | float | avg runrank percentile across prior races |

> Tip: if you don’t trust `position_metrics` completeness early on, start with split-based features only, then add ranks.

### 1.5 Draft-legal pack features (your edge)
From `wtcs_pack_membership` history (prior races only). Define on the **swim** checkpoint primarily.

| Feature | Type | Definition |
|---|---:|---|
| front_pack_rate | float | % of prior WTCS races where athlete was in pack_id==1 at swim (or within X sec of leader) |
| avg_swim_gap_leader | float | mean `gap_to_leader_sec` at swim |
| p90_swim_gap_leader | float | 90th percentile `gap_to_leader_sec` at swim |
| avg_pack_size_swim | float | mean `pack_size` at swim |
| bike_pack_rate | float | % of races in pack_id==1 at bike |
| avg_bike_gap_leader | float | mean gap at bike checkpoint |

**Optional interaction features:**
| Feature | Type | Definition |
|---|---:|---|
| front_pack_rate_recent | float | same as front_pack_rate but last 365d |
| swim_gap_trend | float | slope of swim gap to leader over last N races |

### 1.6 Field-strength context features (uses start list)
From `program_entries` for the upcoming event/program (entry_type='start', active rows). For each athlete, compute relative-in-field features:

| Feature | Type | Definition |
|---|---:|---|
| seed_total_rank | int | rank of athlete’s `ema_total_sec_5` among entrants (lower = better) |
| seed_total_gap_to_best | float | athlete EMA total minus best EMA total in field |
| field_depth_top10_mean | float | mean of best 10 EMAs in field |
| field_spread_top20 | float | p90 - p10 of EMAs among entrants |
| n_entrants | int | field size |

### 1.7 Minimal feature set to launch (MVP)
If you want the smallest reliable v1:

- `ema_swim_sec_5, ema_bike_sec_5, ema_run_sec_5`
- `std_total_sec_24m`
- `days_since_last_race`
- `front_pack_rate, avg_swim_gap_leader`
- `seed_total_rank, n_entrants`
- plus basic event metadata: distance category, wetsuit flag, country

---

## 2) Model strategy (WTCS draft-legal, pre-race)

### 2.1 First pass (fast + strong)
Train 4 regressors:
- `model_swim`: predict `swim_sec`
- `model_bike`: predict `bike_sec`
- `model_run`: predict `run_sec`
- `model_total`: predict `total_sec` (or sum splits and train only splits)

Then:
- predicted_total = `pred_total` (or `pred_swim+pred_t1+pred_bike+pred_t2+pred_run`)
- predicted finishing order = sort ascending predicted_total

### 2.2 Upgrade: pack-conditional modeling
Two-stage for bike/run:
1) Classifier `p_front_pack` (prob athlete is in front pack at swim)
2) Bike/run regressors use `p_front_pack` (or sampled pack scenario in simulation)

This captures the “make the swim pack or die” WTCS reality.

---

## 3) Monte Carlo simulation setup (probabilities)

### 3.1 What you need
For each athlete:
- mean predictions: `mu_swim, mu_t1, mu_bike, mu_t2, mu_run`
- uncertainty estimates: `sigma_swim, ...` OR `sigma_total`

### 3.2 Uncertainty options (choose one)
**Option A (cheap, good enough):** residual SD by athlete over last 24 months  
- `sigma_total = std(residual_total)` where residual = actual - model_pred on prior races

**Option B (even cheaper):** historical SD of total time (conditioned on distance and WTCS-only subset)

**Option C (best):** model variance (train a second model to predict abs error)

### 3.3 Correlations (simple defaults)
- Swim and bike in WTCS are *positively* correlated via pack access
- Bike and run can be *negatively* correlated (hard bike -> worse run)

MVP:
- sample an overall `delta` factor per athlete and add to all splits (introduces correlation)
- add a pack effect to bike if “front pack” is sampled

### 3.4 Simulation algorithm (per race)
For sim in 1..N:
1) For each athlete, sample `front_pack ~ Bernoulli(p_front_pack)`
2) Sample swim: `swim ~ Normal(mu_swim, sigma_swim)`
3) Sample bike: `bike ~ Normal(mu_bike + pack_bonus(front_pack), sigma_bike)`
4) Sample run: `run ~ Normal(mu_run + run_penalty(front_pack), sigma_run)` (optional)
5) total = sum splits
6) rank all by total
Aggregate across simulations:
- win/podium/topK probabilities
- expected rank, rank CI
- time intervals (p10/p50/p90)

---

## 4) Repo scaffolding (files + responsibilities)

```
tri_analysis/
  prediction/
    __init__.py
    sql.py
    build_dataset.py
    features.py
    train.py
    predict.py
    simulate.py
    evaluate.py
    utils_time.py
    schemas.py
notebooks/
  01_build_dataset.ipynb
  02_feature_sanity_checks.ipynb
  03_train_baseline_models.ipynb
  04_backtest_report.ipynb
```

### 4.1 File responsibilities
- `sql.py`: parameterized SQL queries for pulling training rows & prior history
- `utils_time.py`: parsing time strings -> seconds, plus formatting
- `build_dataset.py`: materialize training table/view as parquet or DB table
- `features.py`: feature computation (athlete history, pack metrics, field context)
- `train.py`: training pipelines + model registry serialization
- `predict.py`: produce deterministic predictions for an upcoming program
- `simulate.py`: Monte Carlo simulation and probabilities
- `evaluate.py`: backtests + metrics + saved reports

---

## 5) Function signatures (Copilot-ready)

### 5.1 SQL extraction
```python
# tri_analysis/prediction/sql.py
from dataclasses import dataclass
from datetime import date
import pandas as pd
from sqlalchemy.engine import Engine

@dataclass(frozen=True)
class ProgramKey:
    event_id: int
    prog_id: int

def fetch_program_results(engine: Engine, key: ProgramKey) -> pd.DataFrame:
    """Returns rows for one (event_id, prog_id) with splits, total, finish fields, joined to events."""

def fetch_athlete_history(engine: Engine, athlete_id: int, before_date: date, limit: int = 50) -> pd.DataFrame:
    """All prior races for athlete before 'before_date' with event_date, splits, totals, and joins needed."""

def fetch_pack_history(engine: Engine, athlete_id: int, before_date: date, limit: int = 50) -> pd.DataFrame:
    """Prior pack membership rows for athlete before 'before_date'."""

def fetch_start_list(engine: Engine, key: ProgramKey) -> pd.DataFrame:
    """Active program_entries rows for (event_id, prog_id) for upcoming races."""
```

### 5.2 Dataset build
```python
# tri_analysis/prediction/build_dataset.py
from sqlalchemy.engine import Engine
import pandas as pd

def build_training_rows(engine: Engine, start_date: str, end_date: str) -> pd.DataFrame:
    """Build base labeled dataset (keys + labels + event metadata) for events in window."""

def materialize_training_dataset(engine: Engine, out_path: str, start_date: str, end_date: str) -> str:
    """Writes parquet/csv and returns path."""
```

### 5.3 Feature engineering
```python
# tri_analysis/prediction/features.py
from datetime import date
import pandas as pd
from .sql import ProgramKey
from sqlalchemy.engine import Engine

def compute_athlete_form_features(history_df: pd.DataFrame, event_date: date) -> dict:
    """Return dict of pre-race athlete features from prior results only (no leakage)."""

def compute_pack_features(pack_df: pd.DataFrame, event_date: date) -> dict:
    """Return dict of draft-legal pack access metrics (swim/bike)."""

def compute_field_context_features(start_list_df: pd.DataFrame, athlete_features_df: pd.DataFrame) -> pd.DataFrame:
    """Adds seed rank + field depth features for each athlete in the upcoming race."""

def build_features_for_program(engine: Engine, key: ProgramKey) -> pd.DataFrame:
    """
    For an upcoming program, returns one row per athlete with all features needed for prediction.
    Must NOT use any data from the target event/program results.
    """
```

### 5.4 Training
```python
# tri_analysis/prediction/train.py
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelBundle:
    model_swim: object
    model_t1: object
    model_bike: object
    model_t2: object
    model_run: object
    model_total: object
    model_front_pack: object | None  # optional classifier
    feature_columns: list[str]
    created_at: str
    version: str

def train_baseline_models(train_df: pd.DataFrame, feature_cols: list[str]) -> ModelBundle:
    """Train regressors for splits/total; optionally classifier for front-pack."""

def save_model_bundle(bundle: ModelBundle, path: str) -> str:
    """Serialize to disk (joblib/pickle) and return path."""

def load_model_bundle(path: str) -> ModelBundle:
    """Load from disk."""
```

### 5.5 Prediction
```python
# tri_analysis/prediction/predict.py
import pandas as pd
from .train import ModelBundle

def predict_splits_and_total(features_df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Adds predicted split seconds + predicted_total_sec + deterministic rank."""

def format_prediction_output(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Add pretty time strings and top-k flags for reporting."""
```

### 5.6 Simulation
```python
# tri_analysis/prediction/simulate.py
import numpy as np
import pandas as pd
from .train import ModelBundle

def estimate_uncertainty(features_df: pd.DataFrame, history_df_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Add sigma columns per athlete (simple method: history residual SD or time SD)."""

def run_monte_carlo(pred_df: pd.DataFrame, n_sims: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Returns per-athlete probabilities: win/podium/top5/top10/top20,
    expected_rank, rank_p10/rank_p90, total_p10/p50/p90.
    """
```

### 5.7 Evaluation
```python
# tri_analysis/prediction/evaluate.py
import pandas as pd

def precision_at_k(pred_order: list[int], true_order: list[int], k: int) -> float:
    """Fraction of true top-k contained in predicted top-k."""

def spearman_rank_corr(pred_rank: pd.Series, true_rank: pd.Series) -> float:
    """Rank correlation (handle ties, DNFs by position_sort)."""

def backtest_events(engine, event_ids: list[int], model_params: dict) -> pd.DataFrame:
    """Runs rolling or leave-one-event-out backtests and returns metrics per event."""
```

---

## 6) Minimal definition of done (v1)
For a held-out WTCS event/program:
- `Precision@10 >= baseline` (baseline = rank by last-12m EMA total only)
- Total MAE improves vs baseline
- Simulation outputs are calibrated: actual winner in top-3 probability bucket reasonably often

---

## 7) Copilot prompts (copy/paste)
Use these to accelerate implementation:

- “Create `utils_time.py` that parses time strings like `mm:ss`, `hh:mm:ss`, and returns seconds; handle nulls and ‘DNS/DNF’ gracefully.”
- “Write SQLAlchemy query in `sql.py` to fetch labeled rows for a given (event_id, prog_id) joining events + race_results + position_metrics + pack membership (optional).”
- “Implement `compute_athlete_form_features()` using only prior races before event_date; include EWMA, last, best, std, trend slope.”
- “Implement `train_baseline_models()` using LightGBM regressors (or sklearn HistGradientBoostingRegressor) and persist with joblib.”

---

## 8) Notes specific to your schema
- Your DB already stores pack membership and checkpoint metrics as deterministic tables; these are perfect for draft-legal features.
- Your program entries table already supports upcoming start lists; use it to compute field-context features pre-race.
