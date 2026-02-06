# Triathlon Race Prediction Model - Improvement Brainstorm

## Current State Analysis

The prediction pipeline lives in `tri_analysis/prediction/` with these core modules:
- **sql.py** - Parameterized SQL queries for fetching results, history, packs, start lists
- **features.py** - Feature engineering (athlete form, pack dynamics, field context)
- **train.py** - Model training (LightGBM / sklearn HistGradientBoosting)
- **predict.py** - Deterministic predictions with anchoring
- **simulate.py** - Monte Carlo simulation for probability estimates

### Root Cause of Accuracy Issues

The **core problem** is that absolute-second features are poisoned by cross-distance mixing:

1. **EMA features mix distances**: `ema_total_sec_5` blends ~3600s (sprint) with ~7200s (standard), creating meaningless averages
2. **No distance feature**: `prog_distance_category` is NOT in the feature set — the model literally cannot distinguish sprint from standard
3. **Broken seed rank**: `seed_total_rank` ranks athletes by raw EMA seconds across different distances
4. **Corrupted volatility**: `std_total_sec_24m` conflates distance variation with actual inconsistency
5. **Hardcoded pack effects**: Monte Carlo uses fixed -30s/+20s rather than learned values

---

## Improvement Categories

### CATEGORY 1: Distance-Agnostic Performance Features (Highest Priority)

No separate models needed. Instead, normalize features across distances.

#### A. Finish Percentile EMA (highest impact, easiest)

For each historical race, compute:
```
finish_pct = finish_position / n_finishers_in_race
```
Then EMA this across races. Naturally distance-agnostic. An athlete finishing 5th/50 in WTCS standard and 3rd/40 in World Cup sprint has percentiles of 0.10 and 0.075 — both meaningful, both comparable.

Currently `ema_swim_pos_pct_7` does something similar for swim position, but there's no overall finish percentile EMA.

#### B. Gap-to-Median Ratio (moderate effort, high impact)

For each historical race, compute:
```
performance_ratio = athlete_total_sec / race_median_total_sec
```
A ratio of 0.95 means 5% faster than median regardless of distance. EMA this. Gives the model a distance-normalized speed signal that captures magnitude (unlike pure rank).

Implementation: store or compute race median at feature-build time via SQL window function or subquery.

#### C. Per-Split Pace Percentiles

Same logic applied per-split:
```
swim_pct = swim_position / n_swimmers
bike_pct = bike_position / n_bikers
run_pct  = run_position / n_runners
```
EMA each. Tells the model "this athlete is typically a top-20% swimmer, mid-pack biker, top-10% runner" regardless of distance.

#### D. Cross-Distance Performance Features

Don't discard cross-distance data — use it explicitly:
```
ema_finish_pct_sprint   = EMA of finish_pct in sprint races only
ema_finish_pct_standard = EMA of finish_pct in standard races only
```
When predicting a sprint, the standard-distance percentile still tells the model something about general ability. The model can learn the correlation coefficient.

---

### CATEGORY 2: Tier-Aware Performance Modeling

Current tier features exist but don't give the model what it needs. `tier_delta` (event_tier - athlete_avg_tier) captures "stepping up/down" but doesn't convey how good an athlete actually is at top-level racing.

**Recommendations:**
- **Tier-weighted performance score**: Weight percentiles by tier (WTCS percentiles count 4x)
- **Elo/Glicko ratings**: Head-to-head rating system that naturally handles tier differences
- **Fix field-relative seed rank**: Rank by percentile EMA, not raw seconds

---

### CATEGORY 3: Target Variable Strategy

**Option A (minimal change):** Add `prog_distance_category` as a feature. Model learns different second ranges per distance.

**Option B (recommended):** Predict finish percentile instead of seconds. Then convert back to seconds using field-level calibration.

**Option C (most powerful):** Two-stage model — Stage 1 predicts percentile, Stage 2 converts to seconds using race-specific context.

---

### CATEGORY 4: Monte Carlo Simulation Improvements

1. **Learn pack effects from data**: Query actual time differentials between front-pack and chase athletes instead of hardcoding -30s/+20s
2. **Continuous gap effects**: Replace binary pack membership with continuous function of swim gap
3. **Causal swim→bike→run chain**: Model the sequential dependency — fast swim → front pack → faster bike → fresh legs → faster run
4. **Distance-specific uncertainty**: Sprint races have smaller absolute variance than standard
5. **Race-day conditions**: Temperature, altitude, wetsuit status affect uncertainty

---

### CATEGORY 5: Training Pipeline Improvements

1. **Hyperparameter tuning**: Grid search or Optuna with proper cross-validation
2. **Time-based cross-validation**: Train on years 1-3, validate on year 4, test on year 5. Never leak future data.
3. **Gender-specific models**: Men's and women's racing dynamics differ significantly (pack sizes, split distributions)

---

## Implementation Priority

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| 1 | Add `prog_distance_category` as feature | High | Low |
| 2 | Finish Percentile EMA | Very High | Low |
| 3 | Gap-to-Median Ratio EMA | High | Medium |
| 4 | Per-Split Pace Percentiles | High | Medium |
| 5 | Cross-Distance Features | High | Medium |
| 6 | Compute empirical pack effects | High | Medium |
| 7 | Elo ratings | Very High | High |
| 8 | Predict percentile as target | Very High | High |
| 9 | Hyperparameter tuning + time-based CV | Medium | Medium |
| 10 | Gender-specific models | Medium | Low |

Items 1-5 (Category 1) form the foundation and should be implemented first.


PRIORITY ORDERING (What I'd implement first)
Add prog_distance_category as a model feature -- 5 minutes, immediate improvement
Add overall finish percentile EMA (ema_finish_pct_5) -- replaces broken ema_total_sec_5 as primary performance signal
Add gap-to-median ratio EMA -- distance-normalized speed signal
Compute empirical pack effects from data -- replace hardcoded constants
Implement causal swim→bike→run simulation chain -- realistic Monte Carlo
Add Elo ratings -- best single feature for cross-distance, cross-tier ranking
Predict percentile as target instead of absolute seconds
Hyperparameter tuning with time-based CV
Gender-specific models
Continuous gap-based pack effects

Claude’s Plan
Category 5: Training Pipeline Improvements
Overview
Three improvements: (1) time-based cross-validation, (2) hyperparameter tuning, (3) gender-specific models. All changes are backward-compatible — existing training/backtest commands work unchanged.

Files to Modify
tri_analysis/prediction/train.py — Core changes: parameterize model, add CV/tuning functions, gender filtering
scripts/train_models.py — CLI flags: --tune, --cv, --gender, --train_both_genders
tri_analysis/prediction/evaluate.py — Gender-aware backtest (accept dict of bundles)
scripts/run_backtest.py — CLI flags: --model_men, --model_women
No changes to: features.py, predict.py, sql.py, simulate.py

Step 1: Parameterize create_regressor() (train.py)
Add DEFAULT_PARAMS dict and optional params argument:


DEFAULT_PARAMS = {
    "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
    "num_leaves": 31, "min_child_samples": 20,
}

def create_regressor(params: dict | None = None) -> Pipeline:
    p = {**DEFAULT_PARAMS, **(params or {})}
    # Use p values instead of hardcoded constants
Add model_params: dict | None = None to train_baseline_models(), forward to create_regressor(params=model_params). Store params in bundle.metadata["model_params"].

Step 2: Time-Based CV Splits (train.py)
Add time_based_cv_splits() — pure in-memory date filtering on cached DataFrame:


Fold 1: Train ≤2021-12-31, Val 2022
Fold 2: Train ≤2022-12-31, Val 2023
Fold 3: Train ≤2023-12-31, Val 2024H1
Fold 4: Train ≤2024-06-30, Val 2024H2
Returns list[tuple[train_df, val_df]]. No DB calls — slices the already-built training DataFrame.

Step 3: Cross-Validation Function (train.py)
Add cross_validate() that:

Splits data using time_based_cv_splits()
For each fold: trains a single finish_pct model (fast — one model, not five)
Groups val predictions by (event_id, prog_id) and computes per-race Spearman
Returns {"mean_spearman": float, "mean_mae": float, "fold_results": [...]}
Key: evaluation mirrors backtest logic but skips feature building (already in DataFrame).

Step 4: Hyperparameter Tuning (train.py)
Add tune_hyperparameters() — grid search using cross_validate():


DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [10, 20, 50],
}
54 combinations x 4 folds = 216 model fits. Each is a single LightGBM on ~20K rows (~1-2 sec). Total: ~3-7 minutes.

Optimizes for Spearman on percentile model only (drives ranking metrics). Returns {"best_params": dict, "best_score": float, "all_results": [...]}. Logs top-5 configs.

Step 5: Gender Filtering (train.py)
Add gender: str | None = None parameter to build_training_dataset(). After fetching programs, filter:


if gender == "men": programs_df = programs_df[programs_df["prog_name"] == "Elite Men"]
if gender == "women": programs_df = programs_df[programs_df["prog_name"] == "Elite Women"]
Step 6: CLI Updates (train_models.py)
New flags: --tune, --cv, --gender {men,women}, --train_both_genders

Training loop iterates over genders. For --train_both_genders, outputs bundle_men.joblib and bundle_women.joblib (appends gender suffix to --output path).

Step 7: Gender-Aware Backtest (evaluate.py)
Update backtest_events() to accept bundle_path: str | dict[str, str]. When dict, selects bundle per event by checking prog_name for "women".

Step 8: Backtest CLI (run_backtest.py)
New flags: --model_men, --model_women. When both provided, passes {"men": path, "women": path} dict to backtest_events().

Usage After Implementation

# Existing behavior (unchanged)
python scripts/train_models.py --output models/bundle.joblib
python scripts/run_backtest.py --model models/bundle.joblib

# Cross-validate current defaults
python scripts/train_models.py --cv

# Tune + train
python scripts/train_models.py --tune --output models/bundle_tuned.joblib

# Gender-specific with tuning
python scripts/train_models.py --tune --train_both_genders --output models/bundle_v7.joblib
# -> models/bundle_v7_men.joblib, models/bundle_v7_women.joblib

# Backtest gender-specific
python scripts/run_backtest.py --model_men models/bundle_v7_men.joblib --model_women models/bundle_v7_women.joblib
Verification
Run python scripts/train_models.py --output models/test.joblib — should produce identical results to current behavior
Run python scripts/train_models.py --cv — should print per-fold Spearman/MAE
Run python scripts/train_models.py --tune --output models/tuned.joblib — should print best params and top-5 configs
Run python scripts/run_backtest.py --model models/tuned.joblib — compare to current 72.5% P@10
Run python scripts/train_models.py --tune --train_both_genders --output models/v7.joblib
Run python scripts/run_backtest.py --model_men models/v7_men.joblib --model_women models/v7_women.joblib — compare to unified model

# Existing behavior (unchanged)
python scripts/train_models.py --output models/bundle.joblib

# Cross-validate current defaults
python scripts/train_models.py --cv

# Tune + train
python scripts/train_models.py --tune --output models/bundle_tuned.joblib

# Gender-specific with tuning
python scripts/train_models.py --tune --train_both_genders --output models/bundle_v7.joblib
# -> models/bundle_v7_men.joblib, models/bundle_v7_women.joblib

# Backtest gender-specific
python scripts/run_backtest.py --model_men models/bundle_v7_men.joblib --model_women models/bundle_v7_women.joblib