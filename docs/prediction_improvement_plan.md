# Prediction & Simulation Accuracy Improvement Plan

## Context

The triathlon prediction pipeline (`triathlon-db/tri_analysis/prediction/`) achieves >70% top-10 precision on the deterministic model, but Monte Carlo simulation results feel unreliable. Root cause analysis reveals several compounding issues: missing transition modeling, split-total accounting mismatches, potential pack-effect double-counting, and uncalibrated noise. This plan addresses all of them in priority order, with each phase independently testable.

**All changes are in**: `tri_analysis/prediction/`
**Integration touchpoint**: `PodiumDashboard/app/services/prediction.py`

---

## Phase 1: Fix T1/T2 Transition Accounting (Highest Impact) — COMPLETED

**Problem**: `pred_total_sec` includes T1+T2 (~30-300s) but split models only predict swim/bike/run. The simulation normalizes splits to match total, silently inflating each leg with transition seconds and distorting split ratios and pack formation.

**Results (v26)**: P@10=71.95%, Spearman=0.782, MAE=365s, split-total gap=0.0s

### 1A. Add Transition Features to `features.py` — DONE

- [x] Parse `t1time` and `t2time` to seconds in `compute_athlete_form_features()`
- [x] Compute `ema_t1_sec_5`, `ema_t2_sec_5`, `ema_t1t2_sec_5`, `std_t1t2_sec_24m`
- [x] Add T1/T2 to `EMA_SPLIT_PLAUSIBLE_RANGES`
- [x] Add to both `get_feature_columns()` and `get_split_feature_columns()`
- [x] Add T1/T2 defaults to `fill_missing_features()`

### 1B. Train T1/T2 Models in `train.py` — DONE

- [x] Train `model_t1` and `model_t2` in `train_baseline_models()`
- [x] Train distance-specific T1/T2 in `train_distance_split_models()`
- [x] Train distance-specific total models per distance

### 1C. Predict T1/T2 in `predict.py` — DONE

- [x] Add `pred_t1_sec` and `pred_t2_sec` predictions
- [x] T1/T2 fallback from total model budget when models unavailable
- [x] Use split sum as `pred_total_sec` when all 5 splits available (eliminates gap)
- [x] Add `pred_t1_hms`, `pred_t2_hms` to output

### 1D. Fix Simulation Split Accounting in `simulate.py` — DONE

- [x] Remove normalization block
- [x] Add T1/T2 to causal chain: `total = swim + T1 + bike + T2 + run`
- [x] Add T1/T2 noise (`DEFAULT_SIGMA_T1=5.0`, `DEFAULT_SIGMA_T2=4.0`)
- [x] Update `estimate_uncertainty()` with `sigma_t1` and `sigma_t2`

### 1E. Use Swim + T1 for Pack Formation — DONE

- [x] Pack formation uses `sim_swim + sim_t1` instead of `sim_swim` alone
- [x] Distance-specific pack params (sprint=5s gap, standard=3s gap)
- [x] Distance-specific pack effect learning (sprint, standard separately)
- [x] Dynamic pack merging with distance-specific merge params

### 1F. Additional Phase 1 Fixes (v25-v26)

- [x] Distance-specific total models in `train_distance_split_models()`
- [x] Fix pack display in `predict_program.py` to use `apply_pack_merges()`
- [x] Defensive `pd.to_numeric()` guards in predict.py and features.py for backtest error
- [x] Full traceback logging in evaluate.py backtest except block
- [x] Consolidated debug script (`scripts/debug_diagnostics.py`)

---

## Phase 2: Empirical Residual Covariance, Calibrated Uncertainty & Pack Rate Fix — COMPLETED

**Problem**: Current simulation uses independent noise (90%) + shared `form_factor` (10%) with hardcoded split sigmas (`DEFAULT_SIGMA_SWIM=15, BIKE=45, RUN=30`). Real residuals are correlated (fast swimmers tend to have fast T1s) and the hardcoded sigmas may not match empirical prediction errors. Additionally, `front_pack_rate` is distorted by distance filtering that excludes relevant cross-distance pack data.

**Results (v27)**: P@10=72.18%, Spearman=0.776, MAE=369.1s. MVN noise model active with learned 5×5 covariance. Empirical sigmas (sprint): swim=51.6s, t1=23.1s, bike=129.2s, t2=9.4s, run=80.2s.

---

### 2A. Compute Empirical Residual Covariance Matrix — DONE

**File**: `train.py` — new function `compute_residual_stats()`
**Called from**: `scripts/train_models.py` after split model training

**Implementation**:
1. Add function in `train.py` (~50 lines):
   ```python
   def compute_residual_stats(
       train_df: pd.DataFrame,
       bundle: ModelBundle,
       feature_cols: list[str],
       split_feature_cols: list[str],
   ) -> dict:
   ```
2. For each distance that has trained split models (sprint, standard):
   - Filter `train_df` to that distance
   - Predict all 5 splits using the distance-specific models (swim, t1, bike, t2, run)
   - Compute residuals: `residual[split] = actual[split] - predicted[split]`
   - Drop rows where any split is NaN (need complete 5-split vectors)
   - Build 5x5 covariance matrix: `np.cov(residuals.T)` -> shape (5, 5)
   - Extract per-split std: `np.sqrt(np.diag(cov_matrix))`
   - Also compute overall (all distances combined) as fallback
3. Return dict structure:
   ```python
   {
       "overall": {"cov_matrix": [[...]], "per_split_sigma": {"swim": 12.3, ...}, "n_samples": 4500},
       "sprint": {"cov_matrix": [[...]], "per_split_sigma": {"swim": 8.1, ...}, "n_samples": 2100},
       "standard": {"cov_matrix": [[...]], "per_split_sigma": {"swim": 14.2, ...}, "n_samples": 1800},
   }
   ```

**Store in bundle**: `bundle.metadata["residual_stats"]`

**In `scripts/train_models.py`** (~10 lines after line 287):
- Call `compute_residual_stats()` after distance split models are trained
- Store result in bundle metadata
- Log per-split sigmas to MLflow

---

### 2B. Replace Independent Noise with Multivariate Normal Sampling — DONE

**File**: `simulate.py` — modify `run_monte_carlo()` (lines 1029-1088)

**Current noise model** (lines 1032-1080):
```python
form_factor = rng.normal(0, 1, size=n_athletes)        # shared
swim_form = form_factor * sigma_swim * form_std          # 10%
swim_noise = rng.normal(0, sigma_swim * noise_std)       # 90%
sim_swim = pred_swim + swim_form + swim_noise
# ... repeat for t1, bike, t2, run independently
```

**New noise model**:
```python
# Load 5x5 covariance matrix from bundle metadata
cov_matrix = _get_residual_cov(bundle_metadata, distance_category)  # 5x5

# Per-athlete scaling: scale cov by (athlete_sigma / empirical_sigma)^2
# This preserves correlation structure while adjusting magnitude per athlete
athlete_scale = sigma_total / empirical_sigma_total  # per-athlete vector

# Each simulation: sample correlated noise for all athletes at once
for sim in range(n_sims):
    # Sample noise: shape (n_athletes, 5)
    raw_noise = rng.multivariate_normal(np.zeros(5), cov_matrix, size=n_athletes)
    # Scale per athlete
    noise = raw_noise * athlete_scale[:, np.newaxis]

    sim_swim = pred_swim + noise[:, 0]
    sim_t1   = pred_t1   + noise[:, 1]
    sim_bike = pred_bike + noise[:, 2]  # + pack adjustment (unchanged)
    sim_t2   = pred_t2   + noise[:, 3]
    sim_run  = pred_run  + noise[:, 4]
```

**Key design decisions**:
- `form_share` parameter is **retired** — the covariance matrix naturally captures shared variance (off-diagonal elements encode swim-run correlation etc.)
- Keep `form_share` parameter in function signature for backward compatibility but ignore it when covariance matrix is available
- Fallback: if no covariance matrix in metadata, use current `form_factor + independent noise` model (backward compatible)

**Helper function** (~15 lines):
```python
def _get_residual_cov(metadata: dict, distance_category: str | None) -> np.ndarray | None:
    """Retrieve 5x5 residual covariance matrix from bundle metadata."""
```
- Looks up distance-specific first, falls back to overall
- Returns None if not available (triggers legacy noise model)

---

### 2C. Per-Split Sigma from Empirical Residuals — DONE

**File**: `simulate.py` — modify `estimate_uncertainty()` (lines 818-872)

**Current approach** (lines 866-870):
```python
# Hardcoded defaults scaled by distance and per-athlete ratio
ratio = df["sigma_total"] / (DEFAULT_SIGMA_TOTAL * dist_mult)
df["sigma_swim"] = DEFAULT_SIGMA_SWIM * dist_mult * ratio
df["sigma_bike"] = DEFAULT_SIGMA_BIKE * dist_mult * ratio
# ...
```

**New approach**:
1. Add `bundle_metadata` parameter to `estimate_uncertainty()`
2. If `residual_stats` available in metadata:
   - Use learned `per_split_sigma` instead of `DEFAULT_SIGMA_*`
   - Use learned per-distance sigmas when available
   - Distance multiplier (`DISTANCE_SIGMA_MULTIPLIER`) is **retired** for distances with learned sigmas (the learned values already encode the distance-specific variance)
3. Keep per-athlete scaling via `std_total_sec_24m` ratio (heteroscedastic) — this is valuable and should remain
4. If no learned sigmas, fall back to current hardcoded defaults (backward compatible)

**Updated signature**:
```python
def estimate_uncertainty(
    pred_df, sigma_total_col="std_total_sec_24m", default_sigma=90.0,
    distance_category=None, bundle_metadata=None,  # NEW
) -> pd.DataFrame:
```

**Propagation**: Update callers of `estimate_uncertainty()`:
- `run_monte_carlo()` in simulate.py (internal)
- `predict_program.py` (if called directly)

---

### 2D. Fix Front Pack Rate Distance Filtering — DONE

**File**: `features.py` — modify `build_features_for_program()` (lines 937-949)

**Current code** (lines 943-949):
```python
pack_df = fetch_pack_history(engine, athlete_id, event_date, limit=100,
                             distance_category=distance_category)
# Fallback ONLY if completely empty
if pack_df.empty and distance_category:
    pack_df = fetch_pack_history(engine, athlete_id, event_date, limit=100)
```

**Fix**: Lower the threshold from "empty" to "< 3 records":
```python
pack_df = fetch_pack_history(engine, athlete_id, event_date, limit=100,
                             distance_category=distance_category)
# Fallback when too few records for reliable statistics
if len(pack_df) < 3 and distance_category:
    pack_df = fetch_pack_history(engine, athlete_id, event_date, limit=100)
```

This ensures athletes like Blake Bullard — fast swimmers with limited sprint-specific pack history but abundant cross-distance pack data — get a representative `front_pack_rate`.

---

### 2E. Coverage Calibration in Backtest — DONE

**File**: `evaluate.py` — new function `compute_simulation_calibration()`

**Implementation** (~40 lines):
```python
def compute_simulation_calibration(
    sim_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> dict:
    """Check what fraction of actual results fall within sim [p10, p90]."""
```

1. Merge sim_df with actual results on `athlete_id`
2. For total time: check `total_p10 <= actual_total <= total_p90`
3. For rank: check `rank_p10 <= actual_rank <= rank_p90`
4. Target: ~80% coverage (if higher = uncertainty too wide, if lower = too narrow)
5. Return `{"time_coverage_80": 0.78, "rank_coverage_80": 0.82, ...}`

---

### Phase 2 Files Modified

| File | Changes |
|------|---------|
| `train.py` | New `compute_residual_stats()` function |
| `simulate.py` | MVN noise model in `run_monte_carlo()`, updated `estimate_uncertainty()`, new `_get_residual_cov()` helper |
| `features.py` | Pack history fallback threshold (`pack_df.empty` -> `len(pack_df) < 3`) |
| `evaluate.py` | New `compute_simulation_calibration()` function |
| `scripts/train_models.py` | Call `compute_residual_stats()`, store in bundle, log to MLflow |
| `scripts/predict_program.py` | Pass `bundle.metadata` through to simulation (if needed) |
| `app/services/prediction.py` | Pass `bundle.metadata` to simulation calls (if needed) |

### Phase 2 Verification

1. **Retrain model**: `python scripts/train_models.py` — produces bundle with `residual_stats` in metadata
2. **Inspect covariance**: Check that off-diagonal elements make sense (swim-t1 should be positive, swim-run may be weakly positive)
3. **Run predict_program.py**: Compare sim output with MVN vs legacy noise — should see:
   - Smoother rank distributions (less "spiky")
   - Better sim median - det total consistency
   - More realistic win probability spreads
4. **Backtest**: Run `evaluate.py` and check P@10, Spearman, MAE haven't regressed
5. **Calibration**: Run `compute_simulation_calibration()` on backtest — target ~80% coverage for [p10, p90]
6. **Blake check**: Verify front_pack_rate is higher with the threshold fix

---

## Phase 3: Resolve Pack Effect Double-Counting — IN PROGRESS

**Problem**: Split models already use `ema_bike_pack_7`, `ema_bike_gap_sec_7` etc. as features, so predicted bike time already partially accounts for the athlete's typical pack positioning. The simulation then adds pack effects on top.

**Evidence (v27 diagnostics)**: Sprint bike model has `ema_bike_pack_7` as feature importance #3 (142.0), `ema_bike_gap_sec_7` at #8 (111.0), `ema_bike_pos_pct_7` at #9 (109.0) — confirming pack features are heavily used.

### 3A. Quantify the Double-Counting

- [x] Ablation study: train split models WITH vs WITHOUT pack features
- [x] Measure how much split predictions change when pack features are removed
- [x] Script: `scripts/ablation_pack_features.py` — trains both variants, backtests, compares

### 3B. Implement Clean Separation (Data-Driven Decision)

Train and backtest BOTH approaches, pick the winner:

**Option A — "Ability-based" split models**: IMPLEMENTED
- [x] `PACK_DYNAMIC_FEATURES` constant in features.py
- [x] `get_split_feature_columns(exclude_pack=True)` flag
- [x] `train_distance_split_models(exclude_pack_features=True)` flag
- [x] `train_models.py --exclude_pack_features` CLI flag
- [x] `bundle.metadata["exclude_pack_features"]` stored in bundle

**Option B — "Realized" split models with reduced sim effects**: IMPLEMENTED
- [x] `pack_effect_scale` in `run_monte_carlo()` — auto-derived from bundle metadata
- [x] When models include pack features: `pack_effect_scale=0.5` (half effect)
- [x] When models exclude pack features: `pack_effect_scale=1.0` (full effect)

**Next**: Run `python scripts/ablation_pack_features.py --gender men` to compare.
Pick whichever gives better P@10 + Spearman on H2 2025.

---

## Phase 4: Feature Engineering Improvements

### 4A. Enrich `days_since_last_race`

- [ ] Add `log1p_days_since` = `log(1 + days_since_last_race)`
- [ ] Add `days_since_bucket` (0-14, 15-28, 29-56, 57-112, 113+) as ordinal
- [ ] Add `freshness_curve` = `abs(days_since_last_race - 21)` (U-shape)

### 4B. Position Change / T1-T2 Performance Features

- [ ] New SQL query `fetch_position_metrics_history()` from `position_metrics` table
- [ ] `ema_swim_to_t1_pos_change_5`, `ema_t1_to_bike_pos_change_5`
- [ ] `ema_t1_rank_pct_5`, `ema_t2_rank_pct_5`

### 4C. Weather/Conditions Features

- [ ] Pull `temperature_air`, `humidity`, `wbgt`, `wind` from events table
- [ ] Binary indicators: `is_hot` (wbgt > 25), `is_cold` (temp < 15)

### 4D. Course Laps Feature

- [ ] Add `bike_laps`, `run_laps` from events table

---

## Phase 5: Model Training Improvements — COMPLETED

### 5A. Hyperparameter Re-Tuning — DONE

- [x] Default `n_estimators` increased to 200, grid range [200, 350, 500]
- [x] `max_depth` grid: [4, 6, 8]
- [x] Added `reg_alpha` (L1) and `reg_lambda` (L2) to `DEFAULT_PARAMS` and `create_regressor()`
- [x] Default `reg_alpha=0.1`, `reg_lambda=1.0`; grid searches [0, 0.1, 1.0] / [0, 1, 5]
- [x] LightGBM gets both `reg_alpha`/`reg_lambda`; sklearn gets `l2_regularization`

### 5B. Per-Split Cross-Validation — DONE

- [x] New `cross_validate_splits()` function in `train.py`
- [x] Trains distance-specific split models per fold, evaluates per-split MAE on validation
- [x] Computes split-total consistency metric (split sum MAE vs actual total)
- [x] New `--cv_splits` CLI flag in `train_models.py`

### 5C. Updated Training Date Range — DONE

- [x] Default `--end_date` updated to `2025-12-31`
- [x] Time-decay weighting via `_compute_time_decay_weights()` (exponential, half_life=365 days)
- [x] Applied to both `train_baseline_models()` and `train_distance_split_models()`
- [x] Multiplies with tier weights (recent WTCS > old continental cups)
- [x] `--time_decay_half_life` and `--no_time_decay` CLI flags

---

## Phase 6: Simulation Refinement (After Phases 1-5) — COMPLETED

### 6A. Split-Level Anchoring Consistency — DONE

- [x] Enhanced diagnostic in `predict.py`: warns when |delta| > 60s per athlete
- [x] Logs worst-case athlete name and delta
- [x] Stores `pred_split_sum_sec` and `pred_split_total_delta` columns for downstream

### 6B. Simulation Diagnostics Output — DONE

- [x] Per-split bias tracking (sim mean - prediction) accumulated during sim loop
- [x] Pack count histogram: tracks number of distinct packs per simulation
- [x] `_build_sim_diagnostics()` computes: per_split_bias, total_bias, front_pack_rate_comparison (hist vs sim mean + correlation), avg_packs_per_sim, pack_count_distribution
- [x] Diagnostics stored in `sim_df.attrs["sim_diagnostics"]`
- [x] Public `print_sim_diagnostics()` for formatted console output
- [x] Wired into `predict_program.py` after simulation

---

## Implementation Order & Dependencies

```
Phase 1 (T1/T2) --- COMPLETED (v26)
  |
Phase 2 (Covariance + Calibration) --- COMPLETED (v27)
  |
Phase 3 (Pack double-count) --- COMPLETED (v28)
  |
Phase 4 (Features) --- COMPLETED (v29)
  |
Phase 5 (Training) --- COMPLETED
  |
Phase 6 (Sim refinement) --- COMPLETED
```

## Metrics Tracking

| Version | P@10 | Spearman | MAE (s) | Split-Total Gap | Notes |
|---------|------|----------|---------|-----------------|-------|
| v23 (baseline) | 71.49% | 0.783 | 362.1 | ~125s | Pre-T1/T2 |
| v24 (Phase 1) | 71.61% | 0.781 | 368.7 | ~125s | T1/T2 added |
| v25 (dist-specific) | 71.61% | 0.781 | 368.7 | ~181s | Dist total model overfit |
| v26 (split sum) | 71.95% | 0.782 | 365.0 | 0.0s | Split sum as total |
| v27 (Phase 2) | 72.18% | 0.776 | 369.1 | 0.0s | MVN noise, learned sigmas, pack rate fix |


## Other Ideas 

- model for each tier, different dynamics at various levels of racing. 
