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

## Phase 7: MC Sigma Calibration & Wind Feature Fix (v34-v35) — IN PROGRESS

### Context: v33-v34 Diagnostic Findings

Backtest results (v34, 90 events H2 2025) revealed the MC simulation dramatically underperforms the deterministic model:

| Metric | Deterministic | MC Sim (Packs) | MC Sim (No Packs) | Start # Baseline |
|--------|--------------|----------------|-------------------|-----------------|
| **P@3** | **53.0%** | 31.5% | 31.1% | 46.3% |
| **P@5** | **59.8%** | 41.8% | 42.2% | 52.9% |
| **P@10** | **71.1%** | 59.3% | 58.7% | 65.3% |
| **Spearman** | **0.792** | 0.559 | 0.551 | 0.631 |

**Key finding**: Packs vs no-packs are nearly identical (pack bonus=0.0s for tier2 = no-op). The MC sim is *below* the start number baseline. The noise/sigma model is the dominant source of error — perturbations are so large they randomize the good deterministic rankings.

### 7A. Remove `wind_speed_kmh` from Split Features — DONE (v34)

**Problem**: `wind_speed_kmh` was #1 importance in ALL 5 sprint split models (918-1929 importance) despite being event-level (identical for all athletes), drowning out athlete-ability features.

**Fix**: Removed from `get_split_feature_columns()` in `features.py`. Wind impact preserved via sigma adjustments in `estimate_uncertainty()`. Split features: 39 → 38.

### 7B. Initial Sigma Tightening — DONE (v34)

- [x] `FIELD_SIGMA_SCALE = 0.7` — within-field variance is lower than population residuals
- [x] Power transform `ratio ** 1.5` — amplifies differentiation between consistent/volatile athletes

**Result**: Improved from v33 but still insufficient. MC sim still ~21 P@3 points below deterministic.

### 7C. Incremental Sigma Tightening — DONE (v35a)

First attempt: tighten the global scaling approach further.

- [x] `FIELD_SIGMA_SCALE`: 0.7 → 0.5
- [x] Sigma clamp: `clip(30, 300)` → `clip(15, 90)`
- [x] Power transform: 1.5 → 1.8

**Result (v35a)**: P@3=32.6% (+1.1), P@5=43.3% (+1.5), Spearman=0.555 — marginal improvement only.

**Root cause identified**: Learned sigmas from model training residuals are massive (sprint: swim=53s, bike=143s, run=85s → total=315s). These measure global model error, not within-race noise. Even at 0.5× scale, base sigma is ~157s — half the entire sprint field spread of ~300s. The ratio-based approach fundamentally uses the wrong signal.

### 7D. Per-Athlete Per-Split Variance Sigma — DONE (v35b, ineffective)

**Approach**: Use each athlete's own race-to-race variance (`std_swim_sec_24m`, etc.) × 0.35 as their simulation sigma.

**Result (v35b)**: P@3=31.5%, P@5=41.1%, Spearman=0.551 — no improvement over v34. Raw race-to-race std is dominated by course-to-course variation (hilly vs flat courses), not athlete-specific day-to-day noise. Scaling by 0.35 still produces ~100s total sigma.

**New features retained for model use** (even though sigma strategy changed):
- [x] `std_swim_sec_24m`, `std_bike_sec_24m`, `std_run_sec_24m` — raw per-split variance
- [x] `std_swim_gap_sec_24m` — consistency of swim gap to leader
- [x] `min_swim_gap_sec_24m` — best swim gap in 24m

### 7E. Normalized Percentile Variance Sigma — DONE (v35c, partial improvement)

**Problem**: Raw time variance is course-dependent. A hilly bike course adds 2+ minutes vs flat, inflating std_bike_sec_24m regardless of athlete consistency.

**Solution**: Use `std_swim_pct_24m` — the standard deviation of the athlete's percentile rank across races. This is completely course-agnostic: an athlete who always finishes as a top-10% swimmer has low std regardless of fast/slow courses.

**New features**:
- [x] `std_swim_pct_24m` — std of swim rank percentile (swimrank / n_finishers) over 24m
- [x] `std_bike_pct_24m` — std of bike rank percentile
- [x] `std_run_pct_24m` — std of run rank percentile

**Sigma computation**:
- `sigma_swim = std_swim_pct_24m × pred_swim_sec`
- Example: std_pct=0.05, pred_swim=600s → sigma=30s (consistent swimmer)
- Example: std_pct=0.15, pred_swim=600s → sigma=90s (erratic swimmer)
- Clamped per-split: swim [3-30s], bike [8-60s], run [5-40s]
- Default for unknown athletes: std_pct=0.12 (moderate variability)

**Result (v35c)**: P@3=34.1% (+2.6 over v34), Spearman=0.567 — best MC sim yet. Tier 2 World Cup hits P@3=50.0%. But still 18 points below deterministic.

**Bug identified**: `sigma = std_pct × pred_time` uses the wrong scaling. `std_pct=0.08` means ±8% of field position. If field spread is 80s, that's ±6.4s. But multiplying by pred_time (600s) gives 48s — a ~7× overestimate.

### 7F. Field-Spread Sigma Fix — DONE (v35d, marginal)

**Fix**: Replace `pred_time` with `field_spread` (p90 - p10 of predicted split times).

**Result (v35d)**: P@3=34.4%, Spearman=0.566 — essentially identical to v35c. The clamps were already dominating in both versions, so the different base calculation didn't matter.

**Conclusion on sigma calibration**: After 5 iterations (v34 → v35d), sigma tuning has yielded +3 P@3 points (31.5% → 34.4%). The MC sim remains ~17 points below deterministic. The remaining gap is NOT primarily a sigma magnitude issue — it's structural: the median-rank aggregation over 500 noisy simulations inherently regresses predictions toward mid-field.

### 7G. MC Aggregation Fix — DONE (mean-time, no improvement)

Switched from mean-rank to mean-time aggregation: average simulated total times, then rank by mean time.

**Result**: P@3=33.7%, Spearman=0.565 — no improvement. Mean-time and mean-rank are mathematically near-equivalent with symmetric noise.

**Conclusion**: The MC sim's 17-point P@3 gap to deterministic is fundamental to adding ANY noise to predictions. The noise doesn't add predictive signal — it only degrades it. The MC sim's value is NOT in ranking (deterministic wins) but in **uncertainty quantification**: win probabilities, confidence intervals, and podium probabilities.

**Recommended approach going forward**:
- Use **deterministic model for point predictions / rankings**
- Use **MC simulation for probability outputs only** (prob_win, prob_podium, rank intervals)
- The backtest should evaluate MC on its probability calibration, not ranking accuracy

### 7H. Pack Logic Overhaul — PLANNED

Deferred until aggregation/structural fix stabilizes MC accuracy.

### Phase 7 Files Modified

| File | Change |
|------|--------|
| `tri_analysis/prediction/features.py` | Removed `wind_speed_kmh` from split features (v34) |
| `tri_analysis/prediction/simulate.py` | FIELD_SIGMA_SCALE, power transform, sigma clamp (v34-v35) |
| `tri_analysis/prediction/evaluate.py` | Added `use_pack_effects` parameter (v34) |
| `scripts/predict_program.py` | `--no_packs` flag, smart weather, geocoding (v34) |
| `scripts/run_backtest.py` | `--no_packs` flag (v34) |
| `scripts/debug_diagnostics.py` | Fixed crash, assign_packs_chain, event_meta_override (v34) |

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
  |
Phase 7 (Sigma calibration + wind fix) --- COMPLETED (v34-v35)
  |
Phase 8 (Tier-specific improvements) --- COMPLETED (v36)
  |
Phase 9 (Pack overhaul) --- PLANNED
```

## Metrics Tracking

| Version | P@3 | P@5 | P@10 | Spearman | MAE (s) | Split-Total Gap | Notes |
|---------|------|------|------|----------|---------|-----------------|-------|
| v23 (baseline) | — | — | 71.49% | 0.783 | 362.1 | ~125s | Pre-T1/T2 |
| v24 (Phase 1) | — | — | 71.61% | 0.781 | 368.7 | ~125s | T1/T2 added |
| v25 (dist-specific) | — | — | 71.61% | 0.781 | 368.7 | ~181s | Dist total model overfit |
| v26 (split sum) | — | — | 71.95% | 0.782 | 365.0 | 0.0s | Split sum as total |
| v27 (Phase 2) | — | — | 72.18% | 0.776 | 369.1 | 0.0s | MVN noise, learned sigmas, pack rate fix |
| v34 (Phase 7a-b) | 31.5% | 41.8% | 59.3% | 0.559 | — | 0.0s | Wind removed from splits, sigma 0.7×, power 1.5 |
| v35a (Phase 7c) | 32.6% | 43.3% | 59.8% | 0.555 | — | 0.0s | Sigma 0.5×, clamp 15-90s, power 1.8 (marginal) |
| v35b (Phase 7d) | 31.5% | 41.1% | 58.9% | 0.551 | 373.5 | 0.0s | Per-athlete raw variance (ineffective — course noise) |
| v35c (Phase 7e) | 34.1% | 43.3% | 59.8% | 0.567 | 374.7 | 0.0s | Normalized pct variance (best MC yet, but unit bug) |
| v35d (Phase 7f) | 34.4% | 42.9% | 59.9% | 0.566 | 374.7 | 0.0s | Field-spread sigma (marginal — clamps dominate) |
| v35d+agg (7g) | 33.7% | 42.9% | 59.9% | 0.565 | 374.7 | 0.0s | Mean-time aggregation (no improvement) |

### Deterministic Model Metrics (separate from MC sim above)

| Version | P@3 | P@5 | P@10 | Spearman | MAE (s) | Notes |
|---------|------|------|------|----------|---------|-------|
| v35c (baseline) | 53.0% | — | — | 0.792 | 374.7 | Pre tier-specific |
| v36 (Phase 8) | 51.9% | 59.3% | 71.1% | 0.792 | 374.6 | Tier-conditioned splits + 8× tier1 weight |

### Deterministic Model by Tier (v35c → v36)

| Tier | v35c P@3 | v36 P@3 | Delta | v36 Spearman | Notes |
|------|----------|---------|-------|--------------|-------|
| Tier 1a (WTCS) | 46.7% | **56.7%** | **+10.0%** | 0.716 | Massive improvement — best non-T100 tier now |
| Tier 1b (T100) | 63.9% | 63.9% | 0.0% | 0.730 | Unchanged |
| Tier 2 (World Cup) | 45.2% | 40.5% | -4.7% | 0.777 | Trade-off from tier1 upweighting |
| Tier 3 (Continental) | 51.8% | 50.0% | -1.8% | 0.843 | Minor drop |
| Tier 4 (Other) | 52.1% | 54.2% | +2.1% | 0.780 | Slight gain |

## Phase 8: Tier-Specific Model Improvements (v36) — COMPLETED

### Problem
WTCS (Tier 1a) had the worst deterministic P@3 at 46.7%, despite being the most important tier for prediction quality. The model trained on all tiers together with 4× tier 1 weighting, but 95% of training data was Tier 3-4 with wide ability gaps — the model couldn't differentiate in tight WTCS fields.

### Changes

**A) Increased Tier Sample Weights** (`features.py`):
- Tier 1: 4.0 → **8.0** (WTCS results count 8×)
- Tier 2: 2.0 → **3.0** (World Cup results count 3×)
- Tier 3/4: unchanged (1.5/1.0)

**B) Tier-Conditioned Split Percentiles** (`features.py`, `sql.py`):
Added 3 new features to split and full models:
- `ema_swim_split_pct_tier12_5`: EMA of swim rank percentile at WTCS/World Cup only
- `ema_bike_split_pct_tier12_5`: EMA of bike rank percentile at WTCS/World Cup only
- `ema_run_split_pct_tier12_5`: EMA of run rank percentile at WTCS/World Cup only

These tell the model "when this athlete races at the top level, where do they rank in each discipline?" — critical for differentiating athletes who are all top-5% overall but have very different WTCS-specific performance.

**C) SQL Update** (`sql.py`):
Added `e.event_name` to `fetch_precomputed_race_metrics()` query so `compute_race_relative_features()` can classify event tier for historical races.

### Results
- **WTCS P@3: 46.7% → 56.7% (+10.0%)** — massive improvement
- Overall P@3: 53.0% → 51.9% (-1.1%) — acceptable trade-off
- Tier 2 dropped 4.7% — may recover with moderate weight tuning in v36b
- Split features: 46 → 49 (3 new tier-conditioned split percentiles)
- Full features: 77 → 80

### Files Modified
| File | Change |
|------|--------|
| `tri_analysis/prediction/features.py` | 8.0/3.0 tier weights; 3 tier-conditioned split pct features |
| `tri_analysis/prediction/sql.py` | Added event_name to precomputed metrics query |

## Other Ideas

- model for each tier, different dynamics at various levels of racing. (partially addressed in Phase 8)


Claude’s Plan
Weather API Integration Plan
Context
The prediction pipeline uses weather features (temperature_air, humidity, wbgt, is_hot) sourced from the World Triathlon API's meta field. Most lower-tier events (Continental Cups, etc.) have no weather data, so they get filled with defaults (22°C, 60% humidity, 20°C WBGT). This creates an artificial cluster the model latches onto as a tier proxy rather than actual conditions. Temperature is currently a top-2 feature in both the unified total model and sprint total model — likely inflated by this default-clustering artifact.

Fix: Integrate Open-Meteo's free historical weather API to backfill all events with real weather, fetch weather for upcoming predictions, and incorporate richer weather signals (wind, precipitation, apparent temp) into both the deterministic model and MC simulation.

Architecture Overview
New Module: tri_analysis/weather.py
Core weather fetch and processing:

fetch_historical_weather(lat, lon, date) → dict of aggregated weather variables
fetch_forecast_weather(lat, lon, date) → same interface, forecast endpoint
_estimate_wbgt(temp, wet_bulb, wind, solar_radiation) → WBGT approximation
_aggregate_race_window(hourly_data) → average/max over 7am-3pm local time
Open-Meteo API
Historical: https://archive-api.open-meteo.com/v1/archive
Forecast: https://api.open-meteo.com/v1/forecast
Free: No API key for non-commercial use
Variables (hourly, aggregated to race window):
temperature_2m, relative_humidity_2m, apparent_temperature
wind_speed_10m, wind_gusts_10m
precipitation, cloud_cover
shortwave_radiation, wet_bulb_temperature_2m
WBGT Estimation
Open-Meteo provides wet bulb temp and solar radiation but not WBGT directly. Use simplified Liljegren formula:


WBGT ≈ 0.7 × Twb + 0.2 × Tg + 0.1 × Tdb
Where Tg (globe temp) is estimated from solar radiation + wind speed.

Implementation Steps
Step 1: Weather Module — tri_analysis/weather.py (NEW)

def fetch_historical_weather(lat: float, lon: float, event_date: date) -> dict:
    """Fetch race-day weather from Open-Meteo Archive API.
    Returns: {temperature_air, humidity, apparent_temp, wind_speed_kmh,
              wind_gust_kmh, precipitation_mm, cloud_cover_pct, wbgt_estimate,
              wet_bulb_temp}
    """

def fetch_forecast_weather(lat: float, lon: float, event_date: date) -> dict:
    """Same interface but for upcoming events (within 16 days)."""

def _estimate_wbgt(temp_c, wet_bulb_c, wind_kmh, solar_wm2) -> float:
    """Simplified WBGT from available Open-Meteo variables."""

def _aggregate_race_window(hourly_data: dict, start_hour=7, end_hour=15) -> dict:
    """Average/max/sum hourly data over typical race window."""
Rate limit: time.sleep(0.5) between API calls.

Step 2: Database Schema — tri_analysis/database.py
Add columns to events table:

Column	Type	Description
wind_speed_kmh	Float	Avg wind speed during race window
wind_gust_kmh	Float	Max wind gust during race window
apparent_temp	Float	Feels-like temperature (°C)
precipitation_mm	Float	Total precipitation (mm)
cloud_cover_pct	Float	Avg cloud cover (%)
wet_bulb_temp	Float	Wet bulb temperature (°C)
weather_source	String	'api', 'open_meteo', or 'manual'
weather_fetched_at	DateTime	When weather was last fetched
Keep existing columns (temperature_air, humidity, wbgt, etc.) — backfill updates these when missing. Add migration in ensure_schema() (ALTER TABLE ADD COLUMN IF NOT EXISTS, matching existing pattern at lines 308-346).

Step 3: Backfill Script — scripts/backfill_weather.py (NEW)

python scripts/backfill_weather.py                          # backfill all missing
python scripts/backfill_weather.py --start_date 2020-01-01  # from 2020 onwards
python scripts/backfill_weather.py --overwrite              # re-fetch everything
python scripts/backfill_weather.py --dry_run                # preview counts
Logic:

Query events with lat/lon but missing weather (temperature_air IS NULL OR weather_source IS NULL)
Skip events already having weather_source = 'api' unless --overwrite
For each event: fetch_historical_weather(lat, lon, event_date)
Update events row, set weather_source = 'open_meteo'
Progress bar + summary stats (fetched, skipped, failed)
Step 4: Ingestion Pipeline Integration — tri_analysis/api_handling.py
In process_program_data() (or post-processing in build_database.py), after populating an event row:

If weather fields are empty and lat/lon available → call fetch_historical_weather() for past events or fetch_forecast_weather() for future events
Set weather_source accordingly
Preserves World Triathlon API weather when available (higher trust)
Step 5: New Features — tri_analysis/prediction/features.py
Add to build_features_for_program() event-level features:

wind_speed_kmh — direct (wind impacts bike heavily)
wind_gust_kmh — gusts create more variance
apparent_temp — physiological impact (better than raw temp)
precipitation_mm — rain affects bike handling
cloud_cover_pct — solar radiation / heat load proxy
Update in features.py:

get_feature_columns() — add 5 new features
get_split_feature_columns() — add wind_speed_kmh (bike split), exclude rest
fill_missing_features() — conservative defaults (wind=10 km/h, precip=0, cloud=50%)
Step 6: MC Simulation Weather Sigma — tri_analysis/prediction/simulate.py
Add weather-based sigma modifiers in estimate_uncertainty():


# Domain-knowledge multipliers (can be trained from residuals later)
heat_mult = 1.0 + max(0, (wbgt - 22) * 0.02)           # +2% per °C above 22
wind_bike_mult = 1.0 + max(0, (wind_kmh - 15) * 0.01)   # +1% per km/h above 15
rain_mult = 1.1 if precipitation_mm > 1.0 else 1.0       # +10% in rain

# Apply per-split
df["sigma_bike"] *= wind_bike_mult * heat_mult * rain_mult
df["sigma_run"] *= heat_mult
df["sigma_swim"] *= rain_mult  # minimal heat effect in water
This makes simulations produce wider uncertainty bands for adverse weather — more realistic rank distributions.

Step 7: Prediction Script — scripts/predict_program.py
For upcoming events:

Check if weather exists; if not and within 16 days, fetch forecast
Display weather context in output header (temp, wind, conditions)
Pass weather features through to simulation
Files Summary
File	Change
tri_analysis/weather.py	NEW — Open-Meteo API client, WBGT estimation, backfill logic
tri_analysis/database.py	Add 8 new weather columns to events table + migration
tri_analysis/api_handling.py	Post-fetch weather enrichment when API weather is missing
tri_analysis/prediction/features.py	5 new weather features; update column lists and defaults
tri_analysis/prediction/simulate.py	Weather-adjusted sigma multipliers in estimate_uncertainty()
scripts/backfill_weather.py	NEW — CLI backfill script with progress tracking
scripts/predict_program.py	Forecast weather fetch; display in output
scripts/debug_diagnostics.py	Show weather data quality in overview section
Verification
Backfill dry run: python scripts/backfill_weather.py --dry_run — check event counts and lat/lon coverage
Backfill small batch: Run for 2024 events, spot-check a known hot race (e.g., Abu Dhabi) against reported conditions
Data quality check: Query events table — verify weather_source distribution, check for nulls in new columns
Retrain model: Train with real weather data — temperature_air importance should drop as default clustering disappears
Backtest comparison: P@10 with vs without new weather features
Forecast test: predict_program.py for upcoming event — verify weather fetched and displayed

---

## Phase 10: Deterministic Model v40 → v41+ (Target P@10 = 80%)

### Context
v40 achieves P@10=74.2% overall (P@3=57.0%, Spearman=0.796) but drops to 64% on large fields (50+ athletes). The 6% gap to 80% is driven primarily by Tier 2 (World Cup, avg 50 athletes, P@10=65%) and large Continental events. Improvements needed across data quality, feature engineering, cold-start handling, and modeling.

**v37→v40 iteration history**: +2.3% P@10 via h2h features, LGBMRanker, NDCG@10 truncation, field-size weighting, 50/50 ensemble.

**Key blocker**: P@10 by field size: <20: 82%, 20-30: 81%, 30-40: 74%, 40-50: 71%, 50+: 64%.

---

### Phase 10A: Diagnostics + Quick Wins (~19 hours, expected +2-3% P@10)

#### 10A-1. Residual Analysis by Field Size (3h) — DIAGNOSTIC
**Why**: Confirm whether cold-start athletes or inherent ranking noise causes large-field P@10 drop.
- **File**: `scripts/run_backtest.py`
- **What**: For each backtest event, output: field_size, count of cold-start athletes (n_matched_races < 3), tier, P@10. Compute fraction of top-10 misses that are cold-start athletes.

#### 10A-2. Bayesian Shrinkage for Cold-Start EMAs (8h) — HIGH IMPACT
**Why**: An athlete with 1 race who finished 5th/50 gets ema_finish_pct=0.10. With `min_periods=1` (features.py line 388), this noisy single-race estimate drives ranking. Large fields have more cold-start athletes.
- **File**: `tri_analysis/prediction/features.py` — modify `compute_athlete_form_features()` (line ~370-434) and `compute_race_relative_features()`
- **What**: After computing all EMA features, apply shrinkage:
  ```
  shrinkage = n_races / (n_races + K)  # K ≈ 5
  adjusted_ema = shrinkage * athlete_ema + (1 - shrinkage) * population_default
  ```
- Apply to: `ema_finish_pct_3/5/12`, `ema_swim_split_pct_5`, `ema_bike_split_pct_5`, `ema_run_split_pct_5`, and tier-conditioned percentiles
- Add `n_total_races` as a feature so the model knows confidence level
- If shrinkage alone isn't sufficient, follow up with k-NN similar athlete matching (find 3-5 athletes with similar WT rank/Elo/country and use their stats as priors)

#### 10A-3. Optimized Ensemble Weights via CV (4h)
**Why**: 50/50 ranker/pct is hardcoded in `predict.py` line 318. Optimal weight likely differs.
- **Files**: `tri_analysis/prediction/predict.py`, `tri_analysis/prediction/train.py`
- **What**: Sweep RANKER_WEIGHT from 0.3 to 0.7 in 0.05 steps across CV folds. Store optimal weight in `ModelBundle.metadata`.

#### 10A-4. SHAP-Based Feature Pruning (4h)
**Why**: 86 features likely includes noise. v38 already removed 5 zero-importance features. Systematic SHAP analysis may find more.
- **File**: `scripts/train_models.py`
- **What**: After training, compute SHAP values on both ranker and percentile models. Remove features with mean |SHAP| < threshold.

---

### Phase 10B: Feature Engineering (~21 hours, expected +1.5-3% P@10)

#### 10B-1. Venue History Features (8h) — HIGH IMPACT
**Why**: `event_venue` is in the DB and fetched in `fetch_program_results()` (sql.py line 73) but NOT in `fetch_athlete_history()` (line 91) and never engineered as features. Athletes consistently over/underperform at specific venues.
- **Files**:
  - `tri_analysis/prediction/sql.py` — add `e.event_venue` to `fetch_athlete_history()` SELECT (after line 156)
  - `tri_analysis/prediction/features.py` — new `compute_venue_features()` function
- **Features**:
  - `has_raced_venue`: binary 0/1
  - `n_venue_races`: count of prior races at this venue
  - `ema_finish_pct_venue`: EMA finish pct at this specific venue (if ≥2 prior races)
  - `venue_delta`: finish_pct_venue - finish_pct_overall (negative = overperforms here)

#### 10B-2. Race Density / Fatigue Features (3h)
**Why**: Only `days_since_last_race` and `races_12m` capture frequency. Missing short-term fatigue.
- **File**: `tri_analysis/prediction/features.py` — modify `compute_athlete_form_features()` (add after line ~332)
- **Features**: `races_30d`, `races_60d`, `days_since_2nd_last`

#### 10B-3. WT Ranking as Stronger Prior for Cold Start (4h)
**Why**: Athletes with <3 races but a WT world ranking get conservative defaults (pct=0.65). The WT ranking is highly informative.
- **File**: `tri_analysis/prediction/features.py` — modify `fill_missing_features()` (line ~1661)
- **What**: When `ema_finish_pct_5` is NaN but `wt_rank_position` is available, impute from mapping: rank 1-10→pct~0.08, 11-30→0.15, 31-100→0.30, 100+→0.50

#### 10B-4. DNF/Reliability Features (6h)
**Why**: Currently only finishers in history (sql.py line 124). A 30% DNF rate athlete is less reliable.
- **Files**: `tri_analysis/prediction/sql.py` — new `fetch_athlete_dnf_stats()` query; `features.py` — call in `build_features_for_program()`
- **Features**: `dnf_rate_24m`, `total_starts_24m`

---

### Phase 10C: Advanced Modeling (~30 hours, expected +1-2% P@10)

#### 10C-1. CatBoost as Third Ensemble Member (10h)
Train CatBoostRanker alongside LGBMRanker. Extend ensemble to 3-way. Optimize weights via CV.

#### 10C-2. Two-Stage Model for Large Fields (16h)
Stage 1: binary classifier "top-20?". Stage 2: rank within candidates. Reduces ranking problem from 50→20 athletes.

#### 10C-3. Within-Race Z-Score Outlier Filter (4h)
Per-race z-score on total_sec in `filter_outliers_by_distance()`, drop |z| > 3 to remove back-of-pack noise.

---

### Phase 10D: Optional / Lower Priority
- Comeback Athlete EMA Decay (6h): When `days_since_last_race > 180`, blend EMA toward defaults
- Seasonal Features (2h): Add `race_month`
- Full Ensemble CV (6h): Train both percentile + ranker per fold
- Bayesian Hyperparameter Optimization (8h): Optuna instead of grid search

### Expected Cumulative Trajectory
- Phase 10A: 74.2% → 76-77% P@10
- Phase 10B: 77% → 78-79%
- Phase 10C: 79% → 80-81%

### Critical Files
- `tri_analysis/prediction/features.py` — EMA computation, cold-start defaults, new features
- `tri_analysis/prediction/train.py` — Model training, ranker, ensemble weight optimization
- `tri_analysis/prediction/predict.py` — Ensemble blending, two-stage prediction
- `tri_analysis/prediction/sql.py` — Data queries (venue history, DNF stats)
- `scripts/run_backtest.py` — Diagnostic analysis, evaluation
