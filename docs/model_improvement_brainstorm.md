# Triathlon Race Prediction Model - Improvement Brainstorm

## Status Summary (as of v59, April 2026)

| Category | Status | Notes |
|----------|--------|-------|
| 1: Distance-Agnostic Features | DONE | Finish percentile EMA, gap-to-median, per-split percentiles, cross-distance all implemented |
| 2: Tier-Aware Modeling | PARTIAL | Tier-conditioned features done (v36). Tier-specific sim params done. Hierarchical model done in v59 |
| 3: Target Variable Strategy | DONE | Finish percentile as target implemented in v37+ |
| 4: MC Simulation Improvements | DONE | Learned pack effects, causal chain, distance-specific uncertainty, MVN covariance |
| 5: Training Pipeline | DONE | Hyperparameter tuning, time-based CV, gender models all implemented. `deterministic=True` added v58 to remove ±1.5pp training variance |
| 6: Tier-Stratified Modeling | TESTED | Option A (conditioned features) DONE. Option B (tier sim params) DONE. Option D (hierarchical ability+context) tested in v59 — REJECT (P@3 -2.3pp, P@1 -10pp). Mechanism failure: mis-ordering inside top-3 (boundary metrics tied with v57). |
| 7: Group-First Bike Model | TODO | Largest architectural change, most principled approach for MC sim |
| 8: Feature Coverage & Data Quality | PARTIAL | Cold-start handling planned (Bayesian shrinkage tested in v43, neutral). Field-boundary v54 features partially address it. |
| 9: New Signal Sources | PARTIAL | H2H/Elo partially done (v37). v54 added field-boundary magnitudes. v56 h2h-vs-field-top10 didn't help. Form trajectory done (v52). Course clustering, season phase TODO |
| 10: Architectural alternatives to additive features | EXHAUSTED | Cascade (v58, REJECT — recall-limited). Hierarchical (v59, REJECT — top-3 ordering noise). Conclusion: 74-75% P@10 / ~58% P@3 is the ceiling for this LGBMRanker family on H2 2025. Next axes: different ML library (CatBoost YetiRank), Optuna sweep on v57, group-first bike for MC sim probabilities, more training data. |

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


---

### CATEGORY 6: Tier-Stratified Modeling

**Observation**: WTCS and Continental races are fundamentally different prediction problems. A WTCS sprint has 25-30 world-class athletes with tight ability gaps, large front packs (15+ riders), and small finishing margins. A Continental cup has 40-70 athletes with huge ability spread, 2-3 small packs, and finish gaps of minutes. The model currently treats both identically.

**The two-axis problem**: Performance varies along TWO independent dimensions:
- **Distance axis** (sprint vs standard) — already modeled via distance-specific split models
- **Tier axis** (WTCS vs World Cup vs Continental) — currently only captured via `event_tier` feature and tier sample weights

**Why this matters for accuracy**:
- **Pack dynamics differ by tier**: WTCS sprints routinely see 15+ athlete front packs; Continental cups might have 3-5. The pack effect magnitude, merge probability, and gap thresholds are all tier-dependent
- **Prediction difficulty differs**: Predicting top-3 at WTCS (P@3=43%) is much harder than Continental (P@3=50%) because ability gaps are smaller. The model may be wasting capacity learning patterns that only apply at one tier level
- **Field strength interaction**: An athlete's percentile at WTCS is not directly comparable to their percentile at a Continental cup. A 20th-percentile finisher at WTCS might win a Continental cup

**Implementation approaches** (from lightweight to heavy):

#### Option A: Tier-Conditioned Features (Low effort, moderate impact)
Add tier-aware features without separate models:
- `ema_finish_pct_tier1`: EMA percentile from Tier 1 races only
- `ema_finish_pct_tier3`: EMA percentile from Tier 3+ races only
- `tier_best_pct`: Best percentile at current race's tier level
- `n_races_at_tier`: How many times athlete has raced at this tier
- `tier_step_up`: Boolean — is this a higher tier than athlete's median?

The model learns the correlation between tier-specific performance signals automatically.

#### Option B: Tier-Specific Pack/Sim Parameters (Medium effort, high impact for sim)
Keep one prediction model but learn separate simulation parameters per tier:
- `pack_params_by_tier`: Different gap thresholds, bonus/penalty by tier (WTCS has tighter packs, smaller effects; Continental has wider gaps, larger effects)
- `merge_params_by_tier`: Different merge probabilities (WTCS packs merge more easily due to drafting skill)
- `residual_cov_by_tier`: Different noise structure (WTCS has lower variance)

This directly addresses the MC simulation underperformance at Tier 1a races (P@3=23% currently).

#### Option C: Tier-Stratified Models (High effort, potentially highest impact)
Train separate model ensembles per tier group:
- **Elite models** (Tier 1a + 1b): Trained on WTCS/T100 data only. Tight fields, sophisticated pack dynamics
- **Mid-tier models** (Tier 2): World Cup — intermediate field depth
- **Development models** (Tier 3 + 4): Continental — wide ability spread, simpler dynamics

**Risk**: Data sparsity. Tier 1a has only ~40 events/year. Could mitigate with:
- Train on all data, but use tier as a categorical feature with strong weight
- "Warm start" tier-specific models from the unified model
- Share feature engineering, only separate the final regressor

#### Option D: Hierarchical Model (High effort, most principled)
Two-stage model that explicitly separates ability from field context:
1. **Stage 1 (universal)**: Predict athlete ability score from history features (distance-agnostic, tier-agnostic)
2. **Stage 2 (context-specific)**: Given ability scores for all athletes in the field + tier + distance → predict finishing order

This cleanly separates "how good is this athlete" from "how does this field of athletes interact in this race context." Stage 2 can learn that tight ability clusters at WTCS lead to unpredictable pack battles, while wide spreads at Continental races are more predictable.

**Recommendation**: Start with Option A (tier-conditioned features) + Option B (tier-specific sim params). These are low-risk and directly address the tier-level accuracy gaps visible in the backtest. Option D is the most architecturally sound but is a larger refactor.

---

### CATEGORY 7: Monte Carlo Simulation Rethink — Group-First Bike Model

**Current approach** (individual-first):
1. Predict each athlete's bike time independently
2. Add per-sim noise to individual bike times
3. Compute pack formation from swim+T1
4. Adjust individual bike times by pack effect (bonus/penalty)

**Problem**: This models the bike as an individual time trial with pack bonuses bolted on. But in draft-legal triathlon, the bike is fundamentally a **group phenomenon**. Athletes don't have individual bike splits — they have **group finishes** with tiny within-group variance.

**Proposed approach** (group-first):
1. Simulate swim + T1 → determine **bike entry order** (already done)
2. Form initial packs from swim+T1 gaps (already done)
3. **NEW**: Predict pack evolution during bike — which packs merge, which break away
4. **NEW**: Predict **group finish gaps** — how far apart do the groups finish?
5. **NEW**: Derive individual bike times from group membership — each athlete's bike = their group's aggregate bike time ± small within-group noise (5-15s)

**Why this is better**:
- Models the actual causal structure: you finish with your group, not on your own
- Within-group variance is inherently small (drafting → tight finish). The current model adds individual noise that's too large for drafted athletes
- Between-group variance is the real prediction: "will the chase pack catch the front?" matters more than individual bike power
- Naturally produces realistic pack-based rankings instead of artificially smooth individual distributions

**Key modeling questions**:
- What determines the gap between groups? (Initial swim gap, bike terrain, pack sizes, athlete ability within packs)
- What's the within-group finish spread? (Empirically ~5-15s for front pack, wider for chase)
- How to handle solo riders between groups? (Use individual bike prediction for non-drafted athletes)

**Data available**: `wtcs_pack_membership` table has historical pack data, `position_metrics` has T1-to-bike and bike-to-T2 position changes. Can learn group gap distributions empirically.

---

### CATEGORY 8: Feature Coverage & Data Quality

**Observation**: Many accuracy-limiting issues stem from sparse feature coverage, not model architecture.

**Known coverage gaps**:
- **Athletes with <3 races**: EMA features are unreliable, dominated by defaults. The split sanity check catches the worst cases but doesn't fix the underlying feature quality
- **Cross-distance athletes**: An athlete with 20 standard races but 0 sprint races has good standard features but default sprint features. Distance-specific split models see garbage input
- **Missing weather data**: Many events have NULL temperature/humidity/wbgt, making weather features unreliable
- **T1/T2 data quality**: Some events have nonsensical transition times (0s, or 500s+), polluting EMA features

**Potential improvements**:
- **Feature confidence score**: Compute `n_valid_features / n_total_features` per athlete. Use as a model feature or to weight predictions
- **Imputation from ability**: For athletes with sparse sprint data but good standard data, impute sprint features from standard (using learned distance conversion ratios)
- **Minimum history threshold**: Instead of predicting with garbage features, flag athletes with <N prior races and use a simpler model (e.g., just bib number rank) for them
- **Data quality filters**: Drop/cap transition times outside [10s, 180s] before computing EMAs. Drop race results where total_sec is implausible for the distance

**Impact on accuracy ceiling**: Even perfect models can't predict well for athletes with no history. The ~28% of events where P@10 misses may be dominated by races with many sparse-history athletes. Quantifying this would reveal how much of the accuracy ceiling is data-driven vs model-driven.

---

### CATEGORY 9: New Signal Sources

**Head-to-head records**:
- When athletes A and B have raced 10 times, their H2H record (A beat B 7/10) is a strong predictor
- Can compute pairwise Elo/Bradley-Terry ratings from race results
- Especially valuable at WTCS where the same ~30 athletes race repeatedly

**Form trajectory (momentum)**:
- Not just EMA level but **direction** — is the athlete improving or declining?
- `ema_finish_pct_3 - ema_finish_pct_10` = recent form vs longer-term form
- Captures "peaking for championships" vs "end-of-season fatigue"

**Course characteristics**:
- Hilly vs flat bike, technical vs open-water swim, fast vs slow run course
- Some athletes are course specialists (climbers, technical swimmers)
- Could cluster courses by profile and track per-athlete course-type performance

**DNS/DNF patterns**:
- Athletes who DNF frequently may DNF again, affecting field strength
- Late withdrawals change pack dynamics (removing a fast swimmer changes pack formation)

**Season phase**:
- Early season races have more uncertainty (form unknown)
- Championship finals have more predictable form (athletes peak intentionally)
- Encode race's position in the season calendar

---

## Updated Priority Ordering (Post-Phase 6)

Many items from the original priority list have been implemented. Updated ranking
based on current accuracy (P@10=71.7%, Spearman=0.785) and identified gaps:

### Next Highest Impact (Feature/Signal)
1. **Tier-conditioned features** (Cat 6, Option A) — add tier-specific EMA percentiles, tier_step_up
2. **Form trajectory / momentum** (Cat 9) — `ema_finish_pct_3 - ema_finish_pct_10`
3. **Feature coverage score** (Cat 8) — quantify data quality per athlete, use as feature
4. **Head-to-head / Elo ratings** (Cat 9) — strong signal for WTCS where same athletes repeat
5. **Cross-distance imputation** (Cat 8) — fill sprint features from standard data using learned ratios

### Next Highest Impact (Simulation)
6. **Group-first bike model** (Cat 7) — rethink MC sim to predict group gaps, not individual splits
7. **Tier-specific pack/sim parameters** (Cat 6, Option B) — different pack dynamics per tier
8. **Minimum history threshold** (Cat 8) — simpler model for sparse-data athletes

### Longer-term / Larger Refactors
9. **Hierarchical ability + context model** (Cat 6, Option D) — clean separation of ability vs field dynamics
10. **Course clustering** (Cat 9) — track per-athlete course-type performance
11. **Season phase encoding** (Cat 9) — early season uncertainty vs championship peaking

---

## Original Priority Ordering (Pre-Implementation Reference)

~~Add prog_distance_category as a model feature~~ ✅ DONE
~~Add overall finish percentile EMA (ema_finish_pct_5)~~ ✅ DONE
~~Compute empirical pack effects from data~~ ✅ DONE (Phase 2)
~~Implement causal swim→bike→run simulation chain~~ ✅ DONE (Phase 1)
~~Hyperparameter tuning with time-based CV~~ ✅ DONE (Phase 5)
Add gap-to-median ratio EMA -- distance-normalized speed signal
Add Elo ratings -- best single feature for cross-distance, cross-tier ranking
Gender-specific models
Continuous gap-based pack effects

