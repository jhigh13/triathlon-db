# Prediction System - Current Status

## Champion Model

- **Version**: v57 (`models/bundle_elite_v57.joblib`) — pending v59 hierarchical
- **Architecture**: 50/50 ensemble (LGBMRanker + HistGBR finish-percentile regressor) +
  6 v54 field-boundary features (gap-to-cutoff, density). v57 = v54 retrain that landed
  the better LGBM thread schedule.
- **Features**: 92 total (full v54 set), 49 split-specific
- **Training data**: 2021-01-01 through 2025-06-30
- **Backtest period**: H2 2025 (90 events across all tiers)
- **Determinism**: `deterministic=True, force_col_wise=True` enabled v58+ — eliminates the ±1.5pp
  training variance that surfaced when comparing v54 vs v57 (same recipe, different LGBM threads).
- **Rejected architectural alternatives**:
  - v58 cascade — Stage 1 false negatives unrecoverable.
  - v59 hierarchical (ability + context, with 5-fold OOF) — boundary metrics tied with
    v57 but P@1 −10pp, P@3 −2.3pp; failure mode is mis-ordering inside top-3, not the
    boundary. Likely OOF→inference distribution shift (Stage 1 OOF MAE=0.150 vs final
    in-sample=0.124) compounded by multicollinearity between the 8 derived field-ability
    features and v54's existing field-boundary features.

## Current Metrics (Deterministic, v57)

| Metric | Overall | WTCS (1a) | T100 (1b) | World Cup (2) | Continental (3) | Other (4) |
|--------|---------|-----------|-----------|---------------|-----------------|-----------|
| P@1    | 50.0%   | —         | —         | —             | —               | —         |
| P@3    | 58.2%   | 56.7%     | 69.4%     | 52.4%         | 51.8%           | 70.8%     |
| P@5    | 62.9%   | 64.0%     | 70.0%     | 54.3%         | 61.1%           | 68.8%     |
| P@10   | 74.5%   | 76.0%     | 83.3%     | 66.4%         | 72.4%           | 79.3%     |
| weighted_topk | 0.6276 | — | — | — | — | — |
| Spearman | 0.808 | — | — | — | — | — |
| Spearman within top-10 | 0.547 | — | — | — | — | — |
| NDCG@10 | 0.808 | — | — | — | — | — |
| mean_miss_rank@10 | 14.60 | — | — | — | — | — |
| MAE    | ~326s   | —         | —         | —             | —               | —         |

**vs Start-Number Baseline**: P@10=65.3%, P@3=46.3%, Spearman=0.631 (v57 +9.2pp P@10, +11.9pp P@3)

**Enriched scorecard (added April 2026)** — beyond the original P@K table, the backtest now reports:
- `weighted_topk_score` = 0.5·P@3 + 0.3·P@5 + 0.2·P@10 (matches stated priority)
- `spearman_within_top_K` (ordering quality among actual top-K, not just catch rate)
- `mean_miss_rank@K` / `mean_false_alarm_rank@K` (boundary-distance diagnostics)
- `top_K_displacement@K` (avg |pred_rank − actual_rank| in the relevant region)
- Miss buckets (close / mid / far)

## Monte Carlo Simulation

MC sim P@3 = ~34% (19 points below deterministic). This is a **structural limitation** --
adding noise to predictions inherently degrades ranking accuracy. MC sim's value is
exclusively in **probability outputs** (win%, podium%, top-10%, confidence intervals),
not ranking.

## Known Limitations

1. **Large fields (50+ athletes)**: P@10 drops to ~64% vs 82% for small fields. Cold-start athletes with sparse history are the main driver
2. **Tier 2 trade-off**: Tier weighting (8x for WTCS) improved Tier 1a by +10% P@3 but cost Tier 2 -4.7%
3. **Ensemble plateau**: ~74% P@10 appears to be the ceiling for the current architecture. Further gains likely require architectural changes

## Key Lessons (v1 through v45)

1. **Pack features are high-leverage**: +25% P@10 historically vs no-pack baseline
2. **T1/T2 explicit modeling**: Eliminated 125s split-total accounting gap
3. **MC for probabilities only**: Deterministic always wins for ranking; MC adds value only for uncertainty quantification
4. **Tier weighting works but trades off**: WTCS accuracy improves at the cost of mid-tier races
5. **Ensemble methods plateau**: LGBMRanker + percentile blend improved from 71% to 74% P@10 but further gains are diminishing

## Active Improvement Targets (Phase 10)

### Short-term (high confidence)
- **Bayesian shrinkage for cold-start**: K=5 pseudocount for athletes with <3 races. Directly addresses large-field accuracy drop
- **Ensemble weight optimization via CV**: Replace hardcoded 50/50 with CV-optimized blend. Low effort, moderate upside
- **SHAP-based feature pruning**: 80+ features likely includes noise. Prune to top 30-40 by importance

### Medium-term
- **Venue history features**: Track per-venue performance deltas (some athletes are course specialists)
- **Race density / fatigue features**: races_30d, races_60d, days_since_2nd_last
- **WT ranking as cold-start prior**: Use official ranking as baseline for athletes with sparse race history

### Longer-term (architectural)
- **Group-first bike model** (Category 7 in brainstorm): Predict group gaps instead of individual bike times. Most principled approach to pack dynamics
- **Hierarchical ability + context model** (Category 6D in brainstorm): Clean separation of athlete ability from field dynamics

## Performance Progression

| Version | Change | P@10 | P@3 | Spearman | weighted_topk |
|---------|--------|------|-----|----------|---------------|
| Start # baseline | — | 65.3% | 46.3% | 0.631 | — |
| v26 (Phase 1) | T1/T2 fix | 71.95% | — | 0.782 | — |
| v27 (Phase 2) | MVN covariance | 72.18% | — | 0.776 | — |
| v35c (Phase 7) | Sigma calibration | ~71% | 53.0% | 0.792 | — |
| v36 (Phase 8) | Tier weighting | 71.1% | 51.9% | 0.792 | — |
| v40 (Phase 9) | Ensemble + ranker | 74.2% | 57.0% | 0.796 | — |
| v45 | Elite features | 74.3% | 56.3% | 0.804 | 0.6185 |
| v53 | v45 + ensemble weight resave (w=0.80) | 74.1% | 57.4% | 0.789 | 0.6195 |
| v54 | + 6 field-boundary features (Phase 1) | 75.0% | 55.6% | 0.807 | 0.6099 |
| v56 | + 6 h2h-vs-field features (Phase 2) | 75.1% | 54.8% | 0.805 | 0.6055 |
| **v57 (champion)** | v54 retrain with better thread schedule | **74.5%** | **58.2%** | **0.808** | **0.6276** |
| v58 | Cascade architecture | 73.9% | 53.3% | 0.794 | 0.5968 |
| v59 | Hierarchical ability + context (REJECT) | 73.8% | 55.9% | 0.803 | 0.6114 |

## References

- Detailed version history: `docs/archive/prediction_improvement_plan.md`
- Idea backlog: `docs/model_improvement_brainstorm.md`
- Experiment log: `docs/experiment-log.md`
