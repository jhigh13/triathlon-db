# Prediction System - Current Status

## Champion Model

- **Version**: v45 (`models/bundle_elite_v45.joblib`)
- **Architecture**: 50/50 ensemble (LGBMRanker + HistGBR finish-percentile regressor)
- **Features**: 80+ total, 49 split-specific
- **Training data**: 2021-01-01 through 2025-06-30
- **Backtest period**: H2 2025 (~90 events across all tiers)

## Current Metrics (Deterministic)

| Metric | Overall | WTCS (1a) | T100 (1b) | World Cup (2) | Continental (3) |
|--------|---------|-----------|-----------|---------------|-----------------|
| P@3    | 57.0%   | 56.7%     | 63.9%     | 40.5%         | 50-54%          |
| P@10   | 74.2%   | —         | —         | —             | —               |
| Spearman | 0.796 | —         | —         | —             | —               |
| MAE    | ~370s   | —         | —         | —             | —               |

**vs Start-Number Baseline**: P@10=65.3%, P@3=46.3%, Spearman=0.631

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

| Version | Change | P@10 | P@3 | Spearman |
|---------|--------|------|-----|----------|
| Start # baseline | — | 65.3% | 46.3% | 0.631 |
| v26 (Phase 1) | T1/T2 fix | 71.95% | — | 0.782 |
| v27 (Phase 2) | MVN covariance | 72.18% | — | 0.776 |
| v35c (Phase 7) | Sigma calibration | ~71% | 53.0% | 0.792 |
| v36 (Phase 8) | Tier weighting | 71.1% | 51.9% | 0.792 |
| v40 (Phase 9) | Ensemble + ranker | 74.2% | 57.0% | 0.796 |
| v45 (current) | Elite features | 74.2% | 57.0% | 0.796 |

## References

- Detailed version history: `docs/archive/prediction_improvement_plan.md`
- Idea backlog: `docs/model_improvement_brainstorm.md`
- Experiment log: `docs/experiment-log.md`
