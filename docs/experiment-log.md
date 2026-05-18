# Experiment Log

Track all model experiments with results and verdicts. The /experiment and /backtest slash commands append to this file.

## Promotion Criteria
- **PROMOTE**: weighted_topk_score improves AND no individual P@K regresses >2pp AND
  Spearman doesn't drop >0.02 AND mean_miss_rank@10 doesn't worsen >2 ranks
- **REJECT**: weighted_topk_score regresses, or any guardrail breached
- **INVESTIGATE**: Mixed results (some tiers improve, others regress)

`weighted_topk_score` = 0.5 × P@3 + 0.3 × P@5 + 0.2 × P@10 (matches user's stated priority)

## Variance / determinism notes
- Single-run training has ±1.5pp P@3 / ±0.5pp P@10 variance from LightGBM thread-level
  non-determinism (confirmed v54 vs v57 — same code, different seed in LGBM threading,
  P@3 swung 55.6% → 58.2%).
- Starting with v58 the bundle uses `deterministic=True, force_col_wise=True` on every
  LightGBM model. Single-run comparisons after v58 are valid for A/B claims.
- v45–v57 results should be considered single-sample noisy; the v54+v57 pair gives a
  rough variance band of ±1pp on P@3.

## Log

| Date | Version | Change | P@10 | P@3 | Spearman | Verdict |
|------|---------|--------|------|-----|----------|---------|
| 2026-03-09 | v45 | Champion baseline | 74.2% | 57.0% | 0.796 | CHAMPION |
| 2026-03-12 | v46 | CV-optimized ensemble weight (w=0.35) + SQL weather fix | 70.0% | 49.3% | 0.708 | REJECT — missing xgboost + sklearn 1.8 caused regression |
| 2026-03-13 | v47 | SQL fix only, no xgboost, sklearn 1.8, w=0.50 | 70.7% | 50.7% | 0.709 | REJECT — confirmed xgboost + sklearn version as root cause |
| 2026-03-13 | v48 | SQL fix + xgboost restored, sklearn 1.8, w=0.50 | 70.9% | 54.8% | 0.719 | REJECT — xgboost helped P@3 +4.1%; sklearn 1.8 still hurts |
| 2026-03-14 | v49 | SQL fix + xgboost + sklearn 1.6.1, w=0.50 | 71.2% | 55.9% | 0.720 | NEW BASELINE — reproducible with current DB state |
| 2026-03-14 | v50 | v49 + CV-optimized weight (w=0.35) | 70.8% | 53.3% | 0.719 | REJECT — w=0.35 hurt P@3 -2.6%, WTCS P@3 -10pp vs v49 |
| 2026-04 | v53 | v45 with ensemble weight resaved at w=0.80 (P@3-optimised sweep) | 74.1% | 57.4% | 0.789 | INVESTIGATE — wins P@3 (+1.1) at the cost of Spearman (-0.015) |
| 2026-04 | v54 | + 6 field-boundary features (Phase 1: gap-to-cutoff + density) | 75.0% | 55.6% | 0.807 | INVESTIGATE — broke 75% P@10 ceiling, P@1 +4.4pp, P@3 down 0.7pp |
| 2026-04 | v55 | v54 with P@3-optimal ensemble weight sweep | 75.0% | 55.6% | 0.807 | NEUTRAL — P@3-optimal weight on v54 is w=0.50 (already v54 default) |
| 2026-04 | v56 | + 6 h2h-vs-field-top10 features (Phase 2) | 75.1% | 54.8% | 0.805 | REJECT — features used by model but didn't translate to P@3 gain |
| 2026-04 | v57 | v54 retrain (variance check, same code) | 74.5% | 58.2% | 0.808 | NEW CHAMPION — same recipe as v54, different LGBM thread schedule. weighted_topk_score=0.6276 (best on agreed metric). Confirms v54 architecture is good and the v54 P@3 dip was noise. |
| 2026-04 | v58 | Cascade architecture (Stage 1 binary + Stage 2 focused ranker) | 73.9% | 53.3% | 0.794 | REJECT — Stage 1 false negatives unrecoverable; spearman_within_top_3 +0.04 (only win). weighted_topk_score 0.5968 (worst). |
| 2026-04 | v59 | Hierarchical ability + context (Stage 1 = athlete-level ability, Stage 2 = LGBMRanker + 8 derived field-ability features) | 73.8% | 55.9% | 0.803 | REJECT — P@1 −10pp, P@3 −2.3pp, weighted_topk_score −0.016. Boundary metrics tied (mean_miss_rank@10 14.62 vs 14.60), failure mode is **mis-ordering inside top-3 not boundary**. Likely OOF→inference distribution shift (Stage 1 OOF MAE=0.150 vs final in-sample=0.124) plus multicollinearity with v54 features.
