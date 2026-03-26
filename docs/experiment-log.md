# Experiment Log

Track all model experiments with results and verdicts. The /experiment and /backtest slash commands append to this file.

## Promotion Criteria
- **PROMOTE**: P@10 improves AND P@3 does not regress by >2%
- **REJECT**: Primary metrics regress
- **INVESTIGATE**: Mixed results (some tiers improve, others regress)

## Log

| Date | Version | Change | P@10 | P@3 | Spearman | Verdict |
|------|---------|--------|------|-----|----------|---------|
| 2026-03-09 | v45 | Champion baseline | 74.2% | 57.0% | 0.796 | CHAMPION |
| 2026-03-12 | v46 | CV-optimized ensemble weight (w=0.35) + SQL weather fix | 70.0% | 49.3% | 0.708 | REJECT — missing xgboost + sklearn 1.8 caused regression |
| 2026-03-13 | v47 | SQL fix only, no xgboost, sklearn 1.8, w=0.50 | 70.7% | 50.7% | 0.709 | REJECT — confirmed xgboost + sklearn version as root cause |
| 2026-03-13 | v48 | SQL fix + xgboost restored, sklearn 1.8, w=0.50 | 70.9% | 54.8% | 0.719 | REJECT — xgboost helped P@3 +4.1%; sklearn 1.8 still hurts |
| 2026-03-14 | v49 | SQL fix + xgboost + sklearn 1.6.1, w=0.50 | 71.2% | 55.9% | 0.720 | NEW BASELINE — reproducible with current DB state |
| 2026-03-14 | v50 | v49 + CV-optimized weight (w=0.35) | 70.8% | 53.3% | 0.719 | REJECT — w=0.35 hurt P@3 -2.6%, WTCS P@3 -10pp vs v49 |
