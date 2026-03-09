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
