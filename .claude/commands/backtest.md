Run a backtest evaluation on a model and compare against the champion.

## Context
- Backtest script: scripts/run_backtest.py
- Backtest period: H2 2025 (~90 events across all tiers)
- Champion baseline (from MEMORY.md): P@10=74.2%, P@3=57.0%, Spearman=0.796
- Experiment log: docs/experiment-log.md

## Steps
1. Identify model to test from $ARGUMENTS, or find the latest bundle_elite_v*.joblib in models/
2. Run deterministic backtest (primary metric):
   ```powershell
   python scripts/run_backtest.py --model {model_path} --no_sim
   ```
3. Parse the output and extract:
   - Overall: P@3, P@5, P@10, Spearman, MAE
   - By tier: P@3 and Spearman for each tier (1a, 1b, 2, 3, 4)
   - vs start-number baseline
4. Compare against champion:
   - Flag any metric that REGRESSED by >1% (warning)
   - Flag any metric that IMPROVED by >1% (highlight)
   - Show tier-specific changes
5. Present a clear verdict:
   - **PROMOTE**: P@10 improves AND P@3 does not regress by >2%
   - **REJECT**: Primary metrics regress on both P@10 and P@3
   - **INVESTIGATE**: Mixed results -- some tiers improve, others regress
6. If verdict is PROMOTE:
   - Append a row to docs/experiment-log.md with date, version, change description, metrics, verdict
   - Suggest updating MEMORY.md champion reference
7. If user wants MC simulation evaluation too, run without --no_sim flag

## User Arguments
$ARGUMENTS
