Run a full autonomous experiment cycle: implement a change, train, backtest, and evaluate.

## Context
This command runs the complete experiment loop. The user describes a hypothesis or code change to test. Claude implements the change, trains a new model, runs backtest, and reports results compared to the champion.

## Champion Baseline (from MEMORY.md)
- Model: bundle_elite_v45.joblib
- P@10=74.2%, P@3=57.0%, Spearman=0.796

## Experiment Protocol

### 1. UNDERSTAND
Parse the user's hypothesis or change description from $ARGUMENTS.

### 2. PLAN
- Identify which files need modification
- State the expected impact (which metric should improve and why)
- Identify risks (which metrics might regress)

### 3. IMPLEMENT
- Make focused, minimal code changes
- Do NOT refactor unrelated code

### 4. TRAIN
- Auto-detect next version number from models/ directory
- Run: `python scripts/train_models.py --output models/bundle_elite_v{N}.joblib`
- Report training summary (samples, features, time)

### 5. BACKTEST
- Run: `python scripts/run_backtest.py --model models/bundle_elite_v{N}.joblib --no_sim`
- Extract overall and per-tier metrics

### 6. EVALUATE
Compare against champion:
- Overall: P@3, P@10, Spearman, MAE
- Per-tier breakdown (1a, 1b, 2, 3, 4)
- vs start-number baseline

### 7. REPORT
Present structured results:
```
### Experiment: {description}
- **Hypothesis**: {what we expected to happen}
- **Changes**: {files modified, what changed}
- **Results**:
  | Metric | Champion | Experiment | Delta |
  |--------|----------|------------|-------|
  | P@10   | 74.2%    | XX.X%      | +X.X% |
  | P@3    | 57.0%    | XX.X%      | +X.X% |
  | Spearman | 0.796  | X.XXX      | +X.XXX |
- **Verdict**: PROMOTE / REJECT / INVESTIGATE
- **Next**: {suggested follow-up}
```

### 8. LOG
Append a row to docs/experiment-log.md with date, version, change, metrics, verdict.

## Safety Rules
- ALWAYS create a NEW model file (never overwrite existing)
- NEVER train past 2025-06-30
- If P@10 regresses by >3%, flag immediately and stop
- If training fails, diagnose the error before retrying
- State all code changes clearly before making them

## User Hypothesis
$ARGUMENTS
