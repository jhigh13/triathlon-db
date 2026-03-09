Train a new model version for the triathlon prediction pipeline.

## Context
- Current champion: Check MEMORY.md and models/ for latest bundle_elite_v*.joblib
- Training script: scripts/train_models.py
- Default training window: 2021-01-01 to 2025-06-30

## Steps
1. List models/ directory to find the highest existing version number. Increment by 1 for the new version
2. Run training:
   ```powershell
   python scripts/train_models.py --output models/bundle_elite_v{N}.joblib $ARGUMENTS
   ```
3. After training completes, report:
   - Number of training samples
   - Feature count
   - Per-split model scores from output
   - Training time
   - Any warnings or errors
4. If --cv or --cv_splits was used, summarize cross-validation results
5. Suggest running /backtest next with the new model path

## Safety Rules
- NEVER train with end_date past 2025-06-30 (backtest data leakage)
- ALWAYS increment version number -- never overwrite existing model files
- Training is logged to MLflow automatically

## User Arguments
$ARGUMENTS
