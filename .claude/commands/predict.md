Generate predictions for an upcoming triathlon event.

## Context
- Prediction script: scripts/predict_program.py
- Find events: python scripts/find_events.py
- Champion model: latest bundle_elite_v*.joblib in models/ (check MEMORY.md for current champion)

## Steps
1. If user didn't provide event_id and prog_id, run find_events.py to show upcoming options:
   ```powershell
   python scripts/find_events.py
   ```
   Then ask user to pick an event
2. Identify the champion model path from models/ directory
3. Run prediction:
   ```powershell
   python scripts/predict_program.py --event_id {ID} --prog_id {ID} --model_path {champion_model} $ARGUMENTS
   ```
4. After prediction completes, summarize:
   - Top-20 deterministic rankings with predicted times
   - MC simulation probabilities (win%, podium%, top-10%) for top athletes
   - Pack formation analysis
   - Any cold-start warnings (athletes with sparse history)
5. Note the output CSV path saved to outputs/

## Tips
- Use --no_mc for faster deterministic-only predictions
- Weather data is auto-fetched when available
- For debugging a specific prediction, use /diagnostics

## User Arguments
$ARGUMENTS
