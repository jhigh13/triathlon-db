Run diagnostic analysis on predictions for a specific event.

## Context
- Diagnostic script: scripts/debug_diagnostics.py
- Available sections: overview, importances, field, accounting, perturbation, simulation, athlete

## Steps
1. Parse event_id, prog_id, and model_path from $ARGUMENTS. If not provided, ask the user
2. Determine which sections to run (default: overview)
3. Run diagnostics:
   ```powershell
   python scripts/debug_diagnostics.py --event_id {ID} --prog_id {ID} --model_path {model} --section {section} $ARGUMENTS
   ```
4. Summarize findings:
   - **overview**: Field size, prediction vs actual rankings, biggest movers
   - **importances**: Top features driving predictions, any surprising feature weights
   - **field**: Field composition analysis, tier distribution, cold-start athlete count
   - **accounting**: Split-total consistency check, any accounting gaps
   - **simulation**: Pack formation diagnostics, MC simulation distribution
   - **athlete**: Deep-dive on specific athlete (requires --athlete flag)
5. If issues found, suggest potential improvements or further investigation

## User Arguments
$ARGUMENTS
