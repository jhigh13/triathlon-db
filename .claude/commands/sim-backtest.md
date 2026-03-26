Run a Monte Carlo simulation backtest with swim prediction diagnostics.

## Context
- Backtest script: scripts/run_backtest.py (WITH simulation, no --no_sim flag)
- Swim accuracy script: scripts/eval_swim_accuracy.py
- Sim parameter sweep: scripts/sweep_sim_params.py
- Backtest period: H2 2025 (~90 events across all tiers)
- Champion deterministic baseline: P@10=74.2%, P@3=57.0%, Spearman=0.796
- Known MC gap: sim P@3 is ~19 points below deterministic (structural vs fixable TBD)

## Steps
1. Identify model to test from $ARGUMENTS, or find the latest bundle_elite_v*.joblib in models/
2. Run swim accuracy evaluation (fast, no simulation):
   ```powershell
   python scripts/eval_swim_accuracy.py --model {model_path}
   ```
3. Parse swim accuracy output:
   - Swim Spearman (>0.8 = good, <0.6 = major error source)
   - Front Pack Recall (>0.8 = good, <0.5 = pack formation unreliable)
   - Gap MAE (<5s = good, >15s = noisy)
   - Note tier/distance/field-size breakdowns
4. Run full MC simulation backtest:
   ```powershell
   python scripts/run_backtest.py --model {model_path}
   ```
5. Parse simulation output and extract:
   - Deterministic: P@3, P@10, Spearman, MAE
   - Simulation: sim_P@3, sim_P@10, sim_Spearman, sim_MAE
   - Det vs Sim gap (the 19pt gap we're trying to close)
   - By-tier simulation breakdown
6. Present analysis:
   - Is swim ordering the bottleneck? (swim Spearman vs overall Spearman)
   - Is pack formation reliable? (front pack recall/precision)
   - What is the actual det→sim gap? Has it changed?
   - Which tiers/distances suffer most in simulation?
7. Suggest next steps based on findings:
   - If swim Spearman < 0.7: Focus on improving swim prediction model
   - If front pack recall < 0.6: Focus on pack formation algorithm
   - If gap MAE > 10s: Focus on gap structure modeling
   - If sim gap is small for some tiers: Focus on worst-performing tiers

## Optional: Parameter Sweep
If user requests parameter sensitivity analysis:
```powershell
python scripts/sweep_sim_params.py --model {model_path} --param all --n_sims 500
```
This sweeps sigma_swim_scale, pack_gap_sec, front_pack_bonus_sec, chase_penalty_sec, and pack_effect_scale one at a time. Takes longer but identifies which parameters the simulation is most sensitive to.

## User Arguments
$ARGUMENTS
