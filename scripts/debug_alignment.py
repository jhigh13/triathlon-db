"""
Debug: Check data alignment through the pipeline and diagnose analyze_predictions issues.

1. Verify athlete-feature alignment through predict → simulate → CSV → analyze
2. Check if analyze_predictions mixes up athletes due to model mismatch
3. Show the Det. Rank vs E[Rank] discrepancy root cause
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)

from tri_analysis.database import get_engine
from tri_analysis.prediction.sql import ProgramKey, fetch_event_metadata
from tri_analysis.prediction.features import build_features_for_program, fill_missing_features
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.utils_time import seconds_to_hms

engine = get_engine()
key = ProgramKey(event_id=195253, prog_id=676981)
event_meta = fetch_event_metadata(engine, key)
distance_cat = event_meta.get("prog_distance_category")

# ── BUILD FEATURES ──
print("Building features...")
features_df = build_features_for_program(engine, key, use_start_list=True)

bundle_v22 = load_model_bundle("models/bundle_claudev26_men.joblib")
feature_cols_v22 = bundle_v22.feature_columns
features_df = fill_missing_features(features_df, feature_cols_v22)

# ── PREDICT WITH v22 ──
print("\n" + "=" * 80)
print("STEP 1: Alignment check BEFORE predict_splits_and_total")
print("=" * 80)

# Show that features match athletes before prediction
print("\nBefore prediction (first 5 athletes):")
print(f"  {'athlete_full_name':40s}  ema_run_sec_5  elo_rating")
for _, row in features_df.head(5).iterrows():
    ema_run = seconds_to_hms(int(row["ema_run_sec_5"])) if pd.notna(row["ema_run_sec_5"]) else "N/A"
    print(f"  {row['athlete_full_name']:40s}  {ema_run:>12s}  {row['elo_rating']:.0f}")

pred_df = predict_splits_and_total(features_df, bundle_v22, distance_category=distance_cat)

print("\nAfter prediction (sorted by predicted_rank, first 10):")
print(f"  {'Rank':>4s}  {'athlete_full_name':40s}  ema_run_sec   pred_run   pred_total   finish_pct")
for _, row in pred_df.head(10).iterrows():
    ema_run = seconds_to_hms(int(row["ema_run_sec_5"])) if pd.notna(row["ema_run_sec_5"]) else "N/A"
    pred_run = seconds_to_hms(int(row["pred_run_sec"])) if pd.notna(row["pred_run_sec"]) else "N/A"
    pred_total = seconds_to_hms(int(row["pred_total_sec"])) if pd.notna(row["pred_total_sec"]) else "N/A"
    finish_pct = f"{row.get('pred_finish_pct', float('nan')):.3f}" if pd.notna(row.get('pred_finish_pct')) else "N/A"
    print(f"  {row['predicted_rank']:4d}  {row['athlete_full_name']:40s}  {ema_run:>10s}  {pred_run:>9s}  {pred_total:>11s}  {finish_pct:>10s}")

# ── KEY CHECK: Do split sums match total? ──
print("\n" + "=" * 80)
print("STEP 2: Split sum vs Total model prediction (ALL athletes)")
print("=" * 80)

pred_df["split_sum"] = pred_df["pred_swim_sec"] + pred_df["pred_bike_sec"] + pred_df["pred_run_sec"]
pred_df["split_total_gap"] = pred_df["split_sum"] - pred_df["pred_total_sec"]

print(f"\n  {'Athlete':40s}  {'Split Sum':>10s}  {'Total Model':>12s}  {'Gap':>6s}  {'Run':>8s}  {'EMA Run':>8s}  {'Run Gap':>8s}")
for _, row in pred_df.iterrows():
    split_sum = seconds_to_hms(int(row["split_sum"]))
    pred_total = seconds_to_hms(int(row["pred_total_sec"]))
    gap = f"{row['split_total_gap']:+.0f}s"
    pred_run = seconds_to_hms(int(row["pred_run_sec"]))
    ema_run = seconds_to_hms(int(row["ema_run_sec_5"])) if pd.notna(row["ema_run_sec_5"]) else "N/A"
    run_gap = f"{row['pred_run_sec'] - row['ema_run_sec_5']:+.0f}s" if pd.notna(row["ema_run_sec_5"]) else "N/A"
    print(f"  {row['athlete_full_name']:40s}  {split_sum:>10s}  {pred_total:>12s}  {gap:>6s}  {pred_run:>8s}  {ema_run:>8s}  {run_gap:>8s}")

# ── CHECK: How many athletes have run prediction far from their EMA? ──
print("\n" + "=" * 80)
print("STEP 3: Run prediction vs EMA (sorted by gap)")
print("=" * 80)

valid_run = pred_df[pred_df["ema_run_sec_5"].notna()].copy()
valid_run["run_gap"] = valid_run["pred_run_sec"] - valid_run["ema_run_sec_5"]
valid_run = valid_run.sort_values("run_gap", ascending=False)

print(f"\n  Athletes with run prediction > 120s off from EMA:")
print(f"  {'Athlete':40s}  {'Pred Run':>9s}  {'EMA Run':>8s}  {'Gap':>8s}  {'EMA Total':>10s}")
big_gaps = valid_run[valid_run["run_gap"].abs() > 120]
for _, row in big_gaps.iterrows():
    pred_run = seconds_to_hms(int(row["pred_run_sec"]))
    ema_run = seconds_to_hms(int(row["ema_run_sec_5"]))
    ema_total = seconds_to_hms(int(row["ema_total_sec_5"])) if pd.notna(row["ema_total_sec_5"]) else "N/A"
    print(f"  {row['athlete_full_name']:40s}  {pred_run:>9s}  {ema_run:>8s}  {row['run_gap']:>+7.0f}s  {ema_total:>10s}")

print(f"\n  Total athletes: {len(valid_run)}")
print(f"  Athletes with |run_gap| > 120s: {len(big_gaps)}")
print(f"  Athletes with |run_gap| > 60s: {len(valid_run[valid_run['run_gap'].abs() > 60])}")
print(f"  Median |run_gap|: {valid_run['run_gap'].abs().median():.0f}s")
print(f"  Mean run_gap: {valid_run['run_gap'].mean():.0f}s (positive = model predicts SLOWER than EMA)")

# ── ANALYZE SCRIPT ISSUE: Model mismatch ──
print("\n" + "=" * 80)
print("STEP 4: analyze_predictions model mismatch")
print("=" * 80)

if os.path.exists("models/bundle_claudev16.joblib"):
    bundle_v16 = load_model_bundle("models/bundle_claudev16.joblib")
    pred_df_v16 = predict_splits_and_total(features_df, bundle_v16, distance_category=distance_cat)

    # Find Blake in both
    blake_v21 = pred_df[pred_df["athlete_full_name"].str.contains("Blake Bullard", case=False, na=False)]
    blake_v16 = pred_df_v16[pred_df_v16["athlete_full_name"].str.contains("Blake Bullard", case=False, na=False)]

    print(f"\n  Blake Bullard predictions comparison:")
    print(f"  {'':20s}  {'claudev21':>12s}  {'claudev16':>12s}")
    print(f"  {'Predicted Rank':20s}  {blake_v21['predicted_rank'].iloc[0]:>12d}  {blake_v16['predicted_rank'].iloc[0]:>12d}")
    print(f"  {'Pred Total':20s}  {blake_v21['pred_total_hms'].iloc[0]:>12s}  {blake_v16['pred_total_hms'].iloc[0]:>12s}")
    print(f"  {'Pred Run':20s}  {blake_v21['pred_run_hms'].iloc[0]:>12s}  {blake_v16['pred_run_hms'].iloc[0]:>12s}")
    print(f"  {'Pred Swim':20s}  {blake_v21['pred_swim_hms'].iloc[0]:>12s}  {blake_v16['pred_swim_hms'].iloc[0]:>12s}")

    # Show top 5 from each model
    print(f"\n  Top 5 athletes by Det. Rank — claudev21 vs claudev16:")
    print(f"  {'claudev21 Top 5':40s}  {'claudev16 Top 5':40s}")
    for i in range(5):
        name_v21 = pred_df.iloc[i]["athlete_full_name"] if i < len(pred_df) else ""
        name_v16 = pred_df_v16.iloc[i]["athlete_full_name"] if i < len(pred_df_v16) else ""
        rank_v21 = pred_df.iloc[i]["predicted_rank"] if i < len(pred_df) else ""
        rank_v16 = pred_df_v16.iloc[i]["predicted_rank"] if i < len(pred_df_v16) else ""
        print(f"  {rank_v21:>2}. {name_v21:37s}  {rank_v16:>2}. {name_v16:37s}")

    # Show Fierro specifically
    fierro_v21 = pred_df[pred_df["athlete_full_name"].str.contains("Fierro", case=False, na=False)]
    fierro_v16 = pred_df_v16[pred_df_v16["athlete_full_name"].str.contains("Fierro", case=False, na=False)]
    if not fierro_v21.empty and not fierro_v16.empty:
        print(f"\n  Osvaldo Fierro rank comparison:")
        print(f"    claudev21 rank: {fierro_v21['predicted_rank'].iloc[0]}")
        print(f"    claudev16 rank: {fierro_v16['predicted_rank'].iloc[0]}")
else:
    print("  bundle_claudev16.joblib not found, skipping model comparison")

# ── STEP 5: Is the simulation itself correct? Spot check. ──
print("\n" + "=" * 80)
print("STEP 5: Simulation alignment spot-check")
print("=" * 80)

from tri_analysis.prediction.simulate import run_monte_carlo, PackEffectParams, pack_params_from_dict, merge_params_from_dict

pack_params_dict = bundle_v22.metadata.get("pack_effect_params")
pack_params = pack_params_from_dict(pack_params_dict) if pack_params_dict else PackEffectParams()
merge_params_dict = bundle_v22.metadata.get("merge_params")
merge_params = merge_params_from_dict(merge_params_dict) if merge_params_dict else None

# Run small simulation for spot-checking
sim_df = run_monte_carlo(pred_df, n_sims=100, random_state=42,
                         pack_params=pack_params, merge_params=merge_params,
                         distance_category=distance_cat)

# Check that athlete names still match their features after simulation
print(f"\n  Spot-check: athlete features preserved after simulation?")
blake_sim = sim_df[sim_df["athlete_full_name"].str.contains("Blake Bullard", case=False, na=False)]
if not blake_sim.empty:
    print(f"  Blake ema_run_sec_5 in sim_df: {blake_sim['ema_run_sec_5'].iloc[0]:.0f}s ({seconds_to_hms(int(blake_sim['ema_run_sec_5'].iloc[0]))})")
    print(f"  Blake elo_rating in sim_df: {blake_sim['elo_rating'].iloc[0]:.0f}")
    print(f"  Blake pred_run_sec in sim_df: {blake_sim['pred_run_sec'].iloc[0]:.0f}s ({seconds_to_hms(int(blake_sim['pred_run_sec'].iloc[0]))})")
    print(f"  Blake prob_win: {blake_sim['prob_win'].iloc[0]:.3f}")
    print(f"  Blake expected_rank: {blake_sim['expected_rank'].iloc[0]:.1f}")

# Also check the CSV output alignment
csv_path = "outputs/predictions_195253_676981.csv"
if os.path.exists(csv_path):
    print(f"\n  Checking CSV alignment...")
    csv_df = pd.read_csv(csv_path)
    blake_csv = csv_df[csv_df["athlete_full_name"].str.contains("Blake Bullard", case=False, na=False)]
    if not blake_csv.empty:
        print(f"  Blake in CSV: ema_run={blake_csv['ema_run_sec_5'].iloc[0]:.0f}s, "
              f"pred_run={blake_csv['pred_run_sec'].iloc[0]:.0f}s, "
              f"elo={blake_csv['elo_rating'].iloc[0]:.0f}")
        # Cross-check: does the elo_rating match between pred_df and csv?
        blake_pred_elo = pred_df[pred_df["athlete_full_name"].str.contains("Blake Bullard", case=False, na=False)]["elo_rating"].iloc[0]
        blake_csv_elo = blake_csv["elo_rating"].iloc[0]
        print(f"  Elo match check: pred_df={blake_pred_elo:.0f}, csv={blake_csv_elo:.0f} → {'MATCH' if abs(blake_pred_elo - blake_csv_elo) < 0.1 else 'MISMATCH!'}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
