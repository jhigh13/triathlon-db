"""
Diagnostic: Why do sprint split models predict times so different from EMA?

Hypothesis: The EMA features (ema_run_sec_5 etc.) fall back to ALL distances
when an athlete has no matching-distance history. This means:
- Some training rows have sprint EMA values (reliable: ~900s for run)
- Some training rows have standard/mixed EMA values (unreliable: ~2100s for run)
The model can't distinguish these cases, so it underweights ema_run_sec_5 and
leans on other features (elo, tier_delta) which add noise to predictions.

This script:
1. Builds training data for sprint distance
2. Checks what fraction of sprint training rows have EMA values that look
   like they came from a different distance (likely fallback)
3. Shows how feature importances differ between "clean" and "fallback" subsets
4. Shows the La Paz field's EMA distribution vs sprint training distribution
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
from tri_analysis.prediction.train import load_model_bundle, build_training_dataset
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.utils_time import seconds_to_hms

engine = get_engine()
bundle = load_model_bundle("models/bundle_claudev22_men.joblib")
feature_cols = bundle.feature_columns

# ── 1. Build sprint training data ──
print("=" * 80)
print("STEP 1: Build sprint training dataset")
print("=" * 80)

train_df = build_training_dataset(
    engine,
    start_date="2020-01-01",
    end_date="2025-12-31",
    elite_only=True,
    distance_categories=["sprint"],
    match_distance=True,
    gender="men",
)

sprint_train = train_df[
    train_df["prog_distance_category"].str.lower().str.strip() == "sprint"
].copy()
print(f"\nSprint training rows: {len(sprint_train)}")

# ── 2. Check EMA values distribution ──
print("\n" + "=" * 80)
print("STEP 2: EMA value distribution in sprint training data")
print("=" * 80)

# A sprint 5km run for elite men is typically 14-18 minutes (840-1080s)
# Standard distance 10km run is typically 30-38 minutes (1800-2280s)
# If ema_run_sec_5 > 1500s in sprint training data, it's likely a fallback

for split, ema_col, target_col, sprint_range in [
    ("swim", "ema_swim_sec_5", "swim_sec", (400, 900)),    # sprint swim: 7-15 min
    ("bike", "ema_bike_sec_5", "bike_sec", (1200, 2200)),  # sprint bike: 20-37 min
    ("run", "ema_run_sec_5", "run_sec", (700, 1500)),      # sprint run: 12-25 min
    ("total", "ema_total_sec_5", "total_sec", (2800, 5000)),  # sprint total: 47-83 min
]:
    if ema_col not in sprint_train.columns or target_col not in sprint_train.columns:
        continue

    valid = sprint_train[[ema_col, target_col]].dropna()
    if valid.empty:
        continue

    ema_vals = valid[ema_col]
    target_vals = valid[target_col]

    # Count how many EMA values are outside sprint range (likely fallback)
    lo, hi = sprint_range
    in_range = ((ema_vals >= lo) & (ema_vals <= hi)).sum()
    above_range = (ema_vals > hi).sum()
    below_range = (ema_vals < lo).sum()

    print(f"\n  {split.upper()}: {ema_col}")
    print(f"    Total valid rows: {len(valid)}")
    print(f"    EMA in sprint range ({lo}-{hi}s): {in_range} ({in_range/len(valid)*100:.1f}%)")
    print(f"    EMA above sprint range (>{hi}s, likely fallback): {above_range} ({above_range/len(valid)*100:.1f}%)")
    print(f"    EMA below sprint range (<{lo}s): {below_range} ({below_range/len(valid)*100:.1f}%)")
    print(f"    EMA median: {ema_vals.median():.0f}s ({seconds_to_hms(int(ema_vals.median()))})")
    print(f"    EMA mean: {ema_vals.mean():.0f}s ({seconds_to_hms(int(ema_vals.mean()))})")
    print(f"    Target median: {target_vals.median():.0f}s ({seconds_to_hms(int(target_vals.median()))})")
    print(f"    Target mean: {target_vals.mean():.0f}s ({seconds_to_hms(int(target_vals.mean()))})")

    # Correlation between EMA and target
    corr = ema_vals.corr(target_vals)
    print(f"    Correlation (EMA vs actual): {corr:.3f}")

    # Split by in-range vs fallback
    mask_in = (ema_vals >= lo) & (ema_vals <= hi)
    mask_out = ema_vals > hi
    if mask_in.sum() > 10 and mask_out.sum() > 10:
        corr_in = ema_vals[mask_in].corr(target_vals[mask_in])
        corr_out = ema_vals[mask_out].corr(target_vals[mask_out])
        mae_in = (ema_vals[mask_in] - target_vals[mask_in]).abs().mean()
        mae_out = (ema_vals[mask_out] - target_vals[mask_out]).abs().mean()
        print(f"    In-range subset: corr={corr_in:.3f}, MAE={mae_in:.0f}s")
        print(f"    Fallback subset: corr={corr_out:.3f}, MAE={mae_out:.0f}s")

# ── 3. Compare with La Paz field ──
print("\n" + "=" * 80)
print("STEP 3: La Paz field EMA distribution vs sprint training")
print("=" * 80)

key_lapaz = ProgramKey(event_id=195253, prog_id=676981)
event_meta = fetch_event_metadata(engine, key_lapaz)
dist_cat = event_meta.get("prog_distance_category")

features_df = build_features_for_program(engine, key_lapaz, use_start_list=True)
features_df = fill_missing_features(features_df, feature_cols)

# Check EMA distribution
for ema_col, sprint_hi in [("ema_run_sec_5", 1500), ("ema_bike_sec_5", 2200), ("ema_swim_sec_5", 900)]:
    if ema_col not in features_df.columns:
        continue
    vals = features_df[ema_col].dropna()
    n_fallback = (vals > sprint_hi).sum()
    print(f"\n  La Paz {ema_col}:")
    print(f"    Athletes with valid EMA: {len(vals)}")
    print(f"    Likely fallback (>{sprint_hi}s): {n_fallback} ({n_fallback/len(vals)*100:.1f}%)")
    print(f"    Median: {vals.median():.0f}s ({seconds_to_hms(int(vals.median()))})")
    print(f"    Min: {vals.min():.0f}s ({seconds_to_hms(int(vals.min()))})")
    print(f"    Max: {vals.max():.0f}s ({seconds_to_hms(int(vals.max()))})")

# ── 4. Feature importance analysis for sprint run model ──
print("\n" + "=" * 80)
print("STEP 4: Sprint run model feature importances")
print("=" * 80)

dist_models = bundle.distance_split_models.get("sprint", {})
run_model = dist_models.get("run")

if run_model is not None:
    # Get the estimator from pipeline
    if hasattr(run_model, 'named_steps'):
        estimator = run_model.named_steps.get('regressor') or run_model[-1]
    else:
        estimator = run_model

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        print(f"\n  Top 20 features by importance (sprint run model):")
        for feat_name, imp in feat_imp[:20]:
            print(f"    {feat_name:40s}  importance={imp:.4f}")

        # Key question: how much does ema_run_sec_5 matter?
        ema_run_imp = dict(feat_imp).get("ema_run_sec_5", 0)
        total_imp = sum(imp for _, imp in feat_imp)
        print(f"\n  ema_run_sec_5 importance: {ema_run_imp:.4f} ({ema_run_imp/total_imp*100:.1f}% of total)")

# ── 5. Show what the model predicts for "clean" vs "fallback" athletes ──
print("\n" + "=" * 80)
print("STEP 5: Predictions for clean-EMA vs fallback-EMA athletes (La Paz)")
print("=" * 80)

pred_df = predict_splits_and_total(features_df, bundle, distance_category=dist_cat)

# Classify athletes
ema_run = pred_df["ema_run_sec_5"]
pred_run = pred_df["pred_run_sec"]
is_clean = (ema_run >= 700) & (ema_run <= 1500)  # Sprint range
is_fallback = ema_run > 1500

clean_athletes = pred_df[is_clean]
fallback_athletes = pred_df[is_fallback]
null_athletes = pred_df[ema_run.isna()]

print(f"  Clean EMA athletes (sprint range): {len(clean_athletes)}")
print(f"  Fallback EMA athletes (>1500s): {len(fallback_athletes)}")
print(f"  Null EMA athletes: {len(null_athletes)}")

if not clean_athletes.empty:
    run_gap_clean = (clean_athletes["pred_run_sec"] - clean_athletes["ema_run_sec_5"])
    print(f"\n  Clean athletes: pred_run - ema_run gap:")
    print(f"    Mean: {run_gap_clean.mean():+.0f}s")
    print(f"    Median: {run_gap_clean.median():+.0f}s")
    print(f"    Std: {run_gap_clean.std():.0f}s")

if not fallback_athletes.empty:
    run_gap_fb = (fallback_athletes["pred_run_sec"] - fallback_athletes["ema_run_sec_5"])
    print(f"\n  Fallback athletes: pred_run - ema_run gap:")
    print(f"    Mean: {run_gap_fb.mean():+.0f}s")
    print(f"    Median: {run_gap_fb.median():+.0f}s")
    print(f"    Std: {run_gap_fb.std():.0f}s")

# Show some specific athletes
print(f"\n  {'Athlete':35s}  {'EMA Run':>8s}  {'Pred Run':>9s}  {'Gap':>7s}  {'Type':>8s}")
for _, row in pred_df.head(25).iterrows():
    ema = row["ema_run_sec_5"]
    pred = row["pred_run_sec"]
    if pd.notna(ema):
        ema_str = seconds_to_hms(int(ema))
        gap = f"{pred - ema:+.0f}s"
        etype = "clean" if 700 <= ema <= 1500 else "fallback"
    else:
        ema_str = "N/A"
        gap = "N/A"
        etype = "null"
    pred_str = seconds_to_hms(int(pred))
    print(f"  {row['athlete_full_name']:35s}  {ema_str:>8s}  {pred_str:>9s}  {gap:>7s}  {etype:>8s}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
