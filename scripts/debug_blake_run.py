"""
Diagnostic script: Why is Blake Bullard's predicted run 19:52 when his EMA run is 14:52?

Traces through the full pipeline to find the root cause.
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.sql import ProgramKey, fetch_event_metadata, fetch_athlete_history, fetch_start_list
from tri_analysis.prediction.features import (
    build_features_for_program,
    fill_missing_features,
    compute_athlete_form_features,
)
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.utils_time import seconds_to_hms
from tri_analysis.database import get_engine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

engine = get_engine()
key = ProgramKey(event_id=195253, prog_id=676981)
bundle_path = "models/bundle_claudev26_men.joblib"

event_meta = fetch_event_metadata(engine, key)
event_date = pd.to_datetime(event_meta["event_date"]).date()
distance_category = event_meta.get("prog_distance_category")

print("=" * 80)
print(f"EVENT: {event_meta.get('event_name')} - {event_date}")
print(f"DISTANCE: {distance_category}")
print("=" * 80)

# ── Step 1: Find Blake's athlete_id from start list ──
start_list = fetch_start_list(engine, key)
blake_row = start_list[start_list["athlete_full_name"].str.contains("Blake Bullard", case=False, na=False)]
if blake_row.empty:
    print("Blake Bullard not found in start list!")
    sys.exit(1)

blake_id = int(blake_row["athlete_id"].iloc[0])
print(f"\nBlake Bullard athlete_id: {blake_id}")

# ── Step 2: Check Blake's raw race history ──
print("\n" + "=" * 80)
print("STEP 2: Blake's RAW race history (all distances)")
print("=" * 80)

all_history = fetch_athlete_history(engine, blake_id, event_date, limit=50, distance_category=None, elite_only=True)
if not all_history.empty:
    cols_to_show = ["event_date", "event_name", "prog_name", "prog_distance_category", "swimtime", "biketime", "runtime", "total_time"]
    available = [c for c in cols_to_show if c in all_history.columns]
    print(all_history[available].to_string(index=False))
else:
    print("No history found!")

# ── Step 3: Check distance-filtered history ──
print("\n" + "=" * 80)
print(f"STEP 3: Blake's SPRINT-FILTERED history (distance={distance_category})")
print("=" * 80)

sprint_history = fetch_athlete_history(engine, blake_id, event_date, limit=50, distance_category=distance_category, elite_only=True)
if not sprint_history.empty:
    available = [c for c in cols_to_show if c in sprint_history.columns]
    print(sprint_history[available].to_string(index=False))
else:
    print("No sprint history found! Will fall back to all distances.")

# ── Step 4: Compute form features manually ──
print("\n" + "=" * 80)
print("STEP 4: Computed form features for Blake")
print("=" * 80)

# Replicate what build_features_for_program does
if distance_category and not all_history.empty and "prog_distance_category" in all_history.columns:
    matched_history = all_history[
        all_history["prog_distance_category"].str.lower() == distance_category.lower()
    ].copy()
    if matched_history.empty:
        matched_history = all_history
        print("WARNING: No sprint-specific history, fell back to all distances!")
    else:
        print(f"Using {len(matched_history)} sprint-specific races for form features")
else:
    matched_history = all_history

form_feats = compute_athlete_form_features(matched_history, event_date)
print(f"\n  ema_swim_sec_5: {seconds_to_hms(int(form_feats['ema_swim_sec_5'])) if form_feats['ema_swim_sec_5'] else 'None'} ({form_feats['ema_swim_sec_5']}s)")
print(f"  ema_bike_sec_5: {seconds_to_hms(int(form_feats['ema_bike_sec_5'])) if form_feats['ema_bike_sec_5'] else 'None'} ({form_feats['ema_bike_sec_5']}s)")
print(f"  ema_run_sec_5:  {seconds_to_hms(int(form_feats['ema_run_sec_5'])) if form_feats['ema_run_sec_5'] else 'None'} ({form_feats['ema_run_sec_5']}s)")
print(f"  ema_total_sec_5:{seconds_to_hms(int(form_feats['ema_total_sec_5'])) if form_feats['ema_total_sec_5'] else 'None'} ({form_feats['ema_total_sec_5']}s)")

# Verify: compute EMA manually from the raw data
from tri_analysis.prediction.utils_time import parse_time_to_seconds
df_check = matched_history.copy()
df_check["runtime_sec"] = df_check["runtime"].apply(parse_time_to_seconds)
df_check = df_check.sort_values("event_date", ascending=False)
valid_runs = df_check["runtime_sec"].dropna()
print(f"\n  Raw sprint run times (most recent first):")
for i, (idx, row) in enumerate(df_check.iterrows()):
    rt_sec = row.get("runtime_sec")
    rt_hms = seconds_to_hms(int(rt_sec)) if pd.notna(rt_sec) else "N/A"
    dist = row.get("prog_distance_category", "?")
    print(f"    {row.get('event_date', '?')}: {rt_hms} ({rt_sec}s) [{dist}] - {row.get('event_name', '?')}")
    if i >= 9:
        break

# ── Step 5: Build full features and predict ──
print("\n" + "=" * 80)
print("STEP 5: Full pipeline prediction for Blake")
print("=" * 80)

bundle = load_model_bundle(bundle_path)
features_df = build_features_for_program(engine, key, use_start_list=True)
feature_cols = bundle.feature_columns or []
features_df = fill_missing_features(features_df, feature_cols)

blake_features = features_df[features_df["athlete_id"] == blake_id]
if blake_features.empty:
    print("Blake not found in features_df!")
    sys.exit(1)

# Show all feature values for Blake
print("\nBlake's feature values:")
for col in feature_cols:
    val = blake_features[col].iloc[0]
    if isinstance(val, float) and abs(val) > 100:
        print(f"  {col:40s} = {val:.1f} ({seconds_to_hms(int(val))})")
    else:
        print(f"  {col:40s} = {val}")

# ── Step 6: Predict and show split details ──
print("\n" + "=" * 80)
print("STEP 6: Model predictions for Blake")
print("=" * 80)

pred_df = predict_splits_and_total(features_df, bundle, distance_category=distance_category)
blake_pred = pred_df[pred_df["athlete_id"] == blake_id]

print(f"  Pred Swim:  {blake_pred['pred_swim_hms'].iloc[0]} ({blake_pred['pred_swim_sec'].iloc[0]:.0f}s)")
print(f"  Pred Bike:  {blake_pred['pred_bike_hms'].iloc[0]} ({blake_pred['pred_bike_sec'].iloc[0]:.0f}s)")
print(f"  Pred Run:   {blake_pred['pred_run_hms'].iloc[0]} ({blake_pred['pred_run_sec'].iloc[0]:.0f}s)")
print(f"  Pred Total: {blake_pred['pred_total_hms'].iloc[0]} ({blake_pred['pred_total_sec'].iloc[0]:.0f}s)")
print(f"  Det. Rank:  {blake_pred['predicted_rank'].iloc[0]}")

# ── Step 7: Sprint run model feature importances ──
print("\n" + "=" * 80)
print("STEP 7: Sprint run model - feature importances")
print("=" * 80)

dist_models = bundle.distance_split_models.get("sprint", {})
run_model = dist_models.get("run")

if run_model is None:
    print("No distance-specific sprint run model found! Using unified run model.")
    run_model = bundle.model_run

if run_model is not None:
    # Get the actual estimator from the pipeline
    if hasattr(run_model, 'named_steps'):
        # It's a sklearn Pipeline
        estimator = run_model.named_steps.get('regressor') or run_model.named_steps.get('model')
        if estimator is None:
            # Try last step
            estimator = run_model[-1]
    else:
        estimator = run_model

    # Get feature importances
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        print(f"\n  Top 15 features by importance (sprint run model):")
        for feat_name, imp in feat_imp[:15]:
            val = blake_features[feat_name].iloc[0] if feat_name in blake_features.columns else "N/A"
            if isinstance(val, float) and abs(val) > 100:
                val_str = f"{val:.1f} ({seconds_to_hms(int(val))})"
            else:
                val_str = f"{val}"
            print(f"    {feat_name:40s}  importance={imp:.4f}  blake_val={val_str}")

    # ── Step 8: SHAP-like analysis using tree prediction path ──
    print("\n" + "=" * 80)
    print("STEP 8: Per-feature contribution to Blake's run prediction")
    print("=" * 80)

    X_blake = blake_features[feature_cols].copy()
    X_blake = X_blake.fillna(X_blake.median())

    # For LightGBM, we can get leaf predictions
    if hasattr(estimator, 'predict'):
        # Get prediction through pipeline
        blake_run_pred = run_model.predict(X_blake)
        print(f"\n  Sprint run model prediction for Blake: {seconds_to_hms(int(blake_run_pred[0]))} ({blake_run_pred[0]:.1f}s)")

        # Compare with field
        X_all = features_df[feature_cols].copy().fillna(features_df[feature_cols].median())
        all_run_preds = run_model.predict(X_all)
        print(f"  Field median run prediction: {seconds_to_hms(int(np.median(all_run_preds)))} ({np.median(all_run_preds):.1f}s)")
        print(f"  Field min run prediction: {seconds_to_hms(int(np.min(all_run_preds)))} ({np.min(all_run_preds):.1f}s)")
        print(f"  Field max run prediction: {seconds_to_hms(int(np.max(all_run_preds)))} ({np.max(all_run_preds):.1f}s)")

        # Perturbation analysis: how much does each feature contribute?
        print(f"\n  Perturbation analysis (replace each feature with field median):")
        median_vals = features_df[feature_cols].median()
        baseline_pred = blake_run_pred[0]

        contributions = []
        for col in feature_cols:
            X_perturbed = X_blake.copy()
            X_perturbed[col] = median_vals[col]
            perturbed_pred = run_model.predict(X_perturbed)[0]
            delta = baseline_pred - perturbed_pred  # positive = this feature makes prediction HIGHER
            contributions.append((col, delta, X_blake[col].iloc[0], median_vals[col]))

        contributions.sort(key=lambda x: -abs(x[1]))
        print(f"\n  {'Feature':40s} {'Impact':>8s}   {'Blake':>12s}   {'Median':>12s}")
        print(f"  {'-'*40} {'-'*8}   {'-'*12}   {'-'*12}")
        for feat_name, delta, blake_val, med_val in contributions[:20]:
            sign = "+" if delta > 0 else "-" if delta < 0 else " "
            if isinstance(blake_val, float) and abs(blake_val) > 100:
                bv = seconds_to_hms(int(blake_val))
                mv = seconds_to_hms(int(med_val))
            else:
                bv = f"{blake_val:.3f}" if isinstance(blake_val, float) else str(blake_val)
                mv = f"{med_val:.3f}" if isinstance(med_val, float) else str(med_val)
            print(f"  {feat_name:40s} {sign}{abs(delta):>6.1f}s   {bv:>12s}   {mv:>12s}")

# ── Step 9: Compare Blake with top-ranked athlete ──
print("\n" + "=" * 80)
print("STEP 9: Compare Blake vs top-ranked athlete (Ricardo Morales)")
print("=" * 80)

ricardo = features_df[features_df["athlete_full_name"].str.contains("Ricardo Morales", case=False, na=False)]
if not ricardo.empty:
    print(f"\n  {'Feature':40s} {'Blake':>12s}   {'Ricardo':>12s}")
    print(f"  {'-'*40} {'-'*12}   {'-'*12}")
    for col in feature_cols:
        bv = blake_features[col].iloc[0]
        rv = ricardo[col].iloc[0]
        if isinstance(bv, float) and abs(bv) > 100:
            bv_str = f"{seconds_to_hms(int(bv))}"
            rv_str = f"{seconds_to_hms(int(rv))}" if pd.notna(rv) and isinstance(rv, float) else str(rv)
        else:
            bv_str = f"{bv:.3f}" if isinstance(bv, float) else str(bv)
            rv_str = f"{rv:.3f}" if isinstance(rv, float) and pd.notna(rv) else str(rv)
        print(f"  {col:40s} {bv_str:>12s}   {rv_str:>12s}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
