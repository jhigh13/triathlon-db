"""
Fast diagnostic: Why do sprint split models predict times so different from EMA?

Analyzes the model and La Paz field directly (no training rebuild needed).
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
bundle = load_model_bundle("models/bundle_claudev26_men.joblib")
feature_cols = bundle.feature_columns

# ── Load La Paz data ──
key = ProgramKey(event_id=195253, prog_id=676981)
event_meta = fetch_event_metadata(engine, key)
dist_cat = event_meta.get("prog_distance_category")

print("Building features for La Paz...")
features_df = build_features_for_program(engine, key, use_start_list=True)
features_df = fill_missing_features(features_df, feature_cols)
pred_df = predict_splits_and_total(features_df, bundle, distance_category=dist_cat)

# ── 1. Feature importances for sprint split models ──
print("\n" + "=" * 80)
print("SPRINT MODEL FEATURE IMPORTANCES")
print("=" * 80)

dist_models = bundle.distance_split_models.get("sprint", {})

for split_name in ["swim", "bike", "run"]:
    model = dist_models.get(split_name)
    if model is None:
        print(f"\n  No sprint-specific {split_name} model found")
        continue

    # Extract estimator from pipeline
    if hasattr(model, 'named_steps'):
        estimator = model.named_steps.get('regressor') or model[-1]
    else:
        estimator = model

    if not hasattr(estimator, 'feature_importances_'):
        print(f"\n  {split_name} model has no feature_importances_")
        continue

    importances = estimator.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    print(f"\n  --- Sprint {split_name.upper()} Model: Top 15 Features ---")
    ema_col = f"ema_{split_name}_sec_5"
    for feat_name, imp in feat_imp[:15]:
        marker = " <<<" if feat_name == ema_col else ""
        print(f"    {feat_name:40s}  {imp:.4f}{marker}")

    # Show where the matching EMA ranks
    ema_rank = next((i+1 for i, (f, _) in enumerate(feat_imp) if f == ema_col), None)
    ema_imp = dict(feat_imp).get(ema_col, 0)
    print(f"\n    {ema_col} ranks #{ema_rank} with importance {ema_imp:.4f}")

# ── Also check unified total model ──
if bundle.model_total is not None:
    print(f"\n  --- Unified TOTAL Model: Top 15 Features ---")
    model = bundle.model_total
    if hasattr(model, 'named_steps'):
        estimator = model.named_steps.get('regressor') or model[-1]
    else:
        estimator = model
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        for feat_name, imp in feat_imp[:15]:
            print(f"    {feat_name:40s}  {imp:.4f}")

# ── Also check percentile model ──
if bundle.model_total_pct is not None:
    print(f"\n  --- Percentile (finish_pct) Model: Top 15 Features ---")
    model = bundle.model_total_pct
    if hasattr(model, 'named_steps'):
        estimator = model.named_steps.get('regressor') or model[-1]
    else:
        estimator = model
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        for feat_name, imp in feat_imp[:15]:
            print(f"    {feat_name:40s}  {imp:.4f}")

# ── 2. La Paz field EMA distribution ──
print("\n" + "=" * 80)
print("LA PAZ FIELD: EMA DISTRIBUTION")
print("=" * 80)

for split_name, ema_col, pred_col, sprint_max in [
    ("swim", "ema_swim_sec_5", "pred_swim_sec", 900),
    ("bike", "ema_bike_sec_5", "pred_bike_sec", 2200),
    ("run", "ema_run_sec_5", "pred_run_sec", 1500),
    ("total", "ema_total_sec_5", "pred_total_sec", 5000),
]:
    ema = pred_df[ema_col] if ema_col in pred_df.columns else pd.Series()
    pred = pred_df[pred_col] if pred_col in pred_df.columns else pd.Series()

    valid = pred_df[ema.notna()].copy()
    null_ema = pred_df[ema.isna()].copy()

    n_fallback = (ema > sprint_max).sum()
    print(f"\n  {split_name.upper()}: {ema_col}")
    print(f"    Athletes with EMA: {len(valid)} / {len(pred_df)}")
    print(f"    Athletes with null EMA: {len(null_ema)}")
    if not ema.dropna().empty:
        print(f"    Likely fallback (>{sprint_max}s): {n_fallback} ({n_fallback/len(ema.dropna())*100:.1f}%)")
        print(f"    EMA: median={ema.dropna().median():.0f}s ({seconds_to_hms(int(ema.dropna().median()))}), "
              f"min={ema.dropna().min():.0f}s, max={ema.dropna().max():.0f}s")
    if not pred.empty:
        print(f"    Pred: median={pred.median():.0f}s ({seconds_to_hms(int(pred.median()))}), "
              f"min={pred.min():.0f}s, max={pred.max():.0f}s")

    # Show pred-ema gap distribution
    if not valid.empty:
        gap = valid[pred_col] - valid[ema_col]
        print(f"    Pred-EMA gap: mean={gap.mean():+.0f}s, median={gap.median():+.0f}s, "
              f"std={gap.std():.0f}s")

# ── 3. Perturbation analysis: which features drive the prediction away from EMA? ──
print("\n" + "=" * 80)
print("PERTURBATION: What pushes sprint run predictions away from EMA?")
print("=" * 80)

run_model = dist_models.get("run")
if run_model is not None:
    X = pred_df[feature_cols].copy()
    X = X.fillna(X.median())

    baseline_preds = run_model.predict(X)
    median_vals = X.median()

    # For each feature, replace with field median and see impact on ALL athletes
    feature_impacts = []
    for col in feature_cols:
        X_perturbed = X.copy()
        X_perturbed[col] = median_vals[col]
        perturbed_preds = run_model.predict(X_perturbed)
        # Mean absolute impact across all athletes
        mean_impact = (baseline_preds - perturbed_preds).mean()
        std_impact = (baseline_preds - perturbed_preds).std()
        feature_impacts.append((col, mean_impact, std_impact))

    feature_impacts.sort(key=lambda x: -abs(x[1]))

    print(f"\n  Features ranked by MEAN impact on prediction (across all 64 athletes):")
    print(f"  (positive = feature pushes predictions SLOWER on average)")
    print(f"  {'Feature':40s}  {'Mean Impact':>12s}  {'Std Impact':>12s}")
    for feat_name, mean_imp, std_imp in feature_impacts[:20]:
        sign = "+" if mean_imp > 0 else "-" if mean_imp < 0 else " "
        print(f"  {feat_name:40s}  {sign}{abs(mean_imp):>10.1f}s  {std_imp:>11.1f}s")

    # ── 4. Null vs non-null EMA athletes ──
    print(f"\n" + "=" * 80)
    print("NULL EMA ATHLETES: Do they get reasonable predictions?")
    print("=" * 80)

    null_ema_run = pred_df[pred_df["ema_run_sec_5"].isna()]
    valid_ema_run = pred_df[pred_df["ema_run_sec_5"].notna()]

    print(f"  Athletes with null ema_run_sec_5: {len(null_ema_run)}")
    if not null_ema_run.empty:
        print(f"  Their pred_run: mean={null_ema_run['pred_run_sec'].mean():.0f}s "
              f"({seconds_to_hms(int(null_ema_run['pred_run_sec'].mean()))}), "
              f"median={null_ema_run['pred_run_sec'].median():.0f}s")

    print(f"\n  Athletes with valid ema_run_sec_5: {len(valid_ema_run)}")
    if not valid_ema_run.empty:
        print(f"  Their pred_run: mean={valid_ema_run['pred_run_sec'].mean():.0f}s "
              f"({seconds_to_hms(int(valid_ema_run['pred_run_sec'].mean()))}), "
              f"median={valid_ema_run['pred_run_sec'].median():.0f}s")
        print(f"  Their ema_run: mean={valid_ema_run['ema_run_sec_5'].mean():.0f}s "
              f"({seconds_to_hms(int(valid_ema_run['ema_run_sec_5'].mean()))}), "
              f"median={valid_ema_run['ema_run_sec_5'].median():.0f}s")

    # ── 5. For athletes with BIG pred-ema gaps, what features are driving it? ──
    print(f"\n" + "=" * 80)
    print("ATHLETES WITH BIGGEST PRED-EMA RUN GAPS")
    print("=" * 80)

    valid_run = pred_df[pred_df["ema_run_sec_5"].notna()].copy()
    valid_run["run_gap"] = valid_run["pred_run_sec"] - valid_run["ema_run_sec_5"]
    worst = valid_run.nlargest(5, "run_gap")

    for _, row in worst.iterrows():
        name = row["athlete_full_name"]
        ema_run = seconds_to_hms(int(row["ema_run_sec_5"]))
        pred_run = seconds_to_hms(int(row["pred_run_sec"]))
        gap = row["run_gap"]

        print(f"\n  {name}: EMA={ema_run}, Pred={pred_run}, Gap={gap:+.0f}s")

        # Show this athlete's key feature values
        for feat in ["ema_run_sec_5", "ema_total_sec_5", "elo_rating", "tier_delta",
                      "ema_finish_pct_5", "distance_category_encoded", "n_entrants",
                      "ema_bike_sec_5", "ema_swim_sec_5", "wt_rank_position"]:
            if feat in row.index:
                val = row[feat]
                med = median_vals[feat] if feat in median_vals.index else "N/A"
                if isinstance(val, float) and abs(val) > 100:
                    val_str = f"{val:.0f}s ({seconds_to_hms(int(val))})"
                elif isinstance(val, float):
                    val_str = f"{val:.3f}"
                else:
                    val_str = str(val)
                if isinstance(med, float) and abs(med) > 100:
                    med_str = f"{med:.0f}s ({seconds_to_hms(int(med))})"
                elif isinstance(med, float):
                    med_str = f"{med:.3f}"
                else:
                    med_str = str(med)
                print(f"    {feat:35s}  val={val_str:>20s}  median={med_str:>20s}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
