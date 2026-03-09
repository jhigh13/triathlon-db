#!/usr/bin/env python
"""
Unified diagnostic script for the triathlon prediction pipeline.

Consolidates debug_alignment.py, debug_blake_run.py, debug_simulation.py,
and debug_split_model_fast.py into one comprehensive tool.

Usage:
    python scripts/debug_diagnostics.py --event_id 195253 --prog_id 676981 --model_path models/bundle_claudev26_men.joblib
    python scripts/debug_diagnostics.py --event_id 195253 --prog_id 676981 --model_path models/bundle_claudev26_men.joblib --athlete "Blake Bullard"
    python scripts/debug_diagnostics.py --event_id 195253 --prog_id 676981 --model_path models/bundle_claudev26_men.joblib --section importances

Sections (run all by default, or specify one):
    overview       Model metadata and bundle summary
    importances    Feature importances for all model types
    field          Field-level EMA distributions and prediction stats
    accounting     Split-total accounting diagnostics
    perturbation   Field-wide perturbation analysis (which features move predictions)
    simulation     Simulation diagnostics (pack effects, uncertainty, rank correlation)
    athlete        Deep dive on a specific athlete (requires --athlete)
"""

from __future__ import annotations
import argparse
import logging
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
from tri_analysis.prediction.sql import ProgramKey, fetch_event_metadata, fetch_athlete_history, fetch_start_list
from tri_analysis.prediction.features import (
    build_features_for_program, fill_missing_features, compute_athlete_form_features,
    classify_event_tier,
)
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.simulate import (
    run_monte_carlo, PackEffectParams, pack_params_from_dict,
    merge_params_from_dict, apply_pack_merges, assign_packs_chain,
    estimate_uncertainty, DISTANCE_SIGMA_MULTIPLIER,
    DEFAULT_SIGMA_SWIM, DEFAULT_SIGMA_BIKE, DEFAULT_SIGMA_RUN,
    DEFAULT_SIGMA_T1, DEFAULT_SIGMA_T2,
    get_distance_pack_params, get_distance_merge_params,
    DRAFT_LEGAL_DISTANCES,
)
from tri_analysis.prediction.utils_time import seconds_to_hms, parse_time_to_seconds

logger = logging.getLogger(__name__)

DIVIDER = "=" * 80


def _hms(val):
    """Safe seconds-to-HMS formatting."""
    if pd.isna(val) or val is None:
        return "N/A"
    return seconds_to_hms(int(val))


def _extract_estimator(model):
    """Extract the underlying estimator from a sklearn Pipeline."""
    if model is None:
        return None
    if hasattr(model, 'named_steps'):
        est = model.named_steps.get('regressor') or model.named_steps.get('model')
        if est is None:
            est = model[-1]
        return est
    return model


def _get_feature_importances(model, feature_cols):
    """Get sorted (feature_name, importance) pairs from a model."""
    est = _extract_estimator(model)
    if est is None or not hasattr(est, 'feature_importances_'):
        return []
    # Use the model's actual feature count if it differs from feature_cols
    importances = est.feature_importances_
    if len(importances) != len(feature_cols):
        return [(f"feature_{i}", imp) for i, imp in enumerate(importances)]
    return sorted(zip(feature_cols, importances), key=lambda x: -x[1])


# ──────────────────────────────────────────────────────────────────────
# SECTION: Model Overview
# ──────────────────────────────────────────────────────────────────────

def section_overview(bundle, event_meta, distance_cat):
    """Print model bundle metadata and configuration."""
    print(f"\n{DIVIDER}")
    print("SECTION 1: MODEL OVERVIEW")
    print(DIVIDER)

    print(f"\n  Event: {event_meta.get('event_name')}")
    print(f"  Date:  {event_meta.get('event_date')}")
    print(f"  Distance: {distance_cat}")
    print(f"  Program: {event_meta.get('prog_name')}")

    print(f"\n  Bundle metadata:")
    meta = bundle.metadata or {}
    for k, v in sorted(meta.items()):
        if isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    print(f"      {kk}: {vv}")
                elif isinstance(vv, float):
                    print(f"      {kk}: {vv:.4f}")
                else:
                    print(f"      {kk}: {vv}")
        elif isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        elif not isinstance(v, (np.ndarray, list)):
            print(f"    {k}: {v}")

    print(f"\n  Feature columns: {len(bundle.feature_columns)}")
    split_cols = getattr(bundle, "split_feature_columns", None)
    if split_cols:
        print(f"  Split feature columns: {len(split_cols)}")

    print(f"\n  Models available:")
    for name in ["model_total", "model_total_pct", "model_swim", "model_bike", "model_run", "model_t1", "model_t2"]:
        model = getattr(bundle, name, None)
        status = "YES" if model is not None else "---"
        print(f"    {name:20s}: {status}")

    if hasattr(bundle, "distance_split_models") and bundle.distance_split_models:
        print(f"\n  Distance-specific split models:")
        for dist, models in bundle.distance_split_models.items():
            splits = list(models.keys())
            print(f"    {dist}: {splits}")


# ──────────────────────────────────────────────────────────────────────
# SECTION: Feature Importances
# ──────────────────────────────────────────────────────────────────────

def section_importances(bundle, distance_cat):
    """Print feature importances for all model types."""
    print(f"\n{DIVIDER}")
    print("SECTION 2: FEATURE IMPORTANCES")
    print(DIVIDER)

    feature_cols = bundle.feature_columns
    split_cols = getattr(bundle, "split_feature_columns", None) or feature_cols

    # Percentile model (used for ranking)
    if bundle.model_total_pct is not None:
        fi = _get_feature_importances(bundle.model_total_pct, feature_cols)
        if fi:
            print(f"\n  --- Percentile (ranking) Model: Top 15 ---")
            print(f"  {'Feature':42s}  {'Importance':>10s}")
            for name, imp in fi[:15]:
                print(f"  {name:42s}  {imp:.4f}")

    # Unified total model
    if bundle.model_total is not None:
        fi = _get_feature_importances(bundle.model_total, feature_cols)
        if fi:
            print(f"\n  --- Unified Total Model: Top 15 ---")
            print(f"  {'Feature':42s}  {'Importance':>10s}")
            for name, imp in fi[:15]:
                print(f"  {name:42s}  {imp:.4f}")

    # Distance-specific split models
    dist_key = (distance_cat or "").lower().strip()
    if dist_key == "olympic":
        dist_key = "standard"

    dist_models = {}
    if hasattr(bundle, "distance_split_models") and bundle.distance_split_models:
        dist_models = bundle.distance_split_models.get(dist_key, {})

    for split_name in ["swim", "t1", "bike", "t2", "run", "total"]:
        model = dist_models.get(split_name)
        if model is None:
            # Fall back to unified
            unified = getattr(bundle, f"model_{split_name}", None)
            if unified is not None:
                model = unified
                label = f"Unified {split_name.upper()}"
            else:
                continue
        else:
            label = f"{dist_key.capitalize()} {split_name.upper()}"

        cols = split_cols if split_name not in ("total",) else feature_cols
        fi = _get_feature_importances(model, cols)
        if fi:
            # Show top 10 for each split model
            print(f"\n  --- {label} Model: Top 10 ---")
            ema_col = f"ema_{split_name}_sec_5" if split_name not in ("total",) else "ema_total_sec_5"
            print(f"  {'Feature':42s}  {'Importance':>10s}")
            for name, imp in fi[:10]:
                marker = " <<<" if name == ema_col else ""
                print(f"  {name:42s}  {imp:.4f}{marker}")
            # Show where the matching EMA ranks
            ema_rank = next((i + 1 for i, (f, _) in enumerate(fi) if f == ema_col), None)
            if ema_rank:
                ema_imp = dict(fi).get(ema_col, 0)
                print(f"  {ema_col} ranks #{ema_rank} (importance={ema_imp:.4f})")


# ──────────────────────────────────────────────────────────────────────
# SECTION: Field Analysis
# ──────────────────────────────────────────────────────────────────────

def section_field(pred_df, distance_cat):
    """Analyze field-level EMA distributions and prediction stats."""
    print(f"\n{DIVIDER}")
    print("SECTION 3: FIELD ANALYSIS")
    print(DIVIDER)

    splits = [
        ("swim", "ema_swim_sec_5", "pred_swim_sec"),
        ("t1", "ema_t1_sec_5", "pred_t1_sec"),
        ("bike", "ema_bike_sec_5", "pred_bike_sec"),
        ("t2", "ema_t2_sec_5", "pred_t2_sec"),
        ("run", "ema_run_sec_5", "pred_run_sec"),
        ("total", "ema_total_sec_5", "pred_total_sec"),
    ]

    for split_name, ema_col, pred_col in splits:
        if ema_col not in pred_df.columns or pred_col not in pred_df.columns:
            continue

        ema = pred_df[ema_col]
        pred = pred_df[pred_col]
        n_with_ema = ema.notna().sum()
        n_null_ema = ema.isna().sum()

        print(f"\n  {split_name.upper()}:")
        print(f"    Athletes with EMA: {n_with_ema} / {len(pred_df)}")
        if n_with_ema > 0:
            print(f"    EMA:  median={_hms(ema.median())} ({ema.median():.0f}s), "
                  f"min={_hms(ema.min())}, max={_hms(ema.max())}")
        print(f"    Pred: median={_hms(pred.median())} ({pred.median():.0f}s), "
              f"min={_hms(pred.min())}, max={_hms(pred.max())}")

        # Pred - EMA gap
        valid = pred_df[ema.notna()].copy()
        if not valid.empty:
            gap = valid[pred_col] - valid[ema_col]
            print(f"    Pred-EMA gap: mean={gap.mean():+.0f}s, median={gap.median():+.0f}s, "
                  f"std={gap.std():.0f}s")

    # Athletes with biggest pred-EMA total gaps
    if "ema_total_sec_5" in pred_df.columns:
        valid = pred_df[pred_df["ema_total_sec_5"].notna()].copy()
        if not valid.empty:
            valid["total_gap"] = valid["pred_total_sec"] - valid["ema_total_sec_5"]
            worst = valid.nlargest(5, "total_gap")
            print(f"\n  Top 5 largest pred-EMA total gaps (pred SLOWER than EMA):")
            print(f"  {'Athlete':35s}  {'EMA':>10s}  {'Pred':>10s}  {'Gap':>8s}  {'Rank':>4s}")
            for _, row in worst.iterrows():
                print(f"  {row['athlete_full_name']:35s}  {_hms(row['ema_total_sec_5']):>10s}  "
                      f"{_hms(row['pred_total_sec']):>10s}  {row['total_gap']:>+7.0f}s  "
                      f"{row['predicted_rank']:4d}")


# ──────────────────────────────────────────────────────────────────────
# SECTION: Split-Total Accounting
# ──────────────────────────────────────────────────────────────────────

def section_accounting(pred_df):
    """Check split sum vs total model consistency."""
    print(f"\n{DIVIDER}")
    print("SECTION 4: SPLIT-TOTAL ACCOUNTING")
    print(DIVIDER)

    split_cols = ["pred_swim_sec", "pred_t1_sec", "pred_bike_sec", "pred_t2_sec", "pred_run_sec"]
    available = [c for c in split_cols if c in pred_df.columns]

    if len(available) < 3:
        print("  Insufficient split predictions for accounting check.")
        return

    pred_df = pred_df.copy()
    pred_df["split_sum"] = sum(pred_df[c].fillna(0) for c in available)
    pred_df["split_total_gap"] = pred_df["split_sum"] - pred_df["pred_total_sec"]

    print(f"\n  Split sum components: {available}")
    print(f"  Mean gap (split_sum - pred_total): {pred_df['split_total_gap'].mean():+.1f}s")
    print(f"  Median gap: {pred_df['split_total_gap'].median():+.1f}s")
    print(f"  Min gap: {pred_df['split_total_gap'].min():+.1f}s")
    print(f"  Max gap: {pred_df['split_total_gap'].max():+.1f}s")

    if pred_df['split_total_gap'].abs().mean() < 1.0:
        print(f"\n  GOOD: Split sum matches total (mean |gap| < 1s)")
    else:
        print(f"\n  WARNING: Split-total discrepancy detected. Mean |gap| = {pred_df['split_total_gap'].abs().mean():.1f}s")

    # Rank comparison: det rank vs split-sum rank
    pred_df["split_sum_rank"] = pred_df["split_sum"].rank(method="min", ascending=True).astype(int)
    pred_df["rank_shift"] = pred_df["split_sum_rank"] - pred_df["predicted_rank"]
    rank_corr = pred_df["predicted_rank"].corr(pred_df["split_sum_rank"], method="spearman")
    print(f"\n  Spearman(Det Rank, Split-Sum Rank): {rank_corr:.3f}")

    big_shifts = pred_df[pred_df["rank_shift"].abs() > 10]
    if not big_shifts.empty:
        print(f"  Athletes with >10 rank shift between percentile and time: {len(big_shifts)}")
        for _, row in big_shifts.sort_values("rank_shift", key=abs, ascending=False).head(5).iterrows():
            print(f"    {row['athlete_full_name']:35s}  Det={row['predicted_rank']:2d}  "
                  f"Split={row['split_sum_rank']:2d}  Shift={row['rank_shift']:+d}")


# ──────────────────────────────────────────────────────────────────────
# SECTION: Perturbation Analysis
# ──────────────────────────────────────────────────────────────────────

def section_perturbation(pred_df, bundle, distance_cat):
    """Field-wide perturbation: which features drive predictions away from median."""
    print(f"\n{DIVIDER}")
    print("SECTION 5: PERTURBATION ANALYSIS")
    print(DIVIDER)

    split_cols = getattr(bundle, "split_feature_columns", None) or bundle.feature_columns
    dist_key = (distance_cat or "").lower().strip()
    if dist_key == "olympic":
        dist_key = "standard"

    dist_models = {}
    if hasattr(bundle, "distance_split_models") and bundle.distance_split_models:
        dist_models = bundle.distance_split_models.get(dist_key, {})

    for split_name in ["swim", "bike", "run"]:
        model = dist_models.get(split_name)
        cols = split_cols
        if model is None:
            model = getattr(bundle, f"model_{split_name}", None)
            cols = bundle.feature_columns
        if model is None:
            continue

        X = pred_df[cols].copy().fillna(pred_df[cols].median())
        baseline_preds = model.predict(X)
        median_vals = X.median()

        impacts = []
        for col in cols:
            X_perturbed = X.copy()
            X_perturbed[col] = median_vals[col]
            perturbed_preds = model.predict(X_perturbed)
            mean_impact = (baseline_preds - perturbed_preds).mean()
            std_impact = (baseline_preds - perturbed_preds).std()
            impacts.append((col, mean_impact, std_impact))

        impacts.sort(key=lambda x: -abs(x[1]))

        print(f"\n  --- {split_name.upper()} Model: Top 10 features by mean field impact ---")
        print(f"  (positive = feature pushes field predictions SLOWER)")
        print(f"  {'Feature':42s}  {'Mean':>8s}  {'Std':>8s}")
        for feat, mean_imp, std_imp in impacts[:10]:
            sign = "+" if mean_imp > 0 else "-" if mean_imp < 0 else " "
            print(f"  {feat:42s}  {sign}{abs(mean_imp):>6.1f}s  {std_imp:>7.1f}s")


# ──────────────────────────────────────────────────────────────────────
# SECTION: Simulation Diagnostics
# ──────────────────────────────────────────────────────────────────────

def section_simulation(pred_df, bundle, distance_cat, n_sims=1000, event_tier=None):
    """Run a quick simulation and diagnose pack effects, uncertainty, calibration."""
    print(f"\n{DIVIDER}")
    print("SECTION 6: SIMULATION DIAGNOSTICS")
    print(DIVIDER)

    pack_params = get_distance_pack_params(bundle.metadata, distance_cat, event_tier=event_tier)
    merge_params = get_distance_merge_params(bundle.metadata, distance_cat, event_tier=event_tier)

    is_draft = (distance_cat or "").lower().strip() in DRAFT_LEGAL_DISTANCES
    tier_label = {1: "WTCS", 2: "World Cup", 3: "Continental", 4: "Other"}.get(event_tier, "Unknown")
    print(f"\n  Distance: {distance_cat} ({'draft-legal' if is_draft else 'non-drafting'})")
    print(f"  Event tier: {event_tier} ({tier_label})")

    # Show which param source was used (tier vs distance vs overall)
    meta = bundle.metadata or {}
    tier_key = f"tier{event_tier}" if event_tier else None
    pack_source = "overall"
    if tier_key and tier_key in meta.get("pack_effect_params_by_tier", {}):
        pack_source = f"tier-specific ({tier_label})"
    elif (distance_cat or "").lower().strip() in (meta.get("pack_effect_params_by_distance", {})):
        pack_source = f"distance-specific ({distance_cat})"
    merge_source = "overall"
    if tier_key and tier_key in meta.get("merge_params_by_tier", {}):
        merge_source = f"tier-specific ({tier_label})"
    elif (distance_cat or "").lower().strip() in (meta.get("merge_params_by_distance", {})):
        merge_source = f"distance-specific ({distance_cat})"

    print(f"  Pack params [{pack_source}]: front_bonus={pack_params.front_pack_bonus_sec:.1f}s, chase_penalty={pack_params.chase_penalty_sec:.1f}s, "
          f"max_gap={pack_params.max_gap_sec:.1f}s")
    if merge_params:
        print(f"  Merge params [{merge_source}]: beta_0={merge_params.beta_0:.3f}, beta_gap={merge_params.beta_gap:.3f}, "
              f"beta_chase={merge_params.beta_chase_size:.3f}")

    # Distance sigma
    dist_mult = DISTANCE_SIGMA_MULTIPLIER.get((distance_cat or "").lower().strip(), 1.0)
    print(f"\n  Uncertainty profile:")
    print(f"    Distance multiplier: {dist_mult:.2f}")
    print(f"    Base sigmas: swim={DEFAULT_SIGMA_SWIM}s, t1={DEFAULT_SIGMA_T1}s, "
          f"bike={DEFAULT_SIGMA_BIKE}s, t2={DEFAULT_SIGMA_T2}s, run={DEFAULT_SIGMA_RUN}s")

    # Run MC simulation
    print(f"\n  Running {n_sims} MC simulations...")
    sim_df = run_monte_carlo(
        pred_df, n_sims=n_sims, random_state=42,
        pack_params=pack_params, merge_params=merge_params,
        distance_category=distance_cat,
        bundle_metadata=bundle.metadata,
    )

    # Rank correlation: det vs sim
    merged = pred_df[["athlete_full_name", "predicted_rank"]].merge(
        sim_df[["athlete_full_name", "expected_rank"]], on="athlete_full_name"
    )
    det_sim_corr = merged["predicted_rank"].corr(merged["expected_rank"], method="spearman")
    print(f"\n  Spearman(Det Rank, E[Rank]): {det_sim_corr:.3f}")

    # Biggest rank movers
    merged["rank_delta"] = merged["expected_rank"] - merged["predicted_rank"]
    big_movers = merged.nlargest(5, "rank_delta", keep="first")
    small_movers = merged.nsmallest(5, "rank_delta", keep="first")

    print(f"\n  Biggest rank LOSERS (sim worse than det):")
    for _, row in big_movers.iterrows():
        print(f"    {row['athlete_full_name']:35s}  Det={row['predicted_rank']:2d}  "
              f"E[R]={row['expected_rank']:.1f}  Δ={row['rank_delta']:+.1f}")

    print(f"\n  Biggest rank GAINERS (sim better than det):")
    for _, row in small_movers.iterrows():
        print(f"    {row['athlete_full_name']:35s}  Det={row['predicted_rank']:2d}  "
              f"E[R]={row['expected_rank']:.1f}  Δ={row['rank_delta']:+.1f}")

    # Sim median vs det total
    sim_medians = sim_df.set_index("athlete_full_name")["total_p50"]
    det_totals = pred_df.set_index("athlete_full_name")["pred_total_sec"]
    common = sim_medians.index.intersection(det_totals.index)
    if len(common) > 0:
        offsets = sim_medians[common] - det_totals[common]
        print(f"\n  Sim median vs Det total offset:")
        print(f"    Mean: {offsets.mean():+.1f}s, Median: {offsets.median():+.1f}s, "
              f"Std: {offsets.std():.1f}s")

    # Pack formation preview (pre-simulation)
    if is_draft and merge_params:
        swim_t1 = pred_df["pred_swim_sec"].values + pred_df.get("pred_t1_sec", pd.Series(np.zeros(len(pred_df)))).values
        sorted_idx = np.argsort(swim_t1)
        sorted_times = swim_t1[sorted_idx]

        # assign_packs_chain returns pack IDs in sorted order
        pack_ids = assign_packs_chain(sorted_times, pack_params.max_gap_sec)

        print(f"\n  Predicted pack formation (swim+T1 exit order):")
        for pid in range(pack_ids.max() + 1):
            members_sorted = np.where(pack_ids == pid)[0]  # indices into sorted order
            members_orig = sorted_idx[members_sorted]       # map back to original order
            n = len(members_orig)
            first = pred_df.iloc[members_orig[0]]["athlete_full_name"] if n > 0 else "?"
            gap = sorted_times[members_sorted[0]] - sorted_times[0] if n > 0 else 0
            print(f"    Pack {pid + 1}: {n} athletes, leader gap={gap:+.1f}s "
                  f"(e.g. {first})")

    # Front pack rate comparison
    if "front_pack_rate" in pred_df.columns and "prob_win" in sim_df.columns:
        print(f"\n  Sim front pack % vs historical front_pack_rate:")
        comparison = pred_df[["athlete_full_name", "front_pack_rate"]].merge(
            sim_df[["athlete_full_name", "expected_rank"]], on="athlete_full_name"
        )
        comparison = comparison[comparison["front_pack_rate"].notna()]
        if not comparison.empty:
            comparison = comparison.sort_values("expected_rank").head(15)
            print(f"  {'Athlete':35s}  {'Hist FP%':>8s}  {'E[Rank]':>7s}")
            for _, row in comparison.iterrows():
                fpr = f"{row['front_pack_rate']:.0%}" if pd.notna(row['front_pack_rate']) else "N/A"
                print(f"  {row['athlete_full_name']:35s}  {fpr:>8s}  {row['expected_rank']:7.1f}")


# ──────────────────────────────────────────────────────────────────────
# SECTION: Athlete Deep Dive
# ──────────────────────────────────────────────────────────────────────

def section_athlete(engine, key, pred_df, bundle, distance_cat, athlete_name):
    """Deep trace of a specific athlete through the full pipeline."""
    print(f"\n{DIVIDER}")
    print(f"SECTION 7: ATHLETE DEEP DIVE — {athlete_name}")
    print(DIVIDER)

    event_meta = fetch_event_metadata(engine, key)
    event_date = pd.to_datetime(event_meta["event_date"]).date()

    # Find athlete
    match = pred_df[pred_df["athlete_full_name"].str.contains(athlete_name, case=False, na=False)]
    if match.empty:
        print(f"  Athlete '{athlete_name}' not found in predictions.")
        return

    athlete_id = int(match["athlete_id"].iloc[0])
    print(f"  athlete_id: {athlete_id}")

    # Step 1: Raw race history
    print(f"\n  --- Race History (all distances) ---")
    all_history = fetch_athlete_history(engine, athlete_id, event_date, limit=15, distance_category=None, elite_only=True)
    if not all_history.empty:
        cols = ["event_date", "event_name", "prog_distance_category", "swimtime", "t1time", "biketime", "t2time", "runtime", "total_time"]
        available = [c for c in cols if c in all_history.columns]
        print(all_history[available].to_string(index=False))
    else:
        print("  No history found!")

    # Step 2: Distance-filtered history
    if distance_cat:
        print(f"\n  --- {distance_cat.capitalize()}-Filtered History ---")
        filt = all_history[
            all_history["prog_distance_category"].str.lower() == distance_cat.lower()
        ] if not all_history.empty and "prog_distance_category" in all_history.columns else pd.DataFrame()
        if not filt.empty:
            print(f"  {len(filt)} matching races")
        else:
            print(f"  No {distance_cat} history — will fall back to all distances")

    # Step 3: Feature values
    feature_cols = bundle.feature_columns
    print(f"\n  --- Feature Values ---")
    print(f"  {'Feature':42s}  {'Value':>15s}")
    for col in feature_cols:
        if col in match.columns:
            val = match[col].iloc[0]
            if isinstance(val, float) and pd.notna(val) and abs(val) > 100:
                print(f"  {col:42s}  {val:>10.0f} ({_hms(val)})")
            elif isinstance(val, float) and pd.notna(val):
                print(f"  {col:42s}  {val:>15.3f}")
            else:
                print(f"  {col:42s}  {str(val):>15s}")

    # Step 4: Predictions
    print(f"\n  --- Predictions ---")
    for label, col in [
        ("Swim", "pred_swim_sec"), ("T1", "pred_t1_sec"),
        ("Bike", "pred_bike_sec"), ("T2", "pred_t2_sec"),
        ("Run", "pred_run_sec"), ("Total", "pred_total_sec"),
    ]:
        val = match[col].iloc[0] if col in match.columns else None
        print(f"  Pred {label:5s}: {_hms(val)} ({val:.0f}s)" if pd.notna(val) else f"  Pred {label:5s}: N/A")
    if "predicted_rank" in match.columns:
        print(f"  Det. Rank: {match['predicted_rank'].iloc[0]}")
    if "pred_finish_pct" in match.columns:
        pct = match["pred_finish_pct"].iloc[0]
        print(f"  Finish Pct: {pct:.3f}" if pd.notna(pct) else "  Finish Pct: N/A")

    # Step 5: Perturbation analysis for this athlete
    split_cols = getattr(bundle, "split_feature_columns", None) or feature_cols
    dist_key = (distance_cat or "").lower().strip()
    if dist_key == "olympic":
        dist_key = "standard"

    dist_models = {}
    if hasattr(bundle, "distance_split_models") and bundle.distance_split_models:
        dist_models = bundle.distance_split_models.get(dist_key, {})

    for split_name in ["run", "total"]:
        model = dist_models.get(split_name)
        cols = split_cols if split_name != "total" else feature_cols
        if model is None:
            model = getattr(bundle, f"model_{split_name}", None)
            cols = feature_cols
        if model is None:
            continue

        X_athlete = match[cols].copy().fillna(pred_df[cols].median())
        baseline_pred = model.predict(X_athlete)[0]
        median_vals = pred_df[cols].median()

        contributions = []
        for col in cols:
            X_pert = X_athlete.copy()
            X_pert[col] = median_vals[col]
            pert_pred = model.predict(X_pert)[0]
            delta = baseline_pred - pert_pred
            contributions.append((col, delta, X_athlete[col].iloc[0], median_vals[col]))

        contributions.sort(key=lambda x: -abs(x[1]))

        print(f"\n  --- {split_name.upper()}: Per-feature contribution (top 10) ---")
        print(f"  {'Feature':42s} {'Impact':>8s}   {'Athlete':>12s}   {'Median':>12s}")
        for feat, delta, ath_val, med_val in contributions[:10]:
            sign = "+" if delta > 0 else "-" if delta < 0 else " "
            if isinstance(ath_val, float) and pd.notna(ath_val) and abs(ath_val) > 100:
                av = _hms(ath_val)
                mv = _hms(med_val)
            elif isinstance(ath_val, float) and pd.notna(ath_val):
                av = f"{ath_val:.3f}"
                mv = f"{med_val:.3f}" if pd.notna(med_val) else "N/A"
            else:
                av = str(ath_val)
                mv = str(med_val)
            print(f"  {feat:42s} {sign}{abs(delta):>6.1f}s   {av:>12s}   {mv:>12s}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified diagnostics for triathlon prediction pipeline")
    parser.add_argument("--event_id", type=int, required=True)
    parser.add_argument("--prog_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--athlete", type=str, default=None, help="Athlete name for deep dive (partial match)")
    parser.add_argument("--section", type=str, default=None,
                        choices=["overview", "importances", "field", "accounting",
                                 "perturbation", "simulation", "athlete"],
                        help="Run only this section (default: all)")
    parser.add_argument("--n_sims", type=int, default=1000, help="MC sims for diagnostics (default: 1000)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    engine = get_engine()
    key = ProgramKey(event_id=args.event_id, prog_id=args.prog_id)
    event_meta = fetch_event_metadata(engine, key)
    if event_meta is None:
        print(f"ERROR: No event metadata for event_id={args.event_id}")
        sys.exit(1)

    distance_cat = event_meta.get("prog_distance_category")

    # Geocode + fetch weather (mirrors predict_program.py logic)
    has_coords = event_meta.get("event_latitude") is not None and event_meta.get("event_longitude") is not None
    if not has_coords and event_meta.get("event_venue"):
        try:
            from tri_analysis.weather import geocode_venue
            venue = event_meta["event_venue"]
            country = event_meta.get("event_country")
            coords = geocode_venue(venue, country)
            if coords:
                event_meta["event_latitude"], event_meta["event_longitude"] = coords
                has_coords = True
        except Exception:
            pass

    has_weather = event_meta.get("temperature_air") is not None
    if not has_weather and has_coords:
        try:
            from tri_analysis.weather import fetch_weather_smart
            weather = fetch_weather_smart(
                event_meta["event_latitude"], event_meta["event_longitude"],
                event_meta.get("event_date"),
            )
            if weather:
                for k, v in weather.items():
                    if v is not None:
                        event_meta[k] = v
        except Exception:
            pass

    print(DIVIDER)
    print(f"PREDICTION PIPELINE DIAGNOSTICS")
    print(f"Event: {event_meta.get('event_name')} ({distance_cat})")
    print(f"Model: {args.model_path}")
    print(DIVIDER)

    bundle = load_model_bundle(args.model_path)
    feature_cols = bundle.feature_columns

    # Build features and predict
    print("\nBuilding features...")
    features_df = build_features_for_program(engine, key, use_start_list=True, event_meta_override=event_meta)
    features_df = fill_missing_features(features_df, feature_cols)
    pred_df = predict_splits_and_total(features_df, bundle, distance_category=distance_cat)
    print(f"Athletes: {len(pred_df)}")

    run_all = args.section is None

    if run_all or args.section == "overview":
        section_overview(bundle, event_meta, distance_cat)

    if run_all or args.section == "importances":
        section_importances(bundle, distance_cat)

    if run_all or args.section == "field":
        section_field(pred_df, distance_cat)

    if run_all or args.section == "accounting":
        section_accounting(pred_df)

    if run_all or args.section == "perturbation":
        section_perturbation(pred_df, bundle, distance_cat)

    if run_all or args.section == "simulation":
        event_tier = classify_event_tier(event_meta.get("event_name", ""))
        section_simulation(pred_df, bundle, distance_cat, n_sims=args.n_sims, event_tier=event_tier)

    if args.athlete or args.section == "athlete":
        if args.athlete:
            section_athlete(engine, key, pred_df, bundle, distance_cat, args.athlete)
        else:
            print("\n  --athlete NAME required for athlete deep dive section")

    print(f"\n{DIVIDER}")
    print("DIAGNOSTICS COMPLETE")
    print(DIVIDER)


if __name__ == "__main__":
    main()
