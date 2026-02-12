"""
Diagnostic: Why does the Monte Carlo simulation produce wildly different rankings
from the deterministic model?

Hypothesis: The simulation uses pred_swim + pred_bike + pred_run as center values,
but these split models produce times that are MUCH higher than pred_total_sec
(the total model, which is anchored to EMA). This non-uniform inflation distorts
the relative rankings.

Tests:
1. Show split sum vs pred_total mismatch per athlete
2. Show how mismatch varies by athlete (non-uniform = rank distortion)
3. Simulate with normalized splits to confirm fix
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
from tri_analysis.prediction.simulate import (
    run_monte_carlo, PackEffectParams, pack_params_from_dict,
    merge_params_from_dict, continuous_gap_bike_effect,
    estimate_uncertainty, DISTANCE_SIGMA_MULTIPLIER,
    DEFAULT_SIGMA_SWIM, DEFAULT_SIGMA_BIKE, DEFAULT_SIGMA_RUN,
)
from tri_analysis.prediction.utils_time import seconds_to_hms

engine = get_engine()

# La Paz (broken)
key_lapaz = ProgramKey(event_id=195253, prog_id=676981)
event_meta_lapaz = fetch_event_metadata(engine, key_lapaz)
dist_cat_lapaz = event_meta_lapaz.get("prog_distance_category")

# Try Cuba too if available
key_cuba = None
try:
    # Check what events are available
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT e.event_id, e.event_name, e.event_venue, pe.prog_id, pe.prog_name,
                   pe.prog_distance_category
            FROM events e
            JOIN program_entries pe ON e.event_id = pe.event_id
            WHERE e.event_name ILIKE '%cuba%' OR e.event_venue ILIKE '%cuba%'
               OR e.event_venue ILIKE '%havana%' OR e.event_venue ILIKE '%habana%'
            LIMIT 10
        """))
        cuba_rows = result.fetchall()
        if cuba_rows:
            print("Found Cuba events:")
            for row in cuba_rows:
                print(f"  event_id={row[0]}, name={row[1]}, venue={row[2]}, "
                      f"prog_id={row[3]}, prog={row[4]}, dist={row[5]}")
            # Use first men's sprint
            for row in cuba_rows:
                if 'men' in str(row[4]).lower() or 'elite' in str(row[4]).lower():
                    key_cuba = ProgramKey(event_id=row[0], prog_id=row[3])
                    break
            if key_cuba is None:
                key_cuba = ProgramKey(event_id=cuba_rows[0][0], prog_id=cuba_rows[0][3])
        else:
            print("No Cuba events found in database")
except Exception as e:
    print(f"Error looking up Cuba events: {e}")

bundle = load_model_bundle("models/bundle_claudev26_men.joblib")
# Fall back to v21 if v22 doesn't exist
if bundle is None:
    bundle = load_model_bundle("models/bundle_claudev21_men.joblib")

feature_cols = bundle.feature_columns

# ── Load pack params from bundle ──
pack_params_dict = bundle.metadata.get("pack_effect_params")
pack_params = pack_params_from_dict(pack_params_dict) if pack_params_dict else PackEffectParams()
merge_params_dict = bundle.metadata.get("merge_params")
merge_params = merge_params_from_dict(merge_params_dict) if merge_params_dict else None


def analyze_event(key, event_meta, label):
    """Full analysis of split sum vs total mismatch for one event."""
    dist_cat = event_meta.get("prog_distance_category")
    print(f"\n{'=' * 90}")
    print(f"EVENT: {label} — {event_meta.get('event_name')} ({dist_cat})")
    print(f"{'=' * 90}")

    features_df = build_features_for_program(engine, key, use_start_list=True)
    features_df = fill_missing_features(features_df, feature_cols)
    pred_df = predict_splits_and_total(features_df, bundle, distance_category=dist_cat)

    n = len(pred_df)
    print(f"Athletes: {n}")

    # ── 1. Split sum vs total model ──
    pred_df["split_sum"] = pred_df["pred_swim_sec"] + pred_df["pred_bike_sec"] + pred_df["pred_run_sec"]
    pred_df["split_total_gap"] = pred_df["split_sum"] - pred_df["pred_total_sec"]
    pred_df["split_total_pct"] = pred_df["split_sum"] / pred_df["pred_total_sec"]

    print(f"\n--- Split Sum vs Total Model (pred_total_sec is anchored to EMA) ---")
    print(f"  {'Athlete':40s}  {'Split Sum':>10s}  {'Pred Total':>10s}  {'Gap':>8s}  {'Ratio':>6s}  {'Rank':>5s}")
    for _, row in pred_df.head(20).iterrows():
        split_sum = seconds_to_hms(int(row["split_sum"]))
        pred_total = seconds_to_hms(int(row["pred_total_sec"]))
        gap = f"{row['split_total_gap']:+.0f}s"
        ratio = f"{row['split_total_pct']:.2f}"
        print(f"  {row['athlete_full_name']:40s}  {split_sum:>10s}  {pred_total:>10s}  {gap:>8s}  {ratio:>6s}  {row['predicted_rank']:5d}")

    print(f"\n  FIELD SUMMARY:")
    print(f"    Mean gap: {pred_df['split_total_gap'].mean():+.0f}s")
    print(f"    Median gap: {pred_df['split_total_gap'].median():+.0f}s")
    print(f"    Min gap: {pred_df['split_total_gap'].min():+.0f}s")
    print(f"    Max gap: {pred_df['split_total_gap'].max():+.0f}s")
    print(f"    Mean ratio (split_sum/total): {pred_df['split_total_pct'].mean():.3f}")
    print(f"    Std ratio: {pred_df['split_total_pct'].std():.3f}")

    # ── 2. Rank comparison: total model rank vs split-sum rank ──
    pred_df["split_sum_rank"] = pred_df["split_sum"].rank(method="min", ascending=True).astype(int)
    pred_df["rank_shift"] = pred_df["split_sum_rank"] - pred_df["predicted_rank"]

    print(f"\n--- Rank Comparison: Det. Rank (percentile model) vs Split-Sum Rank ---")
    print(f"  {'Athlete':40s}  {'Det Rank':>8s}  {'Split Rank':>10s}  {'Shift':>6s}")

    # Show top 15 by Det Rank
    for _, row in pred_df.head(15).iterrows():
        print(f"  {row['athlete_full_name']:40s}  {row['predicted_rank']:8d}  {row['split_sum_rank']:10d}  {row['rank_shift']:+6d}")

    big_shifts = pred_df[pred_df["rank_shift"].abs() > 10]
    if not big_shifts.empty:
        print(f"\n  Athletes with >10 rank shift ({len(big_shifts)} total):")
        for _, row in big_shifts.sort_values("rank_shift", key=abs, ascending=False).head(10).iterrows():
            print(f"    {row['athlete_full_name']:40s}  Det={row['predicted_rank']:2d}  Split={row['split_sum_rank']:2d}  Shift={row['rank_shift']:+d}")

    # ── 3. Non-uniformity of inflation ──
    # The key question: does the split inflation VARY by athlete (causing rank distortion)?
    print(f"\n--- Split Inflation Non-Uniformity ---")
    print(f"  If inflation is uniform, split-sum ranks ~ det ranks (no distortion)")
    print(f"  If inflation varies by athlete, ranks diverge (simulation is distorted)")

    rank_correlation = pred_df["predicted_rank"].corr(pred_df["split_sum_rank"], method="spearman")
    print(f"  Spearman rank correlation (Det vs Split-sum): {rank_correlation:.3f}")
    print(f"  (1.0 = perfect agreement, <0.9 = significant distortion)")

    # ── 4. Run simulation with ORIGINAL splits and with NORMALIZED splits ──
    print(f"\n--- Simulation: Original vs Normalized Splits ---")

    # Original simulation
    sim_orig = run_monte_carlo(
        pred_df, n_sims=5000, random_state=42,
        pack_params=pack_params, merge_params=merge_params,
        distance_category=dist_cat,
    )

    # Now normalize splits to match pred_total_sec and re-simulate
    pred_df_norm = pred_df.copy()
    scale = pred_df_norm["pred_total_sec"] / pred_df_norm["split_sum"]
    pred_df_norm["pred_swim_sec"] = pred_df_norm["pred_swim_sec"] * scale
    pred_df_norm["pred_bike_sec"] = pred_df_norm["pred_bike_sec"] * scale
    pred_df_norm["pred_run_sec"] = pred_df_norm["pred_run_sec"] * scale
    # Update HMS columns
    pred_df_norm["pred_swim_hms"] = pred_df_norm["pred_swim_sec"].apply(lambda x: seconds_to_hms(int(x)))
    pred_df_norm["pred_bike_hms"] = pred_df_norm["pred_bike_sec"].apply(lambda x: seconds_to_hms(int(x)))
    pred_df_norm["pred_run_hms"] = pred_df_norm["pred_run_sec"].apply(lambda x: seconds_to_hms(int(x)))

    sim_norm = run_monte_carlo(
        pred_df_norm, n_sims=5000, random_state=42,
        pack_params=pack_params, merge_params=merge_params,
        distance_category=dist_cat,
    )

    print(f"\n  {'Athlete':35s}  {'Det':>4s}  {'E[R] Orig':>9s}  {'E[R] Norm':>9s}  {'Split Gap':>10s}  {'Ratio':>6s}")
    for i in range(min(20, n)):
        name = pred_df.iloc[i]["athlete_full_name"]
        det_rank = pred_df.iloc[i]["predicted_rank"]
        # Find this athlete in simulation results
        orig_row = sim_orig[sim_orig["athlete_full_name"] == name]
        norm_row = sim_norm[sim_norm["athlete_full_name"] == name]
        er_orig = orig_row["expected_rank"].iloc[0] if not orig_row.empty else float("nan")
        er_norm = norm_row["expected_rank"].iloc[0] if not norm_row.empty else float("nan")
        gap = pred_df.iloc[i]["split_total_gap"]
        ratio = pred_df.iloc[i]["split_total_pct"]
        print(f"  {name:35s}  {det_rank:4d}  {er_orig:9.1f}  {er_norm:9.1f}  {gap:+10.0f}s  {ratio:6.2f}")

    # Overall correlation
    merged = pred_df[["athlete_full_name", "predicted_rank"]].merge(
        sim_orig[["athlete_full_name", "expected_rank"]].rename(columns={"expected_rank": "er_orig"}),
        on="athlete_full_name",
    ).merge(
        sim_norm[["athlete_full_name", "expected_rank"]].rename(columns={"expected_rank": "er_norm"}),
        on="athlete_full_name",
    )
    corr_orig = merged["predicted_rank"].corr(merged["er_orig"], method="spearman")
    corr_norm = merged["predicted_rank"].corr(merged["er_norm"], method="spearman")
    print(f"\n  Spearman correlation with Det. Rank:")
    print(f"    Original splits: {corr_orig:.3f}")
    print(f"    Normalized splits: {corr_norm:.3f}")
    print(f"    Improvement: {corr_norm - corr_orig:+.3f}")

    # ── 5. Show pack effect details ──
    print(f"\n--- Pack Effect Baseline (continuous_gap_bike_effect) ---")
    swim_times = pred_df["pred_swim_sec"].values
    effects = continuous_gap_bike_effect(swim_times, pack_params)

    sorted_idx = np.argsort(swim_times)
    print(f"  {'Athlete':35s}  {'Swim':>8s}  {'Gap':>6s}  {'Effect':>8s}")
    for i, idx in enumerate(sorted_idx[:15]):
        name = pred_df.iloc[idx]["athlete_full_name"]
        swim = seconds_to_hms(int(swim_times[idx]))
        gap = f"{swim_times[idx] - swim_times[sorted_idx[0]]:+.1f}s"
        effect = f"{effects[idx]:+.1f}s"
        print(f"  {name:35s}  {swim:>8s}  {gap:>6s}  {effect:>8s}")

    # Show sigma values being used
    dist_mult = DISTANCE_SIGMA_MULTIPLIER.get(dist_cat.lower().strip(), 1.0) if dist_cat else 1.0
    print(f"\n--- Uncertainty (sigma) for {dist_cat} ---")
    print(f"  Distance multiplier: {dist_mult}")
    print(f"  Base sigmas: swim={DEFAULT_SIGMA_SWIM}s, bike={DEFAULT_SIGMA_BIKE}s, run={DEFAULT_SIGMA_RUN}s")
    print(f"  Scaled sigmas: swim={DEFAULT_SIGMA_SWIM*dist_mult:.0f}s, bike={DEFAULT_SIGMA_BIKE*dist_mult:.0f}s, run={DEFAULT_SIGMA_RUN*dist_mult:.0f}s")

    return pred_df


# ── Analyze La Paz ──
print("\n" + "#" * 90)
print("# ANALYZING LA PAZ (problematic event)")
print("#" * 90)
pred_lapaz = analyze_event(key_lapaz, event_meta_lapaz, "La Paz")

# ── Analyze Cuba if available ──
if key_cuba:
    event_meta_cuba = fetch_event_metadata(engine, key_cuba)
    if event_meta_cuba:
        print("\n" + "#" * 90)
        print("# ANALYZING CUBA (working event)")
        print("#" * 90)
        pred_cuba = analyze_event(key_cuba, event_meta_cuba, "Cuba")

print("\n" + "=" * 90)
print("DIAGNOSIS SUMMARY")
print("=" * 90)
print("""
ROOT CAUSE: The Monte Carlo simulation uses pred_swim + pred_bike + pred_run as
center values (line 916 in simulate.py). These come from INDEPENDENT split models
that are NOT anchored to EMA. The total model (pred_total_sec) IS anchored to EMA
(max 110% slowdown cap in predict.py lines 120-132).

When split sums >> pred_total, the simulation operates on inflated center values.
Since the inflation is NON-UNIFORM across athletes (different split_total ratios),
relative rankings get distorted.

FIX: Before simulation, normalize split predictions to sum to pred_total_sec:
    scale = pred_total_sec / (pred_swim + pred_bike + pred_run)
    pred_swim *= scale
    pred_bike *= scale
    pred_run *= scale

This preserves the split model's RELATIVE proportions while using the calibrated
total as the absolute anchor.
""")
