"""
Evaluate swim prediction accuracy and pack formation accuracy.

Measures how well the current model predicts swim exit ordering and gap structure,
which is the critical input to the MC simulation's causal chain.

Metrics computed per event:
  - swim_spearman:           Rank correlation of predicted vs actual swim exit order
  - swim_P@10:               Of top 10 predicted swimmers, how many actually top 10
  - pack_assignment_acc:     Fraction of athletes in the correct pack (pred vs actual swim+T1)
  - front_pack_recall:       Of actual front-pack members, fraction predicted correctly
  - front_pack_precision:    Of predicted front-pack members, fraction actually in front pack
  - gap_mae:                 MAE of predicted gap-to-leader vs actual gap-to-leader (seconds)

Usage:
  python scripts/eval_swim_accuracy.py --model models/bundle_elite_v49.joblib
  python scripts/eval_swim_accuracy.py --model models/bundle_elite_v49.joblib --tier 1a --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from sqlalchemy import text

from tri_analysis.database import get_engine
from tri_analysis.prediction.sql import ProgramKey, fetch_program_results, fetch_event_metadata
from tri_analysis.prediction.features import build_features_for_program, fill_missing_features, get_feature_columns
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.simulate import assign_packs_chain, DISTANCE_PACK_GAP_SEC
from tri_analysis.prediction.utils_time import parse_time_to_seconds

logger = logging.getLogger(__name__)


# ── Tier classification (reuse from run_backtest.py) ──

TIER_CASE_SQL = """
CASE
    WHEN e.event_name ~* 'T100' THEN '1b'
    WHEN e.event_name ~* '(WTCS|Championship Series|World Triathlon Championship Finals)' THEN '1a'
    WHEN e.event_name ~* '(World Triathlon Cup|ITU Triathlon World Cup)' THEN '2'
    WHEN e.event_name ~* '(Continental Cup|Continental Championships|Americas Triathlon Cup|Americas Cup|European Cup|Europe Triathlon Cup|Asian Cup|Asia Triathlon Cup|African Cup|Africa Triathlon Cup|Africa Triathlon Premium Cup|Oceania Cup|Pan-American Cup|Panamerican Cup)' THEN '3'
    ELSE '4'
END
"""

TIER_NAMES = {
    "1a": "WTCS/Finals",
    "1b": "T100",
    "2": "World Cup",
    "3": "Continental",
    "4": "Other",
}


def get_test_events(engine, tier_filter=None):
    """Fetch H2 2025 test events with tier and distance metadata."""
    query = text(f"""
    SELECT DISTINCT
        e.event_id, e.prog_id, e.event_date, e.event_name,
        e.prog_name, e.prog_distance_category,
        {TIER_CASE_SQL} AS event_tier
    FROM events e
    JOIN race_results rr ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
    WHERE e.event_date >= '2025-07-01' AND e.event_date <= '2025-12-31'
    AND e.prog_name ~* '(Elite Men|Elite Women)'
    AND COALESCE(e.prog_distance_category, '') != 'long_distance'
    GROUP BY e.event_id, e.prog_id, e.event_date, e.event_name,
             e.prog_name, e.prog_distance_category
    HAVING COUNT(*) >= 10
    ORDER BY e.event_date, e.prog_name
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if tier_filter:
        df = df[df["event_tier"] == tier_filter]

    return df


def evaluate_swim_for_event(engine, event_id, prog_id, bundle, feature_cols, distance_cat, verbose=False):
    """
    Evaluate swim prediction accuracy for a single event.

    Returns a dict of metrics, or None if the event can't be evaluated.
    """
    key = ProgramKey(event_id=event_id, prog_id=prog_id)

    # Build features
    features_df = build_features_for_program(engine, key, use_start_list=False)
    if features_df.empty:
        return None

    features_df = fill_missing_features(features_df, feature_cols)

    # Get predictions
    pred_df = predict_splits_and_total(features_df, bundle, distance_category=distance_cat)

    # Get actual results
    results_df = fetch_program_results(engine, key)
    results_df["swim_sec"] = results_df["swimtime"].apply(parse_time_to_seconds)
    results_df["t1_sec"] = results_df["t1time"].apply(parse_time_to_seconds)
    results_df["total_sec"] = results_df["total_time"].apply(parse_time_to_seconds)

    # Filter to finishers with valid swim times
    results_df = results_df[results_df["finish_status"] == "FINISH"].copy()
    results_df = results_df[results_df["swim_sec"].notna() & (results_df["swim_sec"] > 0)]

    # Merge predictions with actuals
    merged = pred_df.merge(
        results_df[["athlete_id", "swim_sec", "t1_sec", "total_sec", "finish_position"]],
        on="athlete_id",
        how="inner",
        suffixes=("", "_actual"),
    )

    if len(merged) < 5:
        return None

    # ── Swim Ordering Metrics ──

    # Predicted swim order (by predicted swim time)
    if "pred_swim_sec" not in merged.columns:
        return None

    merged = merged[merged["pred_swim_sec"].notna() & (merged["pred_swim_sec"] > 0)]
    if len(merged) < 5:
        return None

    pred_swim_rank = merged["pred_swim_sec"].rank(method="average").values
    actual_swim_rank = merged["swim_sec"].rank(method="average").values

    # Spearman correlation of swim ordering
    swim_spearman, _ = spearmanr(pred_swim_rank, actual_swim_rank)

    # Swim P@K
    pred_swim_order = merged.sort_values("pred_swim_sec")["athlete_id"].tolist()
    actual_swim_order = merged.sort_values("swim_sec")["athlete_id"].tolist()
    n = len(merged)

    swim_p3 = _precision_at_k(pred_swim_order, actual_swim_order, min(3, n))
    swim_p5 = _precision_at_k(pred_swim_order, actual_swim_order, min(5, n))
    swim_p10 = _precision_at_k(pred_swim_order, actual_swim_order, min(10, n))

    # ── Gap Structure Metrics ──

    # Predicted gaps to leader
    pred_sorted = merged.sort_values("pred_swim_sec")
    pred_leader_time = pred_sorted["pred_swim_sec"].iloc[0]
    pred_gaps = pred_sorted["pred_swim_sec"].values - pred_leader_time

    # Actual gaps to leader
    actual_sorted = merged.sort_values("swim_sec")
    actual_leader_time = actual_sorted["swim_sec"].iloc[0]
    actual_gaps = actual_sorted["swim_sec"].values - actual_leader_time

    # Gap MAE: for each athlete, compare their predicted gap vs actual gap
    # (align by athlete, not by position)
    merged["pred_gap"] = merged["pred_swim_sec"] - merged["pred_swim_sec"].min()
    merged["actual_gap"] = merged["swim_sec"] - merged["swim_sec"].min()
    gap_mae = np.abs(merged["pred_gap"] - merged["actual_gap"]).mean()

    # Gap distribution comparison (KS test)
    if len(pred_gaps) >= 5 and len(actual_gaps) >= 5:
        ks_stat, ks_pval = ks_2samp(pred_gaps, actual_gaps)
    else:
        ks_stat, ks_pval = np.nan, np.nan

    # ── Pack Formation Metrics ──

    # Determine pack gap threshold for this distance
    distance = distance_cat or "standard"
    max_gap = DISTANCE_PACK_GAP_SEC.get(distance, 3.0)

    if max_gap > 0:
        # Predicted packs from predicted swim + predicted T1
        pred_bike_entry = merged["pred_swim_sec"].values
        if "pred_t1_sec" in merged.columns:
            pred_t1 = merged["pred_t1_sec"].fillna(0).values
            pred_bike_entry = pred_bike_entry + pred_t1

        pred_sort_idx = np.argsort(pred_bike_entry)
        pred_sorted_times = pred_bike_entry[pred_sort_idx]
        pred_pack_ids_sorted = assign_packs_chain(pred_sorted_times, max_gap)
        # Map back to original order
        pred_pack_ids = np.empty_like(pred_pack_ids_sorted)
        pred_pack_ids[pred_sort_idx] = pred_pack_ids_sorted

        # Actual packs from actual swim + actual T1
        actual_bike_entry = merged["swim_sec"].values
        if "t1_sec" in merged.columns:
            actual_t1 = merged["t1_sec"].fillna(0).values
            actual_bike_entry = actual_bike_entry + actual_t1

        actual_sort_idx = np.argsort(actual_bike_entry)
        actual_sorted_times = actual_bike_entry[actual_sort_idx]
        actual_pack_ids_sorted = assign_packs_chain(actual_sorted_times, max_gap)
        actual_pack_ids = np.empty_like(actual_pack_ids_sorted)
        actual_pack_ids[actual_sort_idx] = actual_pack_ids_sorted

        # Binary: front pack (0) vs not
        pred_front = (pred_pack_ids == 0)
        actual_front = (actual_pack_ids == 0)

        # Pack assignment accuracy (exact pack match)
        pack_acc = (pred_pack_ids == actual_pack_ids).mean()

        # Front pack recall: of actual front-pack, how many predicted correctly
        if actual_front.sum() > 0:
            front_recall = (pred_front & actual_front).sum() / actual_front.sum()
        else:
            front_recall = np.nan

        # Front pack precision: of predicted front-pack, how many actually were
        if pred_front.sum() > 0:
            front_precision = (pred_front & actual_front).sum() / pred_front.sum()
        else:
            front_precision = np.nan

        # Front pack F1
        if front_recall and front_precision and not np.isnan(front_recall) and not np.isnan(front_precision):
            if (front_recall + front_precision) > 0:
                front_f1 = 2 * front_recall * front_precision / (front_recall + front_precision)
            else:
                front_f1 = 0.0
        else:
            front_f1 = np.nan

        n_pred_packs = len(set(pred_pack_ids))
        n_actual_packs = len(set(actual_pack_ids))
        actual_front_size = int(actual_front.sum())
        pred_front_size = int(pred_front.sum())
    else:
        # Non-drafting distance — pack metrics not applicable
        pack_acc = np.nan
        front_recall = np.nan
        front_precision = np.nan
        front_f1 = np.nan
        n_pred_packs = np.nan
        n_actual_packs = np.nan
        actual_front_size = np.nan
        pred_front_size = np.nan

    # ── Swim Time MAE ──
    swim_time_mae = np.abs(merged["pred_swim_sec"] - merged["swim_sec"]).mean()

    metrics = {
        "n_athletes": n,
        "swim_spearman": swim_spearman,
        "swim_P@3": swim_p3,
        "swim_P@5": swim_p5,
        "swim_P@10": swim_p10,
        "swim_time_mae": swim_time_mae,
        "gap_mae": gap_mae,
        "gap_ks_stat": ks_stat,
        "gap_ks_pval": ks_pval,
        "pack_assignment_acc": pack_acc,
        "front_pack_recall": front_recall,
        "front_pack_precision": front_precision,
        "front_pack_f1": front_f1,
        "n_pred_packs": n_pred_packs,
        "n_actual_packs": n_actual_packs,
        "actual_front_size": actual_front_size,
        "pred_front_size": pred_front_size,
    }

    if verbose:
        print(f"    Athletes: {n}, Swim Spearman: {swim_spearman:.3f}, "
              f"Swim P@10: {swim_p10:.1%}, Pack Acc: {pack_acc:.1%}, "
              f"Front Recall: {front_recall:.1%}, Gap MAE: {gap_mae:.1f}s")

    return metrics


def _precision_at_k(pred_order, true_order, k):
    """Fraction of true top-K in predicted top-K."""
    pred_topk = set(pred_order[:k])
    true_topk = set(true_order[:k])
    return len(pred_topk & true_topk) / k if k > 0 else 0.0


def print_summary(all_results, test_events, label="Overall"):
    """Print aggregate metrics."""
    df = pd.DataFrame(all_results)
    if df.empty:
        print(f"  No results for {label}")
        return df

    print(f"\n{'=' * 75}")
    print(f"  SWIM PREDICTION ACCURACY — {label}")
    print(f"{'=' * 75}")
    print(f"  Events evaluated: {len(df)}")
    print(f"  Avg athletes/event: {df['n_athletes'].mean():.0f}")

    print(f"\n  --- Swim Ordering ---")
    print(f"  Swim Spearman:     {df['swim_spearman'].mean():.3f}  (std {df['swim_spearman'].std():.3f})")
    print(f"  Swim P@3:          {df['swim_P@3'].mean():.1%}")
    print(f"  Swim P@5:          {df['swim_P@5'].mean():.1%}")
    print(f"  Swim P@10:         {df['swim_P@10'].mean():.1%}")
    print(f"  Swim Time MAE:     {df['swim_time_mae'].mean():.1f}s")

    print(f"\n  --- Gap Structure ---")
    print(f"  Gap-to-leader MAE: {df['gap_mae'].mean():.1f}s")
    ks_valid = df["gap_ks_pval"].dropna()
    if len(ks_valid) > 0:
        print(f"  Gap KS p>0.05:     {(ks_valid > 0.05).mean():.0%} of events (gap dist matches actual)")

    pack_valid = df["pack_assignment_acc"].dropna()
    if len(pack_valid) > 0:
        print(f"\n  --- Pack Formation (draft-legal only) ---")
        print(f"  Pack assignment accuracy: {pack_valid.mean():.1%}")
        print(f"  Front pack recall:        {df['front_pack_recall'].dropna().mean():.1%}")
        print(f"  Front pack precision:     {df['front_pack_precision'].dropna().mean():.1%}")
        print(f"  Front pack F1:            {df['front_pack_f1'].dropna().mean():.1%}")
        print(f"  Avg actual front pack:    {df['actual_front_size'].dropna().mean():.0f} athletes")
        print(f"  Avg predicted front pack: {df['pred_front_size'].dropna().mean():.0f} athletes")
        print(f"  Avg actual packs:         {df['n_actual_packs'].dropna().mean():.1f}")
        print(f"  Avg predicted packs:      {df['n_pred_packs'].dropna().mean():.1f}")

    return df


def print_tier_breakdown(all_results, test_events):
    """Print metrics broken down by tier."""
    df = pd.DataFrame(all_results)
    if df.empty:
        return

    merged = df.merge(
        test_events[["event_id", "prog_id", "event_tier", "prog_distance_category"]],
        on=["event_id", "prog_id"],
        how="left",
    )

    # By tier
    print(f"\n{'=' * 75}")
    print(f"  BREAKDOWN BY TIER")
    print(f"{'=' * 75}")
    header = f"  {'Tier':<20} {'N':>4} {'SwimRho':>8} {'SwimP@10':>9} {'GapMAE':>7} {'PackAcc':>8} {'FPRecall':>9}"
    print(header)
    print(f"  {'-' * 70}")

    for tier in sorted(merged["event_tier"].dropna().unique()):
        tier_df = merged[merged["event_tier"] == tier]
        tier_name = TIER_NAMES.get(tier, tier)
        label = f"{tier} ({tier_name})"
        pack_acc = tier_df["pack_assignment_acc"].dropna().mean()
        fp_recall = tier_df["front_pack_recall"].dropna().mean()
        print(f"  {label:<20} {len(tier_df):>4} "
              f"{tier_df['swim_spearman'].mean():>8.3f} "
              f"{tier_df['swim_P@10'].mean():>8.1%} "
              f"{tier_df['gap_mae'].mean():>6.1f}s "
              f"{pack_acc:>7.1%} "
              f"{fp_recall:>8.1%}")

    # By distance
    print(f"\n{'=' * 75}")
    print(f"  BREAKDOWN BY DISTANCE")
    print(f"{'=' * 75}")
    print(header)
    print(f"  {'-' * 70}")

    for dist in sorted(merged["prog_distance_category"].dropna().unique()):
        dist_df = merged[merged["prog_distance_category"] == dist]
        pack_acc = dist_df["pack_assignment_acc"].dropna().mean()
        fp_recall = dist_df["front_pack_recall"].dropna().mean()
        print(f"  {dist:<20} {len(dist_df):>4} "
              f"{dist_df['swim_spearman'].mean():>8.3f} "
              f"{dist_df['swim_P@10'].mean():>8.1%} "
              f"{dist_df['gap_mae'].mean():>6.1f}s "
              f"{pack_acc:>7.1%} "
              f"{fp_recall:>8.1%}")

    # By field size bucket
    print(f"\n{'=' * 75}")
    print(f"  BREAKDOWN BY FIELD SIZE")
    print(f"{'=' * 75}")
    print(header)
    print(f"  {'-' * 70}")

    merged["field_bucket"] = pd.cut(
        merged["n_athletes"],
        bins=[0, 20, 35, 50, 200],
        labels=["<20", "20-35", "35-50", "50+"],
    )

    for bucket in ["<20", "20-35", "35-50", "50+"]:
        bucket_df = merged[merged["field_bucket"] == bucket]
        if bucket_df.empty:
            continue
        pack_acc = bucket_df["pack_assignment_acc"].dropna().mean()
        fp_recall = bucket_df["front_pack_recall"].dropna().mean()
        print(f"  {bucket:<20} {len(bucket_df):>4} "
              f"{bucket_df['swim_spearman'].mean():>8.3f} "
              f"{bucket_df['swim_P@10'].mean():>8.1%} "
              f"{bucket_df['gap_mae'].mean():>6.1f}s "
              f"{pack_acc:>7.1%} "
              f"{fp_recall:>8.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate swim prediction accuracy")
    parser.add_argument("--model", type=str, default="models/bundle_elite_v49.joblib",
                        help="Path to model bundle")
    parser.add_argument("--tier", type=str, default=None,
                        help="Filter to specific tier (1a, 1b, 2, 3, 4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-event details")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    engine = get_engine()
    test_events = get_test_events(engine, tier_filter=args.tier)

    if test_events.empty:
        print("No test events found.")
        return

    print(f"Evaluating swim accuracy on {len(test_events)} events "
          f"using {args.model}...")

    bundle = load_model_bundle(args.model)
    feature_cols = bundle.feature_columns or get_feature_columns()

    all_results = []
    for _, row in test_events.iterrows():
        event_id = int(row["event_id"])
        prog_id = int(row["prog_id"])
        distance_cat = row.get("prog_distance_category")

        if args.verbose:
            gender = "M" if "Men" in str(row.get("prog_name", "")) else "W"
            print(f"  [{gender}] {row['event_date']} {row['event_name']}")

        metrics = evaluate_swim_for_event(
            engine, event_id, prog_id, bundle, feature_cols, distance_cat,
            verbose=args.verbose,
        )
        if metrics is not None:
            metrics["event_id"] = event_id
            metrics["prog_id"] = prog_id
            all_results.append(metrics)

    # Print summary
    tier_label = f"Tier {args.tier} ({TIER_NAMES.get(args.tier, '')})" if args.tier else "All Tiers"
    results_df = print_summary(all_results, test_events, label=tier_label)

    if not args.tier and len(all_results) > 0:
        print_tier_breakdown(all_results, test_events)

    print(f"\n{'=' * 75}")
    print("  INTERPRETATION GUIDE")
    print(f"{'=' * 75}")
    print("  Swim Spearman > 0.8:  Swim ordering is well predicted")
    print("  Swim Spearman < 0.6:  Swim ordering is a major error source")
    print("  Front Pack Recall > 0.8: Most actual front pack members identified")
    print("  Front Pack Recall < 0.5: Pack formation is unreliable")
    print("  Gap MAE < 5s:  Gap structure is well captured")
    print("  Gap MAE > 15s: Gap predictions are noisy, will scramble packs")
    print()


if __name__ == "__main__":
    main()
