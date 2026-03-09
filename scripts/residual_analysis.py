"""
Phase 10A-1: Residual Analysis by Field Size

Diagnostic script to understand WHY large-field events have lower P@10.
Outputs per-event diagnostics: field_size, cold-start athlete counts,
fraction of top-10 misses that are cold-start, and P@10 by field-size bucket.
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

import numpy as np
import pandas as pd
from sqlalchemy import text

from tri_analysis.database import get_engine
from tri_analysis.prediction.sql import ProgramKey, fetch_program_results, fetch_event_metadata
from tri_analysis.prediction.features import (
    build_features_for_program, fill_missing_features, get_feature_columns,
)
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.utils_time import parse_time_to_seconds
from tri_analysis.prediction.evaluate import precision_at_k

COLD_START_THRESHOLD = 3  # athletes with < this many matched races


def analyze_event(engine, event_id, prog_id, bundle, feature_cols):
    """Run prediction for one event and return per-athlete diagnostic data."""
    key = ProgramKey(event_id=event_id, prog_id=prog_id)

    features_df = build_features_for_program(engine, key, use_start_list=False)
    if features_df.empty:
        return None

    features_df = fill_missing_features(features_df, feature_cols)

    # Debug: check if n_matched_races survived fill_missing_features
    if "n_matched_races" not in features_df.columns:
        # n_matched_races may have been filtered; recompute from original
        pass

    event_meta = fetch_event_metadata(engine, key)
    distance_cat = event_meta.get("prog_distance_category") if event_meta else None

    pred_df = predict_splits_and_total(features_df, bundle, distance_category=distance_cat)

    results_df = fetch_program_results(engine, key)
    results_df["total_sec"] = results_df["total_time"].apply(parse_time_to_seconds)

    # Get n_matched_races from features before merge (predict may drop it)
    if "n_matched_races" not in features_df.columns:
        # Fallback: use races_12m if n_matched_races not available
        race_col = "races_12m" if "races_12m" in features_df.columns else None
        if race_col:
            race_counts = features_df[["athlete_id", race_col]].copy()
            race_counts = race_counts.rename(columns={race_col: "n_matched_races"})
        else:
            race_counts = features_df[["athlete_id"]].copy()
            race_counts["n_matched_races"] = 0
    else:
        race_counts = features_df[["athlete_id", "n_matched_races"]].copy()

    # Merge predictions with actuals
    merged = pred_df.merge(
        results_df[["athlete_id", "total_sec", "finish_position", "finish_status"]],
        on="athlete_id",
        how="inner",
    )
    merged = merged[merged["finish_status"] == "FINISH"]

    if merged.empty:
        return None

    field_size = len(merged)

    merged = merged.merge(race_counts, on="athlete_id", how="left")
    # After merge, n_matched_races may have suffix if it existed in both sides
    nmr_col = "n_matched_races"
    if nmr_col not in merged.columns:
        # Check for suffixed version
        for c in merged.columns:
            if "n_matched_races" in c:
                nmr_col = c
                break
        else:
            # No n_matched_races at all; set to 0
            merged["n_matched_races"] = 0
            nmr_col = "n_matched_races"
    merged["is_cold_start"] = merged[nmr_col].fillna(0) < COLD_START_THRESHOLD
    merged["n_matched_races"] = merged[nmr_col].fillna(0)

    # Compute P@10
    pred_order = merged.sort_values("predicted_rank")["athlete_id"].tolist()
    true_order = merged.sort_values("finish_position")["athlete_id"].tolist()
    p10 = precision_at_k(pred_order, true_order, k=min(10, field_size))
    p3 = precision_at_k(pred_order, true_order, k=min(3, field_size))

    # Identify top-10 misses: athletes in actual top-10 but NOT in predicted top-10
    actual_top10 = set(true_order[:10])
    pred_top10 = set(pred_order[:10])
    missed_athletes = actual_top10 - pred_top10

    # For each missed athlete, get their predicted rank and cold-start status
    miss_details = []
    for aid in missed_athletes:
        row = merged[merged["athlete_id"] == aid].iloc[0]
        miss_details.append({
            "athlete_id": aid,
            "predicted_rank": int(row["predicted_rank"]),
            "actual_rank": int(row["finish_position"]),
            "n_matched_races": int(row["n_matched_races"]),
            "is_cold_start": bool(row["is_cold_start"]),
        })

    n_cold_start = int(merged["is_cold_start"].sum())
    n_cold_start_in_top10_actual = int(
        merged[merged["athlete_id"].isin(actual_top10)]["is_cold_start"].sum()
    )
    n_cold_start_misses = sum(1 for m in miss_details if m["is_cold_start"])

    return {
        "event_id": event_id,
        "prog_id": prog_id,
        "event_name": event_meta.get("event_name", ""),
        "prog_name": event_meta.get("prog_name", ""),
        "event_tier": features_df["event_tier"].iloc[0] if "event_tier" in features_df.columns else "",
        "field_size": field_size,
        "n_cold_start": n_cold_start,
        "cold_start_pct": n_cold_start / field_size,
        "p3": p3,
        "p10": p10,
        "n_misses": len(missed_athletes),
        "n_cold_start_in_actual_top10": n_cold_start_in_top10_actual,
        "n_cold_start_misses": n_cold_start_misses,
        "cold_start_miss_frac": n_cold_start_misses / len(missed_athletes) if missed_athletes else 0.0,
        "miss_details": miss_details,
        # Distribution of n_matched_races
        "median_n_races": float(merged["n_matched_races"].median()),
        "q25_n_races": float(merged["n_matched_races"].quantile(0.25)),
        "mean_n_races": float(merged["n_matched_races"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Residual analysis: field size vs P@10")
    parser.add_argument("--model", type=str, default="models/bundle_elite_v40.joblib")
    args = parser.parse_args()

    engine = get_engine()
    bundle = load_model_bundle(args.model)
    feature_cols = bundle.feature_columns or get_feature_columns()

    # Get H2 2025 test events (same query as run_backtest.py)
    query = text("""
    SELECT DISTINCT e.event_id, e.prog_id, e.event_date, e.event_name, e.prog_name,
        CASE
            WHEN e.event_name ~* 'T100' THEN '1b'
            WHEN e.event_name ~* '(WTCS|Championship Series|World Triathlon Championship Finals)' THEN '1a'
            WHEN e.event_name ~* '(World Triathlon Cup|ITU Triathlon World Cup)' THEN '2'
            WHEN e.event_name ~* '(Continental Cup|Continental Championships|Americas Triathlon Cup|Americas Cup|European Cup|Europe Triathlon Cup|Asian Cup|Asia Triathlon Cup|African Cup|Africa Triathlon Cup|Africa Triathlon Premium Cup|Oceania Cup|Pan-American Cup|Panamerican Cup)' THEN '3'
            ELSE '4'
        END AS event_tier
    FROM events e
    JOIN race_results rr ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
    WHERE e.event_date >= '2025-07-01' AND e.event_date <= '2025-12-31'
    AND e.prog_name ~* '(Elite Men|Elite Women)'
    AND COALESCE(e.prog_distance_category, '') != 'long_distance'
    GROUP BY e.event_id, e.prog_id, e.event_date, e.event_name, e.prog_name
    HAVING COUNT(*) >= 10
    ORDER BY event_tier, event_date
    """)
    with engine.connect() as conn:
        test_events = pd.read_sql(query, conn)

    print(f"Analyzing {len(test_events)} events...\n")

    all_results = []
    all_miss_details = []

    for _, row in test_events.iterrows():
        result = analyze_event(engine, row["event_id"], row["prog_id"], bundle, feature_cols)
        if result:
            miss_details = result.pop("miss_details")
            all_results.append(result)
            for md in miss_details:
                md["event_id"] = row["event_id"]
                md["prog_id"] = row["prog_id"]
                md["field_size"] = result["field_size"]
                md["event_tier"] = result["event_tier"]
            all_miss_details.extend(miss_details)

    df = pd.DataFrame(all_results)
    miss_df = pd.DataFrame(all_miss_details) if all_miss_details else pd.DataFrame()

    # ── 1. P@10 by field-size bucket ──
    bins = [0, 20, 30, 40, 50, 200]
    labels = ["<20", "20-30", "30-40", "40-50", "50+"]
    df["field_bucket"] = pd.cut(df["field_size"], bins=bins, labels=labels, right=False)

    print("=" * 70)
    print("1. P@10 BY FIELD SIZE BUCKET")
    print("=" * 70)
    bucket_stats = df.groupby("field_bucket", observed=False).agg(
        n_events=("p10", "count"),
        mean_p10=("p10", "mean"),
        mean_p3=("p3", "mean"),
        mean_field_size=("field_size", "mean"),
        mean_cold_start_pct=("cold_start_pct", "mean"),
        mean_cold_start_miss_frac=("cold_start_miss_frac", "mean"),
    ).round(3)
    print(bucket_stats.to_string())

    # ── 2. Cold-start athlete analysis ──
    print("\n" + "=" * 70)
    print("2. COLD-START ATHLETES (n_matched_races < 3)")
    print("=" * 70)
    print(f"Total events: {len(df)}")
    print(f"Mean cold-start % per field: {df['cold_start_pct'].mean():.1%}")
    print(f"Mean cold-start misses as fraction of all misses: {df['cold_start_miss_frac'].mean():.1%}")
    print(f"\nCorrelation: field_size vs cold_start_pct: {df['field_size'].corr(df['cold_start_pct']):.3f}")
    print(f"Correlation: cold_start_pct vs P@10: {df['cold_start_pct'].corr(df['p10']):.3f}")
    print(f"Correlation: field_size vs P@10: {df['field_size'].corr(df['p10']):.3f}")

    # ── 3. Missed athlete analysis ──
    if not miss_df.empty:
        print("\n" + "=" * 70)
        print("3. MISSED ATHLETE ANALYSIS (actual top-10, predicted outside top-10)")
        print("=" * 70)
        print(f"Total misses across all events: {len(miss_df)}")
        print(f"Cold-start misses: {miss_df['is_cold_start'].sum()} ({miss_df['is_cold_start'].mean():.1%})")
        print(f"\nMissed athletes by predicted rank bucket:")
        miss_bins = [10, 12, 15, 20, 30, 200]
        miss_labels = ["11-12", "13-15", "16-20", "21-30", "30+"]
        miss_df["pred_rank_bucket"] = pd.cut(
            miss_df["predicted_rank"], bins=miss_bins, labels=miss_labels, right=True
        )
        miss_by_rank = miss_df.groupby("pred_rank_bucket", observed=False).agg(
            count=("athlete_id", "count"),
            cold_start_count=("is_cold_start", "sum"),
            mean_n_races=("n_matched_races", "mean"),
        )
        miss_by_rank["pct_of_misses"] = (miss_by_rank["count"] / len(miss_df) * 100).round(1)
        print(miss_by_rank.to_string())

        # By field-size bucket
        print(f"\nMissed athletes by field-size bucket:")
        miss_df["field_bucket"] = pd.cut(miss_df["field_size"], bins=bins, labels=labels, right=False)
        miss_by_field = miss_df.groupby("field_bucket", observed=False).agg(
            count=("athlete_id", "count"),
            cold_start_count=("is_cold_start", "sum"),
            cold_start_pct=("is_cold_start", "mean"),
            mean_n_races=("n_matched_races", "mean"),
        ).round(3)
        print(miss_by_field.to_string())

    # ── 4. Per-tier breakdown ──
    print("\n" + "=" * 70)
    print("4. BY-TIER COLD START ANALYSIS")
    print("=" * 70)
    TIER_NAMES = {"1a": "WTCS", "1b": "T100", "2": "World Cup", "3": "Continental", "4": "Other"}
    for tier in sorted(df["event_tier"].unique()):
        tdf = df[df["event_tier"] == tier]
        tier_name = TIER_NAMES.get(str(tier), "Unknown")
        print(f"\nTier {tier} ({tier_name}): n={len(tdf)}")
        print(f"  Mean field size: {tdf['field_size'].mean():.0f}")
        print(f"  Mean P@10: {tdf['p10'].mean():.1%}")
        print(f"  Mean cold-start %: {tdf['cold_start_pct'].mean():.1%}")
        print(f"  Mean cold-start miss fraction: {tdf['cold_start_miss_frac'].mean():.1%}")
        print(f"  Mean n_matched_races (all athletes): {tdf['mean_n_races'].mean():.1f}")
        print(f"  Q25 n_matched_races: {tdf['q25_n_races'].mean():.1f}")

    # ── 5. Worst events ──
    print("\n" + "=" * 70)
    print("5. WORST 10 EVENTS BY P@10")
    print("=" * 70)
    worst = df.nsmallest(10, "p10")[
        ["event_name", "prog_name", "event_tier", "field_size", "p10",
         "n_cold_start", "cold_start_pct", "cold_start_miss_frac"]
    ]
    for _, r in worst.iterrows():
        print(f"  P@10={r['p10']:.0%} | field={r['field_size']} | "
              f"cold_start={r['n_cold_start']} ({r['cold_start_pct']:.0%}) | "
              f"cold_miss_frac={r['cold_start_miss_frac']:.0%} | "
              f"T{r['event_tier']} | {r['event_name']} [{r['prog_name']}]")

    # ── 6. Summary ──
    print("\n" + "=" * 70)
    print("6. SUMMARY & KEY FINDINGS")
    print("=" * 70)
    large_field = df[df["field_size"] >= 40]
    small_field = df[df["field_size"] < 30]
    print(f"Small fields (<30): P@10={small_field['p10'].mean():.1%}, "
          f"cold-start={small_field['cold_start_pct'].mean():.1%}, n={len(small_field)}")
    print(f"Large fields (40+): P@10={large_field['p10'].mean():.1%}, "
          f"cold-start={large_field['cold_start_pct'].mean():.1%}, n={len(large_field)}")
    gap = small_field['p10'].mean() - large_field['p10'].mean()
    print(f"P@10 gap: {gap:.1%}")

    if not miss_df.empty:
        cold_miss = miss_df[miss_df["is_cold_start"]]
        warm_miss = miss_df[~miss_df["is_cold_start"]]
        print(f"\nAll misses: {len(miss_df)} total")
        print(f"  Cold-start misses: {len(cold_miss)} ({len(cold_miss)/len(miss_df):.1%})")
        print(f"  Warm misses: {len(warm_miss)} ({len(warm_miss)/len(miss_df):.1%})")
        if len(cold_miss) > 0:
            print(f"  Cold-start mean pred rank: {cold_miss['predicted_rank'].mean():.1f}")
        if len(warm_miss) > 0:
            print(f"  Warm mean pred rank: {warm_miss['predicted_rank'].mean():.1f}")


if __name__ == "__main__":
    main()
