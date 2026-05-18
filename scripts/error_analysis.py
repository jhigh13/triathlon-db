"""
Detailed error analysis for backtest predictions.

Identifies WHERE P@10 fails: which athletes are predicted outside top 10
but actually finish in top 10, and vice versa.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from sqlalchemy import text

from tri_analysis.prediction.evaluate import precision_at_k
from tri_analysis.prediction.sql import ProgramKey, fetch_program_results, fetch_event_metadata
from tri_analysis.prediction.features import build_features_for_program, fill_missing_features
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.utils_time import parse_time_to_seconds
from tri_analysis.database import get_engine


def analyze_errors(engine, event_prog_keys, bundle, feature_cols, test_events_df):
    """Run predictions and collect per-athlete error data."""
    all_errors = []
    event_summaries = []

    for i, (event_id, prog_id) in enumerate(event_prog_keys):
        key = ProgramKey(event_id=event_id, prog_id=prog_id)

        try:
            features_df = build_features_for_program(engine, key, use_start_list=False)
            if features_df.empty:
                continue
            features_df = fill_missing_features(features_df, feature_cols)

            event_meta = fetch_event_metadata(engine, key)
            distance_cat = event_meta.get("prog_distance_category") if event_meta else None
            event_name = event_meta.get("event_name", "") if event_meta else ""

            pred_df = predict_splits_and_total(features_df, bundle, distance_category=distance_cat)

            results_df = fetch_program_results(engine, key)
            results_df["total_sec"] = results_df["total_time"].apply(parse_time_to_seconds)

            # Merge predictions with results
            merged = pred_df.merge(
                results_df[["athlete_id", "total_sec", "finish_position", "finish_status",
                            "athlete_full_name"]],
                on="athlete_id", how="inner", suffixes=("_pred", "_actual")
            )
            merged = merged[merged["finish_status"] == "FINISH"].copy()
            if merged.empty:
                continue

            pred_order = merged.sort_values("predicted_rank")["athlete_id"].tolist()
            true_order = merged.sort_values("finish_position")["athlete_id"].tolist()

            p10 = precision_at_k(pred_order, true_order, 10) if len(true_order) >= 10 else np.nan

            # Get tier from test_events_df
            tier_row = test_events_df[
                (test_events_df["event_id"] == event_id) &
                (test_events_df["prog_id"] == prog_id)
            ]
            tier = tier_row["event_tier"].iloc[0] if not tier_row.empty else "?"

            n_field = len(merged)

            # Find misranked athletes for P@10
            if len(true_order) >= 10:
                pred_top10 = set(pred_order[:10])
                true_top10 = set(true_order[:10])
                missed = true_top10 - pred_top10  # Actually top 10 but predicted outside
                false_alarms = pred_top10 - true_top10  # Predicted top 10 but actually outside

                for aid in missed:
                    row = merged[merged["athlete_id"] == aid].iloc[0]
                    all_errors.append({
                        "event_id": event_id,
                        "prog_id": prog_id,
                        "event_name": event_name,
                        "tier": tier,
                        "distance": distance_cat or "",
                        "error_type": "missed",  # should be top10, wasn't predicted
                        "athlete_id": aid,
                        "athlete_full_name": row.get("athlete_full_name", ""),
                        "predicted_rank": int(row["predicted_rank"]),
                        "actual_rank": int(row["finish_position"]),
                        "rank_error": int(row["predicted_rank"] - row["finish_position"]),
                        "pred_total_sec": row.get("pred_total_sec", np.nan),
                        "actual_total_sec": row["total_sec"],
                        "pred_finish_pct": row.get("pred_finish_pct", np.nan),
                        "n_field": n_field,
                        "event_p10": p10,
                    })

                for aid in false_alarms:
                    row = merged[merged["athlete_id"] == aid].iloc[0]
                    all_errors.append({
                        "event_id": event_id,
                        "prog_id": prog_id,
                        "event_name": event_name,
                        "tier": tier,
                        "distance": distance_cat or "",
                        "error_type": "false_alarm",  # predicted top10, wasn't
                        "athlete_id": aid,
                        "athlete_full_name": row.get("athlete_full_name", ""),
                        "predicted_rank": int(row["predicted_rank"]),
                        "actual_rank": int(row["finish_position"]),
                        "rank_error": int(row["predicted_rank"] - row["finish_position"]),
                        "pred_total_sec": row.get("pred_total_sec", np.nan),
                        "actual_total_sec": row["total_sec"],
                        "pred_finish_pct": row.get("pred_finish_pct", np.nan),
                        "n_field": n_field,
                        "event_p10": p10,
                    })

            event_summaries.append({
                "event_id": event_id,
                "prog_id": prog_id,
                "event_name": event_name,
                "tier": tier,
                "distance": distance_cat or "",
                "p10": p10,
                "n_field": n_field,
                "n_missed": len(missed) if len(true_order) >= 10 else 0,
                "n_false_alarm": len(false_alarms) if len(true_order) >= 10 else 0,
            })

            status = "OK" if p10 >= 0.8 else "MISS"
            print(f"  [{i+1:2d}/{len(event_prog_keys)}] {status} P@10={p10:.0%} "
                  f"T{tier} n={n_field} {event_name}")

        except Exception as e:
            print(f"  [{i+1}/{len(event_prog_keys)}] ERROR {event_name}: {e}")
            continue

    return pd.DataFrame(all_errors), pd.DataFrame(event_summaries)


def main():
    parser = argparse.ArgumentParser(description="Detailed backtest error analysis")
    parser.add_argument("--model", type=str, default="models/bundle_elite_v45.joblib")
    args = parser.parse_args()

    engine = get_engine()
    bundle = load_model_bundle(args.model)
    feature_cols = bundle.feature_columns

    # Get test events (same query as run_backtest.py)
    query = text("""
    SELECT DISTINCT
        e.event_id, e.prog_id, e.event_date, e.event_name, e.prog_name,
        e.prog_distance_category,
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
    GROUP BY e.event_id, e.prog_id, e.event_date, e.event_name, e.prog_name, e.prog_distance_category
    HAVING COUNT(*) >= 10
    ORDER BY event_tier, event_date, e.prog_name
    """)
    with engine.connect() as conn:
        test_events = pd.read_sql(query, conn)

    print(f"\n=== Error Analysis: {args.model} ===")
    print(f"Test events: {len(test_events)}\n")

    keys = list(test_events[['event_id', 'prog_id']].itertuples(index=False, name=None))
    errors_df, summaries_df = analyze_errors(engine, keys, bundle, feature_cols, test_events)

    # === Analysis ===
    print("\n" + "=" * 80)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 80)

    # 1. Events where P@10 < 80%
    bad_events = summaries_df[summaries_df["p10"] < 0.8].sort_values("p10")
    print(f"\n--- Events with P@10 < 80% ({len(bad_events)}/{len(summaries_df)}) ---")
    for _, row in bad_events.iterrows():
        print(f"  P@10={row['p10']:.0%} T{row['tier']} [{row['distance']:10s}] "
              f"n={row['n_field']} miss={row['n_missed']} fa={row['n_false_alarm']} "
              f"{row['event_name']}")

    # 2. By tier
    print(f"\n--- P@10 by Tier ---")
    for tier in sorted(summaries_df["tier"].unique()):
        tier_s = summaries_df[summaries_df["tier"] == tier]
        pct_good = (tier_s["p10"] >= 0.8).mean()
        print(f"  Tier {tier}: mean P@10={tier_s['p10'].mean():.1%}, "
              f"{pct_good:.0%} events >=80%, n={len(tier_s)}")

    # 3. By distance
    print(f"\n--- P@10 by Distance ---")
    for dist in sorted(summaries_df["distance"].unique()):
        dist_s = summaries_df[summaries_df["distance"] == dist]
        if len(dist_s) >= 2:
            print(f"  {dist:20s}: mean P@10={dist_s['p10'].mean():.1%}, n={len(dist_s)}")

    # 4. Error patterns: how far off are the misranked athletes?
    if not errors_df.empty:
        missed = errors_df[errors_df["error_type"] == "missed"]
        false_alarms = errors_df[errors_df["error_type"] == "false_alarm"]

        print(f"\n--- Missed Athletes (actual top10, predicted outside) ---")
        print(f"  Count: {len(missed)}")
        print(f"  Mean predicted rank: {missed['predicted_rank'].mean():.1f}")
        print(f"  Median predicted rank: {missed['predicted_rank'].median():.0f}")
        print(f"  Predicted rank distribution:")
        bins = [10, 15, 20, 25, 30, 100]
        for i in range(len(bins) - 1):
            n = ((missed["predicted_rank"] >= bins[i]+1) & (missed["predicted_rank"] <= bins[i+1])).sum()
            print(f"    Rank {bins[i]+1:2d}-{bins[i+1]:3d}: {n} athletes ({n/len(missed)*100:.0f}%)")

        print(f"\n--- False Alarms (predicted top10, actual outside) ---")
        print(f"  Count: {len(false_alarms)}")
        print(f"  Mean actual rank: {false_alarms['actual_rank'].mean():.1f}")
        print(f"  Median actual rank: {false_alarms['actual_rank'].median():.0f}")
        print(f"  Actual rank distribution:")
        for i in range(len(bins) - 1):
            n = ((false_alarms["actual_rank"] >= bins[i]+1) & (false_alarms["actual_rank"] <= bins[i+1])).sum()
            print(f"    Rank {bins[i]+1:2d}-{bins[i+1]:3d}: {n} athletes ({n/len(false_alarms)*100:.0f}%)")

        # 5. Are there repeat offenders?
        print(f"\n--- Repeat Missed Athletes (appear in 3+ events) ---")
        repeat_missed = missed.groupby("athlete_id").agg(
            n_times=("event_id", "count"),
            name=("athlete_full_name", "first"),
            avg_pred_rank=("predicted_rank", "mean"),
            avg_actual_rank=("actual_rank", "mean"),
        ).sort_values("n_times", ascending=False)
        repeats = repeat_missed[repeat_missed["n_times"] >= 3]
        if len(repeats) > 0:
            for _, row in repeats.head(15).iterrows():
                print(f"  {row['name']:30s} missed {row['n_times']}x, "
                      f"avg pred={row['avg_pred_rank']:.0f}, avg actual={row['avg_actual_rank']:.0f}")
        else:
            print("  None (trying 2+ events)...")
            repeats2 = repeat_missed[repeat_missed["n_times"] >= 2]
            for _, row in repeats2.head(15).iterrows():
                print(f"  {row['name']:30s} missed {row['n_times']}x, "
                      f"avg pred={row['avg_pred_rank']:.0f}, avg actual={row['avg_actual_rank']:.0f}")

        # 6. Field size effect
        print(f"\n--- P@10 by Field Size ---")
        bins_field = [0, 20, 30, 40, 50, 100]
        for i in range(len(bins_field) - 1):
            mask = (summaries_df["n_field"] >= bins_field[i]) & (summaries_df["n_field"] < bins_field[i+1])
            sub = summaries_df[mask]
            if len(sub) > 0:
                print(f"  Field {bins_field[i]:2d}-{bins_field[i+1]:2d}: "
                      f"mean P@10={sub['p10'].mean():.1%}, n={len(sub)}")

        # 7. Percentile model analysis: how does pred_finish_pct relate to errors?
        if "pred_finish_pct" in errors_df.columns and errors_df["pred_finish_pct"].notna().any():
            print(f"\n--- Percentile Model Analysis ---")
            missed_pct = missed["pred_finish_pct"].dropna()
            fa_pct = false_alarms["pred_finish_pct"].dropna()
            if len(missed_pct) > 0:
                print(f"  Missed athletes pred_finish_pct:  mean={missed_pct.mean():.3f}, "
                      f"median={missed_pct.median():.3f}")
            if len(fa_pct) > 0:
                print(f"  False alarm pred_finish_pct:      mean={fa_pct.mean():.3f}, "
                      f"median={fa_pct.median():.3f}")

        # 8. Time error analysis
        print(f"\n--- Time Error Analysis ---")
        missed_time_err = (missed["pred_total_sec"] - missed["actual_total_sec"]).dropna()
        fa_time_err = (false_alarms["pred_total_sec"] - false_alarms["actual_total_sec"]).dropna()
        if len(missed_time_err) > 0:
            print(f"  Missed: mean time error = {missed_time_err.mean():.1f}s "
                  f"(positive = predicted too slow)")
        if len(fa_time_err) > 0:
            print(f"  False alarms: mean time error = {fa_time_err.mean():.1f}s "
                  f"(negative = predicted too fast)")

    # Save detailed data for further analysis
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    errors_df.to_csv(out_dir / "error_analysis_v37.csv", index=False)
    summaries_df.to_csv(out_dir / "event_summaries_v37.csv", index=False)
    print(f"\nSaved: outputs/error_analysis_v37.csv ({len(errors_df)} rows)")
    print(f"Saved: outputs/event_summaries_v37.csv ({len(summaries_df)} rows)")


if __name__ == "__main__":
    main()
