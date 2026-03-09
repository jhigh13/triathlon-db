"""
Phase 10: Sweep ensemble weights conditional on field size.

Tests whether large fields benefit from different ranker:pct blend than small fields.
Uses v40 model (no retraining needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.disable(logging.INFO)

import numpy as np
import pandas as pd
from tri_analysis.prediction.evaluate import backtest_events
from tri_analysis.database import get_engine
from sqlalchemy import text
import tri_analysis.prediction.predict as predict_mod


def get_test_event_keys(engine):
    query = text("""
    SELECT DISTINCT e.event_id, e.prog_id
    FROM events e
    JOIN race_results rr ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
    WHERE e.event_date >= '2025-07-01' AND e.event_date <= '2025-12-31'
    AND e.prog_name ~* '(Elite Men|Elite Women)'
    AND COALESCE(e.prog_distance_category, '') != 'long_distance'
    GROUP BY e.event_id, e.prog_id
    HAVING COUNT(*) >= 10
    ORDER BY e.event_id, e.prog_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return list(zip(df["event_id"], df["prog_id"]))


def run_sweep(model_path="models/bundle_elite_v40.joblib", threshold=35):
    engine = get_engine()
    event_keys = get_test_event_keys(engine)
    print(f"Testing field-size conditional weights (threshold={threshold})...", flush=True)
    print(f"Testing on {len(event_keys)} events\n", flush=True)

    original_predict = predict_mod.predict_splits_and_total

    # Test grid: (small_field_weight, large_field_weight)
    combos = []
    for w_small in np.arange(0.35, 0.65, 0.05):
        for w_large in np.arange(0.35, 0.65, 0.05):
            combos.append((round(w_small, 2), round(w_large, 2)))

    results = []

    for w_small, w_large in combos:
        def make_wrapper(ws, wl, thresh):
            def wrapper(features_df, bundle, distance_category=None):
                result = original_predict(features_df, bundle, distance_category=distance_category)
                if "ranker_score" in result.columns and "pred_finish_pct" in result.columns:
                    field_size = len(result)
                    w = wl if field_size >= thresh else ws
                    ranker_rank = result["ranker_score"].rank(method="average", ascending=False)
                    pct_rank = result["pred_finish_pct"].rank(method="average", ascending=True)
                    result["ensemble_rank_score"] = w * ranker_rank + (1 - w) * pct_rank
                    result["predicted_rank"] = result["ensemble_rank_score"].rank(method="min", ascending=True).astype(int)
                return result
            return wrapper

        predict_mod.predict_splits_and_total = make_wrapper(w_small, w_large, threshold)

        results_df = backtest_events(
            engine=engine,
            event_prog_keys=event_keys,
            bundle_path=model_path,
            run_simulation=False,
        )

        p10 = results_df["precision_at_10"].mean()
        p3 = results_df["precision_at_3"].mean()
        spearman = results_df["spearman_corr"].mean()

        results.append({
            "w_small": w_small,
            "w_large": w_large,
            "P@3": p3,
            "P@10": p10,
            "Spearman": spearman,
        })

        marker = " <--" if (w_small == 0.50 and w_large == 0.50) else ""
        if p10 >= 0.742:
            marker += " ***"
        print(f"  small={w_small:.2f} large={w_large:.2f} -> P@3={p3:.1%} P@10={p10:.1%} rho={spearman:.3f}{marker}", flush=True)

    predict_mod.predict_splits_and_total = original_predict

    df = pd.DataFrame(results)
    best = df.loc[df["P@10"].idxmax()]
    baseline = df[(df["w_small"] == 0.50) & (df["w_large"] == 0.50)]

    print("\n" + "=" * 70, flush=True)
    print(f"Best: small={best['w_small']:.2f} large={best['w_large']:.2f} -> P@10={best['P@10']:.1%}", flush=True)
    if not baseline.empty:
        delta = best["P@10"] - baseline.iloc[0]["P@10"]
        print(f"Delta vs 50/50: {delta:+.1%}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/bundle_elite_v40.joblib")
    parser.add_argument("--threshold", type=int, default=35)
    args = parser.parse_args()
    run_sweep(args.model, args.threshold)
