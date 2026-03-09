"""
Phase 10A-3: Sweep RANKER_WEIGHT to find optimal ensemble blend.

Monkey-patches predict_splits_and_total to re-rank with different weights.
No retraining needed — uses the existing v40 model.
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
logging.disable(logging.INFO)  # Suppress verbose per-event logging

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


def run_sweep(model_path="models/bundle_elite_v40.joblib"):
    engine = get_engine()
    event_keys = get_test_event_keys(engine)
    print(f"Sweeping ensemble weights across {len(event_keys)} events...\n", flush=True)

    # Save the original function
    original_predict = predict_mod.predict_splits_and_total

    weights = np.arange(0.30, 0.75, 0.05)
    all_results = []

    for w in weights:
        def make_wrapper(weight_val):
            def wrapper(features_df, bundle, distance_category=None):
                result = original_predict(features_df, bundle, distance_category=distance_category)
                # Re-rank with our weight (overriding the 50/50 applied inside)
                if "ranker_score" in result.columns and "pred_finish_pct" in result.columns:
                    ranker_rank = result["ranker_score"].rank(method="average", ascending=False)
                    pct_rank = result["pred_finish_pct"].rank(method="average", ascending=True)
                    result["ensemble_rank_score"] = weight_val * ranker_rank + (1 - weight_val) * pct_rank
                    result["predicted_rank"] = result["ensemble_rank_score"].rank(method="min", ascending=True).astype(int)
                return result
            return wrapper

        predict_mod.predict_splits_and_total = make_wrapper(w)

        results_df = backtest_events(
            engine=engine,
            event_prog_keys=event_keys,
            bundle_path=model_path,
            run_simulation=False,
        )

        p3 = results_df["precision_at_3"].mean()
        p5 = results_df["precision_at_5"].mean()
        p10 = results_df["precision_at_10"].mean()
        spearman = results_df["spearman_corr"].mean()

        all_results.append({
            "w": round(w, 2),
            "P@3": p3,
            "P@5": p5,
            "P@10": p10,
            "Spearman": spearman,
        })
        print(f"  w={w:.2f} -> P@3={p3:.1%}, P@5={p5:.1%}, P@10={p10:.1%}, Spearman={spearman:.3f}", flush=True)

    # Restore
    predict_mod.predict_splits_and_total = original_predict

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("SWEEP RESULTS (ranker_weight : pct_weight)")
    print("=" * 70)
    for _, r in df.iterrows():
        marker = " <-" if r["w"] == 0.50 else ""
        best_marker = " * BEST" if r["P@10"] == df["P@10"].max() else ""
        print(f"  {r['w']:.2f}:{1-r['w']:.2f}  P@3={r['P@3']:.1%}  P@5={r['P@5']:.1%}  P@10={r['P@10']:.1%}  rho={r['Spearman']:.3f}{marker}{best_marker}")

    best = df.loc[df["P@10"].idxmax()]
    baseline = df[df["w"] == 0.50].iloc[0]
    delta = best["P@10"] - baseline["P@10"]
    print(f"\nBest: w={best['w']:.2f} → P@10={best['P@10']:.1%} (delta vs 50/50: {delta:+.1%})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/bundle_elite_v40.joblib")
    args = parser.parse_args()
    run_sweep(args.model)
