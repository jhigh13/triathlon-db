"""
Phase 10C: Sweep 3-way ensemble weights (LGBM ranker, XGB ranker, percentile).

Tests different weight combinations for the 3-model ensemble.
Uses v45 model (no retraining needed).
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


def run_sweep(model_path="models/bundle_elite_v45.joblib"):
    engine = get_engine()
    event_keys = get_test_event_keys(engine)
    print(f"Sweeping 3-way ensemble weights across {len(event_keys)} events...\n", flush=True)

    original_predict = predict_mod.predict_splits_and_total

    # Weight grid: (lgbm_w, xgb_w, pct_w) where they sum to 1.0
    # Step size 0.1, all combos that sum to 1.0
    combos = []
    for w_lgbm in np.arange(0.1, 0.9, 0.1):
        for w_xgb in np.arange(0.1, 0.9, 0.1):
            w_pct = round(1.0 - w_lgbm - w_xgb, 2)
            if 0.05 <= w_pct <= 0.85:
                combos.append((round(w_lgbm, 2), round(w_xgb, 2), w_pct))

    print(f"Testing {len(combos)} weight combinations\n", flush=True)
    results = []

    for w_lgbm, w_xgb, w_pct in combos:
        def make_wrapper(wl, wx, wp):
            def wrapper(features_df, bundle, distance_category=None):
                result = original_predict(features_df, bundle, distance_category=distance_category)
                if ("ranker_score" in result.columns and
                    "xgb_ranker_score" in result.columns and
                    "pred_finish_pct" in result.columns):
                    lgbm_rank = result["ranker_score"].rank(method="average", ascending=False)
                    xgb_rank = result["xgb_ranker_score"].rank(method="average", ascending=False)
                    pct_rank = result["pred_finish_pct"].rank(method="average", ascending=True)
                    result["ensemble_rank_score"] = wl * lgbm_rank + wx * xgb_rank + wp * pct_rank
                    result["predicted_rank"] = result["ensemble_rank_score"].rank(method="min", ascending=True).astype(int)
                return result
            return wrapper

        predict_mod.predict_splits_and_total = make_wrapper(w_lgbm, w_xgb, w_pct)

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
            "w_lgbm": w_lgbm,
            "w_xgb": w_xgb,
            "w_pct": w_pct,
            "P@3": p3,
            "P@10": p10,
            "Spearman": spearman,
        })

        marker = ""
        if abs(w_lgbm - 0.33) < 0.05 and abs(w_xgb - 0.33) < 0.05:
            marker = " <--"
        if p10 >= 0.745:
            marker += " ***"
        print(f"  lgbm={w_lgbm:.2f} xgb={w_xgb:.2f} pct={w_pct:.2f} -> P@3={p3:.1%} P@10={p10:.1%} rho={spearman:.3f}{marker}", flush=True)

    predict_mod.predict_splits_and_total = original_predict

    df = pd.DataFrame(results)
    best = df.loc[df["P@10"].idxmax()]
    equal = df[(df["w_lgbm"].between(0.3, 0.35)) & (df["w_xgb"].between(0.3, 0.35))]

    print("\n" + "=" * 70, flush=True)
    print(f"Best: lgbm={best['w_lgbm']:.2f} xgb={best['w_xgb']:.2f} pct={best['w_pct']:.2f} -> P@10={best['P@10']:.1%}", flush=True)
    if not equal.empty:
        baseline_p10 = equal.iloc[0]["P@10"]
        delta = best["P@10"] - baseline_p10
        print(f"Delta vs ~equal weights: {delta:+.1%}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/bundle_elite_v45.joblib")
    args = parser.parse_args()
    run_sweep(args.model)
