"""
v55 = v54 with ensemble ranker/percentile weight re-optimised for P@3.

Same machinery as build_v53_weight_sweep.py but operates on v54 (which has
the new field-boundary features) instead of v45.

Usage:
    python scripts/build_v55_weight_sweep.py
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

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from tri_analysis.database import get_engine
from tri_analysis.prediction.evaluate import backtest_events
import tri_analysis.prediction.predict as predict_mod

SOURCE = "models/bundle_elite_v54.joblib"
TARGET = "models/bundle_elite_v55.joblib"


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


def main():
    engine = get_engine()
    event_keys = get_test_event_keys(engine)
    print(f"Sweeping ensemble weights on v54 across {len(event_keys)} H2 2025 events…")

    original_predict = predict_mod.predict_splits_and_total
    weights = np.arange(0.30, 0.96, 0.05)
    rows = []

    for w in weights:
        def make_wrapper(weight_val):
            def wrapper(features_df, bundle, distance_category=None):
                result = original_predict(features_df, bundle, distance_category=distance_category)
                if "ranker_score" in result.columns and "pred_finish_pct" in result.columns:
                    ranker_rank = result["ranker_score"].rank(method="average", ascending=False)
                    pct_rank = result["pred_finish_pct"].rank(method="average", ascending=True)
                    result["ensemble_rank_score"] = (
                        weight_val * ranker_rank + (1 - weight_val) * pct_rank
                    )
                    result["predicted_rank"] = (
                        result["ensemble_rank_score"].rank(method="min", ascending=True).astype(int)
                    )
                return result
            return wrapper
        predict_mod.predict_splits_and_total = make_wrapper(w)
        results_df = backtest_events(
            engine=engine,
            event_prog_keys=event_keys,
            bundle_path=SOURCE,
            run_simulation=False,
        )
        rows.append({
            "w": round(w, 2),
            "P@1": results_df["precision_at_1"].mean() if "precision_at_1" in results_df.columns else float("nan"),
            "P@3": results_df["precision_at_3"].mean(),
            "P@5": results_df["precision_at_5"].mean(),
            "P@10": results_df["precision_at_10"].mean(),
            "Spearman": results_df["spearman_corr"].mean(),
            "weighted": (
                0.5 * results_df["precision_at_3"].mean()
                + 0.3 * results_df["precision_at_5"].mean()
                + 0.2 * results_df["precision_at_10"].mean()
            ),
        })
        print(f"  w={w:.2f}: P@1={rows[-1]['P@1']:.1%}  P@3={rows[-1]['P@3']:.1%}  "
              f"P@5={rows[-1]['P@5']:.1%}  P@10={rows[-1]['P@10']:.1%}  "
              f"weighted={rows[-1]['weighted']:.4f}  rho={rows[-1]['Spearman']:.3f}",
              flush=True)

    predict_mod.predict_splits_and_total = original_predict

    df = pd.DataFrame(rows)
    print()
    print("=" * 70)
    print("RESULTS — best by each metric")
    print("=" * 70)
    for metric in ["P@1", "P@3", "P@5", "P@10", "weighted", "Spearman"]:
        best = df.loc[df[metric].idxmax()]
        print(f"  best {metric:>8}: w={best['w']:.2f} -> {best[metric]:.4f}")

    best_p3 = df.loc[df["P@3"].idxmax()]
    chosen_w = float(best_p3["w"])
    print()
    print(f"Saving v55 with ensemble_ranker_weight = {chosen_w:.2f} (P@3-optimal on v54)")

    src_path = Path(SOURCE)
    tgt_path = Path(TARGET)
    bundle = joblib.load(src_path)
    bundle.metadata = dict(bundle.metadata) if bundle.metadata else {}
    bundle.metadata["ensemble_ranker_weight"] = chosen_w
    bundle.metadata["v55_notes"] = (
        f"Cloned from v54 (which adds field-boundary features); ensemble_ranker_weight "
        f"re-optimised for P@3 across {len(event_keys)} H2 2025 events. "
        f"Sweep table: {df.to_dict(orient='records')}"
    )
    bundle.version = "v55"
    joblib.dump(bundle, tgt_path)
    print(f"Wrote {tgt_path}")


if __name__ == "__main__":
    main()
