"""Show feature importances for every component of a model bundle.

Works with the ensemble bundles (v40+) that contain an LGBMRanker, optional
XGBRanker, a percentile model, a total regressor, and per-distance split
models. Prints:
  1. Per-model ranked importance tables
  2. A combined table with normalised importance per feature across models
  3. The complete feature list from the bundle

Usage (PowerShell):
    .venv/Scripts/python.exe scripts/feature_importance.py
    .venv/Scripts/python.exe scripts/feature_importance.py --model models/bundle_elite_v45.joblib --top 40
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from tri_analysis.prediction.train import load_model_bundle


def _extract(model, feature_names):
    """Return (name, importance) pairs for a model, or None if unavailable."""
    if model is None:
        return None

    # Try sklearn Pipeline: unwrap final estimator
    est = model
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_importances_") or hasattr(step, "booster_"):
                est = step
                break

    importances = None
    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "booster_") and hasattr(est.booster_, "feature_importance"):
        importances = np.asarray(est.booster_.feature_importance(importance_type="gain"), dtype=float)

    if importances is None or importances.size == 0:
        return None

    if len(importances) != len(feature_names):
        # Model was trained on a different feature list (e.g., split-specific)
        # Truncate/pad to align. Usually the split models use split_feature_columns.
        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        feature_names = feature_names[:n]

    total = importances.sum()
    if total <= 0:
        return None
    pct = importances / total * 100.0
    return list(zip(feature_names, pct))


def _as_dataframe(pairs, col):
    if pairs is None:
        return None
    return (
        pd.DataFrame(pairs, columns=["feature", col])
        .sort_values(col, ascending=False)
        .reset_index(drop=True)
    )


def main():
    ap = argparse.ArgumentParser(description="Feature importance across all bundle components")
    ap.add_argument("--model", type=str, default="models/bundle_elite_v45.joblib")
    ap.add_argument("--top", type=int, default=30, help="Top N features to show per table")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path for combined table")
    args = ap.parse_args()

    bundle = load_model_bundle(args.model)
    feats_all = list(bundle.feature_columns)
    feats_split = list(bundle.split_feature_columns) if bundle.split_feature_columns else feats_all

    print("=" * 90)
    print(f"Bundle: {args.model}  (version: {bundle.version})")
    print(f"  {len(feats_all)} full features, {len(feats_split)} split features")
    print("=" * 90)

    # Models keyed by (display_name, model_object, feature_list)
    components: list[tuple[str, object, list[str]]] = [
        ("ranker_lgbm", bundle.model_ranker, feats_all),
        ("ranker_xgb", bundle.model_xgb_ranker, feats_all),
        ("pct_total", bundle.model_total_pct, feats_all),
        ("reg_total", bundle.model_total, feats_all),
        ("reg_swim", bundle.model_swim, feats_all),
        ("reg_bike", bundle.model_bike, feats_all),
        ("reg_run", bundle.model_run, feats_all),
    ]

    # Per-distance split models (dict of dicts)
    for dist, split_dict in (bundle.distance_split_models or {}).items():
        if not isinstance(split_dict, dict):
            continue
        for split_name, m in split_dict.items():
            components.append((f"{dist}_{split_name}", m, feats_split))

    frames: dict[str, pd.DataFrame] = {}
    for name, mdl, feats in components:
        pairs = _extract(mdl, feats)
        df = _as_dataframe(pairs, name)
        if df is not None:
            frames[name] = df

    if not frames:
        print("No models in the bundle expose feature importances.")
        return

    # --- Per-model tables (top N each) ---
    for name, df in frames.items():
        print(f"\n--- {name}  (top {args.top}) ---")
        top = df.head(args.top).copy()
        top[name] = top[name].map(lambda v: f"{v:5.2f}%")
        print(top.to_string(index=False))

    # --- Combined table ---
    combined = None
    for name, df in frames.items():
        if combined is None:
            combined = df.rename(columns={name: name})
        else:
            combined = combined.merge(df, on="feature", how="outer")
    combined = combined.fillna(0.0)

    model_cols = [c for c in combined.columns if c != "feature"]
    combined["mean_pct"] = combined[model_cols].mean(axis=1)
    combined["max_pct"] = combined[model_cols].max(axis=1)
    combined = combined.sort_values("mean_pct", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"COMBINED  (top {args.top} by mean %-importance across {len(model_cols)} models)")
    print("=" * 90)
    show_cols = ["feature", "mean_pct", "max_pct"] + model_cols
    top = combined.head(args.top).copy()
    for c in ["mean_pct", "max_pct"] + model_cols:
        top[c] = top[c].map(lambda v: f"{v:5.2f}")
    print(top[show_cols].to_string(index=False))

    # --- Full feature list ---
    print("\n" + "=" * 90)
    print(f"ALL {len(feats_all)} FEATURES (training column order)")
    print("=" * 90)
    for i, f in enumerate(feats_all, 1):
        print(f"  {i:3d}. {f}")

    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"\nSaved combined table to {args.out}")


if __name__ == "__main__":
    main()
