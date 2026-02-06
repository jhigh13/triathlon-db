"""
Run backtest on 2025 events to evaluate model performance.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running this file directly (e.g. `python .\scripts\run_backtest.py`)
# while still importing from the repo root package folders like `tri_analysis/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

import mlflow
import joblib

from tri_analysis.prediction.evaluate import backtest_events
from tri_analysis.database import get_engine
import pandas as pd
from sqlalchemy import text

def main():
    parser = argparse.ArgumentParser(description="Run backtest on H2 2025 events")
    parser.add_argument(
        "--model",
        type=str,
        default="models/bundle_elite_v5.joblib",
        help="Path to the model file to use (default: models/bundle_elite_v5.joblib)"
    )
    parser.add_argument(
        "--model_men",
        type=str,
        default=None,
        help="Path to men-specific model. Use with --model_women for gender-specific backtesting."
    )
    parser.add_argument(
        "--model_women",
        type=str,
        default=None,
        help="Path to women-specific model. Use with --model_men for gender-specific backtesting."
    )
    args = parser.parse_args()

    # Determine model path(s)
    if args.model_men and args.model_women:
        model_path = {"men": args.model_men, "women": args.model_women}
        model_label = f"men={args.model_men}, women={args.model_women}"
    else:
        model_path = args.model
        model_label = model_path
    engine = get_engine()

    # Get ALL test events from H2 2025 (no per-tier cap)
    # Training should be done through 2025-06-30 to avoid data leakage
    # Tier key: 1a=WTCS/Finals, 1b=T100, 2=World Cup, 3=Continental, 4=Other
    query = text("""
    SELECT DISTINCT
        e.event_id,
        e.prog_id,
        e.event_date,
        e.event_name,
        e.prog_name,
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
    WHERE e.event_date >= '2025-07-01'
    AND e.event_date <= '2025-12-31'
    AND e.prog_name ~* '(Elite Men|Elite Women)'
    AND COALESCE(e.prog_distance_category, '') != 'long_distance'
    GROUP BY e.event_id, e.prog_id, e.event_date, e.event_name, e.prog_name, e.prog_distance_category
    HAVING COUNT(*) >= 10
    ORDER BY event_tier, event_date, e.prog_name
    """)
    with engine.connect() as conn:
        test_events = pd.read_sql(query, conn)
    
    if test_events.empty:
        print("No test events found in H2 2025. Check your database.")
        return
    
    TIER_NAMES = {
        '1a': 'WTCS/Finals',
        '1b': 'T100',
        '2':  'World Cup',
        '3':  'Continental',
        '4':  'Other',
    }

    tier_counts = test_events['event_tier'].value_counts().sort_index()
    men_count = (test_events['prog_name'] == 'Elite Men').sum()
    women_count = (test_events['prog_name'] == 'Elite Women').sum()

    print(f"\n=== Test Set: H2 2025 (all qualifying events) ===")
    print(f"Total programs: {len(test_events)}")
    print(f"  Men programs:    {men_count}")
    print(f"  Women programs:  {women_count}")
    for tier_key in sorted(TIER_NAMES.keys()):
        count = tier_counts.get(tier_key, 0)
        print(f"  Tier {tier_key} ({TIER_NAMES[tier_key]:12s}): {count}")

    print("\n--- Test Events by Tier ---")
    for tier in sorted(test_events['event_tier'].unique()):
        tier_name = TIER_NAMES.get(tier, 'Unknown')
        tier_events = test_events[test_events['event_tier'] == tier].sort_values(['event_date', 'prog_name'])
        print(f"\nTier {tier} ({tier_name}):")
        for _, row in tier_events.iterrows():
            gender_tag = "M" if "Men" in str(row.get('prog_name', '')) else "W"
            dist = row.get('prog_distance_category', '') or ''
            print(f"  {row['event_date']} [{gender_tag}] [{dist:15s}] {row['event_name']}")
    
    print("\nNOTE: Model should be trained through 2025-06-30 to avoid data leakage.")
    print("Command: python scripts/train_models.py --start_date 2018-01-01 --end_date 2025-06-30 --output models/bundle_elite_v6.joblib\n")

    print(f"\nUsing model: {model_label}\n")
    keys = list(test_events[['event_id', 'prog_id']].itertuples(index=False, name=None))
    results = backtest_events(engine, keys, model_path)

    # Compute aggregate metrics
    p3 = results.precision_at_3.mean()
    p5 = results.precision_at_5.mean() if 'precision_at_5' in results.columns else float('nan')
    p10 = results.precision_at_10.mean()
    spearman = results.spearman_corr.mean()
    mae = results.mae_total_sec.mean()

    print("\n=== Backtest Results ===")
    print(f"Events evaluated: {len(results)}")
    print(f"Precision@3:  {p3:.2%}")
    print(f"Precision@5:  {p5:.2%}")
    print(f"Precision@10: {p10:.2%}")
    print(f"Spearman:     {spearman:.3f}")
    print(f"MAE total:    {mae:.1f} sec")

    # Show results by tier
    print("\n--- Results by Tier ---")
    results_with_tier = results.merge(test_events[['event_id', 'prog_id', 'event_tier']],
                                      on=['event_id', 'prog_id'], how='left')
    tier_metrics = {}
    for tier in sorted(results_with_tier['event_tier'].unique()):
        tier_results = results_with_tier[results_with_tier['event_tier'] == tier]
        tier_name = TIER_NAMES.get(tier, 'Unknown')
        t_p3 = tier_results.precision_at_3.mean()
        t_p5 = tier_results.precision_at_5.mean() if 'precision_at_5' in tier_results.columns else float('nan')
        t_p10 = tier_results.precision_at_10.mean()
        t_sp = tier_results.spearman_corr.mean()
        t_mae = tier_results.mae_total_sec.mean()
        tier_metrics[tier] = {
            "p3": t_p3, "p5": t_p5, "p10": t_p10,
            "spearman": t_sp, "mae": t_mae, "n": len(tier_results),
        }
        print(f"  Tier {tier} ({tier_name:12s}): P@3={t_p3:.1%}, P@5={t_p5:.1%}, P@10={t_p10:.1%}, "
              f"Spearman={t_sp:.3f}, MAE={t_mae:.0f}s, n={len(tier_results)}")

    # ── Log backtest metrics to MLflow ──
    mlflow.set_tracking_uri(f"file:///{str(REPO_ROOT / 'mlruns').replace(os.sep, '/')}")
    mlflow.set_experiment("triathlon-prediction")

    # Try to link to the training run via bundle metadata
    parent_run_id = None
    primary_model = args.model_men or args.model
    try:
        bundle = joblib.load(primary_model)
        parent_run_id = getattr(bundle, "metadata", {}).get("mlflow_run_id")
    except Exception:
        pass

    run_name = f"backtest_{os.path.splitext(os.path.basename(primary_model))[0]}"
    with mlflow.start_run(run_name=run_name):
        if parent_run_id:
            mlflow.set_tag("training_run_id", parent_run_id)
        mlflow.set_tag("model_path", model_label)
        mlflow.set_tag("stage", "backtest")

        mlflow.log_metrics({
            "backtest_precision_at_3": p3,
            "backtest_precision_at_5": p5,
            "backtest_precision_at_10": p10,
            "backtest_spearman": spearman,
            "backtest_mae_total_sec": mae,
            "backtest_n_events": len(results),
        })
        for tier, tm in tier_metrics.items():
            mlflow.log_metrics({
                f"backtest_tier{tier}_p3": tm["p3"],
                f"backtest_tier{tier}_p5": tm["p5"],
                f"backtest_tier{tier}_p10": tm["p10"],
                f"backtest_tier{tier}_spearman": tm["spearman"],
                f"backtest_tier{tier}_mae": tm["mae"],
                f"backtest_tier{tier}_n": tm["n"],
            })
    print(f"\nMLflow: backtest metrics logged to experiment 'triathlon-prediction'")

if __name__ == "__main__":
    main()
