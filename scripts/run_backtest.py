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

from tri_analysis.prediction.evaluate import backtest_events, compute_start_num_baseline
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
    parser.add_argument(
        "--no_sim",
        action="store_true",
        help="Skip Monte Carlo simulation (faster, deterministic-only backtest)."
    )
    parser.add_argument(
        "--no_packs",
        action="store_true",
        help="Disable pack effects in MC simulation (pure individual noise). "
             "Useful for isolating whether pack logic helps or hurts accuracy."
    )
    parser.add_argument(
        "--swim_mode",
        type=str,
        default="legacy",
        choices=["legacy", "percentile"],
        help="Swim simulation mode: 'legacy' (add noise to predicted times) or "
             "'percentile' (sample from learned swim exit distributions). "
             "Requires model trained with swim_exit_pct target."
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

    run_sim = not args.no_sim
    use_packs = not args.no_packs
    sim_label = " + Monte Carlo simulation" if run_sim else ""
    if run_sim and not use_packs:
        sim_label += " (NO PACKS)"
    if run_sim and args.swim_mode == "percentile":
        sim_label += " (PERCENTILE SWIM)"
    print(f"\nUsing model: {model_label}{sim_label}\n")
    keys = list(test_events[['event_id', 'prog_id']].itertuples(index=False, name=None))
    results = backtest_events(engine, keys, model_path, run_simulation=run_sim, use_pack_effects=use_packs,
                              swim_mode=args.swim_mode)

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

    # ── Monte Carlo Simulation Results ──
    has_sim = run_sim and 'sim_precision_at_10' in results.columns
    if has_sim:
        sim_p3 = results.sim_precision_at_3.mean()
        sim_p5 = results.sim_precision_at_5.mean() if 'sim_precision_at_5' in results.columns else float('nan')
        sim_p10 = results.sim_precision_at_10.mean()
        sim_sp = results.sim_spearman_corr.mean()
        sim_mae = results.sim_mae_total_sec.mean()

        print("\n=== Monte Carlo Simulation Results ===")
        print(f"Precision@3:  {sim_p3:.2%}")
        print(f"Precision@5:  {sim_p5:.2%}")
        print(f"Precision@10: {sim_p10:.2%}")
        print(f"Spearman:     {sim_sp:.3f}")
        print(f"MAE total:    {sim_mae:.1f} sec")

        # Model vs Sim head-to-head
        print(f"\n--- Model vs Simulation ({len(results)} events) ---")
        header = f"{'Metric':<15} {'Determin.':>10} {'MC Sim':>10} {'Delta':>10} {'Winner':>10}"
        print(header)
        print("-" * len(header))

        for name, det_val, sim_val, higher_better in [
            ("P@3",      p3,      sim_p3,  True),
            ("P@5",      p5,      sim_p5,  True),
            ("P@10",     p10,     sim_p10, True),
            ("Spearman", spearman, sim_sp, True),
            ("MAE (s)",  mae,     sim_mae, False),
        ]:
            delta = sim_val - det_val
            winner = "Sim" if (delta > 0) == higher_better else "Determ." if delta != 0 else "Tie"
            if name in ("Spearman",):
                print(f"{name:<15} {det_val:>10.3f} {sim_val:>10.3f} {delta:>+10.3f} {winner:>10}")
            elif name == "MAE (s)":
                print(f"{name:<15} {det_val:>10.1f} {sim_val:>10.1f} {delta:>+10.1f} {winner:>10}")
            else:
                print(f"{name:<15} {det_val*100:>9.1f}% {sim_val*100:>9.1f}% {delta*100:>+9.1f}% {winner:>10}")

        # Sim by tier
        print("\n--- Simulation Results by Tier ---")
        for tier in sorted(results_with_tier['event_tier'].unique()):
            tier_results = results_with_tier[results_with_tier['event_tier'] == tier]
            tier_name = TIER_NAMES.get(tier, 'Unknown')
            if 'sim_precision_at_3' in tier_results.columns:
                print(f"  Tier {tier} ({tier_name:12s}): "
                      f"P@3={tier_results.sim_precision_at_3.mean():.1%}, "
                      f"P@10={tier_results.sim_precision_at_10.mean():.1%}, "
                      f"Spearman={tier_results.sim_spearman_corr.mean():.3f}, "
                      f"MAE={tier_results.sim_mae_total_sec.mean():.0f}s, n={len(tier_results)}")

    # ── Start Number (Seeding) Baseline ──
    print("\n=== Start Number Baseline (Bib = World Ranking Proxy) ===")
    sn_results = compute_start_num_baseline(engine, keys)

    if sn_results.empty:
        print("  No start_num data available for baseline comparison.")
    else:
        sn_p3 = sn_results.precision_at_3.mean()
        sn_p5 = sn_results.precision_at_5.mean() if 'precision_at_5' in sn_results.columns else float('nan')
        sn_p10 = sn_results.precision_at_10.mean()
        sn_sp = sn_results.spearman_corr.mean()

        print(f"Events with start_num data: {len(sn_results)}")
        print(f"Precision@3:  {sn_p3:.2%}")
        print(f"Precision@5:  {sn_p5:.2%}")
        print(f"Precision@10: {sn_p10:.2%}")
        print(f"Spearman:     {sn_sp:.3f}")

        # Side-by-side comparison
        # Filter model results to same events for fair comparison
        common_events = set(
            zip(sn_results["event_id"], sn_results["prog_id"])
        ) & set(
            zip(results["event_id"], results["prog_id"])
        )
        model_common = results[
            results.apply(lambda r: (r["event_id"], r["prog_id"]) in common_events, axis=1)
        ]

        if not model_common.empty:
            mc_p3 = model_common.precision_at_3.mean()
            mc_p5 = model_common.precision_at_5.mean() if 'precision_at_5' in model_common.columns else float('nan')
            mc_p10 = model_common.precision_at_10.mean()
            mc_sp = model_common.spearman_corr.mean()
            mc_mae = model_common.mae_total_sec.mean()

            # Check if simulation data is available for 3-way comparison
            has_sim_common = has_sim and 'sim_precision_at_3' in model_common.columns
            if has_sim_common:
                sc_sim_p3 = model_common.sim_precision_at_3.mean()
                sc_sim_p5 = model_common.sim_precision_at_5.mean() if 'sim_precision_at_5' in model_common.columns else float('nan')
                sc_sim_p10 = model_common.sim_precision_at_10.mean()
                sc_sim_sp = model_common.sim_spearman_corr.mean()

            n_common = len(common_events)
            if has_sim_common:
                print(f"\n--- 3-Way Comparison ({n_common} common events) ---")
                header = f"{'Metric':<15} {'Determ.':>10} {'MC Sim':>10} {'StartNum':>10} {'Best':>10}"
                print(header)
                print("-" * len(header))

                for name, det_val, sim_val, sn_val, higher_better in [
                    ("P@3",      mc_p3,  sc_sim_p3,  sn_p3,  True),
                    ("P@5",      mc_p5,  sc_sim_p5,  sn_p5,  True),
                    ("P@10",     mc_p10, sc_sim_p10, sn_p10, True),
                    ("Spearman", mc_sp,  sc_sim_sp,  sn_sp,  True),
                ]:
                    vals = {"Determ.": det_val, "MC Sim": sim_val, "StartNum": sn_val}
                    best = max(vals, key=vals.get) if higher_better else min(vals, key=vals.get)
                    if name == "Spearman":
                        print(f"{name:<15} {det_val:>10.3f} {sim_val:>10.3f} {sn_val:>10.3f} {best:>10}")
                    else:
                        print(f"{name:<15} {det_val*100:>9.1f}% {sim_val*100:>9.1f}% {sn_val*100:>9.1f}% {best:>10}")
            else:
                print(f"\n--- Head-to-Head ({n_common} common events) ---")
                header = f"{'Metric':<15} {'Model':>10} {'StartNum':>10} {'Delta':>10} {'Winner':>10}"
                print(header)
                print("-" * len(header))

                for name, model_val, sn_val, higher_better in [
                    ("P@3",      mc_p3,  sn_p3,  True),
                    ("P@5",      mc_p5,  sn_p5,  True),
                    ("P@10",     mc_p10, sn_p10, True),
                    ("Spearman", mc_sp,  sn_sp,  True),
                ]:
                    delta = model_val - sn_val
                    winner = "Model" if (delta > 0) == higher_better else "StartNum" if delta != 0 else "Tie"
                    if name == "Spearman":
                        print(f"{name:<15} {model_val:>10.3f} {sn_val:>10.3f} {delta:>+10.3f} {winner:>10}")
                    else:
                        print(f"{name:<15} {model_val*100:>9.1f}% {sn_val*100:>9.1f}% {delta*100:>+9.1f}% {winner:>10}")

        # By-tier breakdown for start_num baseline
        sn_with_tier = sn_results.merge(test_events[['event_id', 'prog_id', 'event_tier']],
                                         on=['event_id', 'prog_id'], how='left')
        if 'event_tier' in sn_with_tier.columns:
            print("\n--- Start Number Baseline by Tier ---")
            for tier in sorted(sn_with_tier['event_tier'].unique()):
                tier_sn = sn_with_tier[sn_with_tier['event_tier'] == tier]
                tier_name = TIER_NAMES.get(tier, 'Unknown')
                print(f"  Tier {tier} ({tier_name:12s}): P@3={tier_sn.precision_at_3.mean():.1%}, "
                      f"P@10={tier_sn.precision_at_10.mean():.1%}, "
                      f"Spearman={tier_sn.spearman_corr.mean():.3f}, n={len(tier_sn)}")

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
        if has_sim:
            mlflow.log_metrics({
                "backtest_sim_precision_at_3": sim_p3,
                "backtest_sim_precision_at_5": sim_p5,
                "backtest_sim_precision_at_10": sim_p10,
                "backtest_sim_spearman": sim_sp,
                "backtest_sim_mae_total_sec": sim_mae,
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
