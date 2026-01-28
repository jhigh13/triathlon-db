"""
Run backtest on 2025 events to evaluate model performance.
"""
from tri_analysis.prediction.evaluate import backtest_events
from tri_analysis.database import get_engine
import pandas as pd
from sqlalchemy import text

def main():
    engine = get_engine()

    # Get test events from H2 2025 - stratified by tier for balanced evaluation
    # Training should be done through 2025-06-30 to avoid data leakage
    query = text("""
    WITH event_tiers AS (
        SELECT DISTINCT 
            e.event_id, 
            e.prog_id, 
            e.event_date, 
            e.event_name,
            CASE
                WHEN e.event_name ~* '(WTCS|Championship Series|World Championship Finals)' THEN 1
                WHEN e.event_name ~* '(World Triathlon Cup|ITU Triathlon World Cup)' THEN 2
                WHEN e.event_name ~* '(Continental Cup|Continental Championships|Americas Triathlon Cup|Americas Cup|European Cup|Europe Triathlon Cup|Asian Cup|Asia Triathlon Cup|African Cup|Africa Triathlon Cup|Africa Triathlon Premium Cup|Oceania Cup|Pan-American Cup|Panamerican Cup)' THEN 3
                ELSE 4
            END AS event_tier,
            ROW_NUMBER() OVER (PARTITION BY CASE
                WHEN e.event_name ~* '(WTCS|Championship Series|World Championship Finals)' THEN 1
                WHEN e.event_name ~* 'World Cup' THEN 2
                WHEN e.event_name ~* '(Continental Cup|Americas Cup|European Cup|Asian Cup|African Cup|Oceania Cup)' THEN 3
                ELSE 4
            END ORDER BY e.event_date) AS tier_rank
        FROM events e
        JOIN race_results rr ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        WHERE e.event_date >= '2025-07-01' 
        AND e.event_date <= '2025-12-31'
        AND e.prog_name ~* '(Elite Men|Elite Women)'
        GROUP BY e.event_id, e.prog_id, e.event_date, e.event_name
        HAVING COUNT(*) >= 10
    )
    SELECT event_id, prog_id, event_date, event_name, event_tier
    FROM event_tiers
    WHERE tier_rank <= 8
    ORDER BY event_tier, event_date
    """)
    with engine.connect() as conn:
        test_events = pd.read_sql(query, conn)
    
    if test_events.empty:
        print("No test events found in H2 2025. Check your database.")
        return
    
    tier_counts = test_events['event_tier'].value_counts().sort_index()
    
    print(f"\n=== Test Set: H2 2025 (stratified by tier) ===")
    print(f"Total programs: {len(test_events)}")
    print(f"  Tier 1 (WTCS/Champ): {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (World Cup):  {tier_counts.get(2, 0)}")
    print(f"  Tier 3 (Cont Cup):   {tier_counts.get(3, 0)}")
    print(f"  Tier 4 (Other):      {tier_counts.get(4, 0)}")
    
    print("\n--- Test Events by Tier ---")
    for tier in sorted(test_events['event_tier'].unique()):
        tier_name = {1: 'WTCS', 2: 'World Cup', 3: 'Continental', 4: 'Other'}[tier]
        tier_events = test_events[test_events['event_tier'] == tier].sort_values('event_date')
        print(f"\nTier {tier} ({tier_name}):")
        for _, row in tier_events.iterrows():
            print(f"  {row['event_date']} - {row['event_name']}")
    
    print("\nNOTE: Model should be trained through 2025-06-30 to avoid data leakage.")
    print("Command: python scripts/train_models.py --start_date 2018-01-01 --end_date 2025-06-30 --output models/bundle_elite_v6.joblib\n")

    keys = list(test_events[['event_id', 'prog_id']].itertuples(index=False, name=None))
    results = backtest_events(engine, keys, 'models/bundle_elite_v5.joblib')

    print("\n=== Backtest Results (Model v5 with Distance-Filtered Pack + Anchoring) ===")
    print(f"Events evaluated: {len(results)}")
    print(f"Precision@3:  {results.precision_at_3.mean():.2%}")
    print(f"Precision@10: {results.precision_at_10.mean():.2%}")
    print(f"Spearman:     {results.spearman_corr.mean():.3f}")
    print(f"MAE total:    {results.mae_total_sec.mean():.1f} sec")

    # Show results by tier
    print("\n--- Results by Tier ---")
    results_with_tier = results.merge(test_events[['event_id', 'prog_id', 'event_tier']], 
                                      on=['event_id', 'prog_id'], how='left')
    for tier in sorted(results_with_tier['event_tier'].unique()):
        tier_results = results_with_tier[results_with_tier['event_tier'] == tier]
        tier_name = {1: 'WTCS', 2: 'World Cup', 3: 'Continental', 4: 'Other'}[tier]
        print(f"  Tier {tier} ({tier_name:11}): P@10={tier_results.precision_at_10.mean():.1%}, "
              f"Spearman={tier_results.spearman_corr.mean():.3f}, n={len(tier_results)}")
    
    print("\n--- Comparison to Previous Models ---")
    print("Baseline (no pack):      Precision@10 ~22%, Spearman ~0.40")
    print("Model v4 (pack+tier):    Precision@10 ~47.9%, Spearman ~0.11 (H2 2024, Tier 1 only)")
    p10_new = results.precision_at_10.mean()
    sp_new = results.spearman_corr.mean()
    print(f"Current model:           Precision@10 {p10_new:.1%}, Spearman {sp_new:.3f} (H2 2025, stratified)")

if __name__ == "__main__":
    main()
