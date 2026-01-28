"""Quick analysis of Vannerson prediction issue."""
import pandas as pd

df = pd.read_csv('outputs/predictions_195251_676986.csv')

# Print key athletes with their EMA times and predictions
print('Athlete Time Analysis (Top 10 by prediction):')
print('=' * 75)
print(f'{"Athlete":<25} {"EMA Total":>12} {"Best Total":>12} {"Pred Total":>12}')
print('-' * 75)

for _, row in df.sort_values('pred_total_sec').head(10).iterrows():
    ema = row['ema_total_sec_5'] / 60  # convert to minutes
    best = row['best_total_sec_24m'] / 60 if not pd.isna(row['best_total_sec_24m']) else 0
    pred = row['pred_total_sec'] / 60
    name = row['athlete_full_name'][:25]
    print(f'{name:<25} {ema:>10.1f}m {best:>10.1f}m {pred:>10.1f}m')

print()
print('=' * 75)
print('Vannerson Analysis:')
print('=' * 75)
vanerson = df[df['athlete_full_name'].str.contains('Vannerson', case=False)].iloc[0]
print(f"  EMA Total:     {vanerson['ema_total_sec_5']/60:.1f} min ({vanerson['ema_total_sec_5']:.0f} sec)")
print(f"  Best Total:    {vanerson['best_total_sec_24m']/60:.1f} min ({vanerson['best_total_sec_24m']:.0f} sec)")
print(f"  Pred Total:    {vanerson['pred_total_sec']/60:.1f} min ({vanerson['pred_total_sec']:.0f} sec)")
gap = (vanerson['pred_total_sec'] - vanerson['ema_total_sec_5'])/60
print(f"  Gap:           {gap:.1f} min SLOWER than EMA")
print()
print('  Other features:')
print(f"    seed_total_rank: {vanerson['seed_total_rank']}")
print(f"    days_since_last: {vanerson['days_since_last_race']}")
print(f"    races_12m: {vanerson['races_12m']}")
print(f"    std_total_sec_24m: {vanerson['std_total_sec_24m']:.1f}")
print(f"    tier_delta: {vanerson['tier_delta']:.1f}")

print()
print('=' * 75)
print('Lamprecht Analysis (for comparison):')
print('=' * 75)
lamprecht = df[df['athlete_full_name'].str.contains('Lamprecht', case=False)].iloc[0]
print(f"  EMA Total:     {lamprecht['ema_total_sec_5']/60:.1f} min ({lamprecht['ema_total_sec_5']:.0f} sec)")
print(f"  Best Total:    {lamprecht['best_total_sec_24m']/60:.1f} min ({lamprecht['best_total_sec_24m']:.0f} sec)")
print(f"  Pred Total:    {lamprecht['pred_total_sec']/60:.1f} min ({lamprecht['pred_total_sec']:.0f} sec)")
gap2 = (lamprecht['pred_total_sec'] - lamprecht['ema_total_sec_5'])/60
print(f"  Gap:           {gap2:.1f} min SLOWER than EMA")
print()
print('  Other features:')
print(f"    seed_total_rank: {lamprecht['seed_total_rank']}")
print(f"    days_since_last: {lamprecht['days_since_last_race']}")
print(f"    races_12m: {lamprecht['races_12m']}")
print(f"    std_total_sec_24m: {lamprecht['std_total_sec_24m']:.1f}")
print(f"    tier_delta: {lamprecht['tier_delta']:.1f}")
