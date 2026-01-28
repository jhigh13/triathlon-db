"""Analyze feature contributions to Vannerson's slow prediction."""
import joblib
import pandas as pd
import numpy as np

# Load model
bundle = joblib.load('models/bundle_elite_v5.joblib')
model = bundle.model_total
feature_cols = bundle.feature_columns

# Load prediction data
df = pd.read_csv('outputs/predictions_195251_676986.csv')
vanerson = df[df['athlete_full_name'].str.contains('Vannerson', case=False)].iloc[0]
lamprecht = df[df['athlete_full_name'].str.contains('Lamprecht', case=False)].iloc[0]

# Get feature values for both
v_features = vanerson[feature_cols].values.reshape(1, -1)
l_features = lamprecht[feature_cols].values.reshape(1, -1)

# Get predictions
baseline_v = model.predict(v_features)[0]
baseline_l = model.predict(l_features)[0]

print('Feature Contribution Analysis:')
print('(Shows how much replacing V value with L value changes prediction)')
print('=' * 75)
print(f"{'Feature':<25} {'V->L impact':>15} {'V value':>12} {'L value':>12}")
print('-' * 75)

# For each feature, swap Vannerson's value with Lamprecht's and see impact
contributions = []
for i, feat in enumerate(feature_cols):
    # Create modified feature vector
    modified = v_features.copy()
    modified[0, i] = l_features[0, i]  # Replace V's value with L's
    new_pred = model.predict(modified)[0]
    impact = new_pred - baseline_v  # Positive = V would be slower with L's value
    contributions.append((feat, impact, v_features[0, i], l_features[0, i]))

# Sort by absolute impact
contributions.sort(key=lambda x: abs(x[1]), reverse=True)

for feat, impact, v_val, l_val in contributions[:15]:
    if pd.isna(v_val): 
        v_val_str = 'NaN'
    else: 
        v_val_str = f'{v_val:.1f}'
    if pd.isna(l_val): 
        l_val_str = 'NaN'
    else: 
        l_val_str = f'{l_val:.1f}'
    sign = '+' if impact >= 0 else ''
    print(f'{feat:<25} {sign}{impact:>12.0f}s {v_val_str:>12} {l_val_str:>12}')

print()
print(f"Vannerson baseline prediction: {baseline_v:.0f}s ({baseline_v/60:.1f}m)")
print(f"Lamprecht baseline prediction: {baseline_l:.0f}s ({baseline_l/60:.1f}m)")
print(f"Difference: {baseline_v - baseline_l:.0f}s ({(baseline_v - baseline_l)/60:.1f}m)")
