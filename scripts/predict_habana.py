#!/usr/bin/env python
"""Quick simulation for 2026 Americas Triathlon Cup La Habana."""
import os
import sys

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sqlalchemy import text
from tri_analysis.database import get_engine
from tri_analysis.prediction.sql import ProgramKey
from tri_analysis.prediction.features import (
    compute_athlete_form_features,
    compute_pack_features,
    fill_missing_features,
    get_feature_columns,
)
from tri_analysis.prediction.sql import fetch_athlete_history, fetch_pack_history, ProgramKey
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.simulate import run_monte_carlo, format_simulation_output

pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 30)

# La Habana 2026 - Elite Men
EVENT_ID = 195251
PROG_ID = 676986
EVENT_DATE = pd.Timestamp("2026-02-21")
EVENT_NAME = "2026 Americas Triathlon Cup La Habana"
DISTANCE = "sprint"  # Assume sprint for Americas Cup

print(f"=" * 80)
print(f"Prediction for: {EVENT_NAME}")
print(f"Event: {EVENT_ID}, Program: {PROG_ID} (Elite Men)")
print(f"Date: {EVENT_DATE.date()}")
print(f"=" * 80)

engine = get_engine()

# Get athletes from program_entries
with engine.connect() as conn:
    start_list = pd.read_sql(
        text("""
        SELECT pe.athlete_id, a.full_name as athlete_name, 
               a.country
        FROM program_entries pe
        JOIN athlete a ON pe.athlete_id = a.athlete_id
        WHERE pe.event_id = :event_id AND pe.prog_id = :prog_id
          AND pe.is_active = TRUE
        ORDER BY pe.athlete_id
        """),
        conn,
        params={"event_id": EVENT_ID, "prog_id": PROG_ID}
    )

print(f"\nFound {len(start_list)} athletes in start list")

if start_list.empty:
    print("No athletes found!")
    sys.exit(1)

# Build features for each athlete
print("\nBuilding features...")
key = ProgramKey(event_id=EVENT_ID, prog_id=PROG_ID)
rows = []

for _, athlete in start_list.iterrows():
    athlete_id = athlete["athlete_id"]
    
    # Get history and compute form features
    history_df = fetch_athlete_history(
        engine, 
        athlete_id=athlete_id,
        before_date=EVENT_DATE.date(),
        distance_category=DISTANCE,
        elite_only=True
    )
    form = compute_athlete_form_features(history_df, EVENT_DATE.date())
    
    # Get pack history and compute pack features
    pack_df = fetch_pack_history(engine, athlete_id, EVENT_DATE.date())
    pack = compute_pack_features(pack_df, EVENT_DATE.date())
    
    # Americas Cup = Regional = Tier 3
    event_tier = 3
    athlete_avg_tier = form.get("athlete_avg_tier", 3.0)
    tier_delta = event_tier - athlete_avg_tier
    
    row = {
        "athlete_id": athlete_id,
        "athlete_name": athlete["athlete_name"],
        "country": athlete["country"],
        **form,
        **pack,
        "seed_total_rank": None,  # No seed data
        "n_entrants": len(start_list),
        "event_tier": event_tier,
        "tier_delta": tier_delta,
    }
    rows.append(row)

features_df = pd.DataFrame(rows)

# Load model
print("Loading model...")
bundle = load_model_bundle("models/bundle_elite_v5.joblib")
feature_cols = bundle.feature_columns or get_feature_columns()
features_df = fill_missing_features(features_df, feature_cols)

# Predict
print("Generating predictions...")
pred_df = predict_splits_and_total(features_df, bundle)

# Monte Carlo simulation
print("Running Monte Carlo simulation (10,000 iterations)...")
sim_df = run_monte_carlo(pred_df, n_sims=10000, random_state=42)
display_df = format_simulation_output(sim_df)

# Show results
print("\n" + "=" * 80)
print(f"PREDICTIONS: {EVENT_NAME} - Elite Men")
print("=" * 80)
print(display_df.head(30).to_string(index=False))

# Save to CSV
output_path = f"outputs/predictions_{EVENT_ID}_{PROG_ID}.csv"
sim_df.to_csv(output_path, index=False)
print(f"\nFull results saved to: {output_path}")
