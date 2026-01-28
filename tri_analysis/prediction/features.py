"""
Feature engineering for the prediction pipeline.

Computes athlete form features, pack metrics, and field-context features.
All features are computed from data BEFORE the target event_date to prevent leakage.

MVP feature set (v1):
- EWMA swim/bike/run/total over last 5 races
- std_total_sec_24m (stability)
- days_since_last_race
- front_pack_rate, avg_swim_gap_leader (draft-legal edge)
- seed_total_rank, n_entrants (field context)
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from .sql import (
    ProgramKey,
    fetch_athlete_history,
    fetch_pack_history,
    fetch_start_list,
    fetch_event_metadata,
)
from .utils_time import parse_time_to_seconds

logger = logging.getLogger(__name__)

# Event tier classification patterns (Tier 1 = highest prestige)
# Used for both feature engineering and sample weighting during training
EVENT_TIER_PATTERNS = {
    1: [  # WTCS / Championship Series
        "Championship Series", "WTCS", "Championship Finals", "Grand Final",
        "Olympic", "Olympics", "World Championship",
    ],
    2: [  # World Cup
        "World Triathlon Cup", "ITU Triathlon World Cup",
    ],
    3: [  # Regional / Continental Championships & Cups
        "PATCO", "ATU", "OTU", "ASTC", "ETU", "CAMTRI",
        "Continental Cup", "Continental Championships",
        "African Championships", "Asian Championships",
        "Oceania Championships", "Pan-American", "European Championships",
        "European Cup", "Europe Triathlon Cup",
        "Oceania Cup", "Panamerican Cup", "Pan-American Cup",
        "Asia Triathlon Cup", "Africa Triathlon Cup", "Africa Triathlon Premium Cup",
        "Americas Triathlon Cup", "Americas Cup",
    ],
    # Tier 4 = everything else (default)
}

# Sample weights for training by tier (higher = more important)
TIER_SAMPLE_WEIGHTS = {
    1: 4.0,  # WTCS events count 4x
    2: 2.0,  # World Cup events count 2x
    3: 1.5,  # Regional events count 1.5x
    4: 1.0,  # Other events count 1x
}


def classify_event_tier(event_name: str) -> int:
    """Classify an event into a tier based on its name.
    
    Returns:
        1 = WTCS/Championship (highest), 2 = World Cup, 3 = Regional, 4 = Other
    """
    if not event_name:
        return 4
    name_upper = event_name.upper()
    
    for tier in sorted(EVENT_TIER_PATTERNS.keys()):
        for pattern in EVENT_TIER_PATTERNS[tier]:
            if pattern.upper() in name_upper:
                return tier
    return 4

# Default uncertainty for athletes with sparse history (seconds)
DEFAULT_SIGMA_TOTAL = 120.0


def compute_athlete_form_features(history_df: pd.DataFrame, event_date: date) -> dict:
    """
    Compute pre-race athlete form features from prior race history.

    Args:
        history_df: DataFrame from fetch_athlete_history (already filtered to before event_date)
        event_date: Target event date (used for days_since_last_race)

    Returns:
        dict of feature name -> value

    Features computed:
        - ema_swim_sec_5, ema_bike_sec_5, ema_run_sec_5, ema_total_sec_5: EWMA over last 5
        - last_total_sec: Most recent total time
        - best_total_sec_24m: Best total in last 24 months
        - std_total_sec_24m: Std dev of total in last 24 months
        - days_since_last_race: Days from last race to event_date
        - races_12m: Number of races in last 12 months
        - dnf_rate_24m: DNF rate (we don't have DNFs in history_df since it filters to finishers)
        
    Athlete tier features (NEW - competition level indicators):
        - athlete_avg_tier: Average tier of races competed (1=WTCS, 2=WorldCup, 3=Regional, 4=Other)
        - athlete_t1_rate: Fraction of races at Tier 1 (WTCS/Championship)
        - athlete_t1t2_rate: Fraction of races at Tier 1 or 2 (high-level racing)
        - athlete_best_tier: Best (lowest) tier raced at in last 24 months
        - ema_finish_pct_tier1: EMA of finish percentile in Tier 1 races only (if available)
    """
    features = {
        "ema_swim_sec_5": None,
        "ema_bike_sec_5": None,
        "ema_run_sec_5": None,
        "ema_total_sec_5": None,
        "last_total_sec": None,
        "best_total_sec_24m": None,
        "std_total_sec_24m": None,
        "days_since_last_race": None,
        "races_12m": 0,
        "races_24m": 0,
        # Athlete competition level features
        "athlete_avg_tier": 4.0,  # Default to lowest tier
        "athlete_t1_rate": 0.0,
        "athlete_t1t2_rate": 0.0,
        "athlete_best_tier": 4,
        "ema_finish_pct_tier1": None,
    }

    if history_df.empty:
        logger.debug("Empty history_df, returning default features")
        return features

    # Parse time columns to seconds
    df = history_df.copy()
    for col in ["swimtime", "biketime", "runtime", "total_time"]:
        if col in df.columns:
            df[col + "_sec"] = df[col].apply(parse_time_to_seconds)

    # Sort by event_date descending (most recent first)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("event_date", ascending=False).reset_index(drop=True)

    # Filter to valid total times
    valid_df = df[df["total_time_sec"].notna()].copy()

    if valid_df.empty:
        return features

    # ---- Athlete Tier Features ----
    # Classify event tiers if event_name is available
    if "event_name" in valid_df.columns:
        valid_df["event_tier"] = valid_df["event_name"].apply(classify_event_tier)
        
        # Average tier (lower = races at higher level competitions)
        features["athlete_avg_tier"] = float(valid_df["event_tier"].mean())
        
        # Tier 1 rate (fraction of races at WTCS/Championship level)
        features["athlete_t1_rate"] = float((valid_df["event_tier"] == 1).mean())
        
        # Tier 1+2 rate (fraction at high-level: WTCS + World Cup)
        features["athlete_t1t2_rate"] = float((valid_df["event_tier"] <= 2).mean())
        
        # Best tier in last 24 months
        cutoff_24m = pd.Timestamp(event_date) - timedelta(days=730)
        df_24m_tier = valid_df[valid_df["event_date"] >= cutoff_24m]
        if not df_24m_tier.empty:
            features["athlete_best_tier"] = int(df_24m_tier["event_tier"].min())
        
        # EMA of finish percentile in Tier 1 races (WTCS performance)
        tier1_df = valid_df[valid_df["event_tier"] == 1].head(10)
        if len(tier1_df) >= 1 and "finish_position" in tier1_df.columns:
            # Approximate percentile: position / typical WTCS field size (~50)
            tier1_chrono = tier1_df.iloc[::-1].copy()
            tier1_chrono["finish_pct"] = tier1_chrono["finish_position"].astype(float) / 50.0
            tier1_chrono["finish_pct"] = tier1_chrono["finish_pct"].clip(0, 1)
            if not tier1_chrono["finish_pct"].isna().all():
                ema_t1 = tier1_chrono["finish_pct"].ewm(span=5, min_periods=1).mean().iloc[-1]
                features["ema_finish_pct_tier1"] = float(ema_t1)

    # Last race date and days since
    last_race_date = valid_df["event_date"].iloc[0]
    if pd.notna(last_race_date):
        days_since = (pd.Timestamp(event_date) - last_race_date).days
        features["days_since_last_race"] = max(0, days_since)

    # Last total
    features["last_total_sec"] = int(valid_df["total_time_sec"].iloc[0])

    # 24-month window
    cutoff_24m = pd.Timestamp(event_date) - timedelta(days=730)
    df_24m = valid_df[valid_df["event_date"] >= cutoff_24m]

    if not df_24m.empty:
        features["best_total_sec_24m"] = int(df_24m["total_time_sec"].min())
        features["std_total_sec_24m"] = float(df_24m["total_time_sec"].std())
        features["races_24m"] = len(df_24m)

    # 12-month window
    cutoff_12m = pd.Timestamp(event_date) - timedelta(days=365)
    df_12m = valid_df[valid_df["event_date"] >= cutoff_12m]
    features["races_12m"] = len(df_12m)

    # EWMA over last 5 races (span=5 gives alpha≈0.33)
    last_5 = valid_df.head(5)
    if len(last_5) >= 1:
        # Reverse to chronological order for EWMA calculation
        last_5_chrono = last_5.iloc[::-1]

        for split, col in [
            ("swim", "swimtime_sec"),
            ("bike", "biketime_sec"),
            ("run", "runtime_sec"),
            ("total", "total_time_sec"),
        ]:
            if col in last_5_chrono.columns:
                vals = last_5_chrono[col].dropna()
                if len(vals) >= 1:
                    # EWMA with span=5
                    ema_val = vals.ewm(span=5, min_periods=1).mean().iloc[-1]
                    features[f"ema_{split}_sec_5"] = float(ema_val)

    return features


def compute_pack_features(pack_df: pd.DataFrame, event_date: date) -> dict:
    """
    Compute draft-legal pack features from pack membership history.

    Args:
        pack_df: DataFrame from fetch_pack_history (filtered to before event_date)
        event_date: Target event date

    Returns:
        dict of feature name -> value

    Features computed (EMA span=7 races):
        - ema_swim_pos_pct_7: EMA of swim exit position percentile (0-1)
        - ema_bike_pos_pct_7: EMA of bike exit position percentile (0-1)
        - ema_swim_pack_7: EMA of pack number at swim exit (1=front)
        - ema_bike_pack_7: EMA of pack number at bike exit
        - ema_swim_gap_sec_7: EMA of gap to leader at swim (seconds)
        - ema_bike_gap_sec_7: EMA of gap to leader at bike (seconds)

    Legacy features (retained for backward compatibility):
        - front_pack_rate: Fraction of swim checkpoints where pack_id == 1
        - avg_swim_gap_leader: Mean gap_to_leader_sec at swim checkpoint
        - p90_swim_gap_leader: 90th percentile gap at swim
        - avg_pack_size_swim: Mean pack_size at swim
        - bike_pack_rate: Fraction of bike checkpoints in pack_id == 1
        - avg_bike_gap_leader: Mean gap_to_leader_sec at bike
    """
    features = {
        # New EMA-based features (span=7)
        "ema_swim_pos_pct_7": None,
        "ema_bike_pos_pct_7": None,
        "ema_swim_pack_7": None,
        "ema_bike_pack_7": None,
        "ema_swim_gap_sec_7": None,
        "ema_bike_gap_sec_7": None,
        # Legacy features
        "front_pack_rate": None,
        "avg_swim_gap_leader": None,
        "p90_swim_gap_leader": None,
        "avg_pack_size_swim": None,
        "bike_pack_rate": None,
        "avg_bike_gap_leader": None,
    }

    if pack_df.empty:
        return features

    # Sort by event_date descending to get most recent first
    df = pack_df.copy()
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("event_date", ascending=False)

    # Swim checkpoint features
    swim_df = df[df["checkpoint"] == "swim"].copy()
    if not swim_df.empty:
        # Legacy features
        features["front_pack_rate"] = float((swim_df["pack_id"] == 1).mean())
        features["avg_swim_gap_leader"] = float(swim_df["gap_to_leader_sec"].mean())
        features["p90_swim_gap_leader"] = float(swim_df["gap_to_leader_sec"].quantile(0.9))
        features["avg_pack_size_swim"] = float(swim_df["pack_size"].mean())

        # EMA features (last 7 races, chronological order for EWMA)
        swim_last7 = swim_df.head(7).iloc[::-1]  # Reverse to chronological
        
        # Position percentile (pos / field_size approximated by max pos in that race)
        # We'll compute per-race position percentile from pos_at_checkpoint
        swim_last7 = swim_last7.copy()
        if "pos_at_checkpoint" in swim_last7.columns:
            # Group by race (event_id, prog_id) to get field size per race
            swim_pct = []
            for (eid, pid), grp in swim_df.groupby(["event_id", "prog_id"]):
                # Get this athlete's row for this race
                race_row = grp.iloc[0] if len(grp) > 0 else None
                if race_row is not None and pd.notna(race_row["pos_at_checkpoint"]):
                    # Query the full field size from pack_df for this race
                    race_field = pack_df[
                        (pack_df["event_id"] == eid) & 
                        (pack_df["prog_id"] == pid) & 
                        (pack_df["checkpoint"] == "swim")
                    ]
                    # Field size is just this one athlete's position data
                    # Use pack_size as proxy or pos_at_checkpoint relative
                    # Better: use pos_at_checkpoint / total athletes in race
                    # Since we only have this athlete, use pack_size as estimate
                    max_pack = race_row.get("pack_size", 1)
                    pack_id = race_row["pack_id"]
                    # Position percentile: pos / estimated_field
                    # Estimate field as pack_id * avg_pack_size
                    pos = race_row["pos_at_checkpoint"]
                    # Simple: just use position directly, normalize later
                    swim_pct.append({
                        "event_date": race_row["event_date"],
                        "pos": pos,
                        "pack_id": pack_id,
                        "gap_to_leader": race_row["gap_to_leader_sec"]
                    })
            
            if swim_pct:
                swim_pct_df = pd.DataFrame(swim_pct).sort_values("event_date", ascending=True)
                swim_last7_df = swim_pct_df.tail(7)
                
                # EMA of position (lower = better, will normalize at model level)
                if len(swim_last7_df) >= 1:
                    # Compute position percentile assuming avg field of ~50
                    swim_last7_df = swim_last7_df.copy()
                    swim_last7_df["pos_pct"] = swim_last7_df["pos"] / 50.0  # Normalize to ~0-1
                    swim_last7_df["pos_pct"] = swim_last7_df["pos_pct"].clip(0, 1)
                    
                    features["ema_swim_pos_pct_7"] = float(
                        swim_last7_df["pos_pct"].ewm(span=7, min_periods=1).mean().iloc[-1]
                    )
                    features["ema_swim_pack_7"] = float(
                        swim_last7_df["pack_id"].ewm(span=7, min_periods=1).mean().iloc[-1]
                    )
                    features["ema_swim_gap_sec_7"] = float(
                        swim_last7_df["gap_to_leader"].ewm(span=7, min_periods=1).mean().iloc[-1]
                    )

    # Bike checkpoint features
    bike_df = df[df["checkpoint"] == "bike"].copy()
    if not bike_df.empty:
        # Legacy features
        features["bike_pack_rate"] = float((bike_df["pack_id"] == 1).mean())
        features["avg_bike_gap_leader"] = float(bike_df["gap_to_leader_sec"].mean())

        # EMA features for bike
        if "pos_at_checkpoint" in bike_df.columns:
            bike_pct = []
            for (eid, pid), grp in bike_df.groupby(["event_id", "prog_id"]):
                race_row = grp.iloc[0] if len(grp) > 0 else None
                if race_row is not None and pd.notna(race_row["pos_at_checkpoint"]):
                    bike_pct.append({
                        "event_date": race_row["event_date"],
                        "pos": race_row["pos_at_checkpoint"],
                        "pack_id": race_row["pack_id"],
                        "gap_to_leader": race_row["gap_to_leader_sec"]
                    })
            
            if bike_pct:
                bike_pct_df = pd.DataFrame(bike_pct).sort_values("event_date", ascending=True)
                bike_last7_df = bike_pct_df.tail(7)
                
                if len(bike_last7_df) >= 1:
                    bike_last7_df = bike_last7_df.copy()
                    bike_last7_df["pos_pct"] = bike_last7_df["pos"] / 50.0
                    bike_last7_df["pos_pct"] = bike_last7_df["pos_pct"].clip(0, 1)
                    
                    features["ema_bike_pos_pct_7"] = float(
                        bike_last7_df["pos_pct"].ewm(span=7, min_periods=1).mean().iloc[-1]
                    )
                    features["ema_bike_pack_7"] = float(
                        bike_last7_df["pack_id"].ewm(span=7, min_periods=1).mean().iloc[-1]
                    )
                    features["ema_bike_gap_sec_7"] = float(
                        bike_last7_df["gap_to_leader"].ewm(span=7, min_periods=1).mean().iloc[-1]
                    )

    return features


def compute_field_context_features(
    athlete_features_df: pd.DataFrame,
    athlete_id: int
) -> dict:
    """
    Compute field-context features for an athlete within a start list.

    Args:
        athlete_features_df: DataFrame with all athletes and their ema_total_sec_5
        athlete_id: The athlete to compute context for

    Returns:
        dict of feature name -> value

    Features computed:
        - seed_total_rank: Rank of athlete's ema_total_sec_5 among entrants (1=best)
        - seed_total_gap_to_best: Gap to the best seed in field
        - field_depth_top10_mean: Mean of top 10 seeds
        - n_entrants: Field size
    """
    features = {
        "seed_total_rank": None,
        "seed_total_gap_to_best": None,
        "field_depth_top10_mean": None,
        "n_entrants": len(athlete_features_df),
    }

    if athlete_features_df.empty:
        return features

    # Get seeds (lower EMA = better)
    seeds = athlete_features_df[["athlete_id", "ema_total_sec_5"]].copy()
    seeds = seeds.dropna(subset=["ema_total_sec_5"])

    if seeds.empty:
        return features

    # Rank by EMA (ascending = lower time is better)
    seeds["seed_rank"] = seeds["ema_total_sec_5"].rank(method="min", ascending=True)

    best_seed = seeds["ema_total_sec_5"].min()
    features["field_depth_top10_mean"] = float(seeds.nsmallest(10, "ema_total_sec_5")["ema_total_sec_5"].mean())

    # Get this athlete's rank and gap
    athlete_row = seeds[seeds["athlete_id"] == athlete_id]
    if not athlete_row.empty:
        features["seed_total_rank"] = int(athlete_row["seed_rank"].iloc[0])
        athlete_ema = athlete_row["ema_total_sec_5"].iloc[0]
        features["seed_total_gap_to_best"] = float(athlete_ema - best_seed)

    return features


def build_features_for_program(
    engine: Engine,
    key: ProgramKey,
    use_start_list: bool = True,
    match_distance: bool = True,
    elite_only: bool = True
) -> pd.DataFrame:
    """
    Build feature matrix for all athletes in an upcoming program.

    For each athlete, computes form features and pack features from history
    BEFORE the event_date, then adds field-context features.

    Args:
        engine: SQLAlchemy Engine
        key: ProgramKey (event_id, prog_id)
        use_start_list: If True, use program_entries as entrants;
                        if False, use race_results (for training/backtesting)
        match_distance: If True, filter athlete history to same distance category
                        (e.g., only use sprint history for sprint race predictions)
        elite_only: If True, only use Elite race history (excludes Junior, U23, Para)

    Returns:
        DataFrame with one row per athlete, columns for all features plus identifiers

    Raises:
        ValueError: If no event metadata found for the key
    """
    # Get event metadata
    event_meta = fetch_event_metadata(engine, key)
    if event_meta is None:
        raise ValueError(f"No event metadata found for {key}")

    event_date = event_meta["event_date"]
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date).date()
    elif hasattr(event_date, "date"):
        event_date = event_date.date() if callable(getattr(event_date, "date")) else event_date

    # Get distance category for filtering
    distance_category = event_meta.get("prog_distance_category") if match_distance else None
    
    logger.info(f"Building features for {key}, event_date={event_date}, distance={distance_category}, elite_only={elite_only}")

    # Get athlete list
    if use_start_list:
        athletes_df = fetch_start_list(engine, key)
        if athletes_df.empty:
            logger.warning(f"No start list found for {key}")
            return pd.DataFrame()
    else:
        # For training: use actual results
        from .sql import fetch_program_results
        results_df = fetch_program_results(engine, key)
        athletes_df = results_df[["athlete_id", "athlete_full_name"]].drop_duplicates()
        if "athlete_country_name" not in athletes_df.columns:
            athletes_df["athlete_country_name"] = None

    # Compute features for each athlete
    athlete_features_list = []

    for _, row in athletes_df.iterrows():
        athlete_id = row["athlete_id"]
        if pd.isna(athlete_id):
            continue

        athlete_id = int(athlete_id)

        # Fetch history (before event_date to prevent leakage)
        # Filter by distance category and elite status for more accurate comparisons
        history_df = fetch_athlete_history(
            engine, 
            athlete_id, 
            event_date, 
            limit=50,
            distance_category=distance_category,
            elite_only=elite_only
        )
        
        # If no history with matching distance, fall back to all history
        if history_df.empty and distance_category:
            history_df = fetch_athlete_history(
                engine, athlete_id, event_date, limit=50, elite_only=elite_only
            )
        
        # Fetch pack history, filtering by distance to avoid mixing Olympic and Sprint gaps
        pack_df = fetch_pack_history(
            engine, athlete_id, event_date, limit=100, distance_category=distance_category
        )
        
        # If no pack data with matching distance, fall back to all pack history
        if pack_df.empty and distance_category:
            pack_df = fetch_pack_history(engine, athlete_id, event_date, limit=100)

        # Compute features
        form_feats = compute_athlete_form_features(history_df, event_date)
        pack_feats = compute_pack_features(pack_df, event_date)

        # Combine into one row
        athlete_row = {
            "event_id": key.event_id,
            "prog_id": key.prog_id,
            "athlete_id": athlete_id,
            "athlete_full_name": row.get("athlete_full_name"),
            "athlete_country_name": row.get("athlete_country_name"),
            "event_date": event_date,
            "prog_name": event_meta.get("prog_name"),
            "prog_distance_category": event_meta.get("prog_distance_category"),
            "event_country": event_meta.get("event_country"),
            "event_name": event_meta.get("event_name"),
            "event_tier": classify_event_tier(event_meta.get("event_name", "")),
            "wetsuit": event_meta.get("wetsuit"),
            **form_feats,
            **pack_feats,
        }
        athlete_features_list.append(athlete_row)

    if not athlete_features_list:
        logger.warning(f"No athlete features computed for {key}")
        return pd.DataFrame()

    features_df = pd.DataFrame(athlete_features_list)

    # Compute tier_delta: event_tier - athlete_avg_tier
    # Positive value means athlete is "stepping down" to a lower-tier race (advantage)
    # e.g., WTCS athlete (avg_tier ~1.5) at World Cup (tier 2) → tier_delta ≈ +0.5
    if "event_tier" in features_df.columns and "athlete_avg_tier" in features_df.columns:
        features_df["tier_delta"] = features_df["event_tier"] - features_df["athlete_avg_tier"]
    else:
        features_df["tier_delta"] = 0.0

    # Add field-context features
    field_context_rows = []
    for _, row in features_df.iterrows():
        ctx = compute_field_context_features(features_df, row["athlete_id"])
        field_context_rows.append(ctx)

    field_ctx_df = pd.DataFrame(field_context_rows)
    features_df = pd.concat([features_df.reset_index(drop=True), field_ctx_df], axis=1)

    logger.info(f"Built features for {len(features_df)} athletes in {key}")
    return features_df


def get_feature_columns() -> list[str]:
    """
    Return the list of feature column names used for modeling (MVP set).

    These are the columns that should be passed to the model for training/prediction.
    """
    return [
        # Athlete form features
        "ema_swim_sec_5",
        "ema_bike_sec_5",
        "ema_run_sec_5",
        "ema_total_sec_5",
        "std_total_sec_24m",
        "days_since_last_race",
        "races_12m",
        # Athlete competition level features (NEW)
        "athlete_avg_tier",
        "athlete_t1_rate",
        "athlete_t1t2_rate",
        "athlete_best_tier",
        "tier_delta",  # event_tier - athlete_avg_tier (positive = athlete stepping down)
        # Pack features - EMA based (span=7)
        "ema_swim_pos_pct_7",
        "ema_bike_pos_pct_7",
        "ema_swim_pack_7",
        "ema_bike_pack_7",
        "ema_swim_gap_sec_7",
        "ema_bike_gap_sec_7",
        # Legacy pack features
        "front_pack_rate",
        "avg_swim_gap_leader",
        # Field context
        "seed_total_rank",
        "n_entrants",
        # Event context
        "event_tier",
    ]


def fill_missing_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Fill missing feature values with sensible defaults.

    Args:
        df: Feature DataFrame
        feature_cols: List of feature columns to check

    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()

    # Default values for missing features
    defaults = {
        "ema_swim_sec_5": df["ema_swim_sec_5"].median() if "ema_swim_sec_5" in df.columns else 1200,
        "ema_bike_sec_5": df["ema_bike_sec_5"].median() if "ema_bike_sec_5" in df.columns else 3600,
        "ema_run_sec_5": df["ema_run_sec_5"].median() if "ema_run_sec_5" in df.columns else 1800,
        "ema_total_sec_5": df["ema_total_sec_5"].median() if "ema_total_sec_5" in df.columns else 6600,
        "std_total_sec_24m": DEFAULT_SIGMA_TOTAL,
        "days_since_last_race": 60,  # Assume moderate layoff
        "races_12m": 0,
        # Athlete competition level features
        "athlete_avg_tier": 4.0,  # Default to lower-tier
        "athlete_t1_rate": 0.0,
        "athlete_t1t2_rate": 0.0,
        "athlete_best_tier": 4,
        "tier_delta": 0.0,  # Neutral (racing at typical level)
        # New EMA pack features (span=7)
        "ema_swim_pos_pct_7": 0.5,  # Mid-pack default
        "ema_bike_pos_pct_7": 0.5,
        "ema_swim_pack_7": 3.0,  # Pack 3 = mid-field
        "ema_bike_pack_7": 3.0,
        "ema_swim_gap_sec_7": 30.0,  # 30 sec default gap
        "ema_bike_gap_sec_7": 60.0,  # 60 sec default gap at bike
        # Legacy pack features
        "front_pack_rate": 0.5,  # Neutral assumption
        "avg_swim_gap_leader": 30,  # 30 sec gap default
        "seed_total_rank": df["seed_total_rank"].max() if "seed_total_rank" in df.columns else 50,
        "n_entrants": df["n_entrants"].median() if "n_entrants" in df.columns else 50,
        # Event context
        "event_tier": 2,  # Default to World Cup tier (mid-level)
    }

    for col in feature_cols:
        if col in df.columns:
            default_val = defaults.get(col, 0)
            df[col] = df[col].fillna(default_val)
        else:
            df[col] = defaults.get(col, 0)

    return df
