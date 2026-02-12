"""
Feature engineering for the prediction pipeline.

Computes athlete form features, pack metrics, and field-context features.
All features are computed from data BEFORE the target event_date to prevent leakage.

Feature categories:
- Absolute form: EWMA swim/bike/run/total over last 5 races, std, days since last race
- Distance-agnostic: finish percentile EMA, gap-to-median ratio, per-split percentiles
- Cross-distance: per-distance finish percentile EMAs (sprint, standard)
- Competition level: athlete tier stats, tier delta
- Pack dynamics: EMA pack position, gap to leader
- Field context: seed rank (percentile-based), field depth, n_entrants
- Event context: event tier, distance category
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
    fetch_elo_ratings,
    fetch_wt_ranking_points,
    fetch_precomputed_race_metrics,
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

# Plausible split time ranges per distance category (seconds).
# Used to filter out races from matched_history that have split times
# inconsistent with the target distance (e.g., super-sprint results
# mislabeled as sprint). Tighter than SPLIT_OUTLIER_THRESHOLDS in train.py
# because these validate individual race times used for EMA computation.
EMA_SPLIT_PLAUSIBLE_RANGES = {
    "super-sprint": {
        "swimtime_sec": (50, 600),       # ~300m: 0:50 - 10:00
        "t1time_sec":   (5, 180),        # 0:05 - 3:00
        "biketime_sec": (300, 1500),     # ~8km: 5:00 - 25:00
        "t2time_sec":   (5, 120),        # 0:05 - 2:00
        "runtime_sec":  (300, 1200),     # ~2km: 5:00 - 20:00
    },
    "super_sprint": {
        "swimtime_sec": (50, 600),
        "t1time_sec":   (5, 180),
        "biketime_sec": (300, 1500),
        "t2time_sec":   (5, 120),
        "runtime_sec":  (300, 1200),
    },
    "sprint": {
        "swimtime_sec": (400, 1200),     # 750m: 6:40 - 20:00
        "t1time_sec":   (5, 180),        # 0:05 - 3:00
        "biketime_sec": (1100, 2800),    # 20km: 18:20 - 46:40
        "t2time_sec":   (5, 120),        # 0:05 - 2:00
        "runtime_sec":  (700, 2000),     # 5km: 11:40 - 33:20
    },
    "standard": {
        "swimtime_sec": (800, 2400),     # 1500m: 13:20 - 40:00
        "t1time_sec":   (5, 240),        # 0:05 - 4:00
        "biketime_sec": (2400, 5400),    # 40km: 40:00 - 90:00
        "t2time_sec":   (5, 180),        # 0:05 - 3:00
        "runtime_sec":  (1500, 3600),    # 10km: 25:00 - 60:00
    },
    "olympic": {
        "swimtime_sec": (800, 2400),
        "t1time_sec":   (5, 240),
        "biketime_sec": (2400, 5400),
        "t2time_sec":   (5, 180),
        "runtime_sec":  (1500, 3600),
    },
}


def _filter_plausible_history(
    history_df: pd.DataFrame,
    distance_category: str,
) -> pd.DataFrame:
    """
    Remove races from history where split times are outside plausible range
    for the target distance.

    Catches mislabeled races (e.g., super-sprint results tagged as sprint).
    Only filters on splits that are present and non-null in each row;
    missing splits do not disqualify a race.

    Args:
        history_df: Athlete race history DataFrame
        distance_category: Target distance category (e.g., "sprint")

    Returns:
        Filtered DataFrame with implausible races removed.
    """
    dist_key = distance_category.lower().strip()
    ranges = EMA_SPLIT_PLAUSIBLE_RANGES.get(dist_key)
    if ranges is None:
        return history_df  # No ranges defined for this distance

    df = history_df.copy()

    # Parse split times to seconds if not already done
    for col_base in ["swimtime", "biketime", "runtime"]:
        sec_col = col_base + "_sec"
        if sec_col not in df.columns and col_base in df.columns:
            df[sec_col] = df[col_base].apply(parse_time_to_seconds)

    # Build a mask: True = keep this race (all available splits are in range)
    keep = pd.Series(True, index=df.index)

    for sec_col, (lo, hi) in ranges.items():
        if sec_col in df.columns:
            vals = df[sec_col]
            # Only check rows with non-null values
            has_val = vals.notna()
            out_of_range = has_val & ((vals < lo) | (vals > hi))
            keep = keep & ~out_of_range

    n_removed = (~keep).sum()
    if n_removed > 0:
        logger.debug(
            f"Plausibility filter ({dist_key}): removed {n_removed}/{len(df)} "
            f"races with split times outside expected range"
        )

    return df[keep]


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
        - ema_t1_sec_5, ema_t2_sec_5, ema_t1t2_sec_5: EWMA of transition times over last 5
        - std_t1t2_sec_24m: Std dev of combined T1+T2 in last 24 months
        - last_total_sec: Most recent total time
        - best_total_sec_24m: Best total in last 24 months
        - std_total_sec_24m: Std dev of total in last 24 months
        - days_since_last_race: Days from last race to event_date
        - races_12m: Number of races in last 12 months
        
    Athlete tier features (competition level indicators):
        - athlete_avg_tier: Average tier of races competed (1=WTCS, 2=WorldCup, 3=Regional, 4=Other)
        - athlete_t1_rate: Fraction of races at Tier 1 (WTCS/Championship)
        - athlete_t1t2_rate: Fraction of races at Tier 1 or 2 (high-level racing)
        - athlete_best_tier: Best (lowest) tier raced at in last 24 months
        - ema_finish_pct_tier1: EMA of finish percentile in Tier 1 races only (if available)

    Distance-agnostic features:
        - ema_finish_pct_5: EMA of finish percentile (finish_position / n_finishers)
          across last 5 races. Naturally distance-agnostic since percentile is relative.
    """
    features = {
        "ema_swim_sec_5": None,
        "ema_bike_sec_5": None,
        "ema_run_sec_5": None,
        "ema_total_sec_5": None,
        # Transition features
        "ema_t1_sec_5": None,
        "ema_t2_sec_5": None,
        "ema_t1t2_sec_5": None,
        "std_t1t2_sec_24m": None,
        "last_total_sec": None,
        "best_total_sec_24m": None,
        "std_total_sec_24m": None,
        "days_since_last_race": None,
        "races_12m": 0,
        "races_24m": 0,
        # Distance-agnostic performance (multi-span)
        "ema_finish_pct_3": None,   # Short-term form (last ~1 month)
        "ema_finish_pct_5": None,   # Medium-term form (last ~2-3 months)
        "ema_finish_pct_12": None,  # Season baseline (last ~6-9 months)
        "form_trend": None,         # Direction: pct_3 - pct_12 (negative = improving)
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
    for col in ["swimtime", "t1time", "biketime", "t2time", "runtime", "total_time"]:
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
            tier1_chrono = tier1_df.iloc[::-1].copy()
            # Use actual n_finishers if available, otherwise approximate with 50
            if "n_finishers" in tier1_chrono.columns:
                n_fin = tier1_chrono["n_finishers"].astype(float).replace(0, np.nan)
                tier1_chrono["finish_pct"] = tier1_chrono["finish_position"].astype(float) / n_fin
            else:
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

    # Last total — guard against non-numeric values from parse_time_to_seconds
    last_total = valid_df["total_time_sec"].iloc[0]
    if pd.notna(last_total):
        features["last_total_sec"] = int(last_total)

    # 24-month window
    cutoff_24m = pd.Timestamp(event_date) - timedelta(days=730)
    df_24m = valid_df[valid_df["event_date"] >= cutoff_24m]

    if not df_24m.empty:
        best_24m = df_24m["total_time_sec"].min()
        if pd.notna(best_24m):
            features["best_total_sec_24m"] = int(best_24m)
        std_24m = df_24m["total_time_sec"].std()
        features["std_total_sec_24m"] = float(std_24m) if pd.notna(std_24m) else None
        features["races_24m"] = len(df_24m)

        # T1+T2 variability over 24 months
        if "t1time_sec" in df_24m.columns and "t2time_sec" in df_24m.columns:
            t1t2_24m = df_24m["t1time_sec"].add(df_24m["t2time_sec"]).dropna()
            if len(t1t2_24m) >= 2:
                features["std_t1t2_sec_24m"] = float(t1t2_24m.std())

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
            ("t1", "t1time_sec"),
            ("bike", "biketime_sec"),
            ("t2", "t2time_sec"),
            ("run", "runtime_sec"),
            ("total", "total_time_sec"),
        ]:
            if col in last_5_chrono.columns:
                vals = last_5_chrono[col].dropna()
                if len(vals) >= 1:
                    # EWMA with span=5
                    ema_val = vals.ewm(span=5, min_periods=1).mean().iloc[-1]
                    features[f"ema_{split}_sec_5"] = float(ema_val)

        # Combined T1+T2 EWMA
        if "t1time_sec" in last_5_chrono.columns and "t2time_sec" in last_5_chrono.columns:
            t1t2 = last_5_chrono["t1time_sec"].add(last_5_chrono["t2time_sec"])
            t1t2_vals = t1t2.dropna()
            if len(t1t2_vals) >= 1:
                features["ema_t1t2_sec_5"] = float(
                    t1t2_vals.ewm(span=5, min_periods=1).mean().iloc[-1]
                )

    # ---- Finish Percentile EMA (distance-agnostic, multi-span) ----
    # finish_pct = finish_position / n_finishers (0 = best, 1 = last)
    if "finish_position" in valid_df.columns and "n_finishers" in valid_df.columns:
        pct_df = valid_df[["event_date", "finish_position", "n_finishers"]].copy()
        pct_df = pct_df.dropna(subset=["finish_position", "n_finishers"])
        pct_df = pct_df[pct_df["n_finishers"] > 0]
        if not pct_df.empty:
            pct_df["finish_pct"] = pct_df["finish_position"].astype(float) / pct_df["n_finishers"].astype(float)
            pct_df["finish_pct"] = pct_df["finish_pct"].clip(0, 1)

            # Span 3: short-term form (~last month, 3 races)
            last_3_pct = pct_df.head(3).iloc[::-1]
            ema_pct_3 = last_3_pct["finish_pct"].ewm(span=3, min_periods=1).mean().iloc[-1]
            features["ema_finish_pct_3"] = float(ema_pct_3)

            # Span 5: medium-term form (~last 2-3 months)
            last_5_pct = pct_df.head(5).iloc[::-1]
            ema_pct_5 = last_5_pct["finish_pct"].ewm(span=5, min_periods=1).mean().iloc[-1]
            features["ema_finish_pct_5"] = float(ema_pct_5)

            # Span 12: season baseline (~last 6-9 months)
            last_12_pct = pct_df.head(12).iloc[::-1]
            ema_pct_12 = last_12_pct["finish_pct"].ewm(span=12, min_periods=1).mean().iloc[-1]
            features["ema_finish_pct_12"] = float(ema_pct_12)

            # Form trend: negative = improving, positive = declining
            features["form_trend"] = float(ema_pct_3 - ema_pct_12)

    return features


def _filter_corrupt_gaps(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with corrupt gap_to_leader_sec values.

    Detects when the race leader had elapsed=0 (timing system error),
    which causes every athlete's gap to equal their own elapsed time.
    This is detected when gap_to_leader_sec == elapsed_sec (within 1s tolerance).

    Note: Other outliers (DNFs, lapped athletes, bad splits) are now filtered
    at the source in wtcs_pack_metrics.py using the same logic as metrics.py.
    This function provides a safety net for any remaining corrupt races.
    """
    if checkpoint_df.empty:
        return checkpoint_df

    df = checkpoint_df.copy()

    # Filter corrupt races where leader had elapsed=0
    if "elapsed_sec" in df.columns:
        corrupt_leader = (df["gap_to_leader_sec"] - df["elapsed_sec"]).abs() <= 1
        df = df[~corrupt_leader]

    return df


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
        swim_df = _filter_corrupt_gaps(swim_df)

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
        bike_df = _filter_corrupt_gaps(bike_df)

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


def compute_race_relative_features(
    precomputed_metrics: pd.DataFrame,
) -> dict:
    """
    Compute race-relative features from precomputed position_metrics data.

    Uses precomputed split ranks, n_finishers, and median_total_sec from the
    position_metrics table, eliminating the need to fetch full race fields.

    Args:
        precomputed_metrics: DataFrame from fetch_precomputed_race_metrics()
            with columns: event_date, swimrank, bikerank, runrank,
            n_finishers, median_total_sec, elapsedrun

    Returns:
        dict of feature name -> value

    Features computed:
        - ema_perf_ratio_5: EMA of (athlete_total / race_median_total).
          A ratio < 1 means faster than median, > 1 means slower.
        - ema_swim_split_pct_5: EMA of swim split position percentile (0=fastest, 1=slowest)
        - ema_bike_split_pct_5: EMA of bike split position percentile
        - ema_run_split_pct_5: EMA of run split position percentile
    """
    features = {
        "ema_perf_ratio_5": None,
        "ema_swim_split_pct_5": None,
        "ema_bike_split_pct_5": None,
        "ema_run_split_pct_5": None,
    }

    if precomputed_metrics.empty:
        return features

    df = precomputed_metrics.copy()
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("event_date", ascending=False).reset_index(drop=True)

    # Compute percentiles and ratio from precomputed data
    n_fin = df["n_finishers"].astype(float)
    df["swim_pct"] = df["swimrank"].astype(float) / n_fin
    df["bike_pct"] = df["bikerank"].astype(float) / n_fin
    df["run_pct"] = df["runrank"].astype(float) / n_fin

    median_total = df["median_total_sec"].astype(float)
    valid_median = median_total > 0
    df["perf_ratio"] = np.where(
        valid_median,
        df["elapsedrun"].astype(float) / median_total,
        np.nan,
    )

    # Take last 5 races and reverse to chronological for EMA
    def _ema_last5(series):
        """Compute EMA over last 5 non-NaN values (chronological order)."""
        vals = series.head(5).dropna()
        if vals.empty:
            return None
        # Reverse from most-recent-first to chronological
        vals = vals.iloc[::-1]
        return float(vals.ewm(span=5, min_periods=1).mean().iloc[-1])

    features["ema_perf_ratio_5"] = _ema_last5(df["perf_ratio"])
    features["ema_swim_split_pct_5"] = _ema_last5(df["swim_pct"])
    features["ema_bike_split_pct_5"] = _ema_last5(df["bike_pct"])
    features["ema_run_split_pct_5"] = _ema_last5(df["run_pct"])

    return features


def compute_cross_distance_features(
    all_history_df: pd.DataFrame,
    target_distance: str | None,
    event_date: date,
) -> dict:
    """
    Compute cross-distance performance features.

    Uses the athlete's full race history (all distances) to compute
    distance-specific finish percentile EMAs. This allows the model to learn
    how sprint performance correlates with standard performance.

    Args:
        all_history_df: Athlete's full history DataFrame (all distances, from fetch_athlete_history)
        target_distance: The distance category of the target race (e.g., 'sprint', 'standard')
        event_date: Target event date

    Returns:
        dict of feature name -> value

    Features computed:
        - ema_finish_pct_sprint: EMA of finish percentile in sprint races only
        - ema_finish_pct_standard: EMA of finish percentile in standard/olympic races only
    """
    features = {
        "ema_finish_pct_sprint": None,
        "ema_finish_pct_standard": None,
    }

    if all_history_df.empty:
        return features

    df = all_history_df.copy()
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("event_date", ascending=False).reset_index(drop=True)

    # Need finish_position and n_finishers for percentile
    if "finish_position" not in df.columns or "n_finishers" not in df.columns:
        return features

    df = df.dropna(subset=["finish_position", "n_finishers"])
    df = df[df["n_finishers"] > 0].copy()
    df["finish_pct"] = df["finish_position"].astype(float) / df["n_finishers"].astype(float)
    df["finish_pct"] = df["finish_pct"].clip(0, 1)

    # Compute per-distance EMAs
    distance_map = {
        "sprint": ["sprint"],
        "standard": ["standard", "olympic"],
    }

    for feat_name, distance_values in distance_map.items():
        if "prog_distance_category" not in df.columns:
            continue
        dist_df = df[df["prog_distance_category"].str.lower().isin(distance_values)]
        if dist_df.empty:
            continue
        # Last 5 races of this distance, reversed to chronological
        last_5 = dist_df.head(5).iloc[::-1]
        ema_val = last_5["finish_pct"].ewm(span=5, min_periods=1).mean().iloc[-1]
        features[f"ema_finish_pct_{feat_name}"] = float(ema_val)

    return features


# Distance category encoding for model features
# Includes both canonical names and actual DB values (underscore variants)
DISTANCE_CATEGORY_ENCODING = {
    "super-sprint": 0,
    "super_sprint": 0,
    "sprint": 1,
    "standard": 2,
    "olympic": 2,  # Alias for standard
    "middle": 3,
    "middle_distance": 3,
    "long": 4,
    "long_distance": 4,
}


def encode_distance_category(category: str | None) -> int:
    """Encode distance category as integer for model features."""
    if category is None:
        return -1
    return DISTANCE_CATEGORY_ENCODING.get(category.lower().strip(), -1)


def compute_field_context_features(
    athlete_features_df: pd.DataFrame,
    athlete_id: int
) -> dict:
    """
    Compute field-context features for an athlete within a start list.

    Args:
        athlete_features_df: DataFrame with all athletes and their features
        athlete_id: The athlete to compute context for

    Returns:
        dict of feature name -> value

    Features computed:
        - seed_total_rank: Rank among entrants (1=best), using ema_finish_pct_5
          (distance-agnostic percentile) when available, falling back to ema_total_sec_5
        - seed_total_gap_to_best: Gap to the best seed in field (percentile units)
        - field_depth_top10_mean: Mean percentile of top 10 seeds
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

    # Prefer ema_finish_pct_5 for seeding (distance-agnostic)
    # Fall back to ema_total_sec_5 if percentile not available
    use_pct = (
        "ema_finish_pct_5" in athlete_features_df.columns
        and athlete_features_df["ema_finish_pct_5"].notna().sum() >= 3
    )

    if use_pct:
        seed_col = "ema_finish_pct_5"
        ascending = True  # Lower percentile = better
    else:
        seed_col = "ema_total_sec_5"
        ascending = True  # Lower time = better

    seeds = athlete_features_df[["athlete_id", seed_col]].copy()
    seeds = seeds.dropna(subset=[seed_col])

    if seeds.empty:
        return features

    seeds["seed_rank"] = seeds[seed_col].rank(method="min", ascending=ascending)

    best_seed = seeds[seed_col].min()
    features["field_depth_top10_mean"] = float(
        seeds.nsmallest(10, seed_col)[seed_col].mean()
    )

    # Get this athlete's rank and gap
    athlete_row = seeds[seeds["athlete_id"] == athlete_id]
    if not athlete_row.empty:
        features["seed_total_rank"] = int(athlete_row["seed_rank"].iloc[0])
        athlete_val = athlete_row[seed_col].iloc[0]
        if pd.notna(athlete_val) and pd.notna(best_seed):
            features["seed_total_gap_to_best"] = float(athlete_val - best_seed)

    return features


def build_features_for_program(
    engine: Engine,
    key: ProgramKey,
    use_start_list: bool = True,
    match_distance: bool = True,
    elite_only: bool = True,
    athlete_ids_override: list[int] | None = None,
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
    if athlete_ids_override is not None:
        from .sql import fetch_athletes_by_ids
        athletes_df = fetch_athletes_by_ids(engine, athlete_ids_override)
        if athletes_df.empty:
            logger.warning(f"No athletes found for override IDs: {athlete_ids_override}")
            return pd.DataFrame()
        logger.info(f"Using custom athlete list: {len(athletes_df)} athletes")
    elif use_start_list:
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

        # Fetch ALL history (not distance-filtered) for cross-distance features
        all_history_df = fetch_athlete_history(
            engine,
            athlete_id,
            event_date,
            limit=50,
            distance_category=None,  # All distances
            elite_only=elite_only
        )

        # Filter to matching distance in Python for distance-specific features
        used_fallback = True  # Assume fallback unless we get a clean match
        if distance_category and not all_history_df.empty and "prog_distance_category" in all_history_df.columns:
            matched_history = all_history_df[
                all_history_df["prog_distance_category"].str.lower() == distance_category.lower()
            ].copy()
            if matched_history.empty:
                matched_history = all_history_df  # Fallback to all distances
            else:
                used_fallback = False
        else:
            matched_history = all_history_df

        # Filter out races with split times implausible for the target distance
        # (catches mislabeled races, e.g. super-sprint results tagged as sprint)
        if distance_category and not matched_history.empty:
            n_before = len(matched_history)
            matched_history = _filter_plausible_history(matched_history, distance_category)
            if matched_history.empty:
                # All matched races had implausible times — fall back
                matched_history = all_history_df
                used_fallback = True
            elif not used_fallback and len(matched_history) < n_before:
                logger.debug(
                    f"Athlete {athlete_id}: {n_before - len(matched_history)} "
                    f"implausible {distance_category} races filtered out"
                )

        n_matched_races = len(matched_history) if not used_fallback else 0

        # Fetch precomputed race metrics (split ranks, n_finishers, median)
        # from position_metrics table — much faster than fetching full race fields
        precomputed_df = fetch_precomputed_race_metrics(
            engine, athlete_id, event_date, limit=50
        )

        # Fetch pack history, filtering by distance to avoid mixing Olympic and Sprint gaps
        pack_df = fetch_pack_history(
            engine, athlete_id, event_date, limit=100, distance_category=distance_category
        )

        # If no pack data with matching distance, fall back to all pack history
        if pack_df.empty and distance_category:
            pack_df = fetch_pack_history(engine, athlete_id, event_date, limit=100)

        # Compute features
        form_feats = compute_athlete_form_features(matched_history, event_date)
        pack_feats = compute_pack_features(pack_df, event_date)
        race_relative_feats = compute_race_relative_features(precomputed_df)
        cross_distance_feats = compute_cross_distance_features(
            all_history_df, distance_category, event_date
        )

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
            "distance_category_encoded": encode_distance_category(
                event_meta.get("prog_distance_category")
            ),
            "event_country": event_meta.get("event_country"),
            "event_name": event_meta.get("event_name"),
            "event_tier": classify_event_tier(event_meta.get("event_name", "")),
            "wetsuit": int(bool(event_meta.get("wetsuit"))) if event_meta.get("wetsuit") is not None else 0,
            "ema_distance_match": 0 if used_fallback else 1,
            "n_matched_races": n_matched_races,
            **form_feats,
            **pack_feats,
            **race_relative_feats,
            **cross_distance_feats,
        }
        athlete_features_list.append(athlete_row)

    if not athlete_features_list:
        logger.warning(f"No athlete features computed for {key}")
        return pd.DataFrame()

    features_df = pd.DataFrame(athlete_features_list)

    # ---- Elo Ratings & WT Rankings ----
    all_athlete_ids = features_df["athlete_id"].dropna().astype(int).tolist()

    # Fetch Elo ratings
    try:
        elo_df = fetch_elo_ratings(engine, all_athlete_ids)
        if not elo_df.empty:
            elo_df = elo_df[["athlete_id", "elo_rating", "elo_peak", "elo_races"]].copy()
            features_df = features_df.merge(elo_df, on="athlete_id", how="left")
        else:
            features_df["elo_rating"] = None
            features_df["elo_peak"] = None
            features_df["elo_races"] = None
    except Exception as e:
        logger.warning(f"Could not fetch Elo ratings: {e}")
        features_df["elo_rating"] = None
        features_df["elo_peak"] = None
        features_df["elo_races"] = None

    # Fetch WT ranking points
    try:
        # Determine gender from prog_name for ranking category filtering
        prog_name_str = str(event_meta.get("prog_name", "")).lower()
        gender = "female" if "women" in prog_name_str else "male"

        # Use the year before event_date to get most relevant ranking
        ranking_year = event_date.year if isinstance(event_date, date) else None
        wt_df = fetch_wt_ranking_points(
            engine, all_athlete_ids, year=ranking_year, gender=gender
        )
        # Fall back to previous year if no results for current year
        if wt_df.empty and ranking_year:
            wt_df = fetch_wt_ranking_points(
                engine, all_athlete_ids, year=ranking_year - 1, gender=gender
            )

        if not wt_df.empty:
            wt_df = wt_df[[
                "athlete_id", "rank_position", "total_points",
                "continental_rank_position", "continental_total_points",
            ]].copy()
            wt_df = wt_df.rename(columns={
                "rank_position": "wt_rank_position",
                "total_points": "wt_total_points",
                "continental_rank_position": "wt_continental_rank",
                "continental_total_points": "wt_continental_points",
            })
            features_df = features_df.merge(wt_df, on="athlete_id", how="left")
        else:
            features_df["wt_rank_position"] = None
            features_df["wt_total_points"] = None
            features_df["wt_continental_rank"] = None
            features_df["wt_continental_points"] = None
    except Exception as e:
        logger.warning(f"Could not fetch WT rankings: {e}")
        features_df["wt_rank_position"] = None
        features_df["wt_total_points"] = None
        features_df["wt_continental_rank"] = None
        features_df["wt_continental_points"] = None

    # Compute tier_delta: event_tier - athlete_avg_tier
    # Positive value means athlete is "stepping down" to a lower-tier race (advantage)
    # e.g., WTCS athlete (avg_tier ~1.5) at World Cup (tier 2) → tier_delta ≈ +0.5
    if "event_tier" in features_df.columns and "athlete_avg_tier" in features_df.columns:
        # Force numeric dtype to prevent TypeError from mixed int/None object columns
        features_df["tier_delta"] = (
            pd.to_numeric(features_df["event_tier"], errors="coerce") -
            pd.to_numeric(features_df["athlete_avg_tier"], errors="coerce")
        ).fillna(0.0)
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


def get_split_feature_columns() -> list[str]:
    """
    Return the reduced feature set for distance-specific split models.

    Split models predict absolute seconds (swim_sec, bike_sec, run_sec) for
    individual athletes. They should only use features that reflect the
    athlete's own ability and form — NOT field-context or ranking features
    which cause the model to learn "big field → fast times" shortcuts instead
    of "this athlete runs X:XX".

    Excluded from the full feature set:
    - n_entrants, seed_total_rank: field composition doesn't cause faster times
    - elo_rating, elo_peak, elo_races: ranking proxy, causes inverse correlation
    - wt_rank_position, wt_total_points, wt_continental_*: ranking features
    - event_tier: event prestige doesn't change individual split times
    - distance_category_encoded: constant within each distance model
    """
    return [
        # Athlete form features (absolute, distance-specific)
        "ema_swim_sec_5",
        "ema_bike_sec_5",
        "ema_run_sec_5",
        "ema_total_sec_5",
        # Transition features
        "ema_t1_sec_5",
        "ema_t2_sec_5",
        "ema_t1t2_sec_5",
        "std_t1t2_sec_24m",
        "std_total_sec_24m",
        "days_since_last_race",
        "races_12m",
        # Distance-agnostic performance features
        "ema_finish_pct_3",
        "ema_finish_pct_5",
        "ema_finish_pct_12",
        "form_trend",
        "ema_perf_ratio_5",
        # Per-split percentiles (cross-discipline signal: fast swim → bike pack → run)
        "ema_swim_split_pct_5",
        "ema_bike_split_pct_5",
        "ema_run_split_pct_5",
        # Cross-distance form
        "ema_finish_pct_sprint",
        "ema_finish_pct_standard",
        # Competition level (relative, not absolute ranking)
        "athlete_t1t2_rate",
        "tier_delta",
        # Pack dynamics (affect bike/run interaction)
        "ema_swim_pos_pct_7",
        "ema_bike_pos_pct_7",
        "ema_swim_pack_7",
        "ema_bike_pack_7",
        "ema_swim_gap_sec_7",
        "ema_bike_gap_sec_7",
        # Race conditions
        "wetsuit",
        # Distance match quality
        "ema_distance_match",
        "n_matched_races",
    ]


def get_feature_columns() -> list[str]:
    """
    Return the full list of feature column names used for modeling.

    This is the full feature set used by the unified models (total time,
    percentile/ranking). Distance-specific split models use the reduced
    set from get_split_feature_columns().
    """
    return [
        # Athlete form features (absolute, distance-specific)
        "ema_swim_sec_5",
        "ema_bike_sec_5",
        "ema_run_sec_5",
        "ema_total_sec_5",
        # Transition features
        "ema_t1_sec_5",             # EWMA of T1 transition time
        "ema_t2_sec_5",             # EWMA of T2 transition time
        "ema_t1t2_sec_5",           # EWMA of combined T1+T2
        "std_t1t2_sec_24m",         # Std dev of T1+T2 in last 24 months
        "std_total_sec_24m",
        "days_since_last_race",
        "races_12m",
        # Distance-agnostic performance features (multi-span)
        "ema_finish_pct_3",         # Short-term form (~last month)
        "ema_finish_pct_5",         # Medium-term form (~last 2-3 months)
        "ema_finish_pct_12",        # Season baseline (~last 6-9 months)
        "form_trend",               # ema_pct_3 - ema_pct_12 (negative = improving)
        "ema_perf_ratio_5",         # EMA of athlete_total / race_median_total
        "ema_swim_split_pct_5",     # EMA of swim split position percentile
        "ema_bike_split_pct_5",     # EMA of bike split position percentile
        "ema_run_split_pct_5",      # EMA of run split position percentile
        # Cross-distance features
        "ema_finish_pct_sprint",    # EMA of finish percentile in sprint races
        "ema_finish_pct_standard",  # EMA of finish percentile in standard races
        # Athlete competition level features (kept only the most informative)
        "athlete_t1t2_rate",
        "tier_delta",  # event_tier - athlete_avg_tier (positive = athlete stepping down)
        # Pack features - EMA based (span=7)
        "ema_swim_pos_pct_7",
        "ema_bike_pos_pct_7",
        "ema_swim_pack_7",
        "ema_bike_pack_7",
        "ema_swim_gap_sec_7",
        "ema_bike_gap_sec_7",
        # Elo rating features
        "elo_rating",               # Current Elo rating (higher = better)
        "elo_peak",                 # All-time peak Elo rating
        "elo_races",                # Number of races in Elo history
        # World Triathlon ranking features
        "wt_rank_position",         # WT world ranking position (1 = best)
        "wt_total_points",          # WT world ranking points
        "wt_continental_rank",      # WT continental ranking position (1 = best)
        "wt_continental_points",    # WT continental ranking points
        # Field context
        "seed_total_rank",
        "n_entrants",
        # Event context
        "event_tier",
        "distance_category_encoded",  # Integer encoding of prog_distance_category
        "wetsuit",                    # 1 = wetsuit race, 0 = non-wetsuit
        # Distance match quality
        "ema_distance_match",         # 1 = EMA from matching distance, 0 = fallback
        "n_matched_races",            # Number of plausible matching-distance races
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

    # Default values for missing features.
    # Philosophy: unknown athletes should get CONSERVATIVE (below-average) defaults
    # for ability features. Using field median is too generous — an athlete with no
    # data is more likely below-average than exactly average.

    # Helper: 75th percentile of a column (worse than median for "lower is better" features)
    def _q75(col):
        if col in df.columns and df[col].notna().any():
            return float(df[col].quantile(0.75))
        return None

    # Helper: 25th percentile (worse than median for "higher is better" features)
    def _q25(col):
        if col in df.columns and df[col].notna().any():
            return float(df[col].quantile(0.25))
        return None

    defaults = {
        # Absolute form: use field 75th percentile (slower = conservative)
        "ema_swim_sec_5": _q75("ema_swim_sec_5") or 1200,
        "ema_bike_sec_5": _q75("ema_bike_sec_5") or 3600,
        "ema_run_sec_5": _q75("ema_run_sec_5") or 1800,
        "ema_total_sec_5": _q75("ema_total_sec_5") or 6600,
        "std_total_sec_24m": DEFAULT_SIGMA_TOTAL,
        "days_since_last_race": 90,  # Assume long layoff (conservative)
        "races_12m": 0,
        # Distance-agnostic performance features — 0.65 = below mid-field
        "ema_finish_pct_3": 0.65,
        "ema_finish_pct_5": 0.65,
        "ema_finish_pct_12": 0.65,
        "form_trend": 0.0,         # Neutral (no trend)
        "ema_perf_ratio_5": 1.05,  # Slightly slower than race median
        "ema_swim_split_pct_5": 0.65,
        "ema_bike_split_pct_5": 0.65,
        "ema_run_split_pct_5": 0.65,
        # Cross-distance features
        "ema_finish_pct_sprint": 0.65,
        "ema_finish_pct_standard": 0.65,
        # Athlete competition level features
        "athlete_t1t2_rate": 0.0,
        "tier_delta": 0.0,  # Neutral (racing at typical level)
        # EMA pack features (span=7)
        "ema_swim_pos_pct_7": 0.65,  # Below mid-pack
        "ema_bike_pos_pct_7": 0.65,
        "ema_swim_pack_7": 4.0,  # Pack 4 = back of field
        "ema_bike_pack_7": 4.0,
        "ema_swim_gap_sec_7": 40.0,  # 40 sec gap (back of field)
        "ema_bike_gap_sec_7": 80.0,  # 80 sec gap
        # Elo rating features — below-average for unknown athletes
        "elo_rating": _q25("elo_rating") or 1400.0,
        "elo_peak": _q25("elo_peak") or 1400.0,
        "elo_races": 0,
        # World Triathlon ranking features — worst in field or large fallback
        "wt_rank_position": df["wt_rank_position"].max() if "wt_rank_position" in df.columns and df["wt_rank_position"].notna().any() else 500,
        "wt_total_points": 0.0,
        "wt_continental_rank": df["wt_continental_rank"].max() if "wt_continental_rank" in df.columns and df["wt_continental_rank"].notna().any() else 300,
        "wt_continental_points": 0.0,
        # Field context
        "seed_total_rank": df["seed_total_rank"].max() if "seed_total_rank" in df.columns else 50,
        "n_entrants": df["n_entrants"].median() if "n_entrants" in df.columns else 50,
        # Event context
        "event_tier": 2,  # Default to World Cup tier (mid-level)
        "distance_category_encoded": -1,  # Unknown distance
        "wetsuit": 0,  # Default to non-wetsuit
        # Distance match quality
        "ema_distance_match": 0,    # Assume fallback if unknown
        "n_matched_races": 0,
        # T1/T2 features (added in Phase 1)
        "ema_t1_sec_5": _q75("ema_t1_sec_5") or 40,
        "ema_t2_sec_5": _q75("ema_t2_sec_5") or 35,
        "ema_t1t2_sec_5": _q75("ema_t1t2_sec_5") or 75,
        "std_t1t2_sec_24m": 8.0,  # Moderate variability default
        # Pack features
        "front_pack_rate": 0.1,  # Conservative: rarely in front pack
    }

    # First pass: convert any Python None to NaN across all feature columns.
    # Features from compute_athlete_form_features() may return None for missing
    # values which pandas preserves as object dtype. fillna() handles NaN but
    # not always object-dtype None, so we explicitly convert first.
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Second pass: fill NaN with defaults
    for col in feature_cols:
        if col in df.columns:
            default_val = defaults.get(col, 0)
            df[col] = df[col].fillna(default_val)
        else:
            df[col] = defaults.get(col, 0)

    return df
