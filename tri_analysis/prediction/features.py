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
    fetch_head_to_head_results,
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
    1: 8.0,  # WTCS events count 8x (v36: increased from 4.0, +10% WTCS P@3)
    2: 3.0,  # World Cup events count 3x (v36: increased from 2.0)
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
        # Per-split variance (for MC simulation sigma calibration)
        "std_swim_sec_24m": None,
        "std_bike_sec_24m": None,
        "std_run_sec_24m": None,
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
        # Cold-start indicators: let the model learn to be uncertain about new athletes
        "n_swim_results": 0,     # Count of valid swim results (continuous cold-start signal)
        "is_cold_start": 1,      # Binary: 1 if <3 prior races, 0 otherwise
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
        
        # Tier-conditioned finish percentile EMAs
        # Compute EMA of finish percentile separately for each tier group.
        # This lets the model learn "this athlete finishes top-10% at WTCS but
        # top-5% at Continental" — different competitive contexts.
        _tier_groups = {
            "tier1": valid_df[valid_df["event_tier"] == 1],      # WTCS/Championships
            "tier2": valid_df[valid_df["event_tier"] == 2],      # World Cup
            "tier34": valid_df[valid_df["event_tier"] >= 3],     # Continental + Other
        }
        for tier_label, tier_df in _tier_groups.items():
            tier_df_recent = tier_df.head(10)
            if len(tier_df_recent) >= 1 and "finish_position" in tier_df_recent.columns:
                tier_chrono = tier_df_recent.iloc[::-1].copy()
                if "n_finishers" in tier_chrono.columns:
                    n_fin = tier_chrono["n_finishers"].astype(float).replace(0, np.nan)
                    tier_chrono["finish_pct"] = tier_chrono["finish_position"].astype(float) / n_fin
                else:
                    tier_chrono["finish_pct"] = tier_chrono["finish_position"].astype(float) / 50.0
                tier_chrono["finish_pct"] = tier_chrono["finish_pct"].clip(0, 1)
                if not tier_chrono["finish_pct"].isna().all():
                    ema_val = tier_chrono["finish_pct"].ewm(span=5, min_periods=1).mean().iloc[-1]
                    features[f"ema_finish_pct_{tier_label}"] = float(ema_val)

        # Races at each tier level in the last 24 months
        if not df_24m_tier.empty:
            features["races_tier1_24m"] = int((df_24m_tier["event_tier"] == 1).sum())
            features["races_tier2_24m"] = int((df_24m_tier["event_tier"] == 2).sum())
            features["races_tier34_24m"] = int((df_24m_tier["event_tier"] >= 3).sum())

    # Last race date and days since
    last_race_date = valid_df["event_date"].iloc[0]
    if pd.notna(last_race_date):
        days_since = max(0, (pd.Timestamp(event_date) - last_race_date).days)
        features["days_since_last_race"] = days_since
        features["log1p_days_since"] = float(np.log1p(days_since))
        # Freshness curve: ~21 days is optimal; too soon or too long is suboptimal
        features["freshness_curve"] = float(abs(days_since - 21))

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

        # Per-split variability over 24 months (for MC simulation sigma)
        for split, col in [("swim", "swimtime_sec"), ("bike", "biketime_sec"), ("run", "runtime_sec")]:
            if col in df_24m.columns:
                vals = pd.to_numeric(df_24m[col], errors="coerce").dropna()
                if len(vals) >= 2:
                    features[f"std_{split}_sec_24m"] = float(vals.std())

        # T1+T2 variability over 24 months
        if "t1time_sec" in df_24m.columns and "t2time_sec" in df_24m.columns:
            t1_vals = pd.to_numeric(df_24m["t1time_sec"], errors="coerce")
            t2_vals = pd.to_numeric(df_24m["t2time_sec"], errors="coerce")
            t1t2_24m = (t1_vals + t2_vals).dropna()
            if len(t1t2_24m) >= 2:
                features["std_t1t2_sec_24m"] = float(t1t2_24m.std())

    # 12-month window
    cutoff_12m = pd.Timestamp(event_date) - timedelta(days=365)
    df_12m = valid_df[valid_df["event_date"] >= cutoff_12m]
    features["races_12m"] = len(df_12m)

    # Cold-start indicators
    n_total_races = len(valid_df)
    features["is_cold_start"] = 1 if n_total_races < 3 else 0
    # Count valid swim results specifically
    if "swimtime_sec" in valid_df.columns:
        n_swim = int(pd.to_numeric(valid_df["swimtime_sec"], errors="coerce").notna().sum())
        features["n_swim_results"] = n_swim
    else:
        features["n_swim_results"] = 0

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
            t1_vals = pd.to_numeric(last_5_chrono["t1time_sec"], errors="coerce")
            t2_vals = pd.to_numeric(last_5_chrono["t2time_sec"], errors="coerce")
            t1t2 = t1_vals + t2_vals
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

            # Form momentum: recent acceleration (span 3 vs span 5)
            # More responsive than form_trend — captures peaking/fading
            features["form_momentum"] = float(ema_pct_3 - ema_pct_5)

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
        # Swim gap consistency — how reliably close to the front (for pack prediction)
        "std_swim_gap_sec_24m": None,   # Variance of gap (consistent vs erratic swimmer)
        "min_swim_gap_sec_24m": None,   # Best gap in 24m (even bad swims, how close?)
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
        features["front_pack_rate"] = float((swim_df["pack_id"] == 0).mean())
        features["avg_swim_gap_leader"] = float(swim_df["gap_to_leader_sec"].mean())
        features["p90_swim_gap_leader"] = float(swim_df["gap_to_leader_sec"].quantile(0.9))
        features["avg_pack_size_swim"] = float(swim_df["pack_size"].mean())

        # Swim gap consistency features (24-month window)
        swim_gap_vals = pd.to_numeric(swim_df["gap_to_leader_sec"], errors="coerce").dropna()
        if len(swim_gap_vals) >= 2:
            features["std_swim_gap_sec_24m"] = float(swim_gap_vals.std())
        if len(swim_gap_vals) >= 1:
            features["min_swim_gap_sec_24m"] = float(swim_gap_vals.min())

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
        # Long-term swim baseline (12-race EMA of swim percentile)
        "ema_swim_split_pct_12": None,
        # Best swim percentile in last 12 months (peak swim ability)
        "best_swim_pct_12m": None,
        # Swim form trend: pct_5 - pct_12 (negative = swim improving)
        "swim_form_trend": None,
        # Swim gap-to-leader features (course-normalized absolute gap)
        # behindswim = seconds behind swim leader, comparable across courses
        "ema_swim_behind_sec_5": None,   # 5-race EMA of gap-to-leader (short-term)
        "ema_swim_behind_sec_12": None,  # 12-race EMA of gap-to-leader (baseline)
        "best_swim_behind_12m": None,    # Best gap-to-leader in last 12 months
        # Normalized variance: std of split percentile across races (course-agnostic)
        # Low std_swim_pct = consistent swimmer relative to field regardless of course
        "std_swim_pct_24m": None,
        "std_bike_pct_24m": None,
        "std_run_pct_24m": None,
        # Position change features (Phase 4B)
        "ema_t1_rank_pct_5": None,
        "ema_t2_rank_pct_5": None,
        "ema_swim_to_t1_pos_change_5": None,
        "ema_bike_to_t2_pos_change_5": None,
        # Tier-conditioned split percentiles (v36): how athlete ranks in each
        # discipline specifically at Tier 1+2 events (WTCS/World Cup).
        # Critical for WTCS prediction — an athlete may be top-5% swimmer overall
        # but only top-20% at the elite level.
        "ema_swim_split_pct_tier12_5": None,
        "ema_bike_split_pct_tier12_5": None,
        "ema_run_split_pct_tier12_5": None,
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

    # T1/T2 rank percentiles (lower = faster transitioner)
    if "t1rank" in df.columns:
        df["t1_pct"] = pd.to_numeric(df["t1rank"], errors="coerce") / n_fin
    if "t2rank" in df.columns:
        df["t2_pct"] = pd.to_numeric(df["t2rank"], errors="coerce") / n_fin

    # Position changes (negative = gained positions, positive = lost)
    for col in ["swim_to_t1_pos_change", "t1_to_bike_pos_change",
                "bike_to_t2_pos_change", "t2_to_run_pos_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    median_total = df["median_total_sec"].astype(float)
    valid_median = median_total > 0
    df["perf_ratio"] = np.where(
        valid_median,
        df["elapsedrun"].astype(float) / median_total,
        np.nan,
    )

    # Normalized variance: std of percentile rank over 24-month window
    # This is course-agnostic — an athlete who's consistently top-10% swim
    # across fast and slow courses will have low std_swim_pct_24m
    if "event_date" in df.columns:
        cutoff_24m = pd.Timestamp.now() - pd.Timedelta(days=730)
        df_24m = df[df["event_date"] >= cutoff_24m]
    else:
        df_24m = df

    for split, col in [("swim", "swim_pct"), ("bike", "bike_pct"), ("run", "run_pct")]:
        if col in df_24m.columns:
            vals = df_24m[col].dropna()
            if len(vals) >= 3:  # Need at least 3 races for meaningful std
                features[f"std_{split}_pct_24m"] = float(vals.std())

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

    # Long-term swim baseline: 12-race EMA (captures baseline ability vs recent form)
    swim_pct_vals = df["swim_pct"].head(12).dropna()
    if not swim_pct_vals.empty:
        swim_pct_chrono = swim_pct_vals.iloc[::-1]
        features["ema_swim_split_pct_12"] = float(
            swim_pct_chrono.ewm(span=12, min_periods=1).mean().iloc[-1]
        )

    # Best swim percentile in last 12 months (peak swim ability)
    if "event_date" in df.columns:
        cutoff_12m = pd.Timestamp.now() - pd.Timedelta(days=365)
        df_12m = df[df["event_date"] >= cutoff_12m]
        swim_pct_12m = df_12m["swim_pct"].dropna()
        if not swim_pct_12m.empty:
            features["best_swim_pct_12m"] = float(swim_pct_12m.min())  # lower = better

    # Swim form trend: short-term vs long-term (negative = swim improving)
    if features["ema_swim_split_pct_5"] is not None and features["ema_swim_split_pct_12"] is not None:
        features["swim_form_trend"] = features["ema_swim_split_pct_5"] - features["ema_swim_split_pct_12"]

    # Swim gap-to-leader (behindswim): course-normalized absolute gap
    # Unlike raw swim seconds, gap-to-leader is comparable across courses
    # (30s behind leader means the same whether leader swam 8:00 or 10:00)
    if "behindswim" in df.columns:
        behind = pd.to_numeric(df["behindswim"], errors="coerce")
        features["ema_swim_behind_sec_5"] = _ema_last5(behind)

        # 12-race EMA for baseline
        behind_vals = behind.head(12).dropna()
        if not behind_vals.empty:
            behind_chrono = behind_vals.iloc[::-1]
            features["ema_swim_behind_sec_12"] = float(
                behind_chrono.ewm(span=12, min_periods=1).mean().iloc[-1]
            )

        # Best gap in last 12 months (lower = closer to leader = better)
        if "event_date" in df.columns:
            cutoff_12m = pd.Timestamp.now() - pd.Timedelta(days=365)
            behind_12m = behind[df["event_date"] >= cutoff_12m].dropna()
            if not behind_12m.empty:
                features["best_swim_behind_12m"] = float(behind_12m.min())

    # T1/T2 rank percentiles
    if "t1_pct" in df.columns:
        features["ema_t1_rank_pct_5"] = _ema_last5(df["t1_pct"])
    if "t2_pct" in df.columns:
        features["ema_t2_rank_pct_5"] = _ema_last5(df["t2_pct"])

    # Position change EMA: how many positions gained/lost in transitions
    # Negative = athlete gains positions (good transitioner), positive = loses
    if "swim_to_t1_pos_change" in df.columns:
        features["ema_swim_to_t1_pos_change_5"] = _ema_last5(df["swim_to_t1_pos_change"])
    if "bike_to_t2_pos_change" in df.columns:
        features["ema_bike_to_t2_pos_change_5"] = _ema_last5(df["bike_to_t2_pos_change"])

    # Tier-conditioned split percentiles (v36)
    # How the athlete ranks in swim/bike/run specifically at Tier 1+2 events.
    # Filters to WTCS + World Cup races only, computes EMA of split percentiles.
    if "event_name" in df.columns:
        df["_event_tier"] = df["event_name"].apply(classify_event_tier)
        tier12_df = df[df["_event_tier"] <= 2]
        if len(tier12_df) >= 1:
            features["ema_swim_split_pct_tier12_5"] = _ema_last5(tier12_df["swim_pct"])
            features["ema_bike_split_pct_tier12_5"] = _ema_last5(tier12_df["bike_pct"])
            features["ema_run_split_pct_tier12_5"] = _ema_last5(tier12_df["run_pct"])

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


def compute_head_to_head_features(
    h2h_df: pd.DataFrame,
    athlete_ids: list[int],
    event_date=None,
) -> dict[int, dict]:
    """
    Compute recency-weighted head-to-head features for each athlete in the field.

    For each athlete, computes pairwise matchup stats against field opponents,
    with exponential decay weighting (half-life = 365 days) so recent races
    count more than old ones.

    Features:
    - h2h_win_rate: weighted fraction of matchups won against field opponents
    - h2h_avg_position_gap: weighted mean(opponent_pos - athlete_pos)
    - h2h_n_shared_races: number of shared races with any field opponent
    - h2h_n_opponents_faced: how many field opponents they've raced against

    Args:
        h2h_df: DataFrame from fetch_head_to_head_results with columns:
                event_id, prog_id, event_date, athlete_id, finish_position
        athlete_ids: List of athlete IDs in the current field
        event_date: Target event date for recency weighting

    Returns:
        Dict mapping athlete_id -> dict of h2h features
    """
    field_set = set(athlete_ids)
    defaults = {
        "h2h_win_rate": 0.5,
        "h2h_avg_position_gap": 0.0,
        "h2h_n_shared_races": 0,
        "h2h_n_opponents_faced": 0,
    }

    if h2h_df.empty:
        return {aid: defaults.copy() for aid in athlete_ids}

    results = {}

    # Compute recency weights per race (half-life = 365 days)
    HALF_LIFE_DAYS = 365.0
    race_weights = {}
    if event_date is not None and "event_date" in h2h_df.columns:
        ref_date = pd.to_datetime(event_date)
        for (eid, pid), grp in h2h_df.groupby(["event_id", "prog_id"]):
            race_date = pd.to_datetime(grp["event_date"].iloc[0])
            days_ago = max(0, (ref_date - race_date).days)
            race_weights[(eid, pid)] = float(2.0 ** (-days_ago / HALF_LIFE_DAYS))

    # Group by race (event_id, prog_id)
    race_groups = h2h_df.groupby(["event_id", "prog_id"])

    # Build pairwise records: athlete -> opponent -> [(my_pos, opp_pos, weight)]
    matchup_records = {aid: {} for aid in athlete_ids}

    for (eid, pid), race_df in race_groups:
        race_athletes = race_df[race_df["athlete_id"].isin(field_set)]
        if len(race_athletes) < 2:
            continue

        w = race_weights.get((eid, pid), 1.0)

        positions = dict(zip(race_athletes["athlete_id"], race_athletes["finish_position"]))
        athletes_in_race = list(positions.keys())

        for i, a1 in enumerate(athletes_in_race):
            for a2 in athletes_in_race[i + 1:]:
                p1, p2 = positions[a1], positions[a2]
                if a1 in matchup_records:
                    matchup_records[a1].setdefault(a2, []).append((p1, p2, w))
                if a2 in matchup_records:
                    matchup_records[a2].setdefault(a1, []).append((p2, p1, w))

    for aid in athlete_ids:
        records = matchup_records.get(aid, {})
        if not records:
            results[aid] = defaults.copy()
            continue

        total_weighted_wins = 0.0
        total_weight = 0.0
        total_weighted_gap = 0.0
        n_opponents = len(records)

        for _, matchups in records.items():
            for my_pos, opp_pos, w in matchups:
                total_weight += w
                if my_pos < opp_pos:
                    total_weighted_wins += w
                elif my_pos == opp_pos:
                    total_weighted_wins += 0.5 * w
                total_weighted_gap += (opp_pos - my_pos) * w

        # Count unique shared races
        athlete_races = h2h_df[h2h_df["athlete_id"] == aid][["event_id", "prog_id"]].drop_duplicates()
        n_shared_races = len(athlete_races)

        results[aid] = {
            "h2h_win_rate": total_weighted_wins / total_weight if total_weight > 0 else 0.5,
            "h2h_avg_position_gap": total_weighted_gap / total_weight if total_weight > 0 else 0.0,
            "h2h_n_shared_races": n_shared_races,
            "h2h_n_opponents_faced": n_opponents,
        }

    return results


def _parse_weather_numeric(value, default: float = None) -> float | None:
    """Parse a weather string value to float, returning default if unparseable."""
    if value is None:
        return default
    try:
        # Handle strings like "25°C", "65%", "22.5 °C", etc.
        cleaned = str(value).strip().rstrip("°CcFf%").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return default




def compute_venue_features(
    all_history_df: pd.DataFrame,
    target_venue: str | None,
) -> dict:
    """
    Compute venue-specific performance features.

    Athletes may consistently over/underperform at specific venues due to
    course familiarity, altitude, conditions, etc.

    Args:
        all_history_df: Full athlete history (all distances) with event_venue column
        target_venue: The venue of the upcoming race

    Returns:
        Dict of venue feature name -> value
    """
    defaults = {
        "has_raced_venue": 0,
        "n_venue_races": 0,
        "venue_finish_pct_mean": None,
        "venue_delta": None,
    }

    if target_venue is None or all_history_df.empty:
        return defaults

    if "event_venue" not in all_history_df.columns:
        return defaults

    target_venue_lower = str(target_venue).strip().lower()
    if not target_venue_lower:
        return defaults

    venue_mask = all_history_df["event_venue"].fillna("").str.strip().str.lower() == target_venue_lower
    venue_df = all_history_df[venue_mask]

    if venue_df.empty:
        return defaults

    n_venue = len(venue_df)
    result = {
        "has_raced_venue": 1,
        "n_venue_races": min(n_venue, 20),  # Cap to avoid outliers
    }

    # Compute venue-specific finish percentile
    if "finish_position" in venue_df.columns and "n_finishers" in venue_df.columns:
        venue_pcts = venue_df["finish_position"] / venue_df["n_finishers"]
        venue_pcts = venue_pcts.dropna()
        if len(venue_pcts) > 0:
            result["venue_finish_pct_mean"] = float(venue_pcts.mean())

            # Compute overall finish pct for comparison
            if "finish_position" in all_history_df.columns and "n_finishers" in all_history_df.columns:
                all_pcts = (all_history_df["finish_position"] / all_history_df["n_finishers"]).dropna()
                if len(all_pcts) > 0:
                    # Negative delta = overperforms at this venue
                    result["venue_delta"] = float(venue_pcts.mean() - all_pcts.mean())

    return result


def compute_race_density_features(
    all_history_df: pd.DataFrame,
    event_date: date,
) -> dict:
    """
    Compute race density/fatigue features from recent racing history.

    Captures short-term fatigue signals beyond days_since_last_race.

    Args:
        all_history_df: Full athlete history with event_date column
        event_date: The target event date

    Returns:
        Dict of density feature name -> value
    """
    defaults = {
        "races_30d": 0,
        "races_60d": 0,
        "days_since_2nd_last": None,
    }

    if all_history_df.empty or "event_date" not in all_history_df.columns:
        return defaults

    dates = pd.to_datetime(all_history_df["event_date"])
    event_dt = pd.Timestamp(event_date)

    days_ago = (event_dt - dates).dt.days
    # Only look at races before the event
    valid = days_ago > 0

    result = {
        "races_30d": int((days_ago[valid] <= 30).sum()),
        "races_60d": int((days_ago[valid] <= 60).sum()),
    }

    # Days since 2nd most recent race (racing cadence)
    sorted_days = sorted(days_ago[valid])
    if len(sorted_days) >= 2:
        result["days_since_2nd_last"] = float(sorted_days[1])
    else:
        result["days_since_2nd_last"] = None

    return result


def build_features_for_program(
    engine: Engine,
    key: ProgramKey,
    use_start_list: bool = True,
    match_distance: bool = True,
    elite_only: bool = True,
    athlete_ids_override: list[int] | None = None,
    event_meta_override: dict | None = None,
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
        event_meta_override: If provided, merge these values into event metadata
                             (e.g., weather data fetched at prediction time)

    Returns:
        DataFrame with one row per athlete, columns for all features plus identifiers

    Raises:
        ValueError: If no event metadata found for the key
    """
    # Get event metadata
    event_meta = fetch_event_metadata(engine, key)
    if event_meta is None:
        raise ValueError(f"No event metadata found for {key}")

    # Merge any overrides (e.g., weather fetched at prediction time)
    if event_meta_override:
        for k, v in event_meta_override.items():
            if v is not None:
                event_meta[k] = v

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

        # If too few pack records with matching distance, fall back to all pack history
        # Threshold of 3 ensures athletes with limited distance-specific data
        # (e.g., fast swimmers with few sprint races) get representative pack rates
        if len(pack_df) < 3 and distance_category:
            pack_df = fetch_pack_history(engine, athlete_id, event_date, limit=100)

        # Compute features
        form_feats = compute_athlete_form_features(matched_history, event_date)
        pack_feats = compute_pack_features(pack_df, event_date)
        race_relative_feats = compute_race_relative_features(precomputed_df)
        cross_distance_feats = compute_cross_distance_features(
            all_history_df, distance_category, event_date
        )
        venue_feats = compute_venue_features(
            all_history_df, event_meta.get("event_venue")
        )
        density_feats = compute_race_density_features(all_history_df, event_date)

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
            # Weather features (Phase 4C)
            "temperature_air": _parse_weather_numeric(event_meta.get("temperature_air")),
            "humidity": _parse_weather_numeric(event_meta.get("humidity")),
            "wbgt": _parse_weather_numeric(event_meta.get("wbgt")),
            "is_hot": int(_parse_weather_numeric(event_meta.get("wbgt"), default=20.0) > 25),
            # Open-Meteo enriched weather features
            "wind_speed_kmh": _parse_weather_numeric(event_meta.get("wind_speed_kmh")),
            "wind_gust_kmh": _parse_weather_numeric(event_meta.get("wind_gust_kmh")),
            "apparent_temp": _parse_weather_numeric(event_meta.get("apparent_temp")),
            "precipitation_mm": _parse_weather_numeric(event_meta.get("precipitation_mm")),
            "cloud_cover_pct": _parse_weather_numeric(event_meta.get("cloud_cover_pct")),
            # Course laps features (Phase 4D)
            "bike_laps": float(event_meta.get("bike_laps") or 0),
            "run_laps": float(event_meta.get("run_laps") or 0),
            "ema_distance_match": 0 if used_fallback else 1,
            "n_matched_races": n_matched_races,
            **form_feats,
            **pack_feats,
            **race_relative_feats,
            **cross_distance_feats,
            **venue_feats,
            **density_feats,
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
        et = pd.to_numeric(features_df["event_tier"], errors="coerce")
        aat = pd.to_numeric(features_df["athlete_avg_tier"], errors="coerce")
        features_df["tier_delta"] = (et - aat).fillna(0.0)

        # tier_step_up: 1 if racing at a HIGHER tier (lower number) than athlete's median
        # Captures "Continental athlete stepping up to World Cup" penalty
        features_df["tier_step_up"] = (et < aat).astype(int)
    else:
        features_df["tier_delta"] = 0.0
        features_df["tier_step_up"] = 0

    # n_races_at_tier: how many prior races the athlete has at this event's tier level
    # Athletes with 0 races at this tier have less predictable performance
    if "event_tier" in features_df.columns:
        event_tier_val = features_df["event_tier"].iloc[0] if len(features_df) > 0 else 2
        tier_col_map = {1: "races_tier1_24m", 2: "races_tier2_24m"}
        # For tier 3+4, use races_tier34_24m
        tier_races_col = tier_col_map.get(event_tier_val, "races_tier34_24m")
        if tier_races_col in features_df.columns:
            features_df["n_races_at_tier"] = pd.to_numeric(
                features_df[tier_races_col], errors="coerce"
            ).fillna(0).astype(int)
        else:
            features_df["n_races_at_tier"] = 0
    else:
        features_df["n_races_at_tier"] = 0

    # Add field-context features
    field_context_rows = []
    for _, row in features_df.iterrows():
        ctx = compute_field_context_features(features_df, row["athlete_id"])
        field_context_rows.append(ctx)

    field_ctx_df = pd.DataFrame(field_context_rows)
    features_df = pd.concat([features_df.reset_index(drop=True), field_ctx_df], axis=1)

    # ---- Head-to-Head Features ----
    # Compute pairwise win rates against the specific field of competitors.
    # This is a field-level feature (like seed_total_rank) — captures "how does
    # this athlete historically perform against THESE specific opponents?"
    all_athlete_ids = features_df["athlete_id"].dropna().astype(int).tolist()
    try:
        h2h_df = fetch_head_to_head_results(engine, all_athlete_ids, event_date)
        h2h_features = compute_head_to_head_features(h2h_df, all_athlete_ids, event_date=event_date)
        h2h_rows = [h2h_features.get(int(aid), {}) for aid in features_df["athlete_id"]]
        h2h_features_df = pd.DataFrame(h2h_rows)
        features_df = pd.concat([features_df.reset_index(drop=True), h2h_features_df], axis=1)
    except Exception as e:
        logger.warning(f"Could not compute h2h features: {e}")
        for col in ["h2h_win_rate", "h2h_avg_position_gap", "h2h_n_shared_races", "h2h_n_opponents_faced"]:
            features_df[col] = 0.5 if col == "h2h_win_rate" else 0

    # ---- Field-Relative Features ----
    # Encode where each athlete sits relative to THIS specific field.
    # Tree models can't easily learn "my ELO is high relative to the field" from
    # absolute ELO + n_entrants. Pre-computing ranks/percentiles helps ranking.
    if "elo_rating" in features_df.columns:
        elo_vals = pd.to_numeric(features_df["elo_rating"], errors="coerce")
        features_df["elo_rank_in_field"] = elo_vals.rank(method="min", ascending=False)  # 1 = highest ELO
        n = len(features_df)
        features_df["elo_pct_in_field"] = (features_df["elo_rank_in_field"] - 1) / max(n - 1, 1)  # 0 = best
    else:
        features_df["elo_rank_in_field"] = None
        features_df["elo_pct_in_field"] = None

    if "wt_rank_position" in features_df.columns:
        wt_vals = pd.to_numeric(features_df["wt_rank_position"], errors="coerce")
        features_df["wt_rank_in_field"] = wt_vals.rank(method="min", ascending=True)  # 1 = best WT rank
        n = len(features_df)
        features_df["wt_rank_pct_in_field"] = (features_df["wt_rank_in_field"] - 1) / max(n - 1, 1)
    else:
        features_df["wt_rank_in_field"] = None
        features_df["wt_rank_pct_in_field"] = None

    logger.info(f"Built features for {len(features_df)} athletes in {key}")
    return features_df


# Pack dynamics features used in split models.
# These capture historical pack positioning and gaps, which the bike/run models
# use to predict realized times (including drafting effects). Removing them
# produces "ability-only" split models where the simulation adds all pack effects.
PACK_DYNAMIC_FEATURES = [
    "ema_swim_pos_pct_7",
    "ema_bike_pos_pct_7",
    "ema_swim_pack_7",
    "ema_bike_pack_7",
    "ema_swim_gap_sec_7",
    "ema_bike_gap_sec_7",
]


def get_split_feature_columns(exclude_pack: bool = False) -> list[str]:
    """
    Return the reduced feature set for distance-specific split models.

    Split models predict absolute seconds (swim_sec, bike_sec, run_sec) for
    individual athletes. They should only use features that reflect the
    athlete's own ability and form — NOT field-context or ranking features
    which cause the model to learn "big field → fast times" shortcuts instead
    of "this athlete runs X:XX".

    Args:
        exclude_pack: If True, removes pack dynamics features (ema_swim_pack_7,
                     ema_bike_pack_7, ema_bike_gap_sec_7, etc.) to produce
                     "ability-only" models. The simulation then adds all pack
                     effects, avoiding double-counting.

    Excluded from the full feature set:
    - n_entrants, seed_total_rank: field composition doesn't cause faster times
    - elo_rating, elo_peak, elo_races: ranking proxy, causes inverse correlation
    - wt_rank_position, wt_total_points, wt_continental_*: ranking features
    - event_tier: event prestige doesn't change individual split times
    - distance_category_encoded: constant within each distance model
    - temperature_air, humidity, wbgt, is_hot: event-level weather (causes model to
      learn "hot race → slow" instead of athlete ability)
    - bike_laps, run_laps: course configuration (event-level, not athlete-level)
    - tier_delta: event context (event_tier - athlete_avg_tier)
    """
    cols = [
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
        # Per-split variance (athlete consistency — raw seconds, for model features)
        "std_swim_sec_24m",
        "std_bike_sec_24m",
        "std_run_sec_24m",
        # Normalized variance (course-agnostic — std of percentile rank)
        "std_swim_pct_24m",
        "std_bike_pct_24m",
        "std_run_pct_24m",
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
        # Long-term swim baseline + trend (v52: swim accuracy improvement)
        "ema_swim_split_pct_12",      # 12-race EMA of swim percentile (baseline)
        "best_swim_pct_12m",          # Best swim percentile in last 12 months
        "swim_form_trend",            # pct_5 - pct_12 (negative = swim improving)
        # Swim gap-to-leader (v53: course-normalized swim ability)
        "ema_swim_behind_sec_5",      # 5-race EMA of gap-to-swim-leader (seconds)
        "ema_swim_behind_sec_12",     # 12-race EMA (baseline gap)
        "best_swim_behind_12m",       # Best gap-to-leader in 12 months
        # Cold-start indicators (v52: let model learn to be uncertain about new athletes)
        "n_swim_results",             # Count of valid swim results
        "is_cold_start",              # Binary: 1 if <3 prior races
        # Tier-conditioned split percentiles (v36): split performance at elite level
        "ema_swim_split_pct_tier12_5",
        "ema_bike_split_pct_tier12_5",
        "ema_run_split_pct_tier12_5",
        # Cross-distance form
        "ema_finish_pct_sprint",
        "ema_finish_pct_standard",
        # Competition level
        "athlete_t1t2_rate",
        # Tier-conditioned performance (athlete-level, not event-level)
        "ema_finish_pct_tier1",
        "ema_finish_pct_tier2",
        "ema_finish_pct_tier34",
        "n_races_at_tier",
        "races_tier1_24m",
        "races_tier2_24m",
        # Form momentum
        "form_momentum",
        # Pack dynamics (affect bike/run interaction)
        "ema_swim_pos_pct_7",
        "ema_bike_pos_pct_7",
        "ema_swim_pack_7",
        "ema_bike_pack_7",
        "ema_swim_gap_sec_7",
        "ema_bike_gap_sec_7",
        # Swim gap consistency (pack formation signal)
        "std_swim_gap_sec_24m",
        "min_swim_gap_sec_24m",
        # Legacy pack features (v37): computed but previously unused.
        # front_pack_rate has -0.33 correlation with finish position;
        # 76% of top-10 come from front pack in breakaway races.
        "front_pack_rate",
        "avg_swim_gap_leader",
        "p90_swim_gap_leader",
        "bike_pack_rate",
        "avg_bike_gap_leader",
        # Field context for split models (v37): split models need to know
        # competition depth — a 0.10 swim_pct means very different things
        # in a WTCS field vs Continental Cup field.
        "field_depth_top10_mean",
        "seed_total_gap_to_best",
        # Race conditions
        "wetsuit",
        # NOTE: wind_speed_kmh REMOVED from split features (v34). While wind physically
        # affects bike time, it's event-level (identical for all athletes) and was dominating
        # all split models (#1 importance 918-1929), crowding out athlete-ability features.
        # Wind impact is handled via sigma adjustments in estimate_uncertainty() instead.
        # Freshness (athlete-level)
        "log1p_days_since",
        "freshness_curve",
        # Transition skill (position changes — athlete-level)
        "ema_t1_rank_pct_5",
        "ema_t2_rank_pct_5",
        "ema_swim_to_t1_pos_change_5",
        "ema_bike_to_t2_pos_change_5",
        # Distance match quality
        "ema_distance_match",
        "n_matched_races",
    ]

    if exclude_pack:
        cols = [c for c in cols if c not in PACK_DYNAMIC_FEATURES]

    return cols


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
        # Per-split variance (athlete consistency)
        "std_swim_sec_24m",
        "std_bike_sec_24m",
        "std_run_sec_24m",
        # Normalized variance (course-agnostic — std of percentile rank)
        "std_swim_pct_24m",
        "std_bike_pct_24m",
        "std_run_pct_24m",
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
        # Long-term swim baseline + trend (v52)
        "ema_swim_split_pct_12",        # 12-race EMA of swim percentile
        "best_swim_pct_12m",            # Best swim percentile in last 12 months
        "swim_form_trend",              # pct_5 - pct_12 (negative = improving)
        # Swim gap-to-leader (v53: course-normalized swim ability)
        "ema_swim_behind_sec_5",        # 5-race EMA of gap-to-swim-leader (seconds)
        "ema_swim_behind_sec_12",       # 12-race EMA (baseline gap)
        "best_swim_behind_12m",         # Best gap-to-leader in 12 months
        # Cold-start indicators (v52)
        "n_swim_results",               # Count of valid swim results
        "is_cold_start",                # Binary: 1 if <3 prior races
        # Tier-conditioned split percentiles (v36)
        "ema_swim_split_pct_tier12_5",  # Swim split pct at WTCS/World Cup
        "ema_bike_split_pct_tier12_5",  # Bike split pct at WTCS/World Cup
        "ema_run_split_pct_tier12_5",   # Run split pct at WTCS/World Cup
        # Cross-distance features
        "ema_finish_pct_sprint",    # EMA of finish percentile in sprint races
        "ema_finish_pct_standard",  # EMA of finish percentile in standard races
        # Athlete competition level features
        "athlete_t1t2_rate",
        "tier_delta",  # event_tier - athlete_avg_tier (positive = athlete stepping down)
        # "tier_step_up" removed v38: zero importance (tier_delta captures this continuously)
        "n_races_at_tier",  # Prior races at this event's tier level (24m window)
        # Tier-conditioned performance (how athlete performs at each tier level)
        "ema_finish_pct_tier1",   # EMA finish pct at WTCS/Championship level
        "ema_finish_pct_tier2",   # EMA finish pct at World Cup level
        "ema_finish_pct_tier34",  # EMA finish pct at Continental/Other level
        "races_tier1_24m",        # Count of Tier 1 races in 24 months
        "races_tier2_24m",        # Count of Tier 2 races in 24 months
        "races_tier34_24m",       # Count of Tier 3+4 races in 24 months
        # Form momentum
        "form_momentum",          # ema_pct_3 - ema_pct_5 (recent acceleration)
        # Pack features - EMA based (span=7)
        "ema_swim_pos_pct_7",
        "ema_bike_pos_pct_7",
        "ema_swim_pack_7",
        "ema_bike_pack_7",
        "ema_swim_gap_sec_7",
        "ema_bike_gap_sec_7",
        # Swim gap consistency (pack formation signal)
        "std_swim_gap_sec_24m",
        "min_swim_gap_sec_24m",
        # Legacy pack features (v37)
        "front_pack_rate",
        "avg_swim_gap_leader",
        "p90_swim_gap_leader",
        "bike_pack_rate",
        "avg_bike_gap_leader",
        # Elo rating features
        "elo_rating",               # Current Elo rating (higher = better)
        "elo_peak",                 # All-time peak Elo rating
        "elo_races",                # Number of races in Elo history
        # World Triathlon ranking features
        "wt_rank_position",         # WT world ranking position (1 = best)
        "wt_total_points",          # WT world ranking points
        # "wt_continental_rank/points" removed v38: zero importance (wt_rank_position captures this)
        # Field context
        "seed_total_rank",
        "seed_total_gap_to_best",
        "field_depth_top10_mean",
        "n_entrants",
        # Event context
        "event_tier",
        "distance_category_encoded",  # Integer encoding of prog_distance_category
        "wetsuit",                    # 1 = wetsuit race, 0 = non-wetsuit
        # Weather features (Phase 4C)
        "temperature_air",            # Air temperature in °C (numeric)
        "humidity",                   # Humidity percentage
        "wbgt",                       # Wet Bulb Globe Temperature (heat stress)
        # "is_hot" removed v38: zero importance in percentile model (redundant with wbgt)
        # Open-Meteo enriched weather
        "wind_speed_kmh",             # Average wind speed during race (km/h)
        "wind_gust_kmh",              # Max wind gust during race (km/h)
        "apparent_temp",              # Feels-like temperature (°C)
        "precipitation_mm",           # Total precipitation during race (mm)
        "cloud_cover_pct",            # Average cloud cover (%)
        # Course configuration (Phase 4D)
        "bike_laps",                  # Number of bike laps (more laps → more pack dynamics)
        "run_laps",                   # Number of run laps
        # Freshness features (Phase 4A)
        # "log1p_days_since" removed v38: zero importance (redundant with days_since_last_race)
        "freshness_curve",            # abs(days_since - 21), U-shape around optimal ~3 weeks
        # Transition skill features (Phase 4B)
        "ema_t1_rank_pct_5",         # T1 ranking as percentile (0=fastest)
        "ema_t2_rank_pct_5",         # T2 ranking as percentile
        "ema_swim_to_t1_pos_change_5",  # Positions gained/lost in T1 (neg=gained)
        "ema_bike_to_t2_pos_change_5",  # Positions gained/lost in T2
        # Distance match quality
        "ema_distance_match",         # 1 = EMA from matching distance, 0 = fallback
        "n_matched_races",            # Number of plausible matching-distance races
        # Head-to-head features (v38): pairwise records against field opponents
        "h2h_win_rate",               # Fraction of matchups won vs current field (0.5 = neutral)
        "h2h_avg_position_gap",       # Mean(opponent_pos - athlete_pos); positive = beats opponents
        "h2h_n_shared_races",         # Number of shared races with any field opponent
        "h2h_n_opponents_faced",      # How many field opponents they've raced against
        # NOTE: field-relative features (elo_rank_in_field etc.) tested in v40b
        # but regressed P@10 from 74.2% to 72.6% — multicollinear with seed_total_rank
        # NOTE: venue features (has_raced_venue, n_venue_races, venue_finish_pct_mean,
        # venue_delta), races_30d/60d, and days_since_2nd_last tested in v44/v44b
        # but regressed P@10 by -0.8 to -0.9%. All removed.
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
        # Per-split variance defaults (conservative = high variance)
        "std_swim_sec_24m": 50.0,   # ~sprint learned sigma
        "std_bike_sec_24m": 130.0,  # ~sprint learned sigma
        "std_run_sec_24m": 80.0,    # ~sprint learned sigma
        # Normalized percentile variance (course-agnostic, for MC sigma)
        "std_swim_pct_24m": 0.12,   # Moderate variability (~12% of field swing)
        "std_bike_pct_24m": 0.12,
        "std_run_pct_24m": 0.12,
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
        # Tier-conditioned split percentiles (v36)
        "ema_swim_split_pct_tier12_5": 0.70,  # Conservative for unknown at elite level
        "ema_bike_split_pct_tier12_5": 0.70,
        "ema_run_split_pct_tier12_5": 0.70,
        # Cross-distance features
        "ema_finish_pct_sprint": 0.65,
        "ema_finish_pct_standard": 0.65,
        # Athlete competition level features
        "athlete_t1t2_rate": 0.0,
        "tier_delta": 0.0,  # Neutral (racing at typical level)
        "tier_step_up": 0,  # Not stepping up by default
        "n_races_at_tier": 0,  # No prior races at this tier
        # Tier-conditioned performance
        "ema_finish_pct_tier1": 0.70,   # Conservative for unknown at WTCS
        "ema_finish_pct_tier2": 0.65,   # Slightly better at World Cup
        "ema_finish_pct_tier34": 0.60,  # Better at Continental (weaker fields)
        "races_tier1_24m": 0,
        "races_tier2_24m": 0,
        "races_tier34_24m": 0,
        # Form momentum
        "form_momentum": 0.0,  # Neutral (no recent acceleration)
        # EMA pack features (span=7)
        "ema_swim_pos_pct_7": 0.65,  # Below mid-pack
        "ema_bike_pos_pct_7": 0.65,
        "ema_swim_pack_7": 4.0,  # Pack 4 = back of field
        "ema_bike_pack_7": 4.0,
        "ema_swim_gap_sec_7": 40.0,  # 40 sec gap (back of field)
        "ema_bike_gap_sec_7": 80.0,  # 80 sec gap
        # Swim gap consistency
        "std_swim_gap_sec_24m": 20.0,   # Moderate gap variance
        "min_swim_gap_sec_24m": 15.0,   # 15s best gap (mid-field)
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
        "seed_total_gap_to_best": 0.30,  # 30 percentile points behind best seed
        "field_depth_top10_mean": 0.20,  # Moderate field depth
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
        # Pack features (v37: legacy pack features now in feature columns)
        "front_pack_rate": 0.1,  # Conservative: rarely in front pack
        "avg_swim_gap_leader": 30.0,  # 30s behind leader (back of field)
        "p90_swim_gap_leader": 50.0,  # Worst swim exit gap
        "bike_pack_rate": 0.1,  # Rarely in front bike pack
        "avg_bike_gap_leader": 60.0,  # 60s behind on bike
        # Phase 4A: Freshness features
        "log1p_days_since": float(np.log1p(90)),  # Matches days_since default of 90
        "freshness_curve": abs(90 - 21),           # 69 days from optimal
        # Phase 4B: Transition skill features
        "ema_t1_rank_pct_5": 0.65,   # Below mid-field
        "ema_t2_rank_pct_5": 0.65,
        "ema_swim_to_t1_pos_change_5": 0.0,  # Neutral (no gain/loss)
        "ema_bike_to_t2_pos_change_5": 0.0,
        # Phase 4C: Weather features
        "temperature_air": 22.0,   # Mild default
        "humidity": 60.0,          # Moderate default
        "wbgt": 20.0,              # Below heat threshold
        "is_hot": 0,               # Not hot by default
        # Open-Meteo enriched weather
        "wind_speed_kmh": 10.0,    # Light breeze default
        "wind_gust_kmh": 20.0,     # Moderate gust default
        "apparent_temp": 22.0,     # Same as temperature_air default
        "precipitation_mm": 0.0,   # Dry default
        "cloud_cover_pct": 50.0,   # Partly cloudy default
        # Phase 4D: Course laps
        "bike_laps": 0.0,          # Unknown
        "run_laps": 0.0,
        # Head-to-head features (v38)
        "h2h_win_rate": 0.5,               # Neutral: assume 50/50
        "h2h_avg_position_gap": 0.0,       # No gap info
        "h2h_n_shared_races": 0,           # No shared history
        "h2h_n_opponents_faced": 0,        # Haven't faced any opponents
        # Venue history features (Phase 10B-5)
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
