"""Relative Bike Performance Metrics within Packs.

Purpose:
    Quantify bike performance relative to pack mates (athletes who started bike together)
    rather than overall field position. Provides two complementary analyses:
    
    1. Pack-relative bike performance: Compare bike exit position to swim-pack position
    2. First bike split analysis: Speed/rank in the critical T1→first timing mat segment

Key Concepts:
    - Swim Pack: Athletes grouped together exiting the swim (chain-rule, 2s gap threshold)
    - Pack Position: Athlete's rank within their pack (1 = pack leader)
    - Pack Delta: Change in pack position from swim exit to bike exit (negative = gained places)
    - First Bike Split: Time from T1 to first bike timing mat (B1T1 column in detailed results)

Usage:
    from tri_analysis.relative_bike_metrics import (
        load_detailed_results,
        compute_first_bike_split_metrics,
        compute_pack_relative_bike_metrics,
    )
    
    # Load detailed results Excel file
    df = load_detailed_results("data/Detailed results Abu Dhabi2025.xlsx")
    
    # Compute metrics
    first_split = compute_first_bike_split_metrics(df)
    pack_metrics = compute_pack_relative_bike_metrics(engine, event_id, prog_id)

Author: AI Assistant
Created: 2026-01-30
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from tri_analysis.time_utils import time_to_seconds


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default pack threshold in seconds (consistent with wtcs_pack_metrics.py)
DEFAULT_PACK_GAP_THRESHOLD_SEC = 2

# Two-threshold pack parameters
DEFAULT_INITIAL_GAP_THRESHOLD_SEC = 2  # Leader to 2nd person
DEFAULT_CONTINUATION_GAP_THRESHOLD_SEC = 1  # 3rd+ person to previous

# Column name patterns for first bike split detection
FIRST_BIKE_SPLIT_PATTERNS = [
    r"^B1T1$",      # Most common: Bike lap 1, timing point 1
    r"^B1T\d$",     # Variant: B1T2, etc.
    r"^BL1$",       # Bike Lap 1 (time, not cumulative)
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FirstBikeSplitMetrics:
    """Metrics for the first bike split segment (T1 → first bike mat)."""
    
    athlete_name: str
    nationality: str
    bib: Optional[int] = None
    
    # Raw times (seconds)
    swim_time_sec: Optional[int] = None
    t1_time_sec: Optional[int] = None
    first_bike_split_cumulative_sec: Optional[int] = None  # B1T1 (from race start)
    first_bike_segment_sec: Optional[int] = None  # T1→first mat only
    
    # Rankings
    swim_exit_position: Optional[int] = None  # Position after swim
    t1_exit_position: Optional[int] = None    # Position after T1
    first_bike_position: Optional[int] = None  # Position at first bike mat
    first_bike_segment_rank: Optional[int] = None  # Rank for T1→first mat segment time
    
    # Pack info (assigned externally)
    swim_pack_id: Optional[int] = None
    first_bike_pack_id: Optional[int] = None
    first_bike_rank_in_swim_pack: Optional[int] = None


@dataclass
class PackRelativeBikeMetrics:
    """Metrics comparing bike performance relative to swim-pack mates."""
    
    athlete_id: int
    athlete_name: str
    event_id: int
    prog_id: int
    
    # Swim pack info
    swim_pack_id: int
    swim_pack_size: int
    swim_pack_position: int  # Position within swim pack (1 = pack leader)
    
    # Bike exit info (among swim pack mates only)
    bike_exit_pack_position: int  # Position at bike exit among swim-pack mates
    bike_pack_delta: int  # Change in position (negative = gained places)
    bike_pack_pct: float  # Percentile within pack at bike (0% = front, 100% = back)
    
    # Overall context
    swim_overall_position: int
    bike_overall_position: int
    overall_delta: int


@dataclass
class AthletePackComparison:
    """Full comparison of an athlete's bike performance vs pack mates."""
    
    athlete_id: int
    athlete_name: str
    event_id: int
    prog_id: int
    event_name: str = ""
    event_date: str = ""
    
    # Pack context
    swim_pack_id: int = 0
    swim_pack_members: List[str] = field(default_factory=list)
    swim_pack_size: int = 0
    
    # Relative metrics
    swim_position_in_pack: int = 0
    bike_position_in_pack: int = 0
    pack_delta: int = 0  # negative = gained places on pack mates
    
    # Detailed first split (if available from Excel)
    first_bike_segment_rank_in_pack: Optional[int] = None
    first_bike_segment_sec: Optional[int] = None
    
    # Overall positions for context
    swim_overall_position: int = 0
    bike_overall_position: int = 0


# ---------------------------------------------------------------------------
# Excel Parsing Functions
# ---------------------------------------------------------------------------

def find_first_bike_split_column(columns: List[str]) -> Optional[str]:
    """Find the first bike split column in a list of column names.
    
    Returns the column name (e.g., 'B1T1') or None if not found.
    Skips 'Bib' and 'Bike' columns.
    """
    for col in columns:
        col_str = str(col).strip()
        if col_str in ("Bib", "Bike"):
            continue
        for pattern in FIRST_BIKE_SPLIT_PATTERNS:
            if re.match(pattern, col_str, re.IGNORECASE):
                return col_str
    
    # Fallback: find first column starting with 'B' that isn't Bib/Bike
    for col in columns:
        col_str = str(col).strip()
        if col_str.upper().startswith("B") and col_str not in ("Bib", "Bike"):
            return col_str
    
    return None


def load_detailed_results(
    filepath: str | Path,
    *,
    sheet_name: Optional[str] = None,
    gender: Optional[str] = None,
) -> pd.DataFrame:
    """Load detailed results Excel file and normalize column names.
    
    Args:
        filepath: Path to Excel file
        sheet_name: Specific sheet to load (e.g., 'Men', 'Women'). If None, auto-detect.
        gender: If sheet_name is None, use this to find matching sheet ('male'/'female')
    
    Returns DataFrame with standardized columns:
        - rank, bib, name, nat (always present)
        - s1_sec, t1_sec, first_bike_split_sec (converted to seconds)
        - swim_sec, bike_sec, run_sec, total_sec (segment totals)
        - first_bike_col (name of original first bike column)
        - source_sheet (which sheet the data came from)
    """
    xl = pd.ExcelFile(filepath)
    available_sheets = xl.sheet_names
    
    # Determine which sheet to load
    if sheet_name:
        target_sheet = sheet_name
    elif gender:
        # Try to match gender to sheet name
        gender_lower = gender.lower()
        target_sheet = None
        for s in available_sheets:
            s_lower = s.lower()
            if gender_lower in ('male', 'm', 'men'):
                if 'men' in s_lower or s_lower == 'm' or 'u23 m' in s_lower:
                    target_sheet = s
                    break
            elif gender_lower in ('female', 'f', 'women'):
                if 'women' in s_lower or s_lower == 'w' or 'u23 w' in s_lower:
                    target_sheet = s
                    break
        if not target_sheet:
            target_sheet = available_sheets[0]  # Fallback to first sheet
    else:
        target_sheet = available_sheets[0]  # Default to first sheet
    
    df = pd.read_excel(filepath, sheet_name=target_sheet)
    
    # Normalize column names to lowercase
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)
    
    # Store which sheet was loaded
    df.attrs["source_sheet"] = target_sheet
    
    # Handle Gender column if present (Hamburg has it)
    if "gender" in df.columns:
        # Keep gender column but don't require it
        pass
    
    # Find first bike split column
    first_bike_col = find_first_bike_split_column(list(df.columns))
    if first_bike_col:
        first_bike_col = first_bike_col.lower()
    
    # Convert time columns to seconds
    time_cols_to_convert = ["s1", "t1", "swim", "bike", "run", "total"]
    if first_bike_col:
        time_cols_to_convert.append(first_bike_col)
    
    for col in time_cols_to_convert:
        if col in df.columns:
            df[f"{col}_sec"] = df[col].apply(time_to_seconds)
    
    # Compute cumulative times for position tracking
    if "s1_sec" in df.columns and "t1_sec" in df.columns:
        df["swim_exit_cumulative_sec"] = df["s1_sec"]
        df["t1_exit_cumulative_sec"] = df["s1_sec"] + df["t1_sec"]
    
    # First bike split segment time (T1 exit to first bike mat)
    # NOTE: B1T1 in detailed results is already the SEGMENT time (T1→first mat), not cumulative
    if first_bike_col and f"{first_bike_col}_sec" in df.columns:
        # B1T1 is the segment time directly
        df["first_bike_segment_sec"] = df[f"{first_bike_col}_sec"]
        # Compute cumulative time to first bike mat (swim + T1 + first bike segment)
        if "t1_exit_cumulative_sec" in df.columns:
            df["first_bike_cumulative_sec"] = df["t1_exit_cumulative_sec"] + df[f"{first_bike_col}_sec"]
    
    # Store metadata
    df.attrs["first_bike_col"] = first_bike_col
    df.attrs["source_file"] = str(filepath)
    
    return df


def compute_positions_at_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Add position columns at each checkpoint based on cumulative times.
    
    Adds columns:
        - swim_exit_position: Position after swim
        - t1_exit_position: Position after T1
        - first_bike_position: Position at first bike mat
        - first_bike_segment_rank: Rank for T1→first mat segment time only
    """
    result = df.copy()
    
    # Swim exit position (lower cumulative time = better position)
    if "swim_exit_cumulative_sec" in result.columns:
        result["swim_exit_position"] = result["swim_exit_cumulative_sec"].rank(
            method="min", na_option="bottom"
        ).astype("Int64")
    
    # T1 exit position
    if "t1_exit_cumulative_sec" in result.columns:
        result["t1_exit_position"] = result["t1_exit_cumulative_sec"].rank(
            method="min", na_option="bottom"
        ).astype("Int64")
    
    # First bike mat position (cumulative)
    if "first_bike_cumulative_sec" in result.columns:
        result["first_bike_position"] = result["first_bike_cumulative_sec"].rank(
            method="min", na_option="bottom"
        ).astype("Int64")
    
    # First bike segment rank (pure segment speed, ignoring prior position)
    if "first_bike_segment_sec" in result.columns:
        result["first_bike_segment_rank"] = result["first_bike_segment_sec"].rank(
            method="min", na_option="bottom"
        ).astype("Int64")
    
    return result


# ---------------------------------------------------------------------------
# Pack Assignment Functions
# ---------------------------------------------------------------------------

def assign_packs_chain_rule(
    elapsed_times_sec: pd.Series,
    *,
    max_gap_sec: int = DEFAULT_PACK_GAP_THRESHOLD_SEC,
) -> pd.Series:
    """Assign pack IDs using simple chain rule (gap to previous athlete).
    
    NOTE: This is the legacy single-threshold method. For more accurate pack
    detection, use assign_packs_two_threshold() instead.
    
    Args:
        elapsed_times_sec: Series of elapsed times in seconds (with original index)
        max_gap_sec: Maximum gap to stay in same pack (default 2s)
    
    Returns:
        Series of pack IDs (0-indexed) with same index as input
    """
    # Sort by elapsed time while preserving index
    sorted_df = pd.DataFrame({
        "elapsed": elapsed_times_sec,
        "orig_idx": elapsed_times_sec.index,
    }).dropna(subset=["elapsed"]).sort_values("elapsed")
    
    if sorted_df.empty:
        return pd.Series(dtype="Int64")
    
    # Assign pack IDs
    pack_ids = [0]
    current_pack = 0
    prev_time = sorted_df["elapsed"].iloc[0]
    
    for i in range(1, len(sorted_df)):
        curr_time = sorted_df["elapsed"].iloc[i]
        if (curr_time - prev_time) > max_gap_sec:
            current_pack += 1
        pack_ids.append(current_pack)
        prev_time = curr_time
    
    sorted_df["pack_id"] = pack_ids
    
    # Return with original index
    result = pd.Series(index=sorted_df["orig_idx"], data=sorted_df["pack_id"].values)
    return result.astype("Int64")


def assign_packs_two_threshold(
    elapsed_times_sec: pd.Series,
    *,
    initial_gap_sec: int = DEFAULT_INITIAL_GAP_THRESHOLD_SEC,
    continuation_gap_sec: int = DEFAULT_CONTINUATION_GAP_THRESHOLD_SEC,
) -> pd.Series:
    """Assign pack IDs using two-threshold rule for realistic pack detection.
    
    This method accounts for the "noise" where someone goes slightly off the 
    front but isn't really in a breakaway:
    
    - 2nd person joins pack if gap to leader ≤ initial_gap_sec (default 2s)
    - 3rd+ person joins pack if gap to previous ≤ continuation_gap_sec (default 1s)
    - Otherwise, start a new pack
    
    Example with initial=2s, continuation=1s:
        10:00 → Pack 1 (leader)
        10:02 → Pack 1 (2s gap ≤ 2s initial threshold)
        10:03 → Pack 1 (1s gap ≤ 1s continuation)
        10:04 → Pack 1 (1s gap ≤ 1s continuation)
        10:05 → Pack 1 (1s gap ≤ 1s continuation)
        10:07 → Pack 2 (2s gap > 1s continuation → new pack leader)
        10:10 → Pack 3 (3s gap > 2s initial → new pack leader)
        10:11 → Pack 3 (1s gap ≤ 2s initial for 2nd position)
        10:13 → Pack 4 (2s gap > 1s continuation → new pack leader)
        10:15 → Pack 4 (2s gap ≤ 2s initial for 2nd position)
        10:16 → Pack 4 (1s gap ≤ 1s continuation)
    
    Args:
        elapsed_times_sec: Series of elapsed times in seconds (with original index)
        initial_gap_sec: Max gap from pack leader to 2nd person (default 2s)
        continuation_gap_sec: Max gap for 3rd+ person to stay in pack (default 1s)
    
    Returns:
        Series of pack IDs (0-indexed) with same index as input
    """
    # Sort by elapsed time while preserving index
    sorted_df = pd.DataFrame({
        "elapsed": elapsed_times_sec,
        "orig_idx": elapsed_times_sec.index,
    }).dropna(subset=["elapsed"]).sort_values("elapsed")
    
    if sorted_df.empty:
        return pd.Series(dtype="Int64")
    
    n = len(sorted_df)
    pack_ids = [0]  # First person is always pack 0 leader
    current_pack = 0
    position_in_pack = 1  # Leader is position 1
    
    for i in range(1, n):
        curr_time = sorted_df["elapsed"].iloc[i]
        prev_time = sorted_df["elapsed"].iloc[i - 1]
        gap = curr_time - prev_time
        
        # Determine which threshold applies
        if position_in_pack == 1:
            # 2nd person in pack: use initial threshold
            threshold = initial_gap_sec
        else:
            # 3rd+ person: use continuation threshold
            threshold = continuation_gap_sec
        
        if gap <= threshold:
            # Stay in current pack
            pack_ids.append(current_pack)
            position_in_pack += 1
        else:
            # Start new pack
            current_pack += 1
            pack_ids.append(current_pack)
            position_in_pack = 1  # This person is the new pack leader
    
    sorted_df["pack_id"] = pack_ids
    
    # Return with original index
    result = pd.Series(index=sorted_df["orig_idx"], data=sorted_df["pack_id"].values)
    return result.astype("Int64")


def compute_pack_positions(df: pd.DataFrame, pack_col: str, time_col: str) -> pd.DataFrame:
    """Compute position within pack based on elapsed time.
    
    Returns DataFrame with added '{pack_col}_position' column.
    """
    result = df.copy()
    pos_col = f"{pack_col}_position"
    
    # Rank within each pack by elapsed time
    result[pos_col] = result.groupby(pack_col)[time_col].rank(method="min").astype("Int64")
    
    return result


# ---------------------------------------------------------------------------
# First Bike Split Analysis
# ---------------------------------------------------------------------------

def compute_first_bike_split_metrics(
    df: pd.DataFrame,
    *,
    initial_gap_sec: int = DEFAULT_INITIAL_GAP_THRESHOLD_SEC,
    continuation_gap_sec: int = DEFAULT_CONTINUATION_GAP_THRESHOLD_SEC,
    use_two_threshold: bool = True,
) -> pd.DataFrame:
    """Compute first bike split metrics for all athletes in detailed results.
    
    Args:
        df: DataFrame from load_detailed_results()
        initial_gap_sec: Max gap from pack leader to 2nd person (default 2s)
        continuation_gap_sec: Max gap for 3rd+ person to stay in pack (default 1s)
        use_two_threshold: If True, use two-threshold method; otherwise single threshold
    
    Returns DataFrame with:
        - Original athlete info (name, nat, bib, rank)
        - Swim exit position and pack
        - T1 exit position
        - First bike mat position
        - First bike segment time and rank (T1→first mat)
        - Position at first bike mat relative to swim pack mates
    """
    # Ensure positions are computed
    work = compute_positions_at_checkpoints(df.copy())
    
    # Choose pack assignment function
    if use_two_threshold:
        def assign_packs(times):
            return assign_packs_two_threshold(
                times, 
                initial_gap_sec=initial_gap_sec, 
                continuation_gap_sec=continuation_gap_sec
            )
    else:
        def assign_packs(times):
            return assign_packs_chain_rule(times, max_gap_sec=initial_gap_sec)
    
    # Assign swim exit packs
    if "swim_exit_cumulative_sec" in work.columns:
        work["swim_pack_id"] = assign_packs(work["swim_exit_cumulative_sec"])
        work["swim_pack_size"] = work.groupby("swim_pack_id")["swim_pack_id"].transform("size")
        work = compute_pack_positions(work, "swim_pack_id", "swim_exit_cumulative_sec")
        work.rename(columns={"swim_pack_id_position": "swim_pack_position"}, inplace=True)
    
    # Assign first bike mat packs
    if "first_bike_cumulative_sec" in work.columns:
        work["first_bike_pack_id"] = assign_packs(work["first_bike_cumulative_sec"])
    
    # Compute rank in swim pack at first bike mat
    if "swim_pack_id" in work.columns and "first_bike_cumulative_sec" in work.columns:
        work["first_bike_rank_in_swim_pack"] = work.groupby("swim_pack_id")[
            "first_bike_cumulative_sec"
        ].rank(method="min").astype("Int64")
        
        # Compute segment rank within swim pack
        work["first_bike_segment_rank_in_swim_pack"] = work.groupby("swim_pack_id")[
            "first_bike_segment_sec"
        ].rank(method="min").astype("Int64")
    
    return work


# ---------------------------------------------------------------------------
# Database Pack-Relative Metrics
# ---------------------------------------------------------------------------

def fetch_pack_membership_for_event(
    engine: Engine,
    *,
    event_id: int,
    prog_id: int,
) -> pd.DataFrame:
    """Fetch pre-computed pack membership from database for an event.
    
    Returns DataFrame with columns:
        athlete_id, checkpoint, pack_id, pos_at_checkpoint, pack_size, etc.
    """
    # Convert numpy types to Python native types
    event_id = int(event_id)
    prog_id = int(prog_id)
    
    query = text("""
        SELECT 
            pm.athlete_id,
            a.full_name,
            pm.checkpoint,
            pm.pack_id,
            pm.pos_at_checkpoint,
            pm.pack_size,
            pm.gap_to_leader_sec,
            pm.elapsed_sec
        FROM wtcs_pack_membership pm
        JOIN athlete a ON pm.athlete_id = a.athlete_id
        WHERE pm.event_id = :event_id AND pm.prog_id = :prog_id
        ORDER BY pm.checkpoint, pm.pack_id, pm.pos_at_checkpoint
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"event_id": event_id, "prog_id": prog_id})
    
    return df


# Valid reference checkpoints for pack establishment
VALID_REFERENCE_CHECKPOINTS = ("swim", "t1", "first_bike")  # Where pack is established
# "swim" = swim exit (traditional, can be strict before packs form)
# "t1" = T1 exit (after transition, slightly more consolidated)
# "first_bike" = first bike timing mat B1T1 (packs have consolidated)


def compute_pack_relative_bike_metrics(
    engine: Engine,
    *,
    event_id: int,
    prog_id: int,
    reference_checkpoint: str = "swim",
    excel_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute pack-relative bike metrics using database pack membership.
    
    Compares each athlete's bike exit position to their reference checkpoint pack position.
    
    Args:
        engine: Database engine
        event_id: Event ID
        prog_id: Program ID
        reference_checkpoint: Which checkpoint defines the "starting pack"
            - "swim": Pack established at swim exit (traditional, can be strict)
            - "t1": Pack established at T1 exit (slightly more consolidated)
            - "first_bike": Pack established at first bike mat B1T1 (packs consolidated)
        excel_df: DataFrame from load_detailed_results() - required for "t1" and "first_bike" checkpoints
    
    Returns DataFrame with columns:
        athlete_id, full_name, ref_pack_id, ref_pack_size, ref_pack_position,
        bike_exit_pack_position, bike_pack_delta, bike_pack_pct,
        ref_overall_position, bike_overall_position, overall_delta
    """
    if reference_checkpoint not in VALID_REFERENCE_CHECKPOINTS:
        raise ValueError(f"reference_checkpoint must be one of {VALID_REFERENCE_CHECKPOINTS}")
    
    membership = fetch_pack_membership_for_event(engine, event_id=event_id, prog_id=prog_id)
    
    if membership.empty:
        return pd.DataFrame()
    
    # Get bike exit data (always from database)
    bike_df = membership[membership["checkpoint"] == "bike"].copy()
    if bike_df.empty:
        return pd.DataFrame()
    
    # Get reference checkpoint data
    if reference_checkpoint == "first_bike":
        # Use Excel data for first bike timing mat packs
        if excel_df is None or excel_df.empty:
            # Fall back to swim if no Excel data
            ref_df = membership[membership["checkpoint"] == "swim"].copy()
            if ref_df.empty:
                return pd.DataFrame()
        else:
            # Build reference data from Excel first_bike_cumulative_sec
            ref_df = _build_first_bike_reference_packs(excel_df, engine, event_id, prog_id)
            if ref_df.empty:
                # Fall back to swim
                ref_df = membership[membership["checkpoint"] == "swim"].copy()
                if ref_df.empty:
                    return pd.DataFrame()
    elif reference_checkpoint == "t1":
        # Use Excel data for T1 exit packs
        if excel_df is None or excel_df.empty:
            # Fall back to swim if no Excel data
            ref_df = membership[membership["checkpoint"] == "swim"].copy()
            if ref_df.empty:
                return pd.DataFrame()
        else:
            # Build reference data from Excel t1_exit_cumulative_sec
            ref_df = _build_t1_reference_packs(excel_df, engine, event_id, prog_id)
            if ref_df.empty:
                # Fall back to swim
                ref_df = membership[membership["checkpoint"] == "swim"].copy()
                if ref_df.empty:
                    return pd.DataFrame()
    else:
        # Use swim checkpoint from database
        ref_df = membership[membership["checkpoint"] == "swim"].copy()
        if ref_df.empty:
            return pd.DataFrame()
    
    # Rename columns for clarity (use generic "swim_" prefix for backward compatibility)
    ref_df = ref_df.rename(columns={
        "pack_id": "swim_pack_id",
        "pos_at_checkpoint": "swim_overall_position",
        "pack_size": "swim_pack_size",
        "elapsed_sec": "swim_elapsed_sec",
    })
    
    bike_df = bike_df.rename(columns={
        "pack_id": "bike_pack_id",
        "pos_at_checkpoint": "bike_overall_position",
        "elapsed_sec": "bike_elapsed_sec",
    })
    
    # Get reference pack position (rank within reference pack)
    ref_df["swim_pack_position"] = ref_df.groupby("swim_pack_id").cumcount() + 1
    
    # Merge reference and bike data
    merged = ref_df[["athlete_id", "full_name", "swim_pack_id", "swim_pack_size", 
                     "swim_pack_position", "swim_overall_position"]].merge(
        bike_df[["athlete_id", "bike_pack_id", "bike_overall_position", "bike_elapsed_sec"]],
        on="athlete_id",
        how="inner",
    )
    
    # Compute bike pack size and position within bike pack
    merged["bike_pack_size"] = merged.groupby("bike_pack_id")["bike_pack_id"].transform("size")
    merged["bike_pack_position"] = merged.groupby("bike_pack_id")[
        "bike_elapsed_sec"
    ].rank(method="min").astype(int)
    
    # For each athlete, compute their bike position among reference-pack mates
    # (rank by bike_elapsed_sec within swim_pack_id)
    merged["bike_exit_pack_position"] = merged.groupby("swim_pack_id")[
        "bike_elapsed_sec"
    ].rank(method="min").astype(int)
    
    # Compute pack delta and percentile
    merged["bike_pack_delta"] = merged["bike_exit_pack_position"] - merged["swim_pack_position"]
    merged["bike_pack_pct"] = (
        (merged["bike_exit_pack_position"] - 1) / (merged["swim_pack_size"] - 1) * 100
    ).fillna(0).round(1)
    
    # Overall delta
    merged["overall_delta"] = merged["bike_overall_position"] - merged["swim_overall_position"]
    
    # Add metadata about which checkpoint was used
    merged["reference_checkpoint"] = reference_checkpoint
    
    # Select and order columns
    result = merged[[
        "athlete_id", "full_name",
        "swim_pack_id", "swim_pack_size", "swim_pack_position",
        "bike_pack_id", "bike_pack_size", "bike_pack_position",
        "bike_exit_pack_position", "bike_pack_delta", "bike_pack_pct",
        "swim_overall_position", "bike_overall_position", "overall_delta",
        "reference_checkpoint",
    ]].copy()
    
    return result


def _build_first_bike_reference_packs(
    excel_df: pd.DataFrame,
    engine: Engine,
    event_id: int,
    prog_id: int,
) -> pd.DataFrame:
    """Build reference pack data from Excel first_bike_cumulative_sec.
    
    Matches Excel names to database athlete_ids and assigns packs based on
    first bike timing mat (B1T1) cumulative times.
    
    Returns DataFrame with columns matching database format:
        athlete_id, full_name, pack_id, pos_at_checkpoint, pack_size, elapsed_sec
    """
    # Ensure first bike metrics are computed
    if "first_bike_cumulative_sec" not in excel_df.columns:
        excel_df = compute_positions_at_checkpoints(excel_df)
    
    if "first_bike_cumulative_sec" not in excel_df.columns:
        return pd.DataFrame()
    
    # Fetch athlete mapping from database for this event
    query = text("""
        SELECT DISTINCT 
            a.athlete_id, 
            a.full_name,
            LOWER(TRIM(a.full_name)) as name_lower
        FROM race_results rr
        JOIN athlete a ON rr.athlete_id = a.athlete_id
        WHERE rr.event_id = :event_id AND rr.prog_id = :prog_id
    """)
    
    with engine.connect() as conn:
        athletes = pd.read_sql(query, conn, params={"event_id": event_id, "prog_id": prog_id})
    
    if athletes.empty:
        return pd.DataFrame()
    
    # Match Excel names to athlete_ids
    work = excel_df.copy()
    work["name_lower"] = work["name"].str.lower().str.strip()
    
    # Exact match first
    matched = work.merge(
        athletes[["athlete_id", "full_name", "name_lower"]],
        on="name_lower",
        how="left",
        suffixes=("", "_db"),
    )
    
    # Fuzzy match unmatched rows
    unmatched_mask = matched["athlete_id"].isna()
    if unmatched_mask.any():
        for idx in matched[unmatched_mask].index:
            excel_name = matched.loc[idx, "name_lower"]
            for _, athlete in athletes.iterrows():
                db_name = athlete["name_lower"]
                if excel_name in db_name or db_name in excel_name:
                    matched.loc[idx, "athlete_id"] = athlete["athlete_id"]
                    matched.loc[idx, "full_name_db"] = athlete["full_name"]
                    break
                # Try last name match
                if excel_name.split()[-1] in db_name or db_name.split()[-1] in excel_name:
                    matched.loc[idx, "athlete_id"] = athlete["athlete_id"]
                    matched.loc[idx, "full_name_db"] = athlete["full_name"]
                    break
    
    # Filter to matched athletes with valid times
    valid = matched[
        matched["athlete_id"].notna() & 
        matched["first_bike_cumulative_sec"].notna() &
        (matched["first_bike_cumulative_sec"] > 0)
    ].copy()
    
    if valid.empty:
        return pd.DataFrame()
    
    # Assign packs using two-threshold algorithm
    valid["pack_id"] = assign_packs_two_threshold(valid["first_bike_cumulative_sec"])
    
    # Compute pack sizes and positions
    valid["pack_size"] = valid.groupby("pack_id")["pack_id"].transform("size")
    valid["pos_at_checkpoint"] = valid["first_bike_cumulative_sec"].rank(method="min").astype(int)
    
    # Get full name (from DB match or original Excel name)
    # full_name column comes from exact merge, full_name_db from fuzzy match
    if "full_name" in valid.columns:
        full_names = valid["full_name"].fillna(valid["name"])
    elif "full_name_db" in valid.columns:
        full_names = valid["full_name_db"].fillna(valid["name"])
    else:
        full_names = valid["name"]
    
    # Build output DataFrame matching expected format
    result = pd.DataFrame({
        "athlete_id": valid["athlete_id"].astype(int),
        "full_name": full_names,
        "checkpoint": "first_bike",
        "pack_id": valid["pack_id"],
        "pos_at_checkpoint": valid["pos_at_checkpoint"],
        "pack_size": valid["pack_size"],
        "elapsed_sec": valid["first_bike_cumulative_sec"],
    })
    
    return result


def _build_t1_reference_packs(
    excel_df: pd.DataFrame,
    engine: Engine,
    event_id: int,
    prog_id: int,
) -> pd.DataFrame:
    """Build reference pack data from Excel t1_exit_cumulative_sec.
    
    Matches Excel names to database athlete_ids and assigns packs based on
    T1 exit cumulative times.
    
    Returns DataFrame with columns matching database format:
        athlete_id, full_name, pack_id, pos_at_checkpoint, pack_size, elapsed_sec
    """
    # Ensure T1 cumulative times are computed
    if "t1_exit_cumulative_sec" not in excel_df.columns:
        excel_df = compute_positions_at_checkpoints(excel_df)
    
    if "t1_exit_cumulative_sec" not in excel_df.columns:
        return pd.DataFrame()
    
    # Fetch athlete mapping from database for this event
    query = text("""
        SELECT DISTINCT 
            a.athlete_id, 
            a.full_name,
            LOWER(TRIM(a.full_name)) as name_lower
        FROM race_results rr
        JOIN athlete a ON rr.athlete_id = a.athlete_id
        WHERE rr.event_id = :event_id AND rr.prog_id = :prog_id
    """)
    
    with engine.connect() as conn:
        athletes = pd.read_sql(query, conn, params={"event_id": event_id, "prog_id": prog_id})
    
    if athletes.empty:
        return pd.DataFrame()
    
    # Match Excel names to athlete_ids
    work = excel_df.copy()
    work["name_lower"] = work["name"].str.lower().str.strip()
    
    # Exact match first
    matched = work.merge(
        athletes[["athlete_id", "full_name", "name_lower"]],
        on="name_lower",
        how="left",
        suffixes=("", "_db"),
    )
    
    # Fuzzy match unmatched rows
    unmatched_mask = matched["athlete_id"].isna()
    if unmatched_mask.any():
        for idx in matched[unmatched_mask].index:
            excel_name = matched.loc[idx, "name_lower"]
            for _, athlete in athletes.iterrows():
                db_name = athlete["name_lower"]
                if excel_name in db_name or db_name in excel_name:
                    matched.loc[idx, "athlete_id"] = athlete["athlete_id"]
                    matched.loc[idx, "full_name_db"] = athlete["full_name"]
                    break
                # Try last name match
                if excel_name.split()[-1] in db_name or db_name.split()[-1] in excel_name:
                    matched.loc[idx, "athlete_id"] = athlete["athlete_id"]
                    matched.loc[idx, "full_name_db"] = athlete["full_name"]
                    break
    
    # Filter to matched athletes with valid times
    valid = matched[
        matched["athlete_id"].notna() & 
        matched["t1_exit_cumulative_sec"].notna() &
        (matched["t1_exit_cumulative_sec"] > 0)
    ].copy()
    
    if valid.empty:
        return pd.DataFrame()
    
    # Assign packs using two-threshold algorithm
    valid["pack_id"] = assign_packs_two_threshold(valid["t1_exit_cumulative_sec"])
    
    # Compute pack sizes and positions
    valid["pack_size"] = valid.groupby("pack_id")["pack_id"].transform("size")
    valid["pos_at_checkpoint"] = valid["t1_exit_cumulative_sec"].rank(method="min").astype(int)
    
    # Get full name (from DB match or original Excel name)
    if "full_name" in valid.columns:
        full_names = valid["full_name"].fillna(valid["name"])
    elif "full_name_db" in valid.columns:
        full_names = valid["full_name_db"].fillna(valid["name"])
    else:
        full_names = valid["name"]
    
    # Build output DataFrame matching expected format
    result = pd.DataFrame({
        "athlete_id": valid["athlete_id"].astype(int),
        "full_name": full_names,
        "checkpoint": "t1",
        "pack_id": valid["pack_id"],
        "pos_at_checkpoint": valid["pos_at_checkpoint"],
        "pack_size": valid["pack_size"],
        "elapsed_sec": valid["t1_exit_cumulative_sec"],
    })
    
    return result


def get_pack_mates_for_athlete(
    engine: Engine,
    *,
    event_id: int,
    prog_id: int,
    athlete_id: int,
    checkpoint: str = "swim",
) -> pd.DataFrame:
    """Get all athletes in the same pack as a given athlete at a checkpoint.
    
    Useful for showing "who was in your swim pack" comparisons.
    """
    # First get the athlete's pack_id
    query_pack = text("""
        SELECT pack_id
        FROM wtcs_pack_membership
        WHERE event_id = :event_id AND prog_id = :prog_id 
              AND athlete_id = :athlete_id AND checkpoint = :checkpoint
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query_pack, {
            "event_id": event_id, 
            "prog_id": prog_id,
            "athlete_id": athlete_id,
            "checkpoint": checkpoint,
        }).fetchone()
    
    if not result:
        return pd.DataFrame()
    
    pack_id = result[0]
    
    # Get all pack mates
    query_mates = text("""
        SELECT 
            pm.athlete_id,
            a.full_name,
            a.country,
            pm.pos_at_checkpoint,
            pm.gap_to_leader_sec,
            pm.elapsed_sec
        FROM wtcs_pack_membership pm
        JOIN athlete a ON pm.athlete_id = a.athlete_id
        WHERE pm.event_id = :event_id AND pm.prog_id = :prog_id 
              AND pm.checkpoint = :checkpoint AND pm.pack_id = :pack_id
        ORDER BY pm.pos_at_checkpoint
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query_mates, conn, params={
            "event_id": event_id,
            "prog_id": prog_id,
            "checkpoint": checkpoint,
            "pack_id": pack_id,
        })
    
    return df


# ---------------------------------------------------------------------------
# Combined Analysis (Excel + Database)
# ---------------------------------------------------------------------------

def merge_excel_with_database_packs(
    excel_df: pd.DataFrame,
    engine: Engine,
    *,
    event_id: int,
    prog_id: int,
) -> pd.DataFrame:
    """Merge detailed Excel results with database pack membership.
    
    Uses fuzzy name matching to link Excel rows to athlete_id.
    
    Returns DataFrame with both first-split metrics and pack-relative metrics.
    """
    # Fetch athlete mapping from database for this event
    query = text("""
        SELECT DISTINCT 
            a.athlete_id, 
            a.full_name,
            LOWER(TRIM(a.full_name)) as name_lower,
            a.country
        FROM race_results rr
        JOIN athlete a ON rr.athlete_id = a.athlete_id
        WHERE rr.event_id = :event_id AND rr.prog_id = :prog_id
    """)
    
    with engine.connect() as conn:
        athletes = pd.read_sql(query, conn, params={"event_id": event_id, "prog_id": prog_id})
    
    if athletes.empty:
        return excel_df
    
    # Normalize Excel names for matching
    excel_df = excel_df.copy()
    excel_df["name_lower"] = excel_df["name"].str.lower().str.strip()
    
    # Attempt exact match first
    merged = excel_df.merge(
        athletes[["athlete_id", "full_name", "name_lower"]],
        on="name_lower",
        how="left",
        suffixes=("", "_db"),
    )
    
    # For unmatched rows, try fuzzy matching
    unmatched_mask = merged["athlete_id"].isna()
    if unmatched_mask.any():
        # Simple fuzzy: check if DB name is contained in Excel name or vice versa
        for idx in merged[unmatched_mask].index:
            excel_name = merged.loc[idx, "name_lower"]
            for _, athlete in athletes.iterrows():
                db_name = athlete["name_lower"]
                # Check containment both ways
                if excel_name in db_name or db_name in excel_name:
                    merged.loc[idx, "athlete_id"] = athlete["athlete_id"]
                    merged.loc[idx, "full_name_db"] = athlete["full_name"]
                    break
    
    # Now fetch pack-relative metrics and merge
    pack_metrics = compute_pack_relative_bike_metrics(
        engine, event_id=event_id, prog_id=prog_id
    )
    
    if not pack_metrics.empty and "athlete_id" in merged.columns:
        merged = merged.merge(
            pack_metrics[[
                "athlete_id", "swim_pack_id", "swim_pack_size", "swim_pack_position",
                "bike_exit_pack_position", "bike_pack_delta", "bike_pack_pct",
            ]],
            on="athlete_id",
            how="left",
        )
    
    return merged


# ---------------------------------------------------------------------------
# Summary/Export Functions
# ---------------------------------------------------------------------------

def summarize_athlete_bike_performance(
    engine: Engine,
    *,
    athlete_id: int,
    event_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Summarize an athlete's pack-relative bike performance across events.
    
    Returns DataFrame with one row per event showing:
        event_name, event_date, swim_pack_position, bike_exit_pack_position,
        bike_pack_delta, interpretation
    """
    # Get events for athlete
    event_filter = ""
    params: Dict = {"athlete_id": athlete_id}
    if event_ids:
        event_filter = "AND rr.event_id = ANY(:event_ids)"
        params["event_ids"] = event_ids
    
    query = text(f"""
        SELECT DISTINCT rr.event_id, rr.prog_id, e.event_name, e.event_date
        FROM race_results rr
        JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
        WHERE rr.athlete_id = :athlete_id {event_filter}
        ORDER BY e.event_date
    """)
    
    with engine.connect() as conn:
        events = pd.read_sql(query, conn, params=params)
    
    results = []
    for _, event in events.iterrows():
        metrics = compute_pack_relative_bike_metrics(
            engine, event_id=event["event_id"], prog_id=event["prog_id"]
        )
        
        athlete_metrics = metrics[metrics["athlete_id"] == athlete_id]
        if not athlete_metrics.empty:
            row = athlete_metrics.iloc[0].to_dict()
            row["event_name"] = event["event_name"]
            row["event_date"] = event["event_date"]
            
            # Add interpretation
            delta = row["bike_pack_delta"]
            if delta < -2:
                row["interpretation"] = "Strong bike - gained 3+ places on pack"
            elif delta < 0:
                row["interpretation"] = "Good bike - gained places on pack"
            elif delta == 0:
                row["interpretation"] = "Held position in pack"
            elif delta <= 2:
                row["interpretation"] = "Lost ground on pack"
            else:
                row["interpretation"] = "Struggled - lost 3+ places to pack"
            
            results.append(row)
    
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualization Helpers for Season Trend Charts
# ---------------------------------------------------------------------------

def build_season_bike_chart_data(
    engine: Engine,
    *,
    athlete_id: int,
    year: int = 2025,
    include_first_split: bool = True,
    detailed_results_folder: str = "data",
    reference_checkpoint: str = "swim",
) -> pd.DataFrame:
    """Build data for season bike performance chart (like PowerPoint slide).
    
    Args:
        engine: Database engine
        athlete_id: Athlete ID
        year: Season year (default 2025)
        include_first_split: Whether to include first bike split from Excel
        detailed_results_folder: Folder containing detailed results Excel files
        reference_checkpoint: Which checkpoint defines the "starting pack"
            - "swim": Pack established at swim exit (traditional, can be strict)
            - "first_bike": Pack established at first bike mat B1T1 (packs consolidated)
    
    Returns DataFrame with one row per event containing:
        - event_name, event_date (for x-axis labels)
        - bike_overall_position, gap_to_leader_sec (for the dual-axis chart)
        - swim_pack_id, swim_pack_size, swim_pack_position (reference pack info)
        - bike_exit_pack_position, bike_pack_delta (pack-relative performance)
        - first_bike_segment_sec, first_bike_segment_rank (if Excel available)
        
    Perfect for creating the WTCS Bike Positioning slide.
    """
    from pathlib import Path
    
    # Get athlete's WTCS events for the year
    query = text("""
        SELECT DISTINCT 
            rr.event_id, rr.prog_id, 
            e.event_name, e.event_date,
            rr.finish_position
        FROM race_results rr
        JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
        WHERE rr.athlete_id = :athlete_id
          AND EXTRACT(YEAR FROM e.event_date) = :year
          AND (e.event_name ILIKE '%Championship Series%' 
               OR e.event_name ILIKE '%Championship Finals%')
        ORDER BY e.event_date
    """)
    
    with engine.connect() as conn:
        events = pd.read_sql(query, conn, params={"athlete_id": int(athlete_id), "year": int(year)})
    
    if events.empty:
        return pd.DataFrame()
    
    results = []
    for _, event in events.iterrows():
        event_id = int(event["event_id"])
        prog_id = int(event["prog_id"])
        
        row = {
            "event_id": event_id,
            "prog_id": prog_id,
            "event_name": event["event_name"],
            "event_date": event["event_date"],
            "finish_position": event["finish_position"],
            "reference_checkpoint": reference_checkpoint,
        }
        
        # Load Excel data if needed (for first_bike checkpoint or first split metrics)
        excel_df = None
        athlete_gender = _get_athlete_gender(engine, athlete_id)
        excel_path = _find_matching_excel(event["event_name"], event["event_date"], detailed_results_folder)
        if excel_path:
            try:
                excel_df = load_detailed_results(excel_path, gender=athlete_gender)
                excel_df = compute_first_bike_split_metrics(excel_df)
            except Exception:
                excel_df = None
        
        # Get pack-relative metrics (pass Excel data for first_bike checkpoint)
        pack_metrics = compute_pack_relative_bike_metrics(
            engine, event_id=event_id, prog_id=prog_id,
            reference_checkpoint=reference_checkpoint,
            excel_df=excel_df
        )
        athlete_pack = pack_metrics[pack_metrics["athlete_id"] == athlete_id]
        
        if not athlete_pack.empty:
            pm = athlete_pack.iloc[0]
            row.update({
                "swim_pack_id": int(pm["swim_pack_id"]),
                "swim_pack_size": int(pm["swim_pack_size"]),
                "swim_pack_position": int(pm["swim_pack_position"]),
                "bike_pack_id": int(pm["bike_pack_id"]),
                "bike_pack_size": int(pm["bike_pack_size"]),
                "bike_pack_position": int(pm["bike_pack_position"]),
                "bike_exit_pack_position": int(pm["bike_exit_pack_position"]),
                "bike_pack_delta": int(pm["bike_pack_delta"]),
                "swim_overall_position": int(pm["swim_overall_position"]),
            })
        
        # Get position_at_bike and gap from position_metrics (authoritative for race position)
        pos_query = text("""
            SELECT position_at_bike, behindbike, position_at_swim
            FROM position_metrics 
            WHERE event_id = :event_id AND prog_id = :prog_id AND athlete_id = :athlete_id
        """)
        with engine.connect() as conn:
            pos_result = conn.execute(pos_query, {
                "event_id": event_id, "prog_id": prog_id, "athlete_id": int(athlete_id)
            }).fetchone()
        if pos_result:
            if pos_result[0] is not None:
                row["bike_overall_position"] = int(pos_result[0])
            if pos_result[1] is not None:
                row["gap_to_leader_sec"] = int(pos_result[1])
        
        # Add first bike split data from Excel (if available)
        if include_first_split and excel_df is not None:
            # Match athlete by name (fuzzy)
            athlete_name = _get_athlete_name(engine, athlete_id)
            if athlete_name:
                matched = _fuzzy_match_athlete(excel_df, athlete_name)
                if matched is not None:
                    row.update({
                        "first_bike_segment_sec": matched.get("first_bike_segment_sec"),
                        "first_bike_segment_rank": matched.get("first_bike_segment_rank"),
                        "first_bike_rank_in_swim_pack": matched.get("first_bike_rank_in_swim_pack"),
                        "first_bike_segment_rank_in_swim_pack": matched.get("first_bike_segment_rank_in_swim_pack"),
                    })
            
            # Add fastest first bike split and gap to fastest
            if "first_bike_segment_sec" in excel_df.columns:
                valid_times = excel_df["first_bike_segment_sec"].dropna()
                valid_times = valid_times[valid_times > 0]
                if not valid_times.empty:
                    fastest_time = valid_times.min()
                    row["first_bike_fastest_sec"] = float(fastest_time)
                    athlete_time = row.get("first_bike_segment_sec")
                    if athlete_time and athlete_time > 0:
                        row["first_bike_gap_to_fastest_sec"] = float(athlete_time - fastest_time)
        
        results.append(row)
    
    return pd.DataFrame(results)


def _find_matching_excel(event_name: str, event_date, folder: str) -> Optional[str]:
    """Find Excel file matching event name/date in folder."""
    from pathlib import Path
    
    folder_path = Path(folder)
    if not folder_path.exists():
        return None
    
    # Extract location from event name (e.g., "Abu Dhabi", "Hamburg")
    # Common WTCS locations
    locations = [
        "Abu Dhabi", "Hamburg", "Yokohama", "Wollongong", "French Riviera",
        "Weihai", "Karlovy Vary", "Alghero", "Montreal", "Cagliari",
        "Leeds", "Bermuda", "Edmonton", "Chengdu", "Sunderland"
    ]
    
    event_name_upper = event_name.upper()
    matched_location = None
    for loc in locations:
        if loc.upper() in event_name_upper:
            matched_location = loc
            break
    
    if not matched_location:
        return None
    
    # Extract year
    if hasattr(event_date, 'year'):
        year = event_date.year
    else:
        year = int(str(event_date)[:4])
    
    # Search for matching file (handle variations like "Abu Dhabi2025" vs "Abu Dhabi 2025")
    for f in folder_path.glob("Detailed results*.xlsx"):
        fname_upper = f.name.upper()
        # Check location match (handle spaces/no-spaces)
        loc_match = matched_location.upper().replace(" ", "")
        fname_nospace = fname_upper.replace(" ", "")
        if loc_match in fname_nospace and str(year) in f.name:
            return str(f)
    
    return None


def _get_athlete_name(engine: Engine, athlete_id: int) -> Optional[str]:
    """Get athlete full name from database."""
    query = text("SELECT full_name FROM athlete WHERE athlete_id = :aid")
    with engine.connect() as conn:
        result = conn.execute(query, {"aid": int(athlete_id)}).fetchone()
    return result[0] if result else None


def _get_athlete_gender(engine: Engine, athlete_id: int) -> Optional[str]:
    """Get athlete gender from database (e.g., 'male', 'female')."""
    query = text("SELECT gender FROM athlete WHERE athlete_id = :aid")
    with engine.connect() as conn:
        result = conn.execute(query, {"aid": int(athlete_id)}).fetchone()
    return result[0] if result else None


def _fuzzy_match_athlete(df: pd.DataFrame, athlete_name: str) -> Optional[dict]:
    """Find athlete row in DataFrame by fuzzy name match."""
    if "name" not in df.columns:
        return None
    
    name_lower = athlete_name.lower().strip()
    
    # Try exact match first
    exact = df[df["name"].str.lower().str.strip() == name_lower]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    
    # Try partial match
    for _, row in df.iterrows():
        row_name = str(row["name"]).lower().strip()
        if name_lower in row_name or row_name in name_lower:
            return row.to_dict()
        # Try last name match
        if name_lower.split()[-1] in row_name or row_name.split()[-1] in name_lower:
            return row.to_dict()
    
    return None


def format_for_powerpoint_chart(
    chart_data: pd.DataFrame,
    athlete_name: str,
) -> dict:
    """Format chart data for PowerPoint visualization.
    
    Returns a dict with:
        - title: Chart title
        - athlete_name: Athlete name
        - avg_bike_position: Average position off the bike
        - events: List of dicts with event data for the chart
        
    Each event dict contains:
        - label: Short event label (location + date)
        - bike_position: Position off the bike
        - gap_to_leader_sec: Seconds behind leader
        - pack_delta: Change in pack position (interpretation)
        - first_bike_rank: Rank at first bike split (if available)
    """
    if chart_data.empty:
        return {"title": "No data", "events": []}
    
    events = []
    for _, row in chart_data.iterrows():
        # Create short label from event name
        event_name = row.get("event_name", "")
        event_date = row.get("event_date", "")
        
        # Extract location
        location = "Unknown"
        for loc in ["Abu Dhabi", "Hamburg", "Yokohama", "Wollongong", "French Riviera",
                    "Weihai", "Karlovy Vary", "Alghero", "Montreal"]:
            if loc.lower() in event_name.lower():
                location = loc
                break
        
        # Format date
        if hasattr(event_date, 'strftime'):
            date_str = event_date.strftime("%m/%d/%Y")
        else:
            date_str = str(event_date)[:10]
        
        event_entry = {
            "label": f"{location}\n{date_str}",
            "location": location,
            "date": date_str,
            "bike_position": row.get("bike_overall_position"),
            "gap_to_leader_sec": row.get("gap_to_leader_sec"),
            # Reference pack info (swim or first_bike)
            "ref_pack_id": row.get("swim_pack_id"),
            "ref_pack_size": row.get("swim_pack_size"),
            "ref_pack_position": row.get("swim_pack_position"),
            # Bike exit pack info
            "bike_pack_id": row.get("bike_pack_id"),
            "bike_pack_size": row.get("bike_pack_size"),
            "bike_pack_position": row.get("bike_pack_position"),
            # Rank among initial pack at bike exit
            "init_pack_rank_at_bike": row.get("bike_exit_pack_position"),
            "pack_delta": row.get("bike_pack_delta"),
            # First bike split analysis
            "first_bike_segment_sec": row.get("first_bike_segment_sec"),
            "first_bike_segment_rank": row.get("first_bike_segment_rank"),
            "first_bike_fastest_sec": row.get("first_bike_fastest_sec"),
            "first_bike_gap_to_fastest_sec": row.get("first_bike_gap_to_fastest_sec"),
        }
        
        # Add pack-relative interpretation
        delta = row.get("bike_pack_delta")
        if pd.notna(delta):
            if delta < -2:
                event_entry["pack_interpretation"] = f"Gained {abs(int(delta))} on pack"
            elif delta < 0:
                event_entry["pack_interpretation"] = f"Gained {abs(int(delta))} on pack"
            elif delta == 0:
                event_entry["pack_interpretation"] = "Held pack position"
            else:
                event_entry["pack_interpretation"] = f"Lost {int(delta)} to pack"
        
        events.append(event_entry)
    
    # Compute average bike position
    bike_positions = [e["bike_position"] for e in events if e["bike_position"] is not None]
    avg_position = sum(bike_positions) / len(bike_positions) if bike_positions else None
    
    return {
        "title": "WTCS Bike Positioning",
        "athlete_name": athlete_name,
        "avg_bike_position": round(avg_position, 1) if avg_position else None,
        "n_events": len(events),
        "events": events,
    }


def print_season_bike_summary(
    engine: Engine,
    *,
    athlete_id: int,
    year: int = 2025,
    reference_checkpoint: str = "swim",
) -> None:
    """Print a formatted season bike performance summary to console.
    
    Useful for quick analysis before creating PowerPoint slides.
    
    Args:
        engine: Database engine
        athlete_id: Athlete ID
        year: Season year
        reference_checkpoint: Which checkpoint defines the "starting pack"
            - "swim": Pack established at swim exit (can be strict before packs form)
            - "first_bike": Pack established at first bike mat B1T1 (packs consolidated)
    """
    # Get athlete name
    athlete_name = _get_athlete_name(engine, athlete_id) or f"Athlete {athlete_id}"
    
    # Build chart data
    chart_data = build_season_bike_chart_data(
        engine, athlete_id=athlete_id, year=year, include_first_split=True,
        reference_checkpoint=reference_checkpoint
    )
    
    if chart_data.empty:
        print(f"No WTCS data found for {athlete_name} in {year}")
        return
    
    # Format for display
    ppt_data = format_for_powerpoint_chart(chart_data, athlete_name)
    
    # Determine column header based on checkpoint
    ref_labels = {"swim": "Swim", "t1": "T1", "first_bike": "B1T1"}
    ref_label = ref_labels.get(reference_checkpoint, reference_checkpoint)
    
    print("=" * 105)
    print(f"WTCS Bike Positioning - {athlete_name} ({year})")
    print(f"Reference checkpoint: {ref_label} | Avg Bike Position: {ppt_data['avg_bike_position']} | Events: {ppt_data['n_events']}")
    print("=" * 105)
    # Header row 1: groupings
    print(f"{'':16} {'--- Reference Pack ---':^22} {'--- Bike Exit Pack ---':^22} {'--- Delta ---':^12}")
    # Header row 2: columns
    print(f"{'Event':<16} {'Pack':>5} {'Size':>5} {'Rank':>5}   {'Pack':>5} {'Size':>5} {'Rank':>5}   {'Init@Bike':>10} {'Δ':>6}")
    print("-" * 105)
    
    for e in ppt_data["events"]:
        # Reference pack columns
        ref_pack = e.get("ref_pack_id")
        ref_pack_str = str(int(ref_pack)) if ref_pack is not None and pd.notna(ref_pack) else "-"
        ref_size = e.get("ref_pack_size")
        ref_size_str = str(int(ref_size)) if ref_size is not None and pd.notna(ref_size) else "-"
        ref_rank = e.get("ref_pack_position")
        ref_rank_str = str(int(ref_rank)) if ref_rank is not None and pd.notna(ref_rank) else "-"
        
        # Bike exit pack columns
        bike_pack = e.get("bike_pack_id")
        bike_pack_str = str(int(bike_pack)) if bike_pack is not None and pd.notna(bike_pack) else "-"
        bike_size = e.get("bike_pack_size")
        bike_size_str = str(int(bike_size)) if bike_size is not None and pd.notna(bike_size) else "-"
        bike_rank = e.get("bike_pack_position")
        bike_rank_str = str(int(bike_rank)) if bike_rank is not None and pd.notna(bike_rank) else "-"
        
        # Delta columns
        init_at_bike = e.get("init_pack_rank_at_bike")
        init_at_bike_str = str(int(init_at_bike)) if init_at_bike is not None and pd.notna(init_at_bike) else "-"
        pack_delta = e.get("pack_delta")
        if pack_delta is not None and pd.notna(pack_delta):
            delta_str = f"{int(pack_delta):+d}"
        else:
            delta_str = "-"
        
        print(f"{e['location']:<16} {ref_pack_str:>5} {ref_size_str:>5} {ref_rank_str:>5}   {bike_pack_str:>5} {bike_size_str:>5} {bike_rank_str:>5}   {init_at_bike_str:>10} {delta_str:>6}")
    
    print("-" * 105)
    
    # First Bike Split Analysis Section
    # Check if we have first bike split data
    has_first_bike_data = any(
        e.get("first_bike_segment_sec") is not None and pd.notna(e.get("first_bike_segment_sec"))
        for e in ppt_data["events"]
    )
    
    if has_first_bike_data:
        print("\n" + "=" * 75)
        print("First Bike Split Analysis (B1T1 segment)")
        print("=" * 75)
        print(f"{'Event':<16} {'Time':>10} {'Rank':>6} {'Fastest':>10} {'Gap':>10}")
        print("-" * 75)
        
        for e in ppt_data["events"]:
            athlete_time = e.get("first_bike_segment_sec")
            athlete_rank = e.get("first_bike_segment_rank")
            fastest_time = e.get("first_bike_fastest_sec")
            gap_to_fastest = e.get("first_bike_gap_to_fastest_sec")
            
            # Format times as MM:SS
            def fmt_time(sec):
                if sec is None or pd.isna(sec) or sec <= 0:
                    return "-"
                m, s = divmod(int(sec), 60)
                return f"{m}:{s:02d}"
            
            def fmt_gap(sec):
                if sec is None or pd.isna(sec):
                    return "-"
                return f"+{sec:.1f}s"
            
            athlete_time_str = fmt_time(athlete_time)
            rank_str = str(int(athlete_rank)) if athlete_rank is not None and pd.notna(athlete_rank) else "-"
            fastest_str = fmt_time(fastest_time)
            gap_str = fmt_gap(gap_to_fastest)
            
            print(f"{e['location']:<16} {athlete_time_str:>10} {rank_str:>6} {fastest_str:>10} {gap_str:>10}")
        
        print("-" * 75)
    
    print("\nColumn Guide:")
    print(f"  Reference Pack: Pack assignment at {ref_label} checkpoint (Pack #, Size, athlete's Rank in pack)")
    print("  Bike Exit Pack: Pack assignment at bike exit (Pack #, Size, athlete's Rank in that pack)")
    print("  Init@Bike: Rank among ORIGINAL reference pack mates at bike exit")
    print("  Δ (Delta): Init@Bike - Ref Rank (negative = gained on pack, positive = lost to pack)")
    if has_first_bike_data:
        print("  First Bike Split: B1T1 segment time, rank, fastest time in race, and gap to fastest")


def export_powerbi_bike_metrics(
    engine: Engine,
    *,
    year: int = 2025,
    detailed_results_folder: str = "data",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Export comprehensive bike metrics for all athletes at all WTCS events for Power BI.
    
    Creates a denormalized table with one row per athlete per event, containing:
    - Event info (id, name, date, location)
    - Athlete info (id, name, gender)
    - Finish position
    - Pack metrics for ALL checkpoint options (swim, t1, first_bike) as separate columns
    - Bike exit pack info
    - Delta calculations for each checkpoint
    - First bike split analysis (time, rank, fastest, gap)
    - Overall bike position and gap to leader
    
    This format is ideal for Power BI slicing and filtering.
    
    Args:
        engine: Database engine
        year: Season year
        detailed_results_folder: Folder containing detailed results Excel files
        output_path: If provided, saves to CSV
        
    Returns:
        DataFrame with comprehensive bike metrics
    """
    from pathlib import Path
    
    # Get all WTCS events for the year
    events_query = text("""
        SELECT DISTINCT 
            e.event_id, e.prog_id, e.event_name, e.event_date
        FROM events e
        WHERE EXTRACT(YEAR FROM e.event_date) = :year
          AND (e.event_name ILIKE '%Championship Series%' 
               OR e.event_name ILIKE '%Championship Finals%')
        ORDER BY e.event_date, e.prog_id
    """)
    
    with engine.connect() as conn:
        events = pd.read_sql(events_query, conn, params={"year": int(year)})
    
    if events.empty:
        print(f"No WTCS events found for {year}")
        return pd.DataFrame()
    
    all_rows = []
    
    for _, event in events.iterrows():
        event_id = int(event["event_id"])
        prog_id = int(event["prog_id"])
        event_name = event["event_name"]
        event_date = event["event_date"]
        
        # Extract location from event name
        location = "Unknown"
        for loc in ["Abu Dhabi", "Hamburg", "Yokohama", "Wollongong", "French Riviera",
                    "Weihai", "Karlovy Vary", "Alghero", "Montreal", "Cagliari",
                    "Leeds", "Bermuda", "Edmonton", "Chengdu", "Sunderland"]:
            if loc.lower() in event_name.lower():
                location = loc
                break
        
        # Get all athletes in this event WITH gender from athlete table
        athletes_query = text("""
            SELECT DISTINCT 
                rr.athlete_id, a.full_name, a.gender, rr.finish_position
            FROM race_results rr
            JOIN athlete a ON rr.athlete_id = a.athlete_id
            WHERE rr.event_id = :event_id AND rr.prog_id = :prog_id
        """)
        with engine.connect() as conn:
            athletes = pd.read_sql(athletes_query, conn, 
                                   params={"event_id": event_id, "prog_id": prog_id})
        
        if athletes.empty:
            continue
        
        # Determine gender from athletes in the race (majority vote or first non-null)
        gender = None
        gender_counts = athletes["gender"].value_counts()
        if not gender_counts.empty:
            dominant_gender = gender_counts.index[0]
            if dominant_gender and str(dominant_gender).lower() in ["female", "women", "f"]:
                gender = "Women"
            elif dominant_gender and str(dominant_gender).lower() in ["male", "men", "m"]:
                gender = "Men"
        
        # Load Excel data for this event with correct gender filter
        excel_df = None
        excel_path = _find_matching_excel(event_name, event_date, detailed_results_folder)
        if excel_path:
            try:
                excel_df = load_detailed_results(excel_path, gender=gender)
                excel_df = compute_first_bike_split_metrics(excel_df)
            except Exception:
                excel_df = None
        
        # Get pack metrics for all three checkpoints
        swim_packs = compute_pack_relative_bike_metrics(
            engine, event_id=event_id, prog_id=prog_id,
            reference_checkpoint="swim", excel_df=excel_df
        )
        
        t1_packs = compute_pack_relative_bike_metrics(
            engine, event_id=event_id, prog_id=prog_id,
            reference_checkpoint="t1", excel_df=excel_df
        )
        
        first_bike_packs = compute_pack_relative_bike_metrics(
            engine, event_id=event_id, prog_id=prog_id,
            reference_checkpoint="first_bike", excel_df=excel_df
        )
        
        # Get position_metrics for bike position and gap
        pos_query = text("""
            SELECT athlete_id, position_at_bike, behindbike, position_at_swim
            FROM position_metrics
            WHERE event_id = :event_id AND prog_id = :prog_id
        """)
        with engine.connect() as conn:
            pos_metrics = pd.read_sql(pos_query, conn,
                                      params={"event_id": event_id, "prog_id": prog_id})
        pos_dict = pos_metrics.set_index("athlete_id").to_dict("index") if not pos_metrics.empty else {}
        
        # Build first bike split lookup from Excel
        first_bike_lookup = {}
        fastest_first_bike = None
        if excel_df is not None and "first_bike_segment_sec" in excel_df.columns:
            valid_times = excel_df["first_bike_segment_sec"].dropna()
            valid_times = valid_times[valid_times > 0]
            if not valid_times.empty:
                fastest_first_bike = valid_times.min()
            
            # Match Excel names to athletes
            for _, athlete in athletes.iterrows():
                athlete_name = athlete["full_name"]
                matched = _fuzzy_match_athlete(excel_df, athlete_name)
                if matched:
                    first_bike_lookup[athlete["athlete_id"]] = {
                        "first_bike_segment_sec": matched.get("first_bike_segment_sec"),
                        "first_bike_segment_rank": matched.get("first_bike_segment_rank"),
                    }
        
        # Build row for each athlete
        for _, athlete in athletes.iterrows():
            athlete_id = int(athlete["athlete_id"])
            
            row = {
                # Event info
                "event_id": event_id,
                "prog_id": prog_id,
                "event_name": event_name,
                "event_date": event_date,
                "location": location,
                "gender": gender,
                # Athlete info
                "athlete_id": athlete_id,
                "athlete_name": athlete["full_name"],
                "finish_position": athlete["finish_position"],
            }
            
            # Add swim checkpoint pack data
            swim_row = swim_packs[swim_packs["athlete_id"] == athlete_id]
            if not swim_row.empty:
                sr = swim_row.iloc[0]
                row.update({
                    "swim_pack_id": int(sr["swim_pack_id"]) if pd.notna(sr["swim_pack_id"]) else None,
                    "swim_pack_size": int(sr["swim_pack_size"]) if pd.notna(sr["swim_pack_size"]) else None,
                    "swim_pack_rank": int(sr["swim_pack_position"]) if pd.notna(sr["swim_pack_position"]) else None,
                    "swim_to_bike_delta": int(sr["bike_pack_delta"]) if pd.notna(sr["bike_pack_delta"]) else None,
                    "swim_init_rank_at_bike": int(sr["bike_exit_pack_position"]) if pd.notna(sr["bike_exit_pack_position"]) else None,
                })
            
            # Add T1 checkpoint pack data
            t1_row = t1_packs[t1_packs["athlete_id"] == athlete_id]
            if not t1_row.empty:
                tr = t1_row.iloc[0]
                row.update({
                    "t1_pack_id": int(tr["swim_pack_id"]) if pd.notna(tr["swim_pack_id"]) else None,
                    "t1_pack_size": int(tr["swim_pack_size"]) if pd.notna(tr["swim_pack_size"]) else None,
                    "t1_pack_rank": int(tr["swim_pack_position"]) if pd.notna(tr["swim_pack_position"]) else None,
                    "t1_to_bike_delta": int(tr["bike_pack_delta"]) if pd.notna(tr["bike_pack_delta"]) else None,
                    "t1_init_rank_at_bike": int(tr["bike_exit_pack_position"]) if pd.notna(tr["bike_exit_pack_position"]) else None,
                })
            
            # Add first_bike checkpoint pack data
            fb_row = first_bike_packs[first_bike_packs["athlete_id"] == athlete_id]
            if not fb_row.empty:
                fr = fb_row.iloc[0]
                row.update({
                    "first_bike_pack_id": int(fr["swim_pack_id"]) if pd.notna(fr["swim_pack_id"]) else None,
                    "first_bike_pack_size": int(fr["swim_pack_size"]) if pd.notna(fr["swim_pack_size"]) else None,
                    "first_bike_pack_rank": int(fr["swim_pack_position"]) if pd.notna(fr["swim_pack_position"]) else None,
                    "first_bike_to_bike_delta": int(fr["bike_pack_delta"]) if pd.notna(fr["bike_pack_delta"]) else None,
                    "first_bike_init_rank_at_bike": int(fr["bike_exit_pack_position"]) if pd.notna(fr["bike_exit_pack_position"]) else None,
                })
            
            # Add bike exit pack data (same for all checkpoints)
            if not swim_row.empty:
                sr = swim_row.iloc[0]
                row.update({
                    "bike_exit_pack_id": int(sr["bike_pack_id"]) if pd.notna(sr["bike_pack_id"]) else None,
                    "bike_exit_pack_size": int(sr["bike_pack_size"]) if pd.notna(sr["bike_pack_size"]) else None,
                    "bike_exit_pack_rank": int(sr["bike_pack_position"]) if pd.notna(sr["bike_pack_position"]) else None,
                })
            
            # Add position metrics
            if athlete_id in pos_dict:
                pm = pos_dict[athlete_id]
                row["bike_overall_position"] = int(pm["position_at_bike"]) if pd.notna(pm.get("position_at_bike")) else None
                row["gap_to_leader_sec"] = int(pm["behindbike"]) if pd.notna(pm.get("behindbike")) else None
                row["swim_overall_position"] = int(pm["position_at_swim"]) if pd.notna(pm.get("position_at_swim")) else None
            
            # Add first bike split data
            if athlete_id in first_bike_lookup:
                fb = first_bike_lookup[athlete_id]
                row["first_bike_segment_sec"] = fb.get("first_bike_segment_sec")
                row["first_bike_segment_rank"] = fb.get("first_bike_segment_rank")
                row["first_bike_fastest_sec"] = fastest_first_bike
                if fb.get("first_bike_segment_sec") and fastest_first_bike:
                    row["first_bike_gap_to_fastest_sec"] = fb["first_bike_segment_sec"] - fastest_first_bike
            
            all_rows.append(row)
        
        print(f"  Processed {event_name} ({len(athletes)} athletes)")
    
    result = pd.DataFrame(all_rows)
    
    # Reorder columns for clarity
    col_order = [
        # Event
        "event_id", "prog_id", "event_name", "event_date", "location", "gender",
        # Athlete
        "athlete_id", "athlete_name", "finish_position",
        # Overall positions
        "swim_overall_position", "bike_overall_position", "gap_to_leader_sec",
        # Swim checkpoint
        "swim_pack_id", "swim_pack_size", "swim_pack_rank", 
        "swim_init_rank_at_bike", "swim_to_bike_delta",
        # T1 checkpoint
        "t1_pack_id", "t1_pack_size", "t1_pack_rank",
        "t1_init_rank_at_bike", "t1_to_bike_delta",
        # First bike checkpoint
        "first_bike_pack_id", "first_bike_pack_size", "first_bike_pack_rank",
        "first_bike_init_rank_at_bike", "first_bike_to_bike_delta",
        # Bike exit
        "bike_exit_pack_id", "bike_exit_pack_size", "bike_exit_pack_rank",
        # First bike split analysis
        "first_bike_segment_sec", "first_bike_segment_rank", 
        "first_bike_fastest_sec", "first_bike_gap_to_fastest_sec",
    ]
    # Only include columns that exist
    col_order = [c for c in col_order if c in result.columns]
    result = result[col_order]
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"\nExported {len(result)} rows to {output_path}")
        print(f"  Events: {result['event_name'].nunique()}")
        print(f"  Athletes: {result['athlete_id'].nunique()}")
    
    return result


def create_first_bike_split_chart(
    chart_data: pd.DataFrame,
    athlete_name: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> None:
    """Create a dual-axis line chart for first bike split performance.
    
    Creates a PowerPoint-ready visualization with:
    - Primary Y-axis: Athlete's rank at first bike split (inverted, 1 at top)
    - Secondary Y-axis: Gap to fastest split in seconds
    - X-axis: Event locations
    
    Args:
        chart_data: DataFrame from build_season_bike_chart_data()
        athlete_name: Athlete name for title
        output_path: Path to save PNG (if None, displays interactively)
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved image
    """
    import matplotlib.pyplot as plt
    
    # Filter to events with first bike split data
    valid_data = chart_data[
        chart_data["first_bike_segment_rank"].notna() & 
        (chart_data["first_bike_segment_rank"] > 0)
    ].copy()
    
    if valid_data.empty:
        print("No first bike split data available for charting.")
        return
    
    # Extract event labels (location names)
    locations = []
    for _, row in valid_data.iterrows():
        event_name = row.get("event_name", "")
        location = "Unknown"
        for loc in ["Abu Dhabi", "Hamburg", "Yokohama", "Wollongong", "French Riviera",
                    "Weihai", "Karlovy Vary", "Alghero", "Montreal", "Cagliari"]:
            if loc.lower() in event_name.lower():
                location = loc
                break
        locations.append(location)
    
    ranks = valid_data["first_bike_segment_rank"].values
    gaps = valid_data["first_bike_gap_to_fastest_sec"].fillna(0).values
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Style settings for PowerPoint
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Primary axis: Rank (inverted so 1 is at top)
    color1 = '#2E86AB'  # Blue
    ax1.set_xlabel('Event', fontsize=12, fontweight='bold')
    ax1.set_ylabel('First Bike Split Rank', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(locations, ranks, 'o-', color=color1, linewidth=2.5, markersize=10, 
                     label='Rank', markerfacecolor='white', markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
    ax1.tick_params(axis='x', labelsize=10, rotation=45)
    
    # Invert y-axis so rank 1 is at top
    ax1.invert_yaxis()
    
    # Set y-axis limits with padding
    max_rank = int(max(ranks))
    ax1.set_ylim(max_rank + 2, 0)
    ax1.set_yticks(range(1, max_rank + 1, max(1, max_rank // 6)))
    
    # Add rank labels on points
    for i, (loc, rank) in enumerate(zip(locations, ranks)):
        ax1.annotate(f'{int(rank)}', (loc, rank), textcoords="offset points", 
                    xytext=(0, -15), ha='center', fontsize=9, fontweight='bold', color=color1)
    
    # Secondary axis: Gap to fastest
    ax2 = ax1.twinx()
    color2 = '#E94F37'  # Red
    ax2.set_ylabel('Gap to Fastest (seconds)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(locations, gaps, 's--', color=color2, linewidth=2, markersize=8,
                     label='Gap to Fastest', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
    
    # Set secondary y-axis limits
    max_gap = max(gaps) if len(gaps) > 0 else 30
    ax2.set_ylim(0, max_gap * 1.2)
    
    # Add gap labels on points
    for i, (loc, gap) in enumerate(zip(locations, gaps)):
        ax2.annotate(f'+{gap:.0f}s', (loc, gap), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color=color2)
    
    # Title
    year = valid_data["event_date"].iloc[0].year if hasattr(valid_data["event_date"].iloc[0], 'year') else 2025
    ax1.set_title(f'First Bike Split Performance - {athlete_name} ({year})', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    lines = line1 + line2
    labels = ['Rank (lower is better)', 'Gap to Fastest']
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Grid (only on primary axis)
    ax1.grid(True, alpha=0.3)
    ax2.grid(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_bike_position_chart(
    chart_data: pd.DataFrame,
    athlete_name: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> None:
    """Create a dual-axis line chart for overall bike exit position.
    
    Creates a PowerPoint-ready visualization with:
    - Primary Y-axis: Athlete's position off the bike (inverted, 1 at top)
    - Secondary Y-axis: Gap to leader in seconds
    - X-axis: Event locations
    
    Args:
        chart_data: DataFrame from build_season_bike_chart_data()
        athlete_name: Athlete name for title
        output_path: Path to save PNG (if None, displays interactively)
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved image
    """
    import matplotlib.pyplot as plt
    
    # Filter to events with bike position data
    valid_data = chart_data[
        chart_data["bike_overall_position"].notna() & 
        (chart_data["bike_overall_position"] > 0)
    ].copy()
    
    if valid_data.empty:
        print("No bike position data available for charting.")
        return
    
    # Extract event labels (location names)
    locations = []
    for _, row in valid_data.iterrows():
        event_name = row.get("event_name", "")
        location = "Unknown"
        for loc in ["Abu Dhabi", "Hamburg", "Yokohama", "Wollongong", "French Riviera",
                    "Weihai", "Karlovy Vary", "Alghero", "Montreal", "Cagliari"]:
            if loc.lower() in event_name.lower():
                location = loc
                break
        locations.append(location)
    
    positions = valid_data["bike_overall_position"].values
    gaps = valid_data["gap_to_leader_sec"].fillna(0).values
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Style settings for PowerPoint
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Primary axis: Position (inverted so 1 is at top)
    color1 = '#2E86AB'  # Blue
    ax1.set_xlabel('Event', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bike Exit Position', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(locations, positions, 'o-', color=color1, linewidth=2.5, markersize=10,
                     label='Position', markerfacecolor='white', markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
    ax1.tick_params(axis='x', labelsize=10, rotation=45)
    
    # Invert y-axis so position 1 is at top
    ax1.invert_yaxis()
    
    # Set y-axis limits with padding
    max_pos = int(max(positions))
    ax1.set_ylim(max_pos + 2, 0)
    ax1.set_yticks(range(1, max_pos + 1, max(1, max_pos // 6)))
    
    # Add position labels on points
    for i, (loc, pos) in enumerate(zip(locations, positions)):
        ax1.annotate(f'{int(pos)}', (loc, pos), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=9, fontweight='bold', color=color1)
    
    # Secondary axis: Gap to leader
    ax2 = ax1.twinx()
    color2 = '#E94F37'  # Red
    ax2.set_ylabel('Gap to Leader (seconds)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(locations, gaps, 's--', color=color2, linewidth=2, markersize=8,
                     label='Gap to Leader', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
    
    # Set secondary y-axis limits
    max_gap = max(gaps) if len(gaps) > 0 else 60
    ax2.set_ylim(0, max_gap * 1.2)
    
    # Add gap labels on points
    for i, (loc, gap) in enumerate(zip(locations, gaps)):
        ax2.annotate(f'+{int(gap)}s', (loc, gap), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color=color2)
    
    # Title
    year = valid_data["event_date"].iloc[0].year if hasattr(valid_data["event_date"].iloc[0], 'year') else 2025
    ax1.set_title(f'Bike Exit Position - {athlete_name} ({year})',
                  fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    lines = line1 + line2
    labels = ['Position (lower is better)', 'Gap to Leader']
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Grid (only on primary axis)
    ax1.grid(True, alpha=0.3)
    ax2.grid(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "FirstBikeSplitMetrics",
    "PackRelativeBikeMetrics",
    "AthletePackComparison",
    # Constants
    "DEFAULT_PACK_GAP_THRESHOLD_SEC",
    # Excel functions
    "load_detailed_results",
    "find_first_bike_split_column",
    "compute_positions_at_checkpoints",
    "compute_first_bike_split_metrics",
    # Pack functions
    "assign_packs_chain_rule",
    "assign_packs_two_threshold",
    "compute_pack_positions",
    # Constants for pack thresholds
    "DEFAULT_INITIAL_GAP_THRESHOLD_SEC",
    "DEFAULT_CONTINUATION_GAP_THRESHOLD_SEC",
    # Database functions
    "fetch_pack_membership_for_event",
    "compute_pack_relative_bike_metrics",
    "get_pack_mates_for_athlete",
    # Combined analysis
    "merge_excel_with_database_packs",
    "summarize_athlete_bike_performance",
    # Visualization helpers
    "build_season_bike_chart_data",
    "format_for_powerpoint_chart",
    "print_season_bike_summary",
    "create_first_bike_split_chart",
    "create_bike_position_chart",
    # Power BI export
    "export_powerbi_bike_metrics",
]


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def _lookup_athlete_id(engine: Engine, name_or_id: str) -> Optional[int]:
    """Resolve athlete name or ID to athlete_id."""
    # Try as integer ID first
    try:
        return int(name_or_id)
    except ValueError:
        pass
    
    # Search by name (case-insensitive, partial match)
    query = text("""
        SELECT athlete_id, full_name 
        FROM athlete 
        WHERE full_name ILIKE :pattern
        ORDER BY full_name
        LIMIT 10
    """)
    with engine.connect() as conn:
        results = conn.execute(query, {"pattern": f"%{name_or_id}%"}).fetchall()
    
    if not results:
        return None
    if len(results) == 1:
        return results[0][0]
    
    # Multiple matches - print them and ask user to be more specific
    print(f"Multiple athletes match '{name_or_id}':")
    for aid, name in results:
        print(f"  {aid}: {name}")
    print("\nUse the athlete_id number for an exact match.")
    return None


def main():
    """CLI entry point for relative bike metrics analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze athlete bike performance relative to swim pack mates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tri_analysis.relative_bike_metrics --athlete "Taylor Spivey"
  python -m tri_analysis.relative_bike_metrics --athlete 81334 --year 2024
  python -m tri_analysis.relative_bike_metrics --athlete "McQueen" --csv output.csv
  python -m tri_analysis.relative_bike_metrics --athlete 81334 --checkpoint first_bike
        """
    )
    parser.add_argument(
        "--athlete", "-a",
        required=False,
        help="Athlete name (partial match) or athlete_id number (required except for --powerbi-export)"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=2025,
        help="Season year (default: 2025)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        choices=["swim", "t1", "first_bike"],
        default="swim",
        help="Checkpoint where pack is established: 'swim' (strict, pre-T1), 't1' (after transition), or 'first_bike' (B1T1, consolidated) (default: swim)"
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="Export results to CSV file instead of console output"
    )
    parser.add_argument(
        "--json",
        metavar="FILE", 
        help="Export results to JSON file instead of console output"
    )
    parser.add_argument(
        "--chart",
        metavar="FILE",
        help="Generate PNG chart for first bike split (dual-axis: rank & gap to fastest)"
    )
    parser.add_argument(
        "--bike-chart",
        metavar="FILE",
        help="Generate PNG chart for bike exit position (dual-axis: position & gap to leader)"
    )
    parser.add_argument(
        "--powerbi-export",
        metavar="FILE",
        help="Export comprehensive metrics for ALL athletes at ALL WTCS events to CSV for Power BI"
    )
    
    args = parser.parse_args()
    
    # Connect to database
    from tri_analysis.database import get_engine
    engine = get_engine()
    
    # Handle Power BI export (doesn't require athlete)
    if args.powerbi_export:
        export_powerbi_bike_metrics(engine, year=args.year, output_path=args.powerbi_export)
        return 0
    
    # All other modes require athlete
    if not args.athlete:
        print("Error: --athlete is required (unless using --powerbi-export)")
        return 1
    
    # Resolve athlete
    athlete_id = _lookup_athlete_id(engine, args.athlete)
    if athlete_id is None:
        print(f"Could not find athlete matching '{args.athlete}'")
        return 1
    
    athlete_name = _get_athlete_name(engine, athlete_id) or f"Athlete {athlete_id}"
    
    # Build data
    chart_data = build_season_bike_chart_data(
        engine, athlete_id=athlete_id, year=args.year, include_first_split=True,
        reference_checkpoint=args.checkpoint
    )
    
    if chart_data.empty:
        print(f"No WTCS data found for {athlete_name} in {args.year}")
        return 1
    
    # Output
    if args.csv:
        chart_data.to_csv(args.csv, index=False)
        print(f"Exported {len(chart_data)} events to {args.csv}")
    elif args.json:
        # Format for PowerPoint-style output
        ppt_data = format_for_powerpoint_chart(chart_data, athlete_name)
        import json
        with open(args.json, "w") as f:
            json.dump(ppt_data, f, indent=2, default=str)
        print(f"Exported to {args.json}")
    elif args.chart:
        # Generate first bike split chart
        create_first_bike_split_chart(chart_data, athlete_name, output_path=args.chart)
    elif args.bike_chart:
        # Generate bike exit position chart
        create_bike_position_chart(chart_data, athlete_name, output_path=args.bike_chart)
    else:
        # Console output
        print_season_bike_summary(engine, athlete_id=athlete_id, year=args.year,
                                  reference_checkpoint=args.checkpoint)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
