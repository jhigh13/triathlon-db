"""Pack metrics (precompute + persist) for triathlon events.

Purpose
- Compute deterministic pack membership for triathlon events.
- Field is strictly (event_id, prog_id) (no combining).
- Checkpoints are swim/bike/run using elapsed checkpoint times from position_metrics.
- Pack definition is a chain rule: a new pack starts when gap_to_prev_sec > max_gap_to_prev_sec.

Why this exists
- Streamlit should not compute full-field packs on every page load.
- These results are stable, reusable, and fast to query once persisted.

Usage (PowerShell)
- Compute for WTCS events only (default):
  `python -m tri_analysis.wtcs_pack_metrics`

- Compute for ALL events with position_metrics data:
  `python -m tri_analysis.wtcs_pack_metrics --all-events`

- Limit by date range:
  `python -m tri_analysis.wtcs_pack_metrics --start-date 2025-01-01 --end-date 2025-12-31`

Event Tiers (for model weighting):
- WTCS (tier=1): World Triathlon Championship Series, Finals
- World Cup (tier=2): World Triathlon World Cup events
- Regional (tier=3): Continental championships (PATCO, ATU, OTU, ASTC, ETU)
- Other (tier=4): All other events

Notes
- Option A behavior: only athletes with a valid elapsed time at the checkpoint are stored.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

# Allow running as a direct script (python .\tri_analysis\wtcs_pack_metrics.py)
# by ensuring the project root is on sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tri_analysis.database import get_engine
from tri_analysis.wtcs_performance import WTCS_NAME_PATTERNS
from tri_analysis.metrics import adjust_outlier


# Event tier classification for model weighting
# Tier 1 = highest level (WTCS), Tier 4 = lowest (local/other)
EVENT_TIER_PATTERNS: Dict[int, List[str]] = {
    1: [  # WTCS / Championship Series
        "Championship Series",
        "WTCS",
        "Championship Finals",
        "Grand Final",
    ],
    2: [  # World Cup (check BEFORE regional due to "Triathlon Cup" pattern)
        "World Cup",
        "ITU Triathlon World Cup",
        "World Triathlon Cup",  # New naming convention
    ],
    3: [  # Regional / Continental Championships & Cups
        "PATCO",  # Pan American
        "ATU",    # Africa
        "OTU",    # Oceania
        "ASTC",   # Asia
        "ETU",    # Europe
        "CAMTRI", # Central America & Caribbean
        "Continental",
        "African Championships",
        "Asian Championships",
        "Oceania Championships",
        "Pan-American",
        "European Championships",
        "European Cup",
        "Oceania Cup",
        "Panamerican Cup",
        "Pan-American Cup",
        "Asia Triathlon Cup",
        "Africa Triathlon Cup",
        "Americas Triathlon Cup",
    ],
    # Tier 4 = everything else (default)
}


def classify_event_tier(event_name: str) -> int:
    """Classify an event into a tier based on its name.
    
    Returns:
        1 = WTCS (highest), 2 = World Cup, 3 = Regional, 4 = Other (lowest)
    
    Note: Checks tiers in order 1->2->3 to ensure "World Cup" matches tier 2
    before regional patterns like "Cup" might accidentally match.
    """
    if not event_name:
        return 4
    name_upper = event_name.upper()
    
    # Check in tier order (1, 2, 3) to get correct precedence
    for tier in sorted(EVENT_TIER_PATTERNS.keys()):
        patterns = EVENT_TIER_PATTERNS[tier]
        for pattern in patterns:
            if pattern.upper() in name_upper:
                return tier
    return 4


CHECKPOINTS: Dict[str, str] = {
    "swim": "elapsedswim",
    "bike": "elapsedbike",
    "run": "elapsedrun",
}

ALGO_VERSION = "two_threshold_v1"


@dataclass(frozen=True)
class PackAlgoParams:
    """Parameters for pack assignment algorithm.
    
    Two-threshold logic:
    - initial_gap_sec: Max gap from pack leader to 2nd person (default 2s)
    - continuation_gap_sec: Max gap for 3rd+ person to stay in pack (default 1s)
    
    This accounts for the realistic scenario where someone goes slightly off the
    front but isn't really in a breakaway.
    """
    initial_gap_sec: int = 2  # Leader to 2nd person threshold
    continuation_gap_sec: int = 1  # 3rd+ person continuation threshold
    
    # Legacy single-threshold (kept for backward compatibility)
    @property
    def max_gap_to_prev_sec(self) -> int:
        """Legacy property - returns initial_gap_sec for backward compatibility."""
        return self.initial_gap_sec


def _build_wtcs_name_clause(params: Dict[str, object]) -> str:
    """Return SQL snippet for WTCS name match and populate params."""
    pattern_conditions: List[str] = []

    for idx, pat in enumerate(WTCS_NAME_PATTERNS):
        key = f"pat_{idx}"
        params[key] = f"%{pat}%"
        pattern_conditions.append(f"event_name ILIKE :{key}")

    params["broad_world"] = "%World Triathlon%"
    params["broad_series"] = "%Series%"
    params["broad_finals"] = "%Finals%"

    broad_clause = "(event_name ILIKE :broad_world AND (event_name ILIKE :broad_series OR event_name ILIKE :broad_finals))"
    return "( " + broad_clause + " OR " + " OR ".join(pattern_conditions) + " )"


def list_wtcs_event_program_pairs(
    engine: Engine,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Return distinct WTCS (event_id, prog_id) pairs from events."""
    params: Dict[str, object] = {}
    where_parts: List[str] = [_build_wtcs_name_clause(params)]
    if start_date:
        where_parts.append("event_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append("event_date <= :end_date")
        params["end_date"] = end_date

    where_sql = " AND ".join(where_parts)

    query = text(
        f"""
        SELECT DISTINCT event_id, prog_id
        FROM events
        WHERE {where_sql}
        ORDER BY event_id, prog_id
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [(int(r[0]), int(r[1])) for r in rows]


def list_all_event_program_pairs_with_position_metrics(
    engine: Engine,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Return all (event_id, prog_id) pairs that have position_metrics data.
    
    This is more inclusive than list_wtcs_event_program_pairs - it includes
    World Cups, Continental Championships, and any other event with timing data.
    """
    params: Dict[str, object] = {}
    where_parts: List[str] = []
    
    if start_date:
        where_parts.append("e.event_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append("e.event_date <= :end_date")
        params["end_date"] = end_date

    where_sql = " AND ".join(where_parts) if where_parts else "TRUE"

    query = text(
        f"""
        SELECT DISTINCT pm.event_id, pm.prog_id
        FROM position_metrics pm
        JOIN events e ON pm.event_id = e.event_id AND pm.prog_id = e.prog_id
        WHERE {where_sql}
        ORDER BY pm.event_id, pm.prog_id
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [(int(r[0]), int(r[1])) for r in rows]


def assign_pack_ids_chain(sorted_elapsed_sec: Sequence[int], *, max_gap_to_prev_sec: int) -> np.ndarray:
    """Assign pack IDs for elapsed seconds already sorted ascending.

    LEGACY single-threshold chain rule (kept for backward compatibility):
    - pack_id starts at 0
    - for i>0, if elapsed[i] - elapsed[i-1] > threshold => new pack
    """
    n = len(sorted_elapsed_sec)
    if n == 0:
        return np.array([], dtype=int)

    pack_ids = np.zeros(n, dtype=int)
    current_pack = 0
    prev = int(sorted_elapsed_sec[0])

    for i in range(1, n):
        cur = int(sorted_elapsed_sec[i])
        if (cur - prev) > max_gap_to_prev_sec:
            current_pack += 1
        pack_ids[i] = current_pack
        prev = cur

    return pack_ids


def assign_pack_ids_two_threshold(
    sorted_elapsed_sec: Sequence[int],
    *,
    initial_gap_sec: int = 2,
    continuation_gap_sec: int = 1,
) -> np.ndarray:
    """Assign pack IDs using two-threshold rule for realistic pack detection.
    
    This method accounts for the "noise" where someone goes slightly off the 
    front but isn't really in a breakaway:
    
    - 2nd person joins pack if gap to leader ≤ initial_gap_sec (default 2s)
    - 3rd+ person joins pack if gap to previous ≤ continuation_gap_sec (default 1s)
    - Otherwise, start a new pack
    
    Args:
        sorted_elapsed_sec: Elapsed times already sorted ascending
        initial_gap_sec: Max gap from pack leader to 2nd person (default 2s)
        continuation_gap_sec: Max gap for 3rd+ person to stay in pack (default 1s)
    
    Returns:
        Array of pack IDs (0-indexed)
    """
    n = len(sorted_elapsed_sec)
    if n == 0:
        return np.array([], dtype=int)

    pack_ids = np.zeros(n, dtype=int)
    current_pack = 0
    position_in_pack = 1  # Leader is position 1
    prev = int(sorted_elapsed_sec[0])

    for i in range(1, n):
        cur = int(sorted_elapsed_sec[i])
        gap = cur - prev
        
        # Determine which threshold applies
        if position_in_pack == 1:
            # 2nd person in pack: use initial threshold
            threshold = initial_gap_sec
        else:
            # 3rd+ person: use continuation threshold
            threshold = continuation_gap_sec
        
        if gap <= threshold:
            # Stay in current pack
            pack_ids[i] = current_pack
            position_in_pack += 1
        else:
            # Start new pack
            current_pack += 1
            pack_ids[i] = current_pack
            position_in_pack = 1  # This person is the new pack leader
        
        prev = cur

    return pack_ids


def _compute_checkpoint_membership(
    df: pd.DataFrame,
    *,
    event_id: int,
    prog_id: int,
    checkpoint: str,
    elapsed_col: str,
    params: PackAlgoParams,
    computed_at: datetime,
) -> pd.DataFrame:
    """Compute membership rows for a single checkpoint.

    Option A: only include athletes with valid elapsed for this checkpoint.
    Uses two-threshold pack assignment for realistic pack detection.
    """
    if df.empty or elapsed_col not in df.columns:
        return pd.DataFrame()

    work = df[["athlete_id", elapsed_col]].copy()
    work = work.rename(columns={elapsed_col: "elapsed_sec"})
    work["elapsed_sec"] = pd.to_numeric(work["elapsed_sec"], errors="coerce")
    work = work.dropna(subset=["elapsed_sec"]).copy()

    if work.empty:
        return pd.DataFrame()

    work["elapsed_sec"] = work["elapsed_sec"].astype("int64")
    work = work.sort_values("elapsed_sec", ascending=True, kind="mergesort").reset_index(drop=True)

    elapsed = work["elapsed_sec"].to_numpy(dtype=np.int64)
    
    # Use two-threshold algorithm for more realistic pack detection
    pack_ids = assign_pack_ids_two_threshold(
        elapsed,
        initial_gap_sec=params.initial_gap_sec,
        continuation_gap_sec=params.continuation_gap_sec,
    )

    work["pos_at_checkpoint"] = np.arange(1, len(work) + 1, dtype=int)
    work["pack_id"] = pack_ids.astype(int)

    leader_elapsed = int(elapsed[0])
    work["gap_to_leader_sec"] = (work["elapsed_sec"] - leader_elapsed).astype(int)

    # gap_to_prev: leader is NULL
    gap_to_prev = np.empty(len(work), dtype=np.int64)
    gap_to_prev[0] = -1
    gap_to_prev[1:] = elapsed[1:] - elapsed[:-1]
    work["gap_to_prev_sec"] = gap_to_prev.astype(int)
    work.loc[work["pos_at_checkpoint"] == 1, "gap_to_prev_sec"] = pd.NA

    work["pack_size"] = work.groupby("pack_id")["athlete_id"].transform("size").astype(int)

    work["event_id"] = int(event_id)
    work["prog_id"] = int(prog_id)
    work["checkpoint"] = checkpoint
    work["max_gap_to_prev_sec"] = int(params.max_gap_to_prev_sec)
    work["algo_version"] = ALGO_VERSION
    work["computed_at"] = computed_at

    # Column order for readability / stable inserts
    return work[[
        "event_id",
        "prog_id",
        "athlete_id",
        "checkpoint",
        "elapsed_sec",
        "pos_at_checkpoint",
        "gap_to_prev_sec",
        "gap_to_leader_sec",
        "pack_id",
        "pack_size",
        "max_gap_to_prev_sec",
        "algo_version",
        "computed_at",
    ]]


def _parse_time_to_secs(t) -> int:
    """Parse HH:MM:SS or MM:SS time string to seconds."""
    if pd.isna(t) or t == "":
        return 0
    parts = str(t).split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
    except ValueError:
        return 0
    return 0


def compute_pack_membership_for_event_program(
    engine: Engine,
    *,
    event_id: int,
    prog_id: int,
    params: PackAlgoParams,
) -> pd.DataFrame:
    """Compute pack membership for all checkpoints for a single (event_id, prog_id).

    Applies the same outlier filtering as metrics.py:
    1. Exclude athletes with ANY zero/missing split (DNF, lapped, bad data)
    2. Apply per-split outlier detection (suspiciously fast times marked as NA)
    3. Only include athletes with valid elapsed times after filtering

    This ensures gaps are computed only from clean data and corrupt athletes
    don't pollute the pack calculations.
    """
    # Fetch elapsed times from position_metrics AND segment times from race_results
    query = text(
        """
        SELECT pm.athlete_id, pm.elapsedswim, pm.elapsedbike, pm.elapsedrun,
               rr.swimtime, rr.t1time, rr.biketime, rr.t2time, rr.runtime
        FROM position_metrics pm
        JOIN race_results rr ON pm.event_id = rr.event_id
                            AND pm.prog_id = rr.prog_id
                            AND pm.athlete_id = rr.athlete_id
        WHERE pm.event_id = :event_id AND pm.prog_id = :prog_id
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"event_id": event_id, "prog_id": prog_id})  # type: ignore

    if df.empty:
        return pd.DataFrame()

    # Deduplicate: the JOIN can produce multiple rows per athlete if race_results
    # has duplicates.  Keep first occurrence (arbitrary but stable).
    df = df.drop_duplicates(subset=["athlete_id"], keep="first")

    computed_at = datetime.now(timezone.utc)

    # Parse all split times to seconds
    df["swimsecs"] = df["swimtime"].apply(_parse_time_to_secs)
    df["t1secs"] = df["t1time"].apply(_parse_time_to_secs)
    df["bikesecs"] = df["biketime"].apply(_parse_time_to_secs)
    df["t2secs"] = df["t2time"].apply(_parse_time_to_secs)
    df["runsecs"] = df["runtime"].apply(_parse_time_to_secs)

    # Global exclusion: if ANY split is zero, exclude from pack calculations
    # This catches DNFs, lapped athletes, and corrupt data
    valid_all_splits = (
        (df["swimsecs"] > 0) &
        (df["t1secs"] > 0) &
        (df["bikesecs"] > 0) &
        (df["t2secs"] > 0) &
        (df["runsecs"] > 0)
    )

    # Apply outlier detection per split (same as metrics.py)
    # This catches suspiciously fast times that indicate timing errors
    for col in ["swimsecs", "t1secs", "bikesecs", "t2secs", "runsecs"]:
        threshold = 3 if col in ["t1secs", "t2secs"] else 2
        df[col] = adjust_outlier(df[col], threshold)

    # After outlier adjustment, recompute valid_all_splits
    # (outliers are now NA, so they'll fail the > 0 check)
    valid_all_splits = (
        (df["swimsecs"] > 0) &
        (df["t1secs"] > 0) &
        (df["bikesecs"] > 0) &
        (df["t2secs"] > 0) &
        (df["runsecs"] > 0) &
        df["swimsecs"].notna() &
        df["t1secs"].notna() &
        df["bikesecs"].notna() &
        df["t2secs"].notna() &
        df["runsecs"].notna()
    )

    # Filter to only valid athletes
    df_valid = df[valid_all_splits].copy()

    if df_valid.empty:
        return pd.DataFrame()

    # Map checkpoint to elapsed column
    parts: List[pd.DataFrame] = []
    for checkpoint, elapsed_col in CHECKPOINTS.items():
        # All athletes in df_valid have valid data for all checkpoints
        part = _compute_checkpoint_membership(
            df_valid,
            event_id=event_id,
            prog_id=prog_id,
            checkpoint=checkpoint,
            elapsed_col=elapsed_col,
            params=params,
            computed_at=computed_at,
        )
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    # Safety: drop any duplicate (event_id, prog_id, athlete_id, checkpoint) rows
    # that could violate the PK constraint on insert.
    result = result.drop_duplicates(
        subset=["event_id", "prog_id", "athlete_id", "checkpoint"], keep="first"
    )
    return result


def refresh_pack_membership_for_event_program(
    engine: Engine,
    *,
    event_id: int,
    prog_id: int,
    params: PackAlgoParams,
    dry_run: bool = False,
) -> int:
    """Recompute and persist membership for a single WTCS (event_id, prog_id).

    Returns number of rows written.
    """
    df_out = compute_pack_membership_for_event_program(engine, event_id=event_id, prog_id=prog_id, params=params)

    if dry_run:
        return int(len(df_out))

    # Use a single transaction for DELETE + INSERT to avoid race conditions
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM wtcs_pack_membership WHERE event_id = :event_id AND prog_id = :prog_id"),
            {"event_id": int(event_id), "prog_id": int(prog_id)},
        )

        if not df_out.empty:
            df_out.to_sql(
                "wtcs_pack_membership",
                conn,  # Use the connection from the transaction, not the engine
                if_exists="append",
                index=False,
                method="multi",
            )

    return int(len(df_out))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute and persist pack membership for triathlon events.")
    parser.add_argument("--start-date", default=None, help="Optional start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional end date filter (YYYY-MM-DD)")
    parser.add_argument("--initial-gap-sec", type=int, default=2, 
                        help="Max gap from pack leader to 2nd person (default: 2)")
    parser.add_argument("--continuation-gap-sec", type=int, default=1,
                        help="Max gap for 3rd+ person to stay in pack (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Compute counts but do not write to DB")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of (event_id, prog_id) pairs")
    parser.add_argument(
        "--all-events", 
        action="store_true", 
        help="Process ALL events with position_metrics (not just WTCS). Includes World Cups, Continental events, etc."
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    engine = get_engine()
    params = PackAlgoParams(
        initial_gap_sec=int(args.initial_gap_sec),
        continuation_gap_sec=int(args.continuation_gap_sec),
    )

    if args.all_events:
        pairs = list_all_event_program_pairs_with_position_metrics(
            engine, start_date=args.start_date, end_date=args.end_date
        )
        scope_label = "ALL events with position_metrics"
    else:
        pairs = list_wtcs_event_program_pairs(engine, start_date=args.start_date, end_date=args.end_date)
        scope_label = "WTCS events"
    
    if args.limit:
        pairs = pairs[: int(args.limit)]

    print(f"{scope_label} matched: {len(pairs)}")

    total_rows = 0
    for event_id, prog_id in pairs:
        rows = refresh_pack_membership_for_event_program(
            engine,
            event_id=event_id,
            prog_id=prog_id,
            params=params,
            dry_run=bool(args.dry_run),
        )
        total_rows += rows
        print(f"event_id={event_id} prog_id={prog_id}: rows={'(dry)' if args.dry_run else ''}{rows}")

    print(f"Total rows {'computed' if args.dry_run else 'written'}: {total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
