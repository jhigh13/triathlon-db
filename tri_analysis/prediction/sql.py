"""
SQL query functions for the prediction pipeline.

Provides parameterized queries to fetch:
- Program results (for training labels)
- Athlete history (prior races for features)
- Pack membership history (draft-legal features)
- Start lists (upcoming race entrants)

All queries use SQLAlchemy Engine and return pandas DataFrames.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional
import logging

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgramKey:
    """Unique identifier for a race program (event + program)."""
    event_id: int
    prog_id: int

    def __post_init__(self):
        # Ensure native Python int (not numpy.int64)
        object.__setattr__(self, 'event_id', int(self.event_id))
        object.__setattr__(self, 'prog_id', int(self.prog_id))

    def __str__(self) -> str:
        return f"ProgramKey(event_id={self.event_id}, prog_id={self.prog_id})"


def fetch_program_results(engine: Engine, key: ProgramKey) -> pd.DataFrame:
    """
    Fetch race results for a specific (event_id, prog_id) with event metadata.

    Returns one row per athlete with splits, total time, finish status, and event info.
    Used for building training labels.

    Columns returned:
        event_id, prog_id, athlete_id, athlete_full_name,
        swimtime, t1time, biketime, t2time, runtime, total_time,
        finish_status, finish_position, position_sort,
        event_date, prog_name, prog_distance_category, event_country, wetsuit
    """
    query = text("""
        SELECT
            rr.event_id,
            rr.prog_id,
            rr.athlete_id,
            rr.athlete_full_name,
            rr.swimtime,
            rr.t1time,
            rr.biketime,
            rr.t2time,
            rr.runtime,
            rr.total_time,
            rr.finish_status,
            rr.finish_position,
            rr.position_sort,
            e.event_date,
            e.prog_name,
            e.prog_distance_category,
            e.event_country,
            e.event_venue,
            e.wetsuit,
            e.event_latitude,
            e.event_longitude
        FROM race_results rr
        JOIN events e
            ON rr.event_id = e.event_id
            AND rr.prog_id = e.prog_id
        WHERE rr.event_id = :event_id
          AND rr.prog_id = :prog_id
        ORDER BY rr.position_sort
    """)

    df = pd.read_sql(query, engine, params={"event_id": key.event_id, "prog_id": key.prog_id})
    logger.debug(f"fetch_program_results: {len(df)} rows for {key}")
    return df


def fetch_athlete_history(
    engine: Engine,
    athlete_id: int,
    before_date: date,
    limit: int = 50,
    distance_category: Optional[str] = None,
    elite_only: bool = False
) -> pd.DataFrame:
    """
    Fetch all prior race results for an athlete before a given date.

    Used for computing athlete form features (EWMA, std, etc.) without leakage.
    Only includes FINISHERS (finish_status = 'FINISH') with valid total_time.

    Args:
        engine: SQLAlchemy Engine
        athlete_id: Athlete ID
        before_date: Cutoff date (exclusive); only races before this date
        limit: Maximum number of recent races to return (default 50)
        distance_category: If provided, filter to this distance (e.g., 'sprint', 'standard')
        elite_only: If True, filter to Elite programs only (excludes Junior, U23, Para)

    Columns returned:
        event_id, prog_id, athlete_id, event_date,
        swimtime, t1time, biketime, t2time, runtime, total_time,
        finish_status, finish_position, position_sort,
        prog_name, prog_distance_category, event_country, wetsuit
    """
    # Build dynamic WHERE clause
    where_clauses = [
        "rr.athlete_id = :athlete_id",
        "e.event_date < :before_date",
        "rr.finish_status = 'FINISH'"
    ]
    params = {"athlete_id": athlete_id, "before_date": before_date, "limit": limit}
    
    if distance_category:
        where_clauses.append("e.prog_distance_category = :distance_category")
        params["distance_category"] = distance_category
    
    if elite_only:
        where_clauses.append("e.prog_name IN ('Elite Men', 'Elite Women')")
    
    where_sql = " AND ".join(where_clauses)
    
    query = text(f"""
        SELECT
            rr.event_id,
            rr.prog_id,
            rr.athlete_id,
            e.event_date,
            e.event_name,
            rr.swimtime,
            rr.t1time,
            rr.biketime,
            rr.t2time,
            rr.runtime,
            rr.total_time,
            rr.finish_status,
            rr.finish_position,
            rr.position_sort,
            e.prog_name,
            e.prog_distance_category,
            e.event_country,
            e.wetsuit
        FROM race_results rr
        JOIN events e
            ON rr.event_id = e.event_id
            AND rr.prog_id = e.prog_id
        WHERE {where_sql}
        ORDER BY e.event_date DESC
        LIMIT :limit
    """)

    df = pd.read_sql(query, engine, params=params)
    logger.debug(f"fetch_athlete_history: {len(df)} rows for athlete {athlete_id} before {before_date}")
    return df


def fetch_pack_history(
    engine: Engine,
    athlete_id: int,
    before_date: date,
    limit: int = 50,
    distance_category: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch pack membership history for an athlete before a given date.

    Used for computing draft-legal pack features (front_pack_rate, avg_swim_gap_leader, etc.).

    Args:
        engine: SQLAlchemy Engine
        athlete_id: Athlete ID
        before_date: Cutoff date (exclusive)
        limit: Maximum number of records per checkpoint (default 50)
        distance_category: If provided, filter to this distance (e.g., 'sprint', 'standard')
                           to avoid mixing Olympic distance gaps with Sprint gaps.

    Columns returned:
        event_id, prog_id, athlete_id, event_date, checkpoint,
        pack_id, pack_size, gap_to_leader_sec, gap_to_prev_sec, pos_at_checkpoint
    """
    where_clauses = [
        "pm.athlete_id = :athlete_id",
        "e.event_date < :before_date",
    ]
    params = {"athlete_id": athlete_id, "before_date": before_date, "limit": limit}
    
    if distance_category:
        where_clauses.append("e.prog_distance_category = :distance_category")
        params["distance_category"] = distance_category

    where_sql = " AND ".join(where_clauses)
    
    query = text(f"""
        SELECT
            pm.event_id,
            pm.prog_id,
            pm.athlete_id,
            e.event_date,
            pm.checkpoint,
            pm.pack_id,
            pm.pack_size,
            pm.gap_to_leader_sec,
            pm.gap_to_prev_sec,
            pm.pos_at_checkpoint
        FROM wtcs_pack_membership pm
        JOIN events e
            ON pm.event_id = e.event_id
            AND pm.prog_id = e.prog_id
        WHERE {where_sql}
        ORDER BY e.event_date DESC, pm.checkpoint
        LIMIT :limit
    """)

    df = pd.read_sql(query, engine, params=params)
    logger.debug(f"fetch_pack_history: {len(df)} rows for athlete {athlete_id} before {before_date}, distance={distance_category}")
    return df


def fetch_start_list(engine: Engine, key: ProgramKey) -> pd.DataFrame:
    """
    Fetch the start list (program entries) for an upcoming race.

    Filters to active entries with entry_type='start' (actual starters).

    Args:
        engine: SQLAlchemy Engine
        key: ProgramKey (event_id, prog_id)

    Columns returned:
        event_id, prog_id, athlete_id, athlete_full_name, athlete_country_name,
        start_num, entry_type
    """
    query = text("""
        SELECT
            pe.event_id,
            pe.prog_id,
            pe.athlete_id,
            pe.athlete_full_name,
            pe.athlete_country_name,
            pe.start_num,
            pe.entry_type
        FROM program_entries pe
        WHERE pe.event_id = :event_id
          AND pe.prog_id = :prog_id
          AND pe.is_active = TRUE
          AND pe.entry_type = 'start'
          AND pe.athlete_id IS NOT NULL
        ORDER BY pe.start_num
    """)

    df = pd.read_sql(query, engine, params={"event_id": key.event_id, "prog_id": key.prog_id})
    logger.debug(f"fetch_start_list: {len(df)} active entrants for {key}")
    return df


def fetch_event_metadata(engine: Engine, key: ProgramKey) -> Optional[dict]:
    """
    Fetch event metadata for a given program.

    Returns a dict with event-level info, or None if not found.
    """
    query = text("""
        SELECT
            event_id,
            prog_id,
            event_name,
            event_date,
            prog_name,
            prog_distance_category,
            event_country,
            event_venue,
            wetsuit,
            event_latitude,
            event_longitude
        FROM events
        WHERE event_id = :event_id
          AND prog_id = :prog_id
    """)

    df = pd.read_sql(query, engine, params={"event_id": key.event_id, "prog_id": key.prog_id})
    if df.empty:
        logger.warning(f"No event metadata found for {key}")
        return None
    return df.iloc[0].to_dict()


def fetch_training_programs(
    engine: Engine,
    start_date: str,
    end_date: str,
    min_finishers: int = 10,
    elite_only: bool = False,
    distance_categories: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Fetch all programs within a date range that have sufficient finishers.

    Used for building the training dataset.

    Args:
        engine: SQLAlchemy Engine
        start_date: Start date (inclusive), format 'YYYY-MM-DD'
        end_date: End date (inclusive), format 'YYYY-MM-DD'
        min_finishers: Minimum number of finishers required (default 10)
        elite_only: If True, only include Elite Men/Women programs (excludes Junior, U23, Para)
        distance_categories: If provided, filter to these distances (e.g., ['sprint', 'standard'])

    Returns:
        DataFrame with columns: event_id, prog_id, event_date, prog_name, prog_distance_category, finisher_count
    """
    # Build dynamic WHERE clause
    where_clauses = [
        "e.event_date >= :start_date",
        "e.event_date <= :end_date"
    ]
    params = {"start_date": start_date, "end_date": end_date, "min_finishers": min_finishers}
    
    if elite_only:
        where_clauses.append("e.prog_name IN ('Elite Men', 'Elite Women')")
    
    if distance_categories:
        placeholders = ", ".join([f":dist_{i}" for i in range(len(distance_categories))])
        where_clauses.append(f"e.prog_distance_category IN ({placeholders})")
        for i, cat in enumerate(distance_categories):
            params[f"dist_{i}"] = cat
    
    where_sql = " AND ".join(where_clauses)
    
    query = text(f"""
        SELECT
            e.event_id,
            e.prog_id,
            e.event_date,
            e.prog_name,
            e.prog_distance_category,
            COUNT(*) FILTER (WHERE rr.finish_status = 'FINISH') AS finisher_count
        FROM events e
        JOIN race_results rr
            ON e.event_id = rr.event_id
            AND e.prog_id = rr.prog_id
        WHERE {where_sql}
        GROUP BY e.event_id, e.prog_id, e.event_date, e.prog_name, e.prog_distance_category
        HAVING COUNT(*) FILTER (WHERE rr.finish_status = 'FINISH') >= :min_finishers
        ORDER BY e.event_date
    """)

    df = pd.read_sql(query, engine, params=params)
    logger.info(f"fetch_training_programs: {len(df)} programs from {start_date} to {end_date} (elite_only={elite_only})")
    return df
