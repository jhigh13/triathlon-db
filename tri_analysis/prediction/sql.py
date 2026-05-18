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
        prog_name, prog_distance_category, event_country, wetsuit,
        n_finishers
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
            e.event_venue,
            e.wetsuit,
            (SELECT COUNT(*) FROM race_results rr2
             WHERE rr2.event_id = rr.event_id
               AND rr2.prog_id = rr.prog_id
               AND rr2.finish_status = 'FINISH') AS n_finishers
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


def fetch_race_field_results(
    engine: Engine,
    race_keys: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    Fetch all finishers' split times for a set of races.

    Used for computing race-level stats (median total, split ranks).
    Returns all finishers for the given (event_id, prog_id) pairs.

    Args:
        engine: SQLAlchemy Engine
        race_keys: List of (event_id, prog_id) tuples

    Columns returned:
        event_id, prog_id, athlete_id, swimtime, biketime, runtime,
        total_time, finish_position, position_sort
    """
    if not race_keys:
        return pd.DataFrame()

    # Build VALUES list for PostgreSQL tuple IN clause
    values_list = ", ".join(
        [f"({int(eid)}, {int(pid)})" for eid, pid in race_keys]
    )

    query = text(f"""
        SELECT
            rr.event_id,
            rr.prog_id,
            rr.athlete_id,
            rr.swimtime,
            rr.biketime,
            rr.runtime,
            rr.total_time,
            rr.finish_position,
            rr.position_sort
        FROM race_results rr
        WHERE (rr.event_id, rr.prog_id) IN ({values_list})
          AND rr.finish_status = 'FINISH'
          AND rr.total_time IS NOT NULL
        ORDER BY rr.event_id, rr.prog_id, rr.position_sort
    """)

    df = pd.read_sql(query, engine)
    logger.debug(f"fetch_race_field_results: {len(df)} rows for {len(race_keys)} races")
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


def fetch_pack_effect_data(
    engine: Engine,
    start_date: str | None = None,
    end_date: str | None = None,
    distance_category: str | None = None,
    elite_only: bool = True,
    event_tier: int | None = None,
) -> pd.DataFrame:
    """
    Fetch paired swim-pack + bike-time data for learning pack effects.

    Returns one row per athlete per race, joining swim pack membership with
    race results. Used to learn the empirical relationship between swim gap
    and bike performance (drafting benefit vs. solo penalty).

    Args:
        engine: SQLAlchemy Engine
        start_date: Optional start date filter (inclusive)
        end_date: Optional end date filter (inclusive)
        distance_category: Optional filter (e.g., 'sprint', 'standard')
        elite_only: If True, only include Elite Men/Women programs

    Columns returned:
        event_id, prog_id, athlete_id, swim_pack_id, swim_gap_to_leader,
        swim_pack_size, swim_position, swimtime, biketime, runtime,
        total_time, finish_position, prog_distance_category, event_date
    """
    where_clauses = [
        "pm.checkpoint = 'swim'",
        "rr.finish_status = 'FINISH'",
        "rr.total_time IS NOT NULL",
        "rr.biketime IS NOT NULL",
    ]
    params = {}

    if start_date:
        where_clauses.append("e.event_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_clauses.append("e.event_date <= :end_date")
        params["end_date"] = end_date
    if distance_category:
        where_clauses.append("e.prog_distance_category = :distance_category")
        params["distance_category"] = distance_category
    if elite_only:
        where_clauses.append("e.prog_name IN ('Elite Men', 'Elite Women')")

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            pm.event_id,
            pm.prog_id,
            pm.athlete_id,
            pm.pack_id AS swim_pack_id,
            pm.gap_to_leader_sec AS swim_gap_to_leader,
            pm.pack_size AS swim_pack_size,
            pm.pos_at_checkpoint AS swim_position,
            rr.swimtime,
            rr.biketime,
            rr.runtime,
            rr.total_time,
            rr.finish_position,
            e.prog_distance_category,
            e.event_date,
            e.event_name
        FROM wtcs_pack_membership pm
        JOIN race_results rr
            ON pm.event_id = rr.event_id
            AND pm.prog_id = rr.prog_id
            AND pm.athlete_id = rr.athlete_id
        JOIN events e
            ON pm.event_id = e.event_id
            AND pm.prog_id = e.prog_id
        WHERE {where_sql}
        ORDER BY e.event_date, pm.event_id, pm.prog_id, pm.pos_at_checkpoint
    """)

    df = pd.read_sql(query, engine, params=params)

    # Post-query tier filtering (tier is derived from event_name, not a DB column)
    if event_tier is not None and "event_name" in df.columns:
        from .features import classify_event_tier
        df["_tier"] = df["event_name"].apply(classify_event_tier)
        df = df[df["_tier"] == event_tier].drop(columns=["_tier"])

    logger.info(
        f"fetch_pack_effect_data: {len(df)} rows "
        f"({df[['event_id', 'prog_id']].drop_duplicates().shape[0]} races)"
        + (f", tier={event_tier}" if event_tier else "")
    )
    return df


def fetch_swim_to_bike_transitions(
    engine: Engine,
    start_date: str | None = None,
    end_date: str | None = None,
    distance_category: str | None = None,
    elite_only: bool = True,
    event_tier: int | None = None,
) -> pd.DataFrame:
    """
    Fetch paired swim-exit and bike-exit pack membership for learning merge patterns.

    Self-joins wtcs_pack_membership at checkpoint='swim' to checkpoint='bike'
    for the same (event_id, prog_id, athlete_id). This reveals which athletes
    changed packs during the bike leg.

    Args:
        engine: SQLAlchemy Engine
        start_date: Optional start date filter (inclusive)
        end_date: Optional end date filter (inclusive)
        distance_category: Optional filter (e.g., 'sprint', 'standard')
        elite_only: If True, only include Elite Men/Women programs

    Columns returned:
        event_id, prog_id, athlete_id, event_date, prog_distance_category,
        swim_pack_id, swim_pack_size, swim_gap_to_leader_sec, swim_gap_to_prev_sec, swim_pos,
        bike_pack_id, bike_pack_size, bike_gap_to_leader_sec, bike_pos
    """
    where_clauses = [
        "sw.checkpoint = 'swim'",
        "bk.checkpoint = 'bike'",
    ]
    params = {}

    if start_date:
        where_clauses.append("e.event_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_clauses.append("e.event_date <= :end_date")
        params["end_date"] = end_date
    if distance_category:
        where_clauses.append("e.prog_distance_category = :distance_category")
        params["distance_category"] = distance_category
    if elite_only:
        where_clauses.append("e.prog_name IN ('Elite Men', 'Elite Women')")

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            sw.event_id,
            sw.prog_id,
            sw.athlete_id,
            e.event_date,
            e.prog_distance_category,
            sw.pack_id           AS swim_pack_id,
            sw.pack_size         AS swim_pack_size,
            sw.gap_to_leader_sec AS swim_gap_to_leader_sec,
            sw.gap_to_prev_sec   AS swim_gap_to_prev_sec,
            sw.pos_at_checkpoint AS swim_pos,
            bk.pack_id           AS bike_pack_id,
            bk.pack_size         AS bike_pack_size,
            bk.gap_to_leader_sec AS bike_gap_to_leader_sec,
            bk.pos_at_checkpoint AS bike_pos,
            e.event_name
        FROM wtcs_pack_membership sw
        JOIN wtcs_pack_membership bk
            ON sw.event_id = bk.event_id
            AND sw.prog_id = bk.prog_id
            AND sw.athlete_id = bk.athlete_id
            AND bk.checkpoint = 'bike'
        JOIN events e
            ON sw.event_id = e.event_id
            AND sw.prog_id = e.prog_id
        WHERE {where_sql}
        ORDER BY e.event_date, sw.event_id, sw.prog_id, sw.pos_at_checkpoint
    """)

    df = pd.read_sql(query, engine, params=params)

    # Post-query tier filtering
    if event_tier is not None and "event_name" in df.columns:
        from .features import classify_event_tier
        df["_tier"] = df["event_name"].apply(classify_event_tier)
        df = df[df["_tier"] == event_tier].drop(columns=["_tier"])

    logger.info(
        f"fetch_swim_to_bike_transitions: {len(df)} rows "
        f"({df[['event_id', 'prog_id']].drop_duplicates().shape[0]} races)"
        + (f", tier={event_tier}" if event_tier else "")
    )
    return df


def fetch_elo_ratings(
    engine: Engine,
    athlete_ids: list[int],
) -> pd.DataFrame:
    """
    Fetch Elo ratings for a set of athletes.

    Args:
        engine: SQLAlchemy Engine
        athlete_ids: List of athlete IDs to look up

    Columns returned:
        athlete_id, elo_rating, elo_peak, elo_races, last_race_date
    """
    if not athlete_ids:
        return pd.DataFrame()

    id_list = ", ".join(str(int(a)) for a in athlete_ids)

    query = text(f"""
        SELECT
            athlete_id,
            elo_rating,
            elo_peak,
            elo_races,
            last_race_date
        FROM athlete_elo_ratings
        WHERE athlete_id IN ({id_list})
    """)

    df = pd.read_sql(query, engine)
    logger.debug(f"fetch_elo_ratings: {len(df)} rows for {len(athlete_ids)} athletes")
    return df


def fetch_wt_ranking_points(
    engine: Engine,
    athlete_ids: list[int],
    year: int | None = None,
    gender: str | None = None,
) -> pd.DataFrame:
    """
    Fetch World Triathlon ranking points for a set of athletes.

    Returns one row per athlete with both world and continental rankings
    as separate columns. Uses ranking_cat_id to distinguish:
      - World rankings: cat_ids 13-16
      - Continental rankings: cat_ids 35-44

    Args:
        engine: SQLAlchemy Engine
        athlete_ids: List of athlete IDs
        year: If provided, fetch rankings for this year
        gender: If provided, filter to gender-specific ranking categories

    Columns returned:
        athlete_id, rank_position, total_points, year,
        continental_rank_position, continental_total_points
    """
    if not athlete_ids:
        return pd.DataFrame()

    id_list = ", ".join(str(int(a)) for a in athlete_ids)

    where_clauses = [f"ar.athlete_id IN ({id_list})"]
    params: dict = {}

    if year:
        where_clauses.append("ar.year = :year")
        params["year"] = year

    where_sql = " AND ".join(where_clauses)

    # Fetch world rankings (cat_ids 13-16) and continental rankings (cat_ids 35-44)
    # separately, then join per athlete. DISTINCT ON picks the most recent entry.
    query = text(f"""
        WITH world AS (
            SELECT DISTINCT ON (ar.athlete_id)
                ar.athlete_id,
                ar.rank_position,
                ar.total_points,
                ar.year
            FROM athlete_rankings ar
            WHERE {where_sql}
              AND ar.ranking_cat_id IN (13, 14, 15, 16)
            ORDER BY ar.athlete_id, ar.year DESC, ar.retrieved_at DESC
        ),
        continental AS (
            SELECT DISTINCT ON (ar.athlete_id)
                ar.athlete_id,
                ar.rank_position AS continental_rank_position,
                ar.total_points AS continental_total_points
            FROM athlete_rankings ar
            WHERE {where_sql}
              AND ar.ranking_cat_id IN (35, 36, 37, 38, 39, 40, 41, 42, 43, 44)
            ORDER BY ar.athlete_id, ar.year DESC, ar.retrieved_at DESC
        )
        SELECT
            COALESCE(w.athlete_id, c.athlete_id) AS athlete_id,
            w.rank_position,
            w.total_points,
            w.year,
            c.continental_rank_position,
            c.continental_total_points
        FROM world w
        FULL OUTER JOIN continental c ON w.athlete_id = c.athlete_id
    """)

    df = pd.read_sql(query, engine, params=params)
    logger.debug(f"fetch_wt_ranking_points: {len(df)} rows for {len(athlete_ids)} athletes")
    return df


def fetch_batch_athlete_history(
    engine: Engine,
    athlete_ids: list[int],
    before_date: date,
    limit_per_athlete: int = 50,
    elite_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch race history for multiple athletes in a single query.

    Returns the same columns as fetch_athlete_history but for all athletes at once.
    Uses a window function to limit to the most recent `limit_per_athlete` races per athlete.
    """
    if not athlete_ids:
        return pd.DataFrame()

    elite_filter = "AND e.prog_name IN ('Elite Men', 'Elite Women')" if elite_only else ""

    query = text(f"""
        WITH finisher_counts AS (
            SELECT event_id, prog_id, COUNT(*) AS n_finishers
            FROM race_results
            WHERE finish_status = 'FINISH'
            GROUP BY event_id, prog_id
        ),
        ranked AS (
            SELECT
                rr.event_id, rr.prog_id, rr.athlete_id, e.event_date, e.event_name,
                rr.swimtime, rr.t1time, rr.biketime, rr.t2time, rr.runtime,
                rr.total_time, rr.finish_status, rr.finish_position, rr.position_sort,
                e.prog_name, e.prog_distance_category, e.event_country, e.event_venue, e.wetsuit,
                fc.n_finishers,
                ROW_NUMBER() OVER (PARTITION BY rr.athlete_id ORDER BY e.event_date DESC) AS rn
            FROM race_results rr
            JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
            LEFT JOIN finisher_counts fc ON rr.event_id = fc.event_id AND rr.prog_id = fc.prog_id
            WHERE rr.athlete_id = ANY(:athlete_ids)
              AND e.event_date < :before_date
              AND rr.finish_status = 'FINISH'
              {elite_filter}
        )
        SELECT event_id, prog_id, athlete_id, event_date, event_name,
               swimtime, t1time, biketime, t2time, runtime, total_time,
               finish_status, finish_position, position_sort,
               prog_name, prog_distance_category, event_country, event_venue, wetsuit,
               n_finishers
        FROM ranked
        WHERE rn <= :limit_per_athlete
        ORDER BY athlete_id, event_date DESC
    """)

    df = pd.read_sql(query, engine, params={
        "athlete_ids": athlete_ids,
        "before_date": before_date,
        "limit_per_athlete": limit_per_athlete,
    })
    logger.debug(f"fetch_batch_athlete_history: {len(df)} rows for {len(athlete_ids)} athletes")
    return df


def fetch_batch_precomputed_race_metrics(
    engine: Engine,
    athlete_ids: list[int],
    before_date: date,
    limit_per_athlete: int = 50,
) -> pd.DataFrame:
    """
    Fetch precomputed race metrics for multiple athletes in a single query.

    Returns the same columns as fetch_precomputed_race_metrics but for all athletes at once.
    """
    if not athlete_ids:
        return pd.DataFrame()

    query = text("""
        WITH athlete_events AS (
            SELECT DISTINCT event_id, prog_id
            FROM position_metrics
            WHERE athlete_id = ANY(:athlete_ids)
        ),
        finisher_counts AS (
            SELECT rr.event_id, rr.prog_id, COUNT(*) AS n_finishers
            FROM race_results rr
            JOIN athlete_events ae ON rr.event_id = ae.event_id AND rr.prog_id = ae.prog_id
            WHERE rr.finish_status = 'FINISH'
            GROUP BY rr.event_id, rr.prog_id
        ),
        ranked AS (
            SELECT
                pm.event_id, pm.prog_id, pm.athlete_id, e.event_date, e.event_name,
                pm.swimrank, pm.bikerank, pm.runrank, pm.t1rank, pm.t2rank,
                pm.swim_to_t1_pos_change, pm.t1_to_bike_pos_change,
                pm.bike_to_t2_pos_change, pm.t2_to_run_pos_change,
                COALESCE(pm.n_finishers, fc.n_finishers) AS n_finishers,
                pm.median_total_sec, pm.elapsedrun, pm.behindswim,
                ROW_NUMBER() OVER (PARTITION BY pm.athlete_id ORDER BY e.event_date DESC) AS rn
            FROM position_metrics pm
            JOIN events e ON pm.event_id = e.event_id AND pm.prog_id = e.prog_id
            LEFT JOIN finisher_counts fc ON pm.event_id = fc.event_id AND pm.prog_id = fc.prog_id
            WHERE pm.athlete_id = ANY(:athlete_ids)
              AND e.event_date < :before_date
              AND COALESCE(pm.n_finishers, fc.n_finishers) IS NOT NULL
              AND pm.elapsedrun > 0
        )
        SELECT event_id, prog_id, athlete_id, event_date, event_name,
               swimrank, bikerank, runrank, t1rank, t2rank,
               swim_to_t1_pos_change, t1_to_bike_pos_change,
               bike_to_t2_pos_change, t2_to_run_pos_change,
               n_finishers, median_total_sec, elapsedrun, behindswim
        FROM ranked
        WHERE rn <= :limit_per_athlete
        ORDER BY athlete_id, event_date DESC
    """)

    df = pd.read_sql(query, engine, params={
        "athlete_ids": athlete_ids,
        "before_date": before_date,
        "limit_per_athlete": limit_per_athlete,
    })
    logger.debug(f"fetch_batch_precomputed_race_metrics: {len(df)} rows for {len(athlete_ids)} athletes")
    return df


def fetch_batch_pack_history(
    engine: Engine,
    athlete_ids: list[int],
    before_date: date,
    limit_per_athlete: int = 100,
    distance_category: str | None = None,
) -> pd.DataFrame:
    """
    Fetch pack membership history for multiple athletes in a single query.

    Returns the same columns as fetch_pack_history but for all athletes at once.
    """
    if not athlete_ids:
        return pd.DataFrame()

    dist_filter = "AND e.prog_distance_category = :distance_category" if distance_category else ""
    params: dict = {
        "athlete_ids": athlete_ids,
        "before_date": before_date,
        "limit_per_athlete": limit_per_athlete,
    }
    if distance_category:
        params["distance_category"] = distance_category

    query = text(f"""
        WITH ranked AS (
            SELECT
                pm.event_id, pm.prog_id, pm.athlete_id, e.event_date,
                pm.checkpoint, pm.pack_id, pm.pack_size,
                pm.gap_to_leader_sec, pm.gap_to_prev_sec, pm.pos_at_checkpoint,
                ROW_NUMBER() OVER (
                    PARTITION BY pm.athlete_id, pm.checkpoint
                    ORDER BY e.event_date DESC
                ) AS rn
            FROM wtcs_pack_membership pm
            JOIN events e ON pm.event_id = e.event_id AND pm.prog_id = e.prog_id
            WHERE pm.athlete_id = ANY(:athlete_ids)
              AND e.event_date < :before_date
              {dist_filter}
        )
        SELECT event_id, prog_id, athlete_id, event_date,
               checkpoint, pack_id, pack_size,
               gap_to_leader_sec, gap_to_prev_sec, pos_at_checkpoint
        FROM ranked
        WHERE rn <= :limit_per_athlete
        ORDER BY athlete_id, event_date DESC, checkpoint
    """)

    df = pd.read_sql(query, engine, params=params)
    logger.debug(f"fetch_batch_pack_history: {len(df)} rows for {len(athlete_ids)} athletes, distance={distance_category}")
    return df


def fetch_precomputed_race_metrics(
    engine: Engine,
    athlete_id: int,
    before_date: date,
    limit: int = 50,
) -> pd.DataFrame:
    """
    Fetch precomputed race metrics from position_metrics for an athlete's history.

    Returns per-race split ranks, n_finishers, median_total_sec, and elapsedrun.
    This replaces the expensive fetch_race_field_results() + Python computation
    approach, since position_metrics already stores per-athlete-per-race rankings.

    Args:
        engine: SQLAlchemy Engine
        athlete_id: Athlete ID
        before_date: Cutoff date (exclusive)
        limit: Maximum number of recent races (default 50)

    Columns returned:
        event_id, prog_id, event_date, event_name, swimrank, bikerank, runrank,
        t1rank, t2rank, swim_to_t1_pos_change, t1_to_bike_pos_change,
        bike_to_t2_pos_change, t2_to_run_pos_change,
        n_finishers, median_total_sec, elapsedrun, behindswim
    """
    query = text("""
        SELECT
            pm.event_id,
            pm.prog_id,
            e.event_date,
            e.event_name,
            pm.swimrank,
            pm.bikerank,
            pm.runrank,
            pm.t1rank,
            pm.t2rank,
            pm.swim_to_t1_pos_change,
            pm.t1_to_bike_pos_change,
            pm.bike_to_t2_pos_change,
            pm.t2_to_run_pos_change,
            COALESCE(pm.n_finishers, rc.n_finishers) AS n_finishers,
            pm.median_total_sec,
            pm.elapsedrun,
            pm.behindswim
        FROM position_metrics pm
        JOIN events e
            ON pm.event_id = e.event_id
            AND pm.prog_id = e.prog_id
        LEFT JOIN (
            SELECT event_id, prog_id, COUNT(*) AS n_finishers
            FROM race_results
            WHERE finish_status = 'FINISH'
            GROUP BY event_id, prog_id
        ) rc ON pm.event_id = rc.event_id AND pm.prog_id = rc.prog_id
        WHERE pm.athlete_id = :athlete_id
          AND e.event_date < :before_date
          AND COALESCE(pm.n_finishers, rc.n_finishers) IS NOT NULL
          AND pm.elapsedrun > 0
        ORDER BY e.event_date DESC
        LIMIT :limit
    """)

    df = pd.read_sql(query, engine, params={
        "athlete_id": athlete_id,
        "before_date": before_date,
        "limit": limit,
    })
    logger.debug(
        f"fetch_precomputed_race_metrics: {len(df)} rows for athlete {athlete_id}"
    )
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
            event_longitude,
            temperature_air,
            humidity,
            wbgt,
            wind,
            swim_laps,
            bike_laps,
            run_laps,
            weather
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


def search_athletes(
    engine: Engine,
    query: str,
    limit: int = 15,
    exclude_ids: list[int] | None = None,
) -> pd.DataFrame:
    """
    Search for athletes by name (case-insensitive ILIKE).

    Returns distinct athletes with their most recent country info.

    Args:
        engine: SQLAlchemy Engine
        query: Search string (matched with ILIKE %query%)
        limit: Max results to return
        exclude_ids: Athlete IDs to exclude from results

    Returns:
        DataFrame with columns: athlete_id, athlete_full_name, athlete_country_name
    """
    if not query or len(query.strip()) < 2:
        return pd.DataFrame(columns=["athlete_id", "athlete_full_name", "athlete_country_name"])

    exclude_clause = ""
    params: dict = {"query": f"%{query.strip()}%", "limit": limit}
    if exclude_ids:
        exclude_clause = "AND rr.athlete_id != ALL(:exclude_ids)"
        params["exclude_ids"] = exclude_ids

    sql = text(f"""
        SELECT DISTINCT ON (rr.athlete_id)
            rr.athlete_id,
            rr.athlete_full_name,
            pe.athlete_country_name
        FROM race_results rr
        LEFT JOIN LATERAL (
            SELECT pe2.athlete_country_name
            FROM program_entries pe2
            WHERE pe2.athlete_id = rr.athlete_id
              AND pe2.athlete_country_name IS NOT NULL
            ORDER BY pe2.event_id DESC
            LIMIT 1
        ) pe ON TRUE
        WHERE rr.athlete_full_name ILIKE :query
          {exclude_clause}
        ORDER BY rr.athlete_id, rr.event_id DESC
        LIMIT :limit
    """)

    df = pd.read_sql(sql, engine, params=params)
    logger.debug(f"search_athletes: {len(df)} results for query='{query}'")
    return df


def fetch_athletes_by_ids(
    engine: Engine,
    athlete_ids: list[int],
) -> pd.DataFrame:
    """
    Fetch athlete name and country for a list of athlete IDs.

    Used when building features for a custom athlete list (user-curated start list).

    Args:
        engine: SQLAlchemy Engine
        athlete_ids: List of athlete IDs to look up

    Returns:
        DataFrame with columns: athlete_id, athlete_full_name, athlete_country_name
    """
    if not athlete_ids:
        return pd.DataFrame(columns=["athlete_id", "athlete_full_name", "athlete_country_name"])

    sql = text("""
        SELECT DISTINCT ON (rr.athlete_id)
            rr.athlete_id,
            rr.athlete_full_name,
            pe.athlete_country_name
        FROM race_results rr
        LEFT JOIN LATERAL (
            SELECT pe2.athlete_country_name
            FROM program_entries pe2
            WHERE pe2.athlete_id = rr.athlete_id
              AND pe2.athlete_country_name IS NOT NULL
            ORDER BY pe2.event_id DESC
            LIMIT 1
        ) pe ON TRUE
        WHERE rr.athlete_id = ANY(:athlete_ids)
        ORDER BY rr.athlete_id, rr.event_id DESC
    """)

    df = pd.read_sql(sql, engine, params={"athlete_ids": athlete_ids})
    logger.debug(f"fetch_athletes_by_ids: {len(df)} athletes found for {len(athlete_ids)} requested")
    return df


def fetch_head_to_head_results(
    engine: Engine,
    athlete_ids: list[int],
    before_date: date,
    limit_per_athlete: int = 30,
) -> pd.DataFrame:
    """
    Fetch race results for all shared races among a set of athletes.

    For each race where at least 2 of the given athletes competed, returns
    their finish positions. Used to compute pairwise head-to-head records.

    Args:
        engine: SQLAlchemy Engine
        athlete_ids: List of athlete IDs in the field
        before_date: Only races before this date (prevents leakage)
        limit_per_athlete: Max recent races per athlete to consider

    Returns:
        DataFrame with columns: event_id, prog_id, event_date, athlete_id,
                                finish_position, n_field_opponents
    """
    if len(athlete_ids) < 2:
        return pd.DataFrame()

    # Find all recent races for these athletes, then filter to shared races
    sql = text("""
        WITH athlete_races AS (
            SELECT rr.event_id, rr.prog_id, e.event_date,
                   rr.athlete_id, rr.finish_position
            FROM race_results rr
            JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
            WHERE rr.athlete_id = ANY(:athlete_ids)
              AND e.event_date < :before_date
              AND rr.finish_status = 'FINISH'
              AND rr.finish_position IS NOT NULL
        ),
        shared_races AS (
            SELECT event_id, prog_id
            FROM athlete_races
            GROUP BY event_id, prog_id
            HAVING COUNT(DISTINCT athlete_id) >= 2
        )
        SELECT ar.event_id, ar.prog_id, ar.event_date,
               ar.athlete_id, ar.finish_position
        FROM athlete_races ar
        JOIN shared_races sr ON ar.event_id = sr.event_id AND ar.prog_id = sr.prog_id
        ORDER BY ar.event_date DESC, ar.event_id, ar.finish_position
    """)

    df = pd.read_sql(sql, engine, params={
        "athlete_ids": athlete_ids,
        "before_date": before_date,
    })
    logger.debug(
        f"fetch_head_to_head_results: {len(df)} rows across "
        f"{df['event_id'].nunique() if not df.empty else 0} shared races "
        f"for {len(athlete_ids)} athletes"
    )
    return df
