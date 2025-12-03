"""WTCS U.S. Athlete Performance Helpers

Purpose:
    Provide reusable functions to fetch, filter, and aggregate World Triathlon Championship Series
    (WTCS) event data for U.S. athletes. This is a skeleton initial implementation focused on
    structure; detailed metrics and edge-case handling will be iteratively added.

Key Responsibilities:
    - Fetch joined race_results + position_metrics + events + athlete dimension
    - Filter to WTCS events (pattern match on event_name) and USA athletes
    - Aggregate checkpoint metrics (average positions, gaps, ranks, position deltas)
    - Provide helper to identify best and worst race (placeholder logic)

Design Notes:
    - We intentionally avoid heavy caching here; Streamlit layer can cache these pure functions.
    - Keep DB access minimal and column-selective to reduce memory footprint.
    - All averages computed excluding null values; caller can merge counts for display.

Next Enhancements (planned):
    - Pack classification helpers (reusing identify_packs logic currently in streamlit_app)
    - Lead vs chase group participation percentages
    - Export-ready summary (wide + long formats)
    - Optional materialized view for performance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
from sqlalchemy import text, Engine

WTCS_NAME_PATTERNS = [
    "World Triathlon Championship Series",   # Full naming (some seasons)
    "World Triathlon Series",                # Shorter variant (legacy or alternate feed)
    "World Triathlon Championship Finals",   # Season finals naming
    "World Triathlon Finals",                # Alternate finals naming
    "WTCS Finals"                             # Abbreviated finals naming
]
USA_COUNTRY_ALIASES = ["USA", "United States", "United States of America"]


@dataclass
class WTCSFilters:
    start_date: Optional[str] = None  # ISO date strings
    end_date: Optional[str] = None
    gender: Optional[str] = None      # 'M'/'F' or None
    para_filter: Optional[bool] = None  # None = all, True=para only, False=non-para only
    country_codes: Optional[List[str]] = None  # Override list of country names/codes; defaults to US aliases
    min_events: int = 1


def fetch_wtcs_us_dataset(engine: Engine, filters: WTCSFilters) -> pd.DataFrame:
    """Return merged WTCS dataset for USA athletes given filters.

    Columns (baseline): athlete_id, full_name, gender, event_id, prog_id, event_date,
        event_name, position_at_* , behind*, *_pos_change, *rank, finish_position

    Filtering:
        - event_name ILIKE %WTCS_NAME_PATTERN%
        - athlete.country = 'USA'
        - date range if provided
        - gender filter if provided
    """
    # Build flexible WTCS name condition accommodating multiple naming conventions.
    # Strategy: (broad AND broad) OR explicit patterns list
    pattern_conditions = []
    # Country filtering
    countries = filters.country_codes or USA_COUNTRY_ALIASES
    params = {}
    for idx, pat in enumerate(WTCS_NAME_PATTERNS):
        key = f"pat_{idx}"
        params[key] = f"%{pat}%"
        pattern_conditions.append(f"e.event_name ILIKE :{key}")
    # Broad fallback (World Triathlon + Series) catches variants like "2025 World Triathlon Series Hamburg"
    params["broad_world"] = "%World Triathlon%"
    params["broad_series"] = "%Series%"
    params["broad_finals"] = "%Finals%"
    # Accept either Series or Finals with World Triathlon in name
    broad_clause = "(e.event_name ILIKE :broad_world AND (e.event_name ILIKE :broad_series OR e.event_name ILIKE :broad_finals))"
    name_clause = "( " + broad_clause + " OR " + " OR ".join(pattern_conditions) + " )"

    conditions = [name_clause]
    if countries:  # apply only if list provided
        country_placeholders = []
        for idx, cval in enumerate(countries):
            key = f"c{idx}"
            params[key] = cval
            country_placeholders.append(f":{key}")
        conditions.append(f"a.country IN (" + ",".join(country_placeholders) + ")")
    # Para filtering: fall back to name-based heuristic when schema lacks is_para
    # If para_filter True → include events with 'Para' in name; False → exclude them; None → no filter
    if filters.para_filter is True:
        conditions.append("e.event_name ILIKE :para_pat")
        params["para_pat"] = "%Para%"
    elif filters.para_filter is False:
        conditions.append("e.event_name NOT ILIKE :para_pat")
        params["para_pat"] = "%Para%"
    if filters.start_date:
        conditions.append("e.event_date >= :start_date")
        params["start_date"] = filters.start_date
    if filters.end_date:
        conditions.append("e.event_date <= :end_date")
        params["end_date"] = filters.end_date
    if filters.gender and filters.gender.lower() != "all":
        g = filters.gender.strip().lower()
        if g in {"m", "male"}:
            normalized = ["male"]
        elif g in {"f", "female"}:
            normalized = ["female"]
        else:
            normalized = [g]
        gender_placeholders = []
        for idx, gv in enumerate(normalized):
            key = f"g{idx}"
            params[key] = gv
            gender_placeholders.append(f":{key}")
        conditions.append("LOWER(a.gender) IN (" + ",".join(gender_placeholders) + ")")

    where_clause = " AND ".join(conditions)

    # Select only needed columns to keep memory low
    query = text(f"""
        SELECT
            rr.athlete_id,
            a.full_name,
            a.gender,
            a.country,
            rr.event_id,
            rr.prog_id,
            e.event_date,
            e.event_name,
            rr.position as finish_position,
            pm.position_at_swim, pm.position_at_t1, pm.position_at_bike, pm.position_at_t2, pm.position_at_run,
            pm.behindswim, pm.behindt1, pm.behindbike, pm.behindt2, pm.behindrun,
            pm.swim_to_t1_pos_change, pm.t1_to_bike_pos_change, pm.bike_to_t2_pos_change, pm.t2_to_run_pos_change,
            pm.swimrank, pm.t1rank, pm.bikerank, pm.t2rank, pm.runrank
        FROM race_results rr
        JOIN athlete a ON rr.athlete_id = a.athlete_id
        JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
        LEFT JOIN position_metrics pm ON pm.event_id = rr.event_id AND pm.prog_id = rr.prog_id AND pm.athlete_id = rr.athlete_id
        WHERE {where_clause}
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["event_date"])  # type: ignore

    return df


def list_wtcs_event_names(engine: Engine, filters: WTCSFilters) -> pd.DataFrame:
    """Return distinct WTCS event names that match the flexible patterns (debug aid for UI)."""
    params = {
        "broad_world": "%World Triathlon%",
        "broad_series": "%Series%",
        "broad_finals": "%Finals%",
    }
    for idx, pat in enumerate(WTCS_NAME_PATTERNS):
        params[f"pat_{idx}"] = f"%{pat}%"
    date_filters = []
    if filters.start_date:
        date_filters.append("event_date >= :start_date")
        params["start_date"] = filters.start_date
    if filters.end_date:
        date_filters.append("event_date <= :end_date")
        params["end_date"] = filters.end_date
    # Accept either Series or Finals combined with World Triathlon, or any explicit patterns
    name_clause = "( (event_name ILIKE :broad_world AND (event_name ILIKE :broad_series OR event_name ILIKE :broad_finals)) OR " + " OR ".join([f"event_name ILIKE :pat_{i}" for i in range(len(WTCS_NAME_PATTERNS))]) + " )"
    where_parts = [name_clause]
    # Para filtering fallback (no is_para column): use event_name heuristic
    if filters.para_filter is True:
        where_parts.append("event_name ILIKE :para_pat")
        params["para_pat"] = "%Para%"
    elif filters.para_filter is False:
        where_parts.append("event_name NOT ILIKE :para_pat")
        params["para_pat"] = "%Para%"
    if date_filters:
        where_parts.extend(date_filters)
    where_sql = " AND ".join(where_parts)
    query = text(f"""
        SELECT DISTINCT event_name, event_date
        FROM events
        WHERE {where_sql}
        ORDER BY event_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["event_date"])  # type: ignore
    return df


def wtcs_diagnostics(engine: Engine, filters: WTCSFilters) -> dict:
    """Return lightweight diagnostics (distinct genders, countries, row counts) to aid empty-result debugging."""
    base_events = list_wtcs_event_names(engine, filters)
    result = {"matched_events": len(base_events)}
    try:
        sample_df = fetch_wtcs_us_dataset(engine, filters)
        result["rows_after_filters"] = len(sample_df)
        if not sample_df.empty:
            result["distinct_genders"] = sorted([str(x) for x in sample_df['gender'].dropna().unique()])
            if 'country' in sample_df.columns:
                result["distinct_countries"] = sorted([str(x) for x in sample_df['country'].dropna().unique()])
        else:
            result["distinct_genders"] = []
            result["distinct_countries"] = []
    except Exception as e:
        result["error"] = str(e)
    return result


def aggregate_checkpoint_metrics(df: pd.DataFrame, filters: WTCSFilters) -> pd.DataFrame:
    """Aggregate athlete-level averages across selected WTCS races.

    Returns a DataFrame with columns:
        athlete_id, full_name, races,
        avg_finish_position,
        avg_pos_swim, avg_pos_t1, avg_pos_bike, avg_pos_t2, avg_pos_run,
        avg_gap_swim, avg_gap_t1, avg_gap_bike, avg_gap_t2, avg_gap_run,
        avg_swim_rank, avg_bike_rank, avg_run_rank,
        avg_delta_swim_t1, avg_delta_t1_bike, avg_delta_bike_t2, avg_delta_t2_run
    Rows where races < filters.min_events are retained (caller can gray out) but flagged.
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = ["athlete_id", "full_name"]
    agg_map = {
        "finish_position": "mean",
        "position_at_swim": "mean",
        "position_at_t1": "mean",
        "position_at_bike": "mean",
        "position_at_t2": "mean",
        "position_at_run": "mean",
        "behindswim": "mean",
        "behindt1": "mean",
        "behindbike": "mean",
        "behindt2": "mean",
        "behindrun": "mean",
        "swimrank": "mean",
        "bikerank": "mean",
        "runrank": "mean",
        "swim_to_t1_pos_change": "mean",
        "t1_to_bike_pos_change": "mean",
        "bike_to_t2_pos_change": "mean",
        "t2_to_run_pos_change": "mean",
    }

    # --- Defensive Type Coercion ---------------------------------------------------------
    # Some columns (notably finish_position) can contain non-numeric tokens like 'DNF'.
    # Pandas groupby(mean) on object-dtype columns raises: TypeError: agg function failed.
    # We coerce all aggregation targets to numeric (NaN on failure) before grouping so
    # 'DNF' / blanks are excluded from averages gracefully.
    work = df.copy()
    for col in agg_map.keys():
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    summary = work.groupby(group_cols).agg(agg_map).reset_index()
    summary.rename(columns={
        "finish_position": "avg_finish_position",
        "position_at_swim": "avg_pos_swim",
        "position_at_t1": "avg_pos_t1",
        "position_at_bike": "avg_pos_bike",
        "position_at_t2": "avg_pos_t2",
        "position_at_run": "avg_pos_run",
        "behindswim": "avg_gap_swim",
        "behindt1": "avg_gap_t1",
        "behindbike": "avg_gap_bike",
        "behindt2": "avg_gap_t2",
        "behindrun": "avg_gap_run",
        "swimrank": "avg_swim_rank",
        "bikerank": "avg_bike_rank",
        "runrank": "avg_run_rank",
        "swim_to_t1_pos_change": "avg_delta_swim_t1",
        "t1_to_bike_pos_change": "avg_delta_t1_bike",
        "bike_to_t2_pos_change": "avg_delta_bike_t2",
        "t2_to_run_pos_change": "avg_delta_t2_run",
    }, inplace=True)

    counts = work.groupby(group_cols).size().reset_index(name="races")
    summary = summary.merge(counts, on=group_cols, how="left")
    summary["meets_min_events"] = summary["races"] >= filters.min_events

    # Order columns for readability
    ordered = [
        "athlete_id", "full_name", "races", "meets_min_events", "avg_finish_position",
        "avg_pos_swim", "avg_pos_t1", "avg_pos_bike", "avg_pos_t2", "avg_pos_run",
        "avg_gap_swim", "avg_gap_t1", "avg_gap_bike", "avg_gap_t2", "avg_gap_run",
        "avg_swim_rank", "avg_bike_rank", "avg_run_rank",
        "avg_delta_swim_t1", "avg_delta_t1_bike", "avg_delta_bike_t2", "avg_delta_t2_run"
    ]
    summary = summary[ordered]
    return summary


def select_best_worst_races(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small table mapping athlete -> best_event_id / worst_event_id (placeholder logic).

    Best = lowest finish_position; Worst = highest (excluding null). If ties, earliest event_date.
    """
    if df.empty:
        return pd.DataFrame()
    # Ensure finish_position numeric (remove non-numeric e.g., DNF)
    def to_numeric_pos(x):
        try:
            return int(x)
        except Exception:
            return None
    work = df.copy()
    work["finish_position_int"] = work["finish_position"].apply(to_numeric_pos)
    work = work[~work["finish_position_int"].isna()]
    if work.empty:
        return pd.DataFrame()
    # Sort for deterministic selection
    work.sort_values(["athlete_id", "finish_position_int", "event_date"], inplace=True)
    best = work.groupby("athlete_id").first().reset_index()[["athlete_id", "event_id"]]
    best.rename(columns={"event_id": "best_event_id"}, inplace=True)

    work.sort_values(["athlete_id", "finish_position_int", "event_date"], ascending=[True, False, True], inplace=True)
    worst = work.groupby("athlete_id").first().reset_index()[["athlete_id", "event_id"]]
    worst.rename(columns={"event_id": "worst_event_id"}, inplace=True)

    out = best.merge(worst, on="athlete_id", how="outer")
    return out


__all__ = [
    "WTCSFilters",
    "fetch_wtcs_us_dataset",
    "aggregate_checkpoint_metrics",
    "select_best_worst_races",
    "list_wtcs_event_names",
    "wtcs_diagnostics",
]


def coerce_finish_position(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy of df with numeric_finish_position column (NaN for non-numeric)."""
    out = df.copy()
    def _c(v):
        try:
            return int(str(v))
        except Exception:
            return pd.NA
    out["numeric_finish_position"] = out["finish_position"].apply(_c)
    return out


def melt_checkpoint_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form positions for line charts.

    Produces columns: athlete_id, full_name, event_date, event_id, prog_id, checkpoint, position
    Checkpoints included: swim, t1, bike, t2, run, finish (using numeric finish position)
    """
    base = coerce_finish_position(df)
    pos_cols = {
        "position_at_swim": "Swim",
        "position_at_t1": "T1",
        "position_at_bike": "Bike",
        "position_at_t2": "T2",
        "position_at_run": "Run",
    }
    records = base.melt(
        id_vars=["athlete_id", "full_name", "event_date", "event_name", "event_id", "prog_id", "numeric_finish_position"],
        value_vars=list(pos_cols.keys()),
        var_name="checkpoint_col",
        value_name="position"
    )
    records["checkpoint"] = records["checkpoint_col"].map(pos_cols)
    # Build explicit, unique column set
    cols = ["athlete_id", "full_name", "event_date", "event_name", "event_id", "prog_id", "checkpoint", "position"]
    records_sel = records[cols].copy()

    # Append finish as separate rows; derive once per (athlete,event,prog)
    finish_rows = (
        base.drop_duplicates(subset=["athlete_id", "event_id", "prog_id"]) [[
            "athlete_id", "full_name", "event_date", "event_name", "event_id", "prog_id", "numeric_finish_position"
        ]].rename(columns={"numeric_finish_position": "position"})
    )
    finish_rows["checkpoint"] = "Finish"
    finish_sel = finish_rows[["athlete_id", "full_name", "event_date", "event_name", "event_id", "prog_id", "checkpoint", "position"]].copy()

    long_df = pd.concat([records_sel, finish_sel], ignore_index=True)
    long_df.dropna(subset=["position"], inplace=True)
    return long_df

