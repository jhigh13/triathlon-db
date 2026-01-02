"""WTCS radar scoring utilities.

Implements an end-of-season "report card" style radar profile for athletes.

Scoring rules (locked per user spec):
- WTCS only; compare athlete to full WTCS field
- Separate by gender
- No combining programs: field is (event_id, prog_id)
- Categories: swim, bike, run, transitions
- Each category is a 0..1 "weakness" score per race (0 best), aggregated over season
- Minimum 3 races with valid category scores required; otherwise category is missing
- Convert to radar scale: 10 = best, 1 = weakest via: 10 - 9 * season_weakness

Notes:
- All per-race percentiles are computed within each (event_id, prog_id) group.
- For gap and delta metrics, "smaller is better" (more negative deltas are better).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from tri_analysis.wtcs_performance import WTCSFilters, WTCS_NAME_PATTERNS


@dataclass(frozen=True)
class WtcsRadarProfile:
    athlete_id: int
    full_name: str
    gender: str
    season_start: Optional[str]
    season_end: Optional[str]
    swim: Optional[float]
    bike: Optional[float]
    run: Optional[float]
    transitions: Optional[float]
    n_swim: int
    n_bike: int
    n_run: int
    n_transitions: int


def _group_non_null_count(df: pd.DataFrame, group_cols: list[str], col: str) -> pd.Series:
    return df.groupby(group_cols, dropna=False)[col].transform(lambda s: int(s.notna().sum()))


def _build_wtcs_name_clause() -> tuple[str, dict]:
    """Mirror WTCS name matching used elsewhere (events.event_name)."""
    params: dict = {}
    pattern_conditions = []
    for idx, pat in enumerate(WTCS_NAME_PATTERNS):
        key = f"pat_{idx}"
        params[key] = f"%{pat}%"
        pattern_conditions.append(f"e.event_name ILIKE :{key}")

    params["broad_world"] = "%World Triathlon%"
    params["broad_series"] = "%Series%"
    params["broad_finals"] = "%Finals%"
    broad_clause = "(e.event_name ILIKE :broad_world AND (e.event_name ILIKE :broad_series OR e.event_name ILIKE :broad_finals))"
    name_clause = "( " + broad_clause + " OR " + " OR ".join(pattern_conditions) + " )"
    return name_clause, params


def _fetch_athlete_identity(engine: Engine, athlete_id: int) -> tuple[str, str]:
    query = text("SELECT full_name, gender FROM athlete WHERE athlete_id = :aid")
    with engine.connect() as conn:
        row = conn.execute(query, {"aid": int(athlete_id)}).fetchone()
    if not row:
        # Fallbacks (should be rare if athlete table is populated for the athlete of interest)
        return str(athlete_id), ""
    return str(row[0]), str(row[1] or "")


def _list_athlete_wtcs_event_program_pairs(
    engine: Engine,
    athlete_id: int,
    season_start: Optional[str],
    season_end: Optional[str],
    para_filter: Optional[bool],
) -> pd.DataFrame:
    name_clause, params = _build_wtcs_name_clause()
    params["aid"] = int(athlete_id)
    conditions = ["rr.athlete_id = :aid", name_clause]

    if para_filter is True:
        conditions.append("e.event_name ILIKE :para_pat")
        params["para_pat"] = "%Para%"
    elif para_filter is False:
        conditions.append("e.event_name NOT ILIKE :para_pat")
        params["para_pat"] = "%Para%"
    if season_start:
        conditions.append("e.event_date >= :start_date")
        params["start_date"] = season_start
    if season_end:
        conditions.append("e.event_date <= :end_date")
        params["end_date"] = season_end

    where_clause = " AND ".join(conditions)
    query = text(
        f"""
        SELECT DISTINCT rr.event_id, rr.prog_id
        FROM race_results rr
        JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
        WHERE {where_clause}
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)  # type: ignore


def _fetch_field_for_event_program_pairs(
    engine: Engine,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    tuples = [(int(r.event_id), int(r.prog_id)) for r in pairs.itertuples(index=False)]
    # Build a VALUES list for efficient join
    values_sql = ", ".join([f"(:e{i}, :p{i})" for i in range(len(tuples))])
    params: dict = {}
    for i, (e, p) in enumerate(tuples):
        params[f"e{i}"] = e
        params[f"p{i}"] = p

    query = text(
        f"""
        WITH pairs(event_id, prog_id) AS (
            VALUES {values_sql}
        )
        SELECT
            rr.athlete_id,
            rr.event_id,
            rr.prog_id,
            e.event_date,
            e.event_name,
            pm.position_at_swim,
            pm.position_at_bike,
            pm.behindswim,
            pm.behindbike,
            pm.behindrun,
            pm.behindt1,
            pm.behindt2,
            pm.swimrank,
            pm.bikerank,
            pm.runrank,
            pm.t1rank,
            pm.t2rank,
            pm.elapsedswim,
            pm.elapsedbike,
            pm.elapsedrun
        FROM pairs
        JOIN race_results rr ON rr.event_id = pairs.event_id AND rr.prog_id = pairs.prog_id
        JOIN events e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        LEFT JOIN position_metrics pm
            ON pm.event_id = rr.event_id
           AND pm.prog_id = rr.prog_id
           AND pm.athlete_id = rr.athlete_id
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params, parse_dates=["event_date"])  # type: ignore


def _compute_scored_field_df(
    engine: Engine,
    athlete_id: int,
    season_start: Optional[str],
    season_end: Optional[str],
    para_filter: Optional[bool],
    bike_drop_lead_pack_penalty: float,
) -> tuple[pd.DataFrame, str, str]:
    """Fetch WTCS field for the athlete's WTCS races and compute per-race scores + percentiles.

    IMPORTANT: This uses race_results + position_metrics for the field so we don't depend on
    athlete dimension rows existing for all non-USA athletes.
    """
    full_name, gender = _fetch_athlete_identity(engine, athlete_id)

    pairs = _list_athlete_wtcs_event_program_pairs(
        engine,
        athlete_id=athlete_id,
        season_start=season_start,
        season_end=season_end,
        para_filter=para_filter,
    )
    if pairs.empty:
        raise ValueError(f"No WTCS event/program pairs found for athlete_id={athlete_id} in selected season/window")

    df = _fetch_field_for_event_program_pairs(engine, pairs)
    if df.empty:
        raise ValueError("No WTCS field rows found for the athlete's event/program pairs")

    # Normalize types
    numeric_cols = [
        "swimrank",
        "bikerank",
        "runrank",
        "t1rank",
        "t2rank",
        "behindswim",
        "behindbike",
        "behindrun",
        "behindt1",
        "behindt2",
        "position_at_swim",
        "position_at_bike",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = _to_numeric(df[c])

    # Derived deltas
    df["bike_gap_delta"] = df["behindbike"] - df["behindswim"]
    df["bike_pos_delta"] = df["position_at_bike"] - df["position_at_swim"]
    df["t1_loss"] = df["behindt1"] - df["behindswim"]
    df["t2_loss"] = df["behindt2"] - df["behindbike"]

    have_packs = all(c in df.columns for c in ["pack_id_swim", "pack_id_bike"])
    if have_packs:
        df["pack_id_swim"] = _to_numeric(df["pack_id_swim"])
        df["pack_id_bike"] = _to_numeric(df["pack_id_bike"])
        df["dropped_from_lead_pack"] = (df["pack_id_swim"] == 0) & (df["pack_id_bike"] > 0)
    else:
        df["dropped_from_lead_pack"] = pd.NA

    group_cols = ["event_id", "prog_id"]

    # Full-field group sizes (after WTCS filters). Helpful for debugging.
    df["field_rows_n"] = df.groupby(group_cols, dropna=False)["athlete_id"].transform("size")
    df["field_athletes_n"] = df.groupby(group_cols, dropna=False)["athlete_id"].transform("nunique")

    def gpct(col: str) -> pd.Series:
        return df.groupby(group_cols, dropna=False)[col].transform(_pct_rank_within_group)

    # Percentiles for inputs (0 best)
    df["swimrank_pct"] = gpct("swimrank")
    df["bikerank_pct"] = gpct("bikerank")
    df["runrank_pct"] = gpct("runrank")
    df["t1rank_pct"] = gpct("t1rank")
    df["t2rank_pct"] = gpct("t2rank")

    df["behindswim_pct"] = gpct("behindswim")
    df["behindbike_pct"] = gpct("behindbike")
    df["behindrun_pct"] = gpct("behindrun")

    df["bike_gap_delta_pct"] = gpct("bike_gap_delta")
    df["bike_pos_delta_pct"] = gpct("bike_pos_delta")

    df["t1_loss_pct"] = gpct("t1_loss")
    df["t2_loss_pct"] = gpct("t2_loss")

    # Group sizes used for each percentile (helps debug small fields / missing values)
    df["swimrank_n"] = _group_non_null_count(df, group_cols, "swimrank")
    df["bikerank_n"] = _group_non_null_count(df, group_cols, "bikerank")
    df["runrank_n"] = _group_non_null_count(df, group_cols, "runrank")
    df["t1rank_n"] = _group_non_null_count(df, group_cols, "t1rank")
    df["t2rank_n"] = _group_non_null_count(df, group_cols, "t2rank")

    df["behindswim_n"] = _group_non_null_count(df, group_cols, "behindswim")
    df["behindbike_n"] = _group_non_null_count(df, group_cols, "behindbike")
    df["behindrun_n"] = _group_non_null_count(df, group_cols, "behindrun")
    df["behindt1_n"] = _group_non_null_count(df, group_cols, "behindt1")
    df["behindt2_n"] = _group_non_null_count(df, group_cols, "behindt2")

    df["bike_gap_delta_n"] = _group_non_null_count(df, group_cols, "bike_gap_delta")
    df["bike_pos_delta_n"] = _group_non_null_count(df, group_cols, "bike_pos_delta")
    df["t1_loss_n"] = _group_non_null_count(df, group_cols, "t1_loss")
    df["t2_loss_n"] = _group_non_null_count(df, group_cols, "t2_loss")

    # Category scores per race (0..1 weakness)
    df["swim_score"] = _weighted_mean_ignore_na(
        {"rank": df["swimrank_pct"], "gap": df["behindswim_pct"]}, {"rank": 0.7, "gap": 0.3}
    )
    df["run_score"] = _weighted_mean_ignore_na(
        {"rank": df["runrank_pct"], "gap": df["behindrun_pct"]}, {"rank": 0.7, "gap": 0.3}
    )

    engine_score = _weighted_mean_ignore_na(
        {"rank": df["bikerank_pct"], "gap": df["behindbike_pct"]}, {"rank": 0.5, "gap": 0.5}
    )
    racecraft_score = _weighted_mean_ignore_na(
        {"gap_delta": df["bike_gap_delta_pct"], "pos_delta": df["bike_pos_delta_pct"]},
        {"gap_delta": 0.5, "pos_delta": 0.5},
    )
    df["bike_score"] = _weighted_mean_ignore_na(
        {"engine": engine_score, "racecraft": racecraft_score}, {"engine": 0.5, "racecraft": 0.5}
    )

    if have_packs and bike_drop_lead_pack_penalty > 0:
        penalty = float(bike_drop_lead_pack_penalty)
        df.loc[df["dropped_from_lead_pack"] == True, "bike_score"] = (
            df.loc[df["dropped_from_lead_pack"] == True, "bike_score"] + penalty
        ).clip(upper=1.0)

    df["transitions_score"] = _weighted_mean_ignore_na(
        {"t1": df["t1_loss_pct"], "t2": df["t2_loss_pct"]}, {"t1": 0.5, "t2": 0.5}
    )

    return df, full_name, gender


def compute_wtcs_radar_swim_breakdown(
    engine: Engine,
    athlete_id: int,
    season_start: Optional[str] = None,
    season_end: Optional[str] = None,
    para_filter: Optional[bool] = False,
) -> pd.DataFrame:
    """Return per-race swim components for debugging radar scores."""
    df, _, _ = _compute_scored_field_df(
        engine,
        athlete_id=athlete_id,
        season_start=season_start,
        season_end=season_end,
        para_filter=para_filter,
        bike_drop_lead_pack_penalty=0.10,
    )
    a = df[df["athlete_id"] == athlete_id].copy()
    if a.empty:
        return pd.DataFrame()

    a["swim_strength"] = 10.0 - 9.0 * _to_numeric(a["swim_score"])  # per-race mapping

    cols = [
        "event_date",
        "event_name",
        "event_id",
        "prog_id",
        "field_rows_n",
        "field_athletes_n",
        "swimrank",
        "swimrank_n",
        "swimrank_pct",
        "behindswim",
        "behindswim_n",
        "behindswim_pct",
        "swim_score",
        "swim_strength",
    ]
    cols = [c for c in cols if c in a.columns]
    out = a[cols].sort_values(["event_date", "event_name"]).reset_index(drop=True)
    return out


def compute_wtcs_radar_bike_breakdown(
    engine: Engine,
    athlete_id: int,
    season_start: Optional[str] = None,
    season_end: Optional[str] = None,
    para_filter: Optional[bool] = False,
    bike_drop_lead_pack_penalty: float = 0.10,
) -> pd.DataFrame:
    """Return per-race bike components for debugging radar scores."""
    df, _, _ = _compute_scored_field_df(
        engine,
        athlete_id=athlete_id,
        season_start=season_start,
        season_end=season_end,
        para_filter=para_filter,
        bike_drop_lead_pack_penalty=bike_drop_lead_pack_penalty,
    )
    a = df[df["athlete_id"] == athlete_id].copy()
    if a.empty:
        return pd.DataFrame()

    a["bike_strength"] = 10.0 - 9.0 * _to_numeric(a["bike_score"])  # per-race mapping

    cols = [
        "event_date",
        "event_name",
        "event_id",
        "prog_id",
        "field_rows_n",
        "field_athletes_n",
        "bikerank",
        "bikerank_n",
        "bikerank_pct",
        "behindbike",
        "behindbike_n",
        "behindbike_pct",
        "bike_gap_delta",
        "bike_gap_delta_n",
        "bike_gap_delta_pct",
        "bike_pos_delta",
        "bike_pos_delta_n",
        "bike_pos_delta_pct",
        "dropped_from_lead_pack",
        "bike_score",
        "bike_strength",
    ]
    cols = [c for c in cols if c in a.columns]
    out = a[cols].sort_values(["event_date", "event_name"]).reset_index(drop=True)
    return out


def compute_wtcs_radar_run_breakdown(
    engine: Engine,
    athlete_id: int,
    season_start: Optional[str] = None,
    season_end: Optional[str] = None,
    para_filter: Optional[bool] = False,
) -> pd.DataFrame:
    """Return per-race run components for debugging radar scores."""
    df, _, _ = _compute_scored_field_df(
        engine,
        athlete_id=athlete_id,
        season_start=season_start,
        season_end=season_end,
        para_filter=para_filter,
        bike_drop_lead_pack_penalty=0.10,
    )
    a = df[df["athlete_id"] == athlete_id].copy()
    if a.empty:
        return pd.DataFrame()

    a["run_strength"] = 10.0 - 9.0 * _to_numeric(a["run_score"])  # per-race mapping

    cols = [
        "event_date",
        "event_name",
        "event_id",
        "prog_id",
        "field_rows_n",
        "field_athletes_n",
        "runrank",
        "runrank_n",
        "runrank_pct",
        "behindrun",
        "behindrun_n",
        "behindrun_pct",
        "run_score",
        "run_strength",
    ]
    cols = [c for c in cols if c in a.columns]
    out = a[cols].sort_values(["event_date", "event_name"]).reset_index(drop=True)
    return out


def compute_wtcs_radar_transitions_breakdown(
    engine: Engine,
    athlete_id: int,
    season_start: Optional[str] = None,
    season_end: Optional[str] = None,
    para_filter: Optional[bool] = False,
) -> pd.DataFrame:
    """Return per-race transitions components for debugging radar scores."""
    df, _, _ = _compute_scored_field_df(
        engine,
        athlete_id=athlete_id,
        season_start=season_start,
        season_end=season_end,
        para_filter=para_filter,
        bike_drop_lead_pack_penalty=0.10,
    )
    a = df[df["athlete_id"] == athlete_id].copy()
    if a.empty:
        return pd.DataFrame()

    a["transitions_strength"] = 10.0 - 9.0 * _to_numeric(a["transitions_score"])  # per-race mapping

    cols = [
        "event_date",
        "event_name",
        "event_id",
        "prog_id",
        "field_rows_n",
        "field_athletes_n",
        "t1rank",
        "t1rank_n",
        "t1rank_pct",
        "t2rank",
        "t2rank_n",
        "t2rank_pct",
        "behindt1",
        "behindt1_n",
        "behindt2",
        "behindt2_n",
        "t1_loss",
        "t1_loss_n",
        "t1_loss_pct",
        "t2_loss",
        "t2_loss_n",
        "t2_loss_pct",
        "transitions_score",
        "transitions_strength",
    ]
    cols = [c for c in cols if c in a.columns]
    out = a[cols].sort_values(["event_date", "event_name"]).reset_index(drop=True)
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pct_rank_within_group(values: pd.Series) -> pd.Series:
    """Return 0..1 percentile rank within group, 0 best.

    Uses rank(method='min') so ties get the best percentile for that value.
    If group has fewer than 2 non-null values, returns 0.0 for those non-null rows.
    """
    s = _to_numeric(values)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    mask = s.notna()
    if mask.sum() == 0:
        return out

    n = int(mask.sum())
    if n <= 1:
        out.loc[mask] = 0.0
        return out

    r = s.loc[mask].rank(method="min", ascending=True)
    out.loc[mask] = (r - 1.0) / (n - 1.0)
    return out


def _weighted_mean_ignore_na(values: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """Row-wise weighted mean, renormalizing weights when some values are NaN."""
    keys = [k for k in weights.keys()]
    df = pd.DataFrame({k: values[k] for k in keys})
    w = pd.Series({k: float(weights[k]) for k in keys})

    present = df.notna()
    w_present = present.mul(w, axis=1)
    denom = w_present.sum(axis=1)
    numer = (df.fillna(0.0) * w_present).sum(axis=1)
    out = numer / denom
    out = out.where(denom > 0)
    return out


def compute_wtcs_radar_profile(
    engine: Engine,
    athlete_id: int,
    season_start: Optional[str] = None,
    season_end: Optional[str] = None,
    para_filter: Optional[bool] = False,
    bike_drop_lead_pack_penalty: float = 0.10,
) -> WtcsRadarProfile:
    """Compute WTCS radar profile for one athlete vs full WTCS field.

    `para_filter` default False (Championship / non-para WTCS only) to match typical WTCS focus.
    Pass None to include para, or True for para-only.

    Returns radar values on 1..10 (10 best), or None for categories with <3 races.
    """
    df, full_name, gender = _compute_scored_field_df(
        engine,
        athlete_id=athlete_id,
        season_start=season_start,
        season_end=season_end,
        para_filter=para_filter,
        bike_drop_lead_pack_penalty=bike_drop_lead_pack_penalty,
    )

    # Filter to athlete rows; then aggregate season with min-3 rule
    a = df[df["athlete_id"] == athlete_id].copy()

    def agg_category(col: str) -> tuple[Optional[float], int]:
        s = _to_numeric(a[col]).dropna()
        n = int(s.shape[0])
        if n < 3:
            return None, n
        season_weakness = float(s.mean())
        # 10 = best (lowest weakness), 1 = weakest (highest weakness)
        radar_value = 10.0 - 9.0 * season_weakness
        return float(round(radar_value, 2)), n

    swim, n_swim = agg_category("swim_score")
    bike, n_bike = agg_category("bike_score")
    run, n_run = agg_category("run_score")
    transitions, n_transitions = agg_category("transitions_score")

    return WtcsRadarProfile(
        athlete_id=int(athlete_id),
        full_name=full_name,
        gender=gender,
        season_start=season_start,
        season_end=season_end,
        swim=swim,
        bike=bike,
        run=run,
        transitions=transitions,
        n_swim=n_swim,
        n_bike=n_bike,
        n_run=n_run,
        n_transitions=n_transitions,
    )
