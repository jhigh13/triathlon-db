#!/usr/bin/env python
"""
venue_analysis.py — Historical race analysis for a given venue.

Queries the triathlon-db PostgreSQL database and produces a PowerPoint report
with swim/bike/run split analysis, pack dynamics, and weather history.

Usage:
    python scripts/venue_analysis.py --venue Yokohama --years 8
    python scripts/venue_analysis.py --venue Yokohama --years 8 --gender men
    python scripts/venue_analysis.py --venue Yokohama --years 8 --output report.pptx
"""

import argparse
import io
import os
import re
import sys
from collections import Counter
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Allow running from repo root or scripts/ subdirectory
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "ppt files")
USAT_LOGO_PATH = os.path.join(REPO_ROOT, "docs", "power_bi_files", "USA_Triathlon_Logo.jpg")

# ── Brand colours (matching USAT reference deck) ──────────────────────────────
NAVY = RGBColor(0x00, 0x20, 0x60)       # #002060
RED = RGBColor(0xC0, 0x00, 0x00)        # #C00000
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
MID_GRAY = RGBColor(0xBF, 0xBF, 0xBF)
DARK_GRAY = RGBColor(0x26, 0x26, 0x26)
LIGHT_NAVY = RGBColor(0x00, 0x47, 0xAB)  # accent blue for charts

# Chart colour palette
C_NAVY = "#002060"
C_RED = "#C00000"
C_LIGHT_BLUE = "#4472C4"
C_LIGHT_RED = "#FF6B6B"
C_GRAY = "#808080"
C_ORANGE = "#E76F51"
C_GREEN = "#2A9D8F"
C_GOLD = "#F4A261"
C_VIOLET = "#9B5DE5"

FONT = "Arial"
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# Matplotlib global style
plt.rcParams.update({
    "font.family": "Arial",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_engine():
    load_dotenv(override=True)
    uri = os.environ.get(
        "DB_URI",
        "postgresql+psycopg://postgres:Bc020406%21@localhost:5432/triathlon_results",
    )
    return create_engine(uri)


def parse_time_to_seconds(t) -> float | None:
    """Convert 'H:MM:SS', 'MM:SS', or raw-second strings to float seconds."""
    if t is None or str(t).strip() in ("", "None", "nan"):
        return None
    t = str(t).strip()
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(t)
    except (ValueError, IndexError):
        return None


def adjust_outlier(series: pd.Series, threshold: float = 2.0) -> pd.Series:
    """If the smallest positive value is >threshold× smaller than the 2nd smallest, mark it NA.
    Mirrors the logic in tri_analysis/metrics.py."""
    valid = series[series > 0]
    if len(valid) < 2:
        return series
    sorted_vals = valid.sort_values()
    if sorted_vals.iloc[0] * threshold < sorted_vals.iloc[1]:
        return series.mask(series == sorted_vals.iloc[0], pd.NA)
    return series


def seconds_to_mmss(s: float | None, always_hours: bool = False) -> str:
    if s is None:
        return "—"
    s = int(round(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0 or always_hours:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def query_venue_events(engine, venue: str, years_back: int, gender_filter: str) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=years_back * 365)
    gender_clauses = []
    if gender_filter in ("men", "both"):
        gender_clauses.append("(e.prog_name ILIKE '%Elite Men%' AND e.prog_name NOT ILIKE '%Women%')")
    if gender_filter in ("women", "both"):
        gender_clauses.append("e.prog_name ILIKE '%Elite Women%'")
    gender_sql = " OR ".join(gender_clauses)

    sql = text(f"""
        SELECT
            e.event_id, e.prog_id,
            e.event_name, e.event_date, e.event_venue,
            e.cat_name, e.prog_name, e.prog_distance_category,
            e.swim_distance, e.bike_distance, e.run_distance,
            e.swim_laps, e.bike_laps, e.run_laps,
            e.temperature_air, e.temperature_water,
            e.wind, e.wetsuit, e.weather,
            e.wind_speed_kmh, e.wind_gust_kmh,
            e.apparent_temp, e.precipitation_mm
        FROM events e
        WHERE e.event_venue ILIKE :venue_pat
          AND e.event_date >= :cutoff
          AND e.event_date <= :today
          AND ({gender_sql})
          AND (e.is_para IS NULL OR e.is_para IS NOT TRUE)
        ORDER BY e.event_date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={
            "venue_pat": f"%{venue}%",
            "cutoff": cutoff,
            "today": date.today(),
        })
    return df


def query_splits(engine, event_id: int, prog_id: int) -> dict:
    sql = text("""
        SELECT swimtime, biketime, runtime, total_time
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid AND finish_status = 'FINISH'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"eid": event_id, "pid": prog_id})

    result = {}
    for col, key in [("swimtime", "swim"), ("biketime", "bike"), ("runtime", "run"), ("total_time", "total")]:
        secs = df[col].apply(parse_time_to_seconds).dropna()
        result[f"{key}_fastest"] = float(secs.min()) if len(secs) else None
        result[f"{key}_avg"] = float(secs.mean()) if len(secs) else None
        result[f"{key}_n"] = len(secs)
    return result


def query_winner(engine, event_id: int, prog_id: int) -> dict:
    sql = text("""
        SELECT athlete_full_name, total_time
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid AND finish_position = 1
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()
    if row:
        return {"winner_name": row[0], "winner_time": row[1]}
    return {"winner_name": None, "winner_time": None}


def query_swim_exit_group(engine, event_id: int, prog_id: int) -> int | None:
    sql = text("""
        SELECT COUNT(*)
        FROM position_metrics pm
        JOIN race_results rr USING (event_id, prog_id, athlete_id)
        WHERE pm.event_id = :eid AND pm.prog_id = :pid
          AND rr.finish_status = 'FINISH'
          AND pm.behindswim <= 15
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()
    return int(row[0]) if row and row[0] else None


def query_bike_evolution(engine, event_id: int, prog_id: int) -> dict:
    """Lead pack = athletes within 15s of leader at T1/T2.
    Chase group = athletes within 15s of the first athlete outside the lead pack.
    All data from position_metrics for consistency across all race tiers."""
    sql = text("""
        SELECT
            COUNT(CASE WHEN pm.behindt1   <= 15 THEN 1 END)               AS t1_lead_n,
            MAX(CASE WHEN pm.behindt1     <= 15 THEN pm.behindt1   END)   AS t1_lead_tail,
            COUNT(CASE WHEN pm.behindbike <= 15 THEN 1 END)               AS t2_lead_n,
            MAX(CASE WHEN pm.behindbike   <= 15 THEN pm.behindbike END)   AS t2_lead_tail,
            MIN(CASE WHEN pm.behindbike   >  15 THEN pm.behindbike END)   AS t2_chase_head
        FROM position_metrics pm
        JOIN race_results rr USING (event_id, prog_id, athlete_id)
        WHERE pm.event_id = :eid AND pm.prog_id = :pid
          AND rr.finish_status = 'FINISH'
    """)
    result: dict = {
        "t1_lead_pack": None, "t2_lead_pack": None,
        "t2_chase_gap": None, "t2_chase_pack": None,
    }

    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()

    if not row or row[0] is None:
        return result

    result["t1_lead_pack"] = int(row[0]) if row[0] else None
    result["t2_lead_pack"] = int(row[2]) if row[2] else None

    # Chase gap: seconds from last lead-pack athlete to first chase athlete at T2
    if row[4] is not None and row[3] is not None:
        result["t2_chase_gap"] = max(0, int(row[4]) - int(row[3]))

    # Chase group: athletes within 15s of the first chase athlete (symmetric window)
    if row[4] is not None:
        chase_head = float(row[4])
        sql2 = text("""
            SELECT COUNT(*)
            FROM position_metrics pm
            JOIN race_results rr USING (event_id, prog_id, athlete_id)
            WHERE pm.event_id = :eid AND pm.prog_id = :pid
              AND rr.finish_status = 'FINISH'
              AND pm.behindbike > 15
              AND pm.behindbike <= :chase_window
        """)
        with engine.connect() as conn:
            row2 = conn.execute(sql2, {
                "eid": event_id, "pid": prog_id,
                "chase_window": chase_head + 15,
            }).fetchone()
        result["t2_chase_pack"] = int(row2[0]) if row2 and row2[0] else None

    return result


def query_dropout_profile(engine, event_id: int, prog_id: int) -> list[dict]:
    """For each DNF/LAP athlete, determine their last valid checkpoint and
    whether they were in the lead group at that point.

    Valid = non-zero split AND not an outlier (>2× faster than 2nd fastest).
    Mirrors the logic in tri_analysis/metrics.py:adjust_outlier.
    """
    sql = text("""
        SELECT athlete_id, athlete_full_name, finish_status,
               swimtime, t1time, biketime, t2time, runtime
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid
          AND finish_status IN ('DNF', 'LAP', 'DNS')
    """)
    sql_lead = text("""
        SELECT swimtime, t1time, biketime, t2time, runtime
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid
          AND finish_status = 'FINISH'
    """)

    with engine.connect() as conn:
        dnf_df = pd.read_sql(sql, conn, params={"eid": event_id, "pid": prog_id})
        fin_df = pd.read_sql(sql_lead, conn, params={"eid": event_id, "pid": prog_id})

    if dnf_df.empty or fin_df.empty:
        return []

    def _to_secs(series: pd.Series) -> pd.Series:
        return series.apply(lambda t: parse_time_to_seconds(t) or 0.0)

    # Build cumulative elapsed for finishers to get leader time at each checkpoint
    fs = pd.DataFrame()
    fs["swim"]  = _to_secs(fin_df["swimtime"])
    fs["t1"]    = fs["swim"] + _to_secs(fin_df["t1time"])
    fs["bike"]  = fs["t1"]  + _to_secs(fin_df["biketime"])

    # Apply outlier filter per checkpoint (same threshold as metrics.py)
    for col in ["swim", "t1", "bike"]:
        fs[col] = adjust_outlier(fs[col][fs[col] > 0].reindex(fs.index, fill_value=0))

    leader_swim = float(fs["swim"][fs["swim"] > 0].min())
    leader_t1   = float(fs["t1"][fs["t1"] > 0].min())
    leader_bike = float(fs["bike"][fs["bike"] > 0].min())

    results = []
    for _, row in dnf_df.iterrows():
        swim_s  = parse_time_to_seconds(row["swimtime"])  or 0.0
        t1_s    = parse_time_to_seconds(row["t1time"])    or 0.0
        bike_s  = parse_time_to_seconds(row["biketime"])  or 0.0

        elapsed_swim = swim_s
        elapsed_t1   = swim_s + t1_s
        elapsed_bike = elapsed_t1 + bike_s

        # Determine last valid checkpoint (non-zero, non-outlier)
        # A split is suspicious if it's >2× faster than the fastest finisher split
        def _is_outlier_vs_leader(elapsed: float, leader: float) -> bool:
            return leader > 0 and elapsed > 0 and elapsed * 2 < leader

        last_cp   = None
        behind    = None

        if elapsed_swim > 0 and not _is_outlier_vs_leader(elapsed_swim, leader_swim):
            last_cp = "swim"
            behind  = round(elapsed_swim - leader_swim)
        if elapsed_t1 > 0 and t1_s > 0 and not _is_outlier_vs_leader(elapsed_t1, leader_t1):
            last_cp = "t1"
            behind  = round(elapsed_t1 - leader_t1)
        if elapsed_bike > 0 and bike_s > 0 and not _is_outlier_vs_leader(elapsed_bike, leader_bike):
            last_cp = "bike"
            behind  = round(elapsed_bike - leader_bike)

        if last_cp is None:
            continue  # no valid data

        results.append({
            "name":        row["athlete_full_name"],
            "status":      row["finish_status"],
            "last_cp":     last_cp,
            "behind":      behind,
            "in_lead_grp": behind is not None and behind <= 15,
        })

    return results


def query_field_size(engine, event_id: int, prog_id: int) -> tuple[int | None, int | None]:
    """Returns (finishers, non_finishers) where non_finishers = DNS+DNF+LAP."""
    sql = text("""
        SELECT
            COUNT(*) FILTER (WHERE finish_status = 'FINISH')     AS finishers,
            COUNT(*) FILTER (WHERE finish_status != 'FINISH')    AS non_finishers
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()
    if row:
        return (int(row[0]) or None, int(row[1]) or None)
    return (None, None)


def query_position_times(engine, event_id: int, prog_id: int) -> dict:
    """Return total_time in seconds for positions 1, 2, 3, 5, 10, 20."""
    positions = [1, 2, 3, 5, 10, 20]
    sql = text("""
        SELECT finish_position, total_time, athlete_full_name
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid
          AND finish_position = ANY(:positions)
          AND finish_status = 'FINISH'
        ORDER BY finish_position
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"eid": event_id, "pid": prog_id, "positions": positions})

    result = {}
    winner_secs = None
    for _, row in df.iterrows():
        pos = int(row.finish_position)
        secs = parse_time_to_seconds(row.total_time)
        result[pos] = {"secs": secs, "name": row.athlete_full_name, "time_str": row.total_time}
        if pos == 1:
            winner_secs = secs

    # Compute gaps from winner
    for pos in positions:
        if pos in result and winner_secs and result[pos]["secs"]:
            result[pos]["gap"] = result[pos]["secs"] - winner_secs
        elif pos in result:
            result[pos]["gap"] = None

    return result


def collect_race_data(engine, events_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, ev in events_df.iterrows():
        eid, pid = int(ev.event_id), int(ev.prog_id)
        splits = query_splits(engine, eid, pid)
        if splits.get("total_n", 0) == 0:
            print(f"  Skipping {ev.event_date} {ev.prog_name} — no results (cancelled or unavailable)")
            continue
        winner = query_winner(engine, eid, pid)
        swim_group = query_swim_exit_group(engine, eid, pid)
        bike = query_bike_evolution(engine, eid, pid)
        dropouts = query_dropout_profile(engine, eid, pid)
        field_size, non_finishers = query_field_size(engine, eid, pid)
        field_size = splits.get("total_n") or field_size or None
        pos_times = query_position_times(engine, eid, pid)

        rows.append({
            "event_id": eid,
            "prog_id": pid,
            "year": pd.to_datetime(ev.event_date).year,
            "date": ev.event_date,
            "event_name": ev.event_name,
            "cat_name": ev.cat_name or "",
            "prog_name": ev.prog_name or "",
            "prog_distance_category": ev.prog_distance_category or "",
            "swim_km": ev.swim_distance,
            "bike_km": ev.bike_distance,
            "run_km": ev.run_distance,
            "bike_laps": ev.bike_laps,
            "run_laps": ev.run_laps,
            "field_size": field_size,
            "non_finishers": non_finishers,
            "swim_exit_group": swim_group,
            "t1_lead_pack":   bike["t1_lead_pack"],
            "t2_lead_pack":   bike["t2_lead_pack"],
            "t2_chase_gap":   bike["t2_chase_gap"],
            "t2_chase_pack":  bike["t2_chase_pack"],
            "dropouts":       dropouts,
            "temp_air": ev.temperature_air,
            "temp_water": ev.temperature_water,
            "wind_raw": ev.wind,
            "wind_kmh": ev.wind_speed_kmh,
            "wetsuit": ev.wetsuit,
            "weather": ev.weather,
            # Open-Meteo enriched fields (filled later by enrich_rows_with_openmeteo)
            "humidity": None,
            "precip": None,
            "pm25": None,
            "aqi": None,
            "uv_index": None,
            "winner_name": winner["winner_name"],
            "winner_time": winner["winner_time"],
            "pos_times": pos_times,
            **splits,
        })
    return rows


# ── Deep-dive helpers (detailed splits, season norms, startlist) ──────────────

def load_detailed_splits(xlsx_path: str) -> dict[str, pd.DataFrame]:
    """Load detailed per-lap splits Excel for a single race.

    Returns {'men': df, 'women': df} with columns:
        Rank, Bib, Name, Nat,
        swim_sec, bike_sec, run_sec, total_sec,
        end_swim_elapsed, end_bike_elapsed, end_run_elapsed,
        dnf (bool — True if Total contained DNF/LAP)

    Handles both Male/female sheet naming variants and both
    T1.1/T2.1 (women) and TA1/TA2 (men) discipline-column variants.
    """
    xl = pd.ExcelFile(xlsx_path)
    sheets_lc = {n.lower(): n for n in xl.sheet_names}
    out: dict[str, pd.DataFrame] = {}
    for gender, candidates in [
        ("men",   ["male", "men", "m"]),
        ("women", ["female", "women", "f"]),
    ]:
        sheet = next((sheets_lc[c] for c in candidates if c in sheets_lc), None)
        if not sheet:
            continue
        df = pd.read_excel(xl, sheet_name=sheet)
        cols = list(df.columns)

        t1_col = "T1.1" if "T1.1" in cols else ("TA1" if "TA1" in cols else None)
        t2_col = "T2.1" if "T2.1" in cols else ("TA2" if "TA2" in cols else None)

        for col, target in [("Swim", "swim_sec"), ("Bike", "bike_sec"), ("Run", "run_sec")]:
            df[target] = df[col].apply(parse_time_to_seconds) if col in cols else None

        df["t1_sec"] = df[t1_col].apply(parse_time_to_seconds) if t1_col else 0
        df["t2_sec"] = df[t2_col].apply(parse_time_to_seconds) if t2_col else 0

        # Non-finishers are marked DNF/DNS/LAP in the Rank column (Total is 'n.a.')
        rank_str = df["Rank"].astype(str)
        total_str = df["Total"].astype(str)
        df["dnf"] = (
            rank_str.str.contains(r"DNF|DNS|LAP", case=False, na=False)
            | total_str.str.contains(r"DNF|DNS|LAP", case=False, na=False)
        )
        df["total_sec"] = df["Total"].apply(parse_time_to_seconds)

        df["end_swim_elapsed"] = df["swim_sec"]
        bike_cumulative = (
            df["swim_sec"].fillna(0)
            + df["t1_sec"].fillna(0)
            + df["bike_sec"].fillna(0)
        )
        df["end_bike_elapsed"] = bike_cumulative.where(
            df["swim_sec"].notna() & df["bike_sec"].notna(),
            pd.NA,
        )
        df["end_run_elapsed"] = df["total_sec"]

        keep = ["Rank", "Bib", "Name", "Nat",
                "swim_sec", "bike_sec", "run_sec", "total_sec",
                "end_swim_elapsed", "end_bike_elapsed", "end_run_elapsed",
                "dnf"]
        out[gender] = df[[c for c in keep if c in df.columns]].copy()
    return out


def compute_pack_scatter(df: pd.DataFrame) -> dict[str, dict]:
    """For each leg, compute scatter data: gap-to-leader vs placement-at-checkpoint.

    Returns dict keyed by leg label ('Swim', 'Bike', 'Run') with:
        {"points": [{"name": ..., "gap": ..., "placement": ..., "in_lead_swim": bool}],
         "leader_time": float | None}

    Swim-exit lead pack = athletes within 15s of the swim leader.
    Run leg includes finishers only (DNF/LAP excluded).
    """
    out: dict[str, dict] = {}

    swim_valid = df[df["end_swim_elapsed"].notna()].copy()
    if swim_valid.empty:
        lead_swim_names: set = set()
    else:
        swim_leader = float(swim_valid["end_swim_elapsed"].min())
        lead_swim_names = set(
            swim_valid.loc[swim_valid["end_swim_elapsed"] - swim_leader <= 15, "Name"]
        )

    for leg_col, label, finishers_only in [
        ("end_swim_elapsed", "Swim", False),
        ("end_bike_elapsed", "Bike", False),
        ("end_run_elapsed",  "Run",  True),
    ]:
        sub = df[df[leg_col].notna()].copy()
        if finishers_only:
            sub = sub[~sub["dnf"]]
        if sub.empty:
            out[label] = {"points": [], "leader_time": None}
            continue
        sub = sub.sort_values(leg_col).reset_index(drop=True)
        leader_time = float(sub[leg_col].iloc[0])
        sub["gap"] = sub[leg_col].astype(float) - leader_time
        sub["placement"] = sub.index + 1
        out[label] = {
            "points": [
                {
                    "name": row["Name"],
                    "gap": float(row["gap"]),
                    "placement": int(row["placement"]),
                    "in_lead_swim": row["Name"] in lead_swim_names,
                }
                for _, row in sub.iterrows()
            ],
            "leader_time": leader_time,
        }
    return out


def map_excel_names_to_athlete_ids(engine, event_id: int, prog_id: int, names: list[str]) -> dict[str, int]:
    """Match Excel athlete names to athlete_ids via race_results for the same historical race."""
    sql = text("""
        SELECT athlete_id, athlete_full_name
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid AND athlete_id IS NOT NULL
    """)
    df = pd.read_sql(sql, engine, params={"eid": event_id, "pid": prog_id})
    lookup = {str(n).lower().strip(): int(aid) for aid, n in zip(df.athlete_id, df.athlete_full_name)}
    return {n: lookup[str(n).lower().strip()] for n in names if str(n).lower().strip() in lookup}


def query_wtcs_season_splits(engine, athlete_ids: list[int], venue_date,
                             gender: str) -> pd.DataFrame:
    """WTCS Standard-distance splits for the given athletes in the 12 months
    ending at venue_date. Excludes DNF/DNS/LAP/DSQ rows and the venue race itself.

    Returns columns: athlete_id, event_date, event_name, swim_sec, bike_sec, run_sec.
    """
    if not athlete_ids:
        return pd.DataFrame(columns=["athlete_id", "event_date", "event_name",
                                     "swim_sec", "bike_sec", "run_sec"])
    gender_clause = (
        "(e.prog_name ILIKE '%Elite Men%' AND e.prog_name NOT ILIKE '%Women%')"
        if gender.lower().startswith("m")
        else "e.prog_name ILIKE '%Elite Women%'"
    )
    sql = text(f"""
        SELECT rr.athlete_id, e.event_date, e.event_name, e.event_venue,
               rr.swimtime, rr.biketime, rr.runtime
        FROM race_results rr
        JOIN events e USING (event_id, prog_id)
        WHERE rr.athlete_id = ANY(:athlete_ids)
          AND e.cat_name ILIKE '%Championship Series%'
          AND (e.prog_distance_category IS NULL OR e.prog_distance_category ILIKE 'standard')
          AND e.event_date BETWEEN (:venue_date - INTERVAL '12 months') AND :venue_date
          AND {gender_clause}
          AND (rr.finish_status IS NULL OR rr.finish_status NOT IN ('DNF', 'DNS', 'DSQ', 'LAP'))
    """)
    df = pd.read_sql(sql, engine, params={"athlete_ids": list(athlete_ids),
                                          "venue_date": venue_date})
    for src, dst in [("swimtime", "swim_sec"), ("biketime", "bike_sec"), ("runtime", "run_sec")]:
        df[dst] = df[src].apply(parse_time_to_seconds)
    return df


def find_upcoming_event(engine, venue: str) -> dict | None:
    """Find the upcoming/most-recent-future event_id at this venue (men's prog as anchor)."""
    sql = text("""
        SELECT event_id, MIN(event_date) AS event_date, MIN(event_name) AS event_name
        FROM events
        WHERE (event_venue ILIKE :v OR event_name ILIKE :v)
          AND event_date >= CURRENT_DATE
        GROUP BY event_id
        ORDER BY event_date ASC
        LIMIT 1
    """)
    df = pd.read_sql(sql, engine, params={"v": f"%{venue}%"})
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def query_top_by_world_ranking(engine, gender: str, on_startlist_event_id: int | None,
                               limit: int = 3) -> pd.DataFrame:
    """Top N athletes by current world ranking. If on_startlist_event_id is given,
    restrict to athletes on that event's startlist."""
    cat_id = 13 if gender.lower().startswith("m") else 14
    if on_startlist_event_id:
        sql = text("""
            WITH latest AS (
              SELECT MAX(retrieved_at) AS dt FROM athlete_rankings WHERE ranking_cat_id = :cat
            )
            SELECT DISTINCT ar.athlete_id, ar.athlete_name,
                   ar.rank_position, ar.total_points
            FROM athlete_rankings ar
            JOIN latest l ON ar.retrieved_at = l.dt
            JOIN program_entries pe
              ON pe.athlete_id = ar.athlete_id
             AND pe.event_id = :eid
             AND pe.is_active = TRUE
             AND pe.entry_type = 'start'
            WHERE ar.ranking_cat_id = :cat
            ORDER BY ar.rank_position ASC
            LIMIT :lim
        """)
        return pd.read_sql(sql, engine, params={"cat": cat_id, "eid": on_startlist_event_id,
                                                "lim": limit})
    sql = text("""
        SELECT ar.athlete_id, ar.athlete_name, ar.rank_position, ar.total_points
        FROM athlete_rankings ar
        WHERE ar.ranking_cat_id = :cat
          AND ar.retrieved_at = (SELECT MAX(retrieved_at) FROM athlete_rankings WHERE ranking_cat_id = :cat)
        ORDER BY ar.rank_position ASC
        LIMIT :lim
    """)
    return pd.read_sql(sql, engine, params={"cat": cat_id, "lim": limit})


def query_prior_podium_on_startlist(engine, prior_event_id: int, prior_prog_id: int,
                                    upcoming_event_id: int) -> pd.DataFrame:
    """Athletes on the upcoming startlist who finished top-3 at the prior race."""
    sql = text("""
        SELECT rr.athlete_id, rr.athlete_full_name,
               rr.position_sort AS position, rr.total_time
        FROM race_results rr
        JOIN program_entries pe
          ON pe.athlete_id = rr.athlete_id
         AND pe.event_id = :upcoming
         AND pe.is_active = TRUE
         AND pe.entry_type = 'start'
        WHERE rr.event_id = :prior_eid AND rr.prog_id = :prior_pid
          AND rr.position_sort IS NOT NULL
          AND rr.position_sort <= 3
        ORDER BY rr.position_sort
    """)
    return pd.read_sql(sql, engine, params={"upcoming": upcoming_event_id,
                                            "prior_eid": prior_event_id,
                                            "prior_pid": prior_prog_id})


# ── Open-Meteo / geocoding helpers ────────────────────────────────────────────

_GEOCODE_CACHE: dict = {}
_WEATHER_CACHE: dict = {}
_AQI_CACHE:     dict = {}


def geocode_venue(venue: str) -> tuple | None:
    """Geocode a venue string via Nominatim OSM. Returns (lat, lon) or None."""
    if venue in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[venue]
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": venue, "format": "json", "limit": 1},
            headers={"User-Agent": "triathlon-venue-analysis/1.0"},
            timeout=10,
        )
        if r.ok and r.json():
            d = r.json()[0]
            result = (float(d["lat"]), float(d["lon"]))
            print(f"  [Geocode] '{venue}' -> {result[0]:.3f}, {result[1]:.3f}")
            _GEOCODE_CACHE[venue] = result
            return result
    except Exception as exc:
        print(f"  [Geocode] Failed for '{venue}': {exc}")
    _GEOCODE_CACHE[venue] = None
    return None


def _ometa_race_avg(hourly: dict, times: list, col: str) -> float | None:
    """Average an Open-Meteo hourly field over the 08:00–11:59 race window."""
    idxs = [i for i, t in enumerate(times) if len(t) >= 13 and "08" <= t[11:13] <= "11"]
    if not idxs:
        idxs = list(range(8, 12))
    vals = [hourly[col][i] for i in idxs
            if i < len(hourly.get(col, [])) and hourly[col][i] is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def fetch_openmeteo_weather(lat: float, lon: float, race_date: str) -> dict:
    """Fetch race-window weather from Open-Meteo archive (hours 08–11 local)."""
    key = (round(lat, 2), round(lon, 2), race_date)
    if key in _WEATHER_CACHE:
        return _WEATHER_CACHE[key]
    result: dict = {}
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": race_date, "end_date": race_date,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                "wind_speed_unit": "kmh",
                "timezone": "auto",
            },
            timeout=15,
        )
        if r.ok:
            hrs = r.json().get("hourly", {})
            times = hrs.get("time", [])
            result = {
                "temp_air_om":  _ometa_race_avg(hrs, times, "temperature_2m"),
                "humidity_om":  _ometa_race_avg(hrs, times, "relative_humidity_2m"),
                "wind_kmh_om":  _ometa_race_avg(hrs, times, "wind_speed_10m"),
                "precip_om":    _ometa_race_avg(hrs, times, "precipitation"),
            }
    except Exception as exc:
        print(f"  [OpenMeteo weather] {race_date}: {exc}")
    _WEATHER_CACHE[key] = result
    return result


def fetch_openmeteo_airquality(lat: float, lon: float, race_date: str) -> dict:
    """Fetch race-window air quality from Open-Meteo AQ archive."""
    key = (round(lat, 2), round(lon, 2), race_date)
    if key in _AQI_CACHE:
        return _AQI_CACHE[key]
    result: dict = {}
    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": race_date, "end_date": race_date,
                "hourly": "pm2_5,european_aqi,uv_index",
                "timezone": "auto",
            },
            timeout=15,
        )
        if r.ok:
            hrs = r.json().get("hourly", {})
            times = hrs.get("time", [])
            result = {
                "pm25": _ometa_race_avg(hrs, times, "pm2_5"),
                "aqi":  _ometa_race_avg(hrs, times, "european_aqi"),
                "uv":   _ometa_race_avg(hrs, times, "uv_index"),
            }
    except Exception as exc:
        print(f"  [OpenMeteo AQI] {race_date}: {exc}")
    _AQI_CACHE[key] = result
    return result


_MARINE_CACHE: dict = {}


def fetch_openmeteo_marine(lat: float, lon: float, race_date: str) -> dict:
    """Fetch sea surface temperature from Open-Meteo Marine API (global coverage)."""
    key = (round(lat, 2), round(lon, 2), race_date)
    if key in _MARINE_CACHE:
        return _MARINE_CACHE[key]
    result: dict = {}
    try:
        r = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": race_date, "end_date": race_date,
                "hourly": "sea_surface_temperature",
                "timezone": "auto",
            },
            timeout=15,
        )
        if r.ok:
            hrs = r.json().get("hourly", {})
            times = hrs.get("time", [])
            sst = _ometa_race_avg(hrs, times, "sea_surface_temperature")
            if sst is not None:
                result["sst"] = sst
    except Exception as exc:
        print(f"  [OpenMeteo Marine] {race_date}: {exc}")
    _MARINE_CACHE[key] = result
    return result


def enrich_rows_with_openmeteo(rows: list[dict], coords: tuple | None) -> None:
    """Fill weather + AQ + water temp fields in-place using Open-Meteo where WT API left NULLs."""
    if coords is None:
        return
    lat, lon = coords
    for row in rows:
        race_date = str(pd.to_datetime(row["date"]).date())
        wx     = fetch_openmeteo_weather(lat, lon, race_date)
        aq     = fetch_openmeteo_airquality(lat, lon, race_date)
        marine = fetch_openmeteo_marine(lat, lon, race_date)
        # Fill temp/wind only if WT API returned nothing (mark with * so slide notes source)
        if not row.get("temp_air") and wx.get("temp_air_om") is not None:
            row["temp_air"] = f"{wx['temp_air_om']}°*"
        if not row.get("wind_kmh") and wx.get("wind_kmh_om") is not None:
            row["wind_kmh"] = wx["wind_kmh_om"]
        # Water temp from Marine API where WT API is absent
        if not row.get("temp_water") and marine.get("sst") is not None:
            row["temp_water"] = f"{marine['sst']:.1f}°*"
        # Humidity and precip always from Open-Meteo
        row["humidity"] = wx.get("humidity_om")
        row["precip"]   = wx.get("precip_om")
        # Air quality always from Open-Meteo
        row["pm25"]     = aq.get("pm25")
        row["aqi"]      = aq.get("aqi")
        row["uv_index"] = aq.get("uv")


# ── Venue content helpers ──────────────────────────────────────────────────────

def _build_wt_event_url(event_name: str) -> str:
    """Convert an event name to a triathlon.org event URL slug."""
    slug = event_name.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug.strip())
    return f"https://www.triathlon.org/events/event/{slug}"


def fetch_venue_content(venue: str, events_df: pd.DataFrame) -> dict:
    """Build WT event URLs from DB data; try the API for the upcoming event date."""
    content: dict = {
        "wt_event_urls": {},   # {year: url}
        "last_year_url": None,
        "upcoming_event_url": None,
    }

    # Build WT event URLs from DB event names (deduped by event_id)
    for _, row in events_df.drop_duplicates(subset=["event_id"]).iterrows():
        yr = pd.to_datetime(row.event_date).year
        content["wt_event_urls"][yr] = _build_wt_event_url(str(row.event_name))

    if content["wt_event_urls"]:
        last_yr = max(content["wt_event_urls"])
        content["last_year_url"] = content["wt_event_urls"][last_yr]

    this_year = date.today().year
    content["upcoming_event_url"] = (
        content["wt_event_urls"].get(this_year)
        or content["wt_event_urls"].get(this_year + 1)
    )

    # Try the WT API for the upcoming event slug (more reliable than constructed URL)
    load_dotenv(override=True)
    api_key = os.getenv("TRI_API_KEY")
    if api_key:
        try:
            r = requests.get(
                "https://api.triathlon.org/v1/events",
                headers={"apikey": api_key},
                params={"q": venue, "limit": 20},
                timeout=10,
            )
            if r.ok:
                body = r.json()
                # API may return {"data": {"events": [...]}} or {"data": [...]}
                data_block = body.get("data", {})
                if isinstance(data_block, list):
                    event_list = data_block
                elif isinstance(data_block, dict):
                    event_list = data_block.get("events", [])
                else:
                    event_list = []
                for ev in event_list:
                    ev_date = str(ev.get("event_date", ""))
                    ev_name = str(ev.get("event_name", ""))
                    if ev_date >= str(date.today()) and "championship series" in ev_name.lower():
                        slug = ev.get("event_slug") or _build_wt_event_url(ev_name).split("/")[-1]
                        content["upcoming_event_url"] = (
                            f"https://www.triathlon.org/events/event/{slug}"
                        )
                        break
        except Exception as exc:
            print(f"  [Info] WT API fetch failed: {exc}")

    return content


def _derive_keys_to_success(rows: list[dict]) -> list[str]:
    """Analyze race data patterns and return 3 key success factors."""
    keys = []

    # 1. Swim exit importance
    swim_groups = [r["swim_exit_group"] for r in rows if r.get("swim_exit_group")]
    if swim_groups:
        avg = sum(swim_groups) / len(swim_groups)
        if avg >= 15:
            keys.append(
                f"Swim positioning matters — an average of {int(avg)} athletes exit the water "
                "within 15 sec of the leader. Entering the bike in the front pack is critical; "
                "conserve energy with a disciplined swim line rather than sprinting alone."
            )
        else:
            keys.append(
                f"Fast swimming is decisive — only ~{int(avg)} athletes typically leave the water "
                "together. A strong swim directly builds a race-winning gap before T1."
            )
    else:
        keys.append(
            f"[PLACEHOLDER: Key 1 — Describe swim strategy and how the swim exit shapes the race at {rows[0]['event_name'] if rows else 'this venue'}, "
            "e.g. whether a large group exits together or swim speed is decisive.]"
        )

    # 2. Bike / breakaway tendency
    packs = [r["lead_bike_pack"] for r in rows if r.get("lead_bike_pack")]
    if packs:
        avg = sum(packs) / len(packs)
        small_count = sum(1 for p in packs if p < 10)
        if small_count >= len(packs) / 2:
            keys.append(
                f"Breakaway bike legs are common — {small_count} of {len(packs)} races had fewer than 10 "
                "riders in the lead pack. Strong cyclists can create decisive gaps; athletes need "
                "the engine to push a high pace from the gun."
            )
        elif avg >= 20:
            keys.append(
                f"Group bike racing dominates (avg lead pack: {int(avg)} athletes). Surviving the bike "
                "in the front group is non-negotiable — the race is ultimately decided on the run."
            )
        else:
            keys.append(
                f"Mixed bike dynamics — the lead group averages {int(avg)} riders. Bike endurance keeps "
                "athletes in contention, but the run determines who stands on the podium."
            )
    else:
        keys.append(
            "[PLACEHOLDER: Key 2 — Describe bike strategy: is this a pack race or do breakaways form? "
            "What power/tactics are needed to stay in the lead group?]"
        )

    # 3. Run
    run_fastest = [r["run_fastest"] for r in rows if r.get("run_fastest")]
    run_avg = [r["run_avg"] for r in rows if r.get("run_avg")]
    if run_fastest and run_avg:
        avg_fast = sum(run_fastest) / len(run_fastest)
        avg_mean = sum(run_avg) / len(run_avg)
        pace_km = avg_fast / 10  # assumes 10 km run
        spread = int(avg_mean - avg_fast)
        keys.append(
            f"Run speed separates the podium — the fastest run pace averages ~{seconds_to_mmss(pace_km)}/km "
            f"with a ~{spread}s spread between fastest and average finisher. "
            "Either a pre-built lead from the bike or exceptional run speed is required to win."
        )
    else:
        keys.append(
            "[PLACEHOLDER: Key 3 — Describe run dynamics: is this a sprint finish or does the winner "
            "typically build a lead? What run pace is required to podium?]"
        )

    return keys


# ── PPTX helpers ───────────────────────────────────────────────────────────────

def _set_cell(cell, text_val: str, bold=False, font_size=10, color=None,
              align=PP_ALIGN.CENTER, bg_color=None, italic=False):
    """Set text, font, and background of a table cell."""
    tf = cell.text_frame
    tf.word_wrap = False
    para = tf.paragraphs[0]
    para.alignment = align
    # Clear any existing runs
    for run in para.runs:
        run.text = ""
    run = para.runs[0] if para.runs else para.add_run()
    run.text = str(text_val)
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONT
    run.font.color.rgb = color or DARK_GRAY

    if bg_color:
        from pptx.oxml.ns import qn
        from lxml import etree
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        # Remove existing fills
        for existing in tcPr.findall(qn("a:solidFill")):
            tcPr.remove(existing)
        solidFill = etree.SubElement(tcPr, qn("a:solidFill"))
        srgbClr = etree.SubElement(solidFill, qn("a:srgbClr"))
        srgbClr.set("val", str(bg_color))


def _add_textbox(slide, text: str, left, top, width, height,
                 font_size=12, bold=False, color=None, align=PP_ALIGN.LEFT,
                 italic=False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONT
    run.font.color.rgb = color or DARK_GRAY
    return txb


def add_slide_chrome(slide, title: str, subtitle: str = "", show_logo: bool = True):
    """Add the standard chrome: navy header bar, red accent line, USAT logo."""
    # Navy header bar
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # Title text in header
    _add_textbox(slide, title, Inches(0.25), Inches(0.05), Inches(10.5), Inches(0.65),
                 font_size=28, bold=True, color=WHITE)

    if subtitle:
        _add_textbox(slide, subtitle, Inches(0.25), Inches(0.68), Inches(10.5), Inches(0.35),
                     font_size=12, color=RGBColor(0xAA, 0xBB, 0xDD))

    # Red accent line
    accent = slide.shapes.add_shape(1, Inches(0), Inches(1.05), SLIDE_W, Inches(0.07))
    accent.fill.solid()
    accent.fill.fore_color.rgb = RED
    accent.line.fill.background()

    # USAT logo top-right
    if show_logo and os.path.exists(USAT_LOGO_PATH):
        slide.shapes.add_picture(USAT_LOGO_PATH, Inches(11.9), Inches(0.03), Inches(1.35), Inches(1.0))

    # Thin navy footer strip
    footer = slide.shapes.add_shape(1, Inches(0), Inches(7.35), SLIDE_W, Inches(0.15))
    footer.fill.solid()
    footer.fill.fore_color.rgb = NAVY
    footer.line.fill.background()


def fig_to_image(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Slide builders ─────────────────────────────────────────────────────────────

def add_title_slide(prs: Presentation, venue: str, years_back: int):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Full navy background
    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    # Red diagonal accent bar at right
    accent = slide.shapes.add_shape(1, Inches(12.5), Inches(0), Inches(0.83), SLIDE_H)
    accent.fill.solid()
    accent.fill.fore_color.rgb = RED
    accent.line.fill.background()

    # White horizontal rule
    rule = slide.shapes.add_shape(1, Inches(0.5), Inches(4.1), Inches(11.8), Inches(0.06))
    rule.fill.solid()
    rule.fill.fore_color.rgb = WHITE
    rule.line.fill.background()

    # USAT logo centered
    if os.path.exists(USAT_LOGO_PATH):
        slide.shapes.add_picture(USAT_LOGO_PATH, Inches(0.6), Inches(0.4), Inches(2.1), Inches(1.6))

    # Main title
    _add_textbox(slide, f"{venue.upper()} RACE HISTORY",
                 Inches(0.5), Inches(2.0), Inches(11.8), Inches(1.4),
                 font_size=54, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    _add_textbox(slide, f"Elite Analysis  ·  Past {years_back} Years  ·  {date.today().strftime('%B %Y')}",
                 Inches(0.5), Inches(4.3), Inches(11.8), Inches(0.6),
                 font_size=18, color=RGBColor(0xAA, 0xBB, 0xDD), align=PP_ALIGN.CENTER)

    _add_textbox(slide, "USA Triathlon High Performance",
                 Inches(0.5), Inches(5.0), Inches(11.8), Inches(0.5),
                 font_size=14, color=RGBColor(0x80, 0x90, 0xB0), align=PP_ALIGN.CENTER)


def add_section_divider(prs: Presentation, gender: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    red_bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.5), SLIDE_H)
    red_bar.fill.solid()
    red_bar.fill.fore_color.rgb = RED
    red_bar.line.fill.background()

    rule = slide.shapes.add_shape(1, Inches(0.8), Inches(4.2), Inches(11.5), Inches(0.06))
    rule.fill.solid()
    rule.fill.fore_color.rgb = WHITE
    rule.line.fill.background()

    _add_textbox(slide, f"ELITE {gender.upper()}",
                 Inches(0.8), Inches(2.3), Inches(11.5), Inches(1.8),
                 font_size=60, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    if os.path.exists(USAT_LOGO_PATH):
        slide.shapes.add_picture(USAT_LOGO_PATH, Inches(11.0), Inches(6.3), Inches(2.0), Inches(1.1))


def add_overview_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Results Summary  —  Elite {gender.title()}", venue)

    # Column definitions: (header, width_inches, align)
    cols = [
        ("Year",          0.65,  PP_ALIGN.CENTER),
        ("Date",          0.95,  PP_ALIGN.CENTER),
        ("Category",      1.9,   PP_ALIGN.LEFT),
        ("Field",         0.55,  PP_ALIGN.CENTER),
        ("DNF/LAP/DNS",   0.75,  PP_ALIGN.CENTER),
        ("Winner",        2.0,   PP_ALIGN.LEFT),
        ("Winner Time",   1.0,   PP_ALIGN.CENTER),
        ("Air Temp",      0.8,   PP_ALIGN.CENTER),
        ("Water Temp",    0.85,  PP_ALIGN.CENTER),
        ("Swim km",       0.7,   PP_ALIGN.CENTER),
        ("Bike km",       0.7,   PP_ALIGN.CENTER),
        ("Run km",        0.7,   PP_ALIGN.CENTER),
    ]
    n_rows = len(rows) + 1
    n_cols = len(cols)
    row_h = min(0.56, 5.9 / n_rows)
    total_w = sum(c[1] for c in cols)
    scale = 12.73 / total_w  # fit within slide width
    scaled_widths = [c[1] * scale for c in cols]

    tbl_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(0.3), Inches(1.25),
        Inches(12.73), Inches(row_h * n_rows)
    )
    tbl = tbl_shape.table
    for i, w in enumerate(scaled_widths):
        tbl.columns[i].width = Inches(w)

    # Header row
    for ci, (col_name, _, align) in enumerate(cols):
        _set_cell(tbl.cell(0, ci), col_name, bold=True, font_size=14,
                  color=WHITE, align=PP_ALIGN.CENTER, bg_color=NAVY)

    def _fmt_dist(val) -> str:
        """Format a distance value stored in either km or metres."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        # Values > 100 are assumed to be stored in metres; convert to km
        if val > 100:
            val = val / 1000.0
        return f"{val:.2f}" if val < 2 else f"{val:.1f}"

    def _dims_equal(a, b) -> bool:
        """Compare two distance tuples treating NaN as equal."""
        for x, y in zip(a, b):
            x_nan = x is None or (isinstance(x, float) and np.isnan(x))
            y_nan = y is None or (isinstance(y, float) and np.isnan(y))
            if x_nan and y_nan:
                continue
            if x_nan != y_nan or x != y:
                return False
        return True

    prev_dims = None
    footnote_needed = False
    for ri, row in enumerate(rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE

        current_dims = (row["swim_km"], row["bike_km"], row["run_km"])
        course_flag = ""
        if prev_dims is not None and not _dims_equal(prev_dims, current_dims):
            course_flag = " *"
            footnote_needed = True
        prev_dims = current_dims

        cat = (row["cat_name"] or row["prog_distance_category"] or "—").replace("World Triathlon Championship Series", "WTCS")
        winner_last = ""
        if row["winner_name"]:
            parts = row["winner_name"].split()
            winner_last = parts[-1] if parts else row["winner_name"]

        air   = str(row.get("temp_air")   or "—").strip()
        water = str(row.get("temp_water") or "—").strip()

        fs  = row.get("field_size") or 0
        dnf = row.get("non_finishers") or 0
        total_field = (fs + dnf) if (fs or dnf) else None

        vals = [
            str(row["year"]),
            pd.to_datetime(row["date"]).strftime("%b %d"),
            cat + course_flag,
            str(total_field) if total_field else "—",
            str(dnf) if dnf else "—",
            winner_last,
            seconds_to_mmss(parse_time_to_seconds(row["winner_time"])),
            air,
            water,
            _fmt_dist(row["swim_km"]),
            _fmt_dist(row["bike_km"]),
            _fmt_dist(row["run_km"]),
        ]
        for ci, (v, (col_name, _, align)) in enumerate(zip(vals, cols)):
            cell_font = 12 if col_name == "Category" else 14
            _set_cell(tbl.cell(ri, ci), v, font_size=cell_font, align=align, bg_color=bg)

    if footnote_needed:
        _add_textbox(slide, "* Course distance changed vs. prior year",
                     Inches(0.3), Inches(7.1), Inches(6), Inches(0.28),
                     font_size=8, color=MID_GRAY, italic=True)


def add_swim_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Swim Exit Dynamics  —  Elite {gender.title()}", venue)

    years = [str(r["year"]) for r in rows]
    group_sizes = [r["swim_exit_group"] or 0 for r in rows]
    fastest = [r["swim_fastest"] for r in rows]
    avg_swim = [r["swim_avg"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("white")

    # Left: swim exit group size
    ax1 = axes[0]
    bar_colors = [C_NAVY if g >= 20 else C_RED for g in group_sizes]
    bars = ax1.bar(years, group_sizes, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, group_sizes):
        if val:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(val), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333")
    ax1.set_title("Athletes Within 15 sec of Swim Leader", fontsize=13, pad=8)
    ax1.set_ylabel("# Athletes", fontsize=11)
    ax1.set_ylim(0, max(group_sizes + [1]) * 1.3)
    ax1.tick_params(axis="x", rotation=45, labelsize=10)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.set_facecolor("white")

    # Right: fastest vs avg swim time — dot/lollipop chart, zoomed y-axis
    ax2 = axes[1]
    x = np.arange(len(years))
    fast_secs = [f for f in fastest]
    avg_secs = [a for a in avg_swim]
    all_secs = [v for v in fast_secs + avg_secs if v]

    ax2.vlines(x, fast_secs, avg_secs, color=C_GRAY, linewidth=1.5, alpha=0.5, zorder=1)
    ax2.scatter(x, fast_secs, s=110, color=C_NAVY, zorder=3, label="Fastest", edgecolor="white", linewidth=1.2)
    ax2.scatter(x, avg_secs, s=110, color=C_LIGHT_BLUE, zorder=3, label="Average", edgecolor="white", linewidth=1.2)

    for xi, secs in zip(x, fast_secs):
        if secs:
            ax2.annotate(seconds_to_mmss(secs), (xi, secs), xytext=(0, -14),
                         textcoords="offset points", ha="center", fontsize=9,
                         fontweight="bold", color=C_NAVY)
    for xi, secs in zip(x, avg_secs):
        if secs:
            ax2.annotate(seconds_to_mmss(secs), (xi, secs), xytext=(0, 9),
                         textcoords="offset points", ha="center", fontsize=9, color=C_LIGHT_BLUE)

    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45, fontsize=10)
    ax2.set_title("Fastest vs. Average Swim Time", fontsize=13, pad=8)
    ax2.set_ylabel("Time (min:sec)", fontsize=11)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_facecolor("white")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v // 60)}:{int(round(v % 60)):02d}")
    )
    if all_secs:
        pad = max(8, (max(all_secs) - min(all_secs)) * 0.25)
        ax2.set_ylim(min(all_secs) - pad, max(all_secs) + pad)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout(pad=1.5)
    slide.shapes.add_picture(fig_to_image(fig), Inches(0.3), Inches(1.25), Inches(12.73), Inches(5.9))


def add_bike_evolution_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    """Bike pack evolution: lead group size at T1 vs T2, plus chase gap per year."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Bike Pack Evolution  —  Elite {gender.title()}", venue)

    _add_textbox(
        slide,
        "Lead pack = athletes within 15 sec of leader. "
        "T1 = lead group entering the bike; T2 = lead group exiting the bike.",
        Inches(0.3), Inches(1.18), Inches(12.73), Inches(0.38),
        font_size=10.5, color=DARK_GRAY, italic=True, align=PP_ALIGN.CENTER,
    )

    years       = [str(r["year"]) for r in rows]
    t1_sizes    = [r.get("t1_lead_pack") or 0 for r in rows]
    t2_sizes    = [r.get("t2_lead_pack") or 0 for r in rows]
    chase_gaps  = [r.get("t2_chase_gap")       for r in rows]  # seconds, may be None

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(years))

    # ── Left: T1 vs T2 pack size per year ─────────────────────────────────────
    # Draw connecting line (consolidation arrow) first
    for xi, (t1, t2) in enumerate(zip(t1_sizes, t2_sizes)):
        if t1 and t2:
            clr = C_GREEN if t2 > t1 else (C_RED if t2 < t1 else C_GRAY)
            ax_left.plot([xi, xi], [t1, t2], color=clr, linewidth=2, alpha=0.6, zorder=1)

    sc1 = ax_left.scatter(x, t1_sizes, s=120, color=C_LIGHT_BLUE, zorder=3,
                          label="T1 (swim exit)", edgecolor="white", linewidth=1.2)
    sc2 = ax_left.scatter(x, t2_sizes, s=120, color=C_NAVY, zorder=3,
                          label="T2 (bike exit)", edgecolor="white", linewidth=1.2)

    for xi, (t1, t2) in enumerate(zip(t1_sizes, t2_sizes)):
        if t1:
            ax_left.annotate(str(t1), (xi, t1), xytext=(0, 8),
                             textcoords="offset points", ha="center",
                             fontsize=9, color=C_LIGHT_BLUE, fontweight="bold")
        if t2:
            ax_left.annotate(str(t2), (xi, t2), xytext=(0, -14),
                             textcoords="offset points", ha="center",
                             fontsize=9, color=C_NAVY, fontweight="bold")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(years, rotation=45, fontsize=10)
    ax_left.set_title("Lead Pack Size: Entering T1 vs Exiting T2", fontsize=12, pad=8)
    ax_left.set_ylabel("Athletes in lead group", fontsize=11)
    ax_left.legend(fontsize=10, loc="upper left")
    ax_left.set_facecolor("white")
    ax_left.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    valid = [v for v in t1_sizes + t2_sizes if v]
    if valid:
        ax_left.set_ylim(0, max(valid) * 1.35)

    # ── Right: chase group gap at T2 ──────────────────────────────────────────
    # If lead pack > 75% of field, treat as one main group (show 7s sentinel)
    MAIN_GROUP_SENTINEL = 7
    adjusted_gaps = []
    is_main_group = []
    for r, g in zip(rows, chase_gaps):
        fs = r.get("field_size") or 0
        t2 = r.get("t2_lead_pack") or 0
        if fs and t2 and t2 >= 0.75 * fs:
            adjusted_gaps.append(MAIN_GROUP_SENTINEL)
            is_main_group.append(True)
        else:
            adjusted_gaps.append(g)
            is_main_group.append(False)

    gap_vals = [g if g is not None else 0 for g in adjusted_gaps]
    bar_colors = []
    for g, main in zip(adjusted_gaps, is_main_group):
        if g is None:    bar_colors.append(C_GRAY)
        elif main:       bar_colors.append(C_NAVY)   # one main group
        elif g <= 30:    bar_colors.append(C_GREEN)
        elif g <= 60:    bar_colors.append(C_GOLD)
        else:            bar_colors.append(C_RED)

    bars = ax_right.bar(years, gap_vals, color=bar_colors, edgecolor="white",
                        linewidth=0.5, width=0.55)
    for bar, g, main in zip(bars, adjusted_gaps, is_main_group):
        if g:
            label = "Main group" if main else f"{g}s"
            ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_right.set_title("Gap: Lead Pack → Chase Group at T2", fontsize=12, pad=8)
    ax_right.set_ylabel("Seconds behind lead pack", fontsize=11)
    ax_right.tick_params(axis="x", rotation=45, labelsize=10)
    ax_right.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_right.set_facecolor("white")

    from matplotlib.patches import Patch
    ax_right.legend(handles=[
        Patch(facecolor=C_NAVY,  label="One main group (>75% of field)"),
        Patch(facecolor=C_GREEN, label="≤30s (tight)"),
        Patch(facecolor=C_GOLD,  label="31–60s (gap)"),
        Patch(facecolor=C_RED,   label=">60s (breakaway)"),
        Patch(facecolor=C_GRAY,  label="No data"),
    ], fontsize=9, loc="upper right")

    plt.tight_layout(pad=1.5)
    slide.shapes.add_picture(fig_to_image(fig),
                             Inches(0.3), Inches(1.62), Inches(12.73), Inches(4.1))

    # ── Summary table ──────────────────────────────────────────────────────────
    tbl_cols = ["Year", "Lead Size @ T1", "Lead Size @ T2", "Change",
                "Chase Group @ T2", "Chase Gap*"]
    n_rows = 1 + len(rows)
    tbl_shape = slide.shapes.add_table(
        n_rows, len(tbl_cols),
        Inches(0.3), Inches(5.78), Inches(12.73), Inches(1.55),
    )
    tbl = tbl_shape.table
    col_w = [Inches(0.75), Inches(2.0), Inches(2.0), Inches(1.3),
             Inches(2.5), Inches(4.18)]
    for ci, w in enumerate(col_w):
        tbl.columns[ci].width = w

    for ci, hdr in enumerate(tbl_cols):
        _set_cell(tbl.cell(0, ci), hdr, bold=True, color=WHITE, bg_color=NAVY,
                  font_size=10)

    for ri, row in enumerate(rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        t1 = row.get("t1_lead_pack")
        t2 = row.get("t2_lead_pack")
        chg = ""
        if t1 and t2:
            diff = t2 - t1
            chg = f"+{diff}" if diff > 0 else str(diff)
        cg  = str(row.get("t2_chase_pack") or "—")
        gap = f"{row['t2_chase_gap']}s" if row.get("t2_chase_gap") is not None else "—"
        vals = [str(row["year"]), str(t1 or "—"), str(t2 or "—"),
                chg or "—", cg, gap]
        for ci, v in enumerate(vals):
            _set_cell(tbl.cell(ri, ci), v, font_size=9.5, bg_color=bg,
                      align=PP_ALIGN.CENTER)

    _add_textbox(slide,
                 "Lead pack = athletes within 15 sec of leader. "
                 "Chase group = athletes within 15 sec of the first athlete outside the lead pack. "
                 "* Chase Gap = seconds from the last lead-pack athlete to the first chase-group athlete at T2.",
                 Inches(0.3), Inches(7.18), Inches(12.73), Inches(0.22),
                 font_size=8, color=MID_GRAY, italic=True)

    # Dropout annotation: summarise DNF/LAP athletes per year that were in the lead group
    dropout_notes = []
    for r in rows:
        d_list = r.get("dropouts") or []
        if not d_list:
            continue
        in_lead = [d for d in d_list if d["in_lead_grp"]]
        total   = len(d_list)
        note = f"{r['year']}: {total} DNF/LAP"
        if in_lead:
            cps = ", ".join(sorted({d['last_cp'] for d in in_lead}))
            note += f" ({len(in_lead)} in lead group at {cps})"
        dropout_notes.append(note)

    if dropout_notes:
        _add_textbox(slide,
                     "Dropouts: " + "  |  ".join(dropout_notes),
                     Inches(0.3), Inches(7.35), Inches(12.73), Inches(0.18),
                     font_size=7.5, color=MID_GRAY, italic=True)


def add_run_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Run Split Analysis  —  Elite {gender.title()}", venue)

    years = [str(r["year"]) for r in rows]
    fastest = [r["run_fastest"] for r in rows]
    avg_run  = [r["run_avg"]     for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.3))
    fig.patch.set_facecolor("white")
    x = np.arange(len(years))

    fast_secs = [f for f in fastest]
    avg_secs = [a for a in avg_run]
    all_secs = [v for v in fast_secs + avg_secs if v]

    ax.vlines(x, fast_secs, avg_secs, color=C_GRAY, linewidth=1.5, alpha=0.5, zorder=1)
    ax.scatter(x, fast_secs, s=130, color=C_NAVY, zorder=3, label="Fastest", edgecolor="white", linewidth=1.2)
    ax.scatter(x, avg_secs, s=130, color=C_LIGHT_BLUE, zorder=3, label="Average", edgecolor="white", linewidth=1.2)

    for xi, secs in zip(x, fast_secs):
        if secs:
            ax.annotate(seconds_to_mmss(secs), (xi, secs), xytext=(0, -16),
                        textcoords="offset points", ha="center", fontsize=10,
                        fontweight="bold", color=C_NAVY)
    for xi, secs in zip(x, avg_secs):
        if secs:
            ax.annotate(seconds_to_mmss(secs), (xi, secs), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=10, color=C_LIGHT_BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, fontsize=11)
    ax.set_title("Fastest vs. Average Run Split by Year", fontsize=14, pad=10)
    ax.set_ylabel("Time (min:sec)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_facecolor("white")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v // 60)}:{int(round(v % 60)):02d}")
    )
    if all_secs:
        pad = max(15, (max(all_secs) - min(all_secs)) * 0.25)
        ax.set_ylim(min(all_secs) - pad, max(all_secs) + pad)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout(pad=1.5)
    slide.shapes.add_picture(fig_to_image(fig), Inches(0.5), Inches(1.22), Inches(12.3), Inches(5.95))


def add_position_times_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    """Line chart + table: times for 1st, 2nd, 3rd, 5th, 10th, 20th place per year."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Key Position Times  —  Elite {gender.title()}", venue)

    positions = [1, 2, 3, 5, 10, 20]
    pos_labels = {1: "1st", 2: "2nd", 3: "3rd", 5: "5th", 10: "10th", 20: "20th"}
    colors_pos = {1: C_NAVY, 2: C_RED, 3: C_ORANGE, 5: C_LIGHT_BLUE, 10: C_GREEN, 20: C_GRAY}

    years = [str(r["year"]) for r in rows]

    # ── Left: line chart of gap from winner (seconds) ──
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 3.4))
    fig.patch.set_facecolor("white")

    for pos in [2, 3, 5, 10, 20]:
        gaps = []
        for row in rows:
            pt = row["pos_times"].get(pos)
            gaps.append(pt["gap"] if pt and pt.get("gap") is not None else np.nan)
        ax_left.plot(years, gaps, marker="o", label=pos_labels[pos],
                     color=colors_pos[pos], linewidth=2, markersize=6)
        # Label endpoint
        last_valid = [(y, g) for y, g in zip(years, gaps) if not np.isnan(g)]
        if last_valid:
            ly, lg = last_valid[-1]
            ax_left.annotate(f"{int(round(lg))}s", xy=(ly, lg),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=7.5, color=colors_pos[pos], fontweight="bold")

    ax_left.set_title("Gap from Winner (seconds)", fontsize=11, pad=8)
    ax_left.set_ylabel("Seconds behind 1st place", fontsize=10)
    ax_left.tick_params(axis="x", rotation=45, labelsize=9)
    ax_left.set_facecolor("white")
    ax_left.legend(fontsize=9, loc="upper left")
    ax_left.axhline(0, color=C_NAVY, linewidth=0.8, alpha=0.4)

    # ── Right: bar chart of average gaps per position ──
    avg_gaps = []
    for pos in [2, 3, 5, 10, 20]:
        all_gaps = []
        for row in rows:
            pt = row["pos_times"].get(pos)
            if pt and pt.get("gap") is not None:
                all_gaps.append(pt["gap"])
        avg_gaps.append(np.mean(all_gaps) if all_gaps else 0)

    bar_labels = [pos_labels[p] for p in [2, 3, 5, 10, 20]]
    bar_clrs = [colors_pos[p] for p in [2, 3, 5, 10, 20]]
    bars = ax_right.bar(bar_labels, avg_gaps, color=bar_clrs, edgecolor="white", width=0.55)
    for bar, val in zip(bars, avg_gaps):
        if val:
            ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          f"{int(round(val))}s", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_right.set_title("Avg Gap from 1st Place (all years)", fontsize=11, pad=8)
    ax_right.set_ylabel("Seconds behind winner", fontsize=10)
    ax_right.set_facecolor("white")
    ax_right.tick_params(axis="x", labelsize=10)

    plt.tight_layout(pad=1.2)
    slide.shapes.add_picture(fig_to_image(fig), Inches(0.3), Inches(1.18), Inches(12.73), Inches(3.3))

    # ── Summary table: actual times per position per year ──
    tbl_rows = 1 + len(rows)
    tbl_cols = 1 + len(positions)
    tbl_top = Inches(4.58)
    row_h = min(0.38, 2.7 / tbl_rows)
    tbl_h = Inches(row_h * tbl_rows)
    tbl_shape = slide.shapes.add_table(tbl_rows, tbl_cols,
                                       Inches(0.3), tbl_top, Inches(12.73), tbl_h)
    tbl = tbl_shape.table
    col_widths = [Inches(0.7)] + [Inches(12.03 / len(positions))] * len(positions)
    for ci, w in enumerate(col_widths):
        tbl.columns[ci].width = w

    _set_cell(tbl.cell(0, 0), "Year", bold=True, color=WHITE, bg_color=NAVY)
    for ci, pos in enumerate(positions, start=1):
        _set_cell(tbl.cell(0, ci), pos_labels[pos], bold=True, color=WHITE, bg_color=NAVY)

    for ri, row in enumerate(rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        _set_cell(tbl.cell(ri, 0), str(row["year"]), bold=True, color=DARK_GRAY, bg_color=bg)
        for ci, pos in enumerate(positions, start=1):
            pt = row["pos_times"].get(pos)
            if pt and pt.get("secs"):
                time_str = seconds_to_mmss(pt["secs"])
                gap_str = f"+{int(round(pt['gap']))}s" if pt.get("gap") else ""
                val = f"{time_str}\n{gap_str}" if gap_str else time_str
            else:
                val = "—"
            _set_cell(tbl.cell(ri, ci), val, font_size=8.5, bg_color=bg)


def add_weather_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Race Conditions  —  Elite {gender.title()}", venue)

    cols = [
        ("Year",        0.65, PP_ALIGN.CENTER),
        ("Date",        0.95, PP_ALIGN.CENTER),
        ("Air Temp",    1.0,  PP_ALIGN.CENTER),
        ("Humidity",    0.9,  PP_ALIGN.CENTER),
        ("Wind (km/h)", 1.0,  PP_ALIGN.CENTER),
        ("Precip (mm)", 0.95, PP_ALIGN.CENTER),
        ("Water Temp",  0.95, PP_ALIGN.CENTER),
        ("Wetsuit",     0.9,  PP_ALIGN.CENTER),
        ("Conditions",  5.49, PP_ALIGN.LEFT),
    ]
    n_rows = len(rows) + 1
    row_h = min(0.50, 5.0 / n_rows)
    tbl_shape = slide.shapes.add_table(
        n_rows, len(cols),
        Inches(0.3), Inches(1.25),
        Inches(12.73), Inches(row_h * n_rows)
    )
    tbl = tbl_shape.table
    total_w = sum(c[1] for c in cols)
    scale = 12.73 / total_w
    for ci, (_, w, _) in enumerate(cols):
        tbl.columns[ci].width = Inches(w * scale)

    for ci, (name, _, _) in enumerate(cols):
        _set_cell(tbl.cell(0, ci), name, bold=True, font_size=12,
                  color=WHITE, align=PP_ALIGN.CENTER, bg_color=NAVY)

    for ri, row in enumerate(rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        wind_val = (f"{row['wind_kmh']:.0f}" if row.get("wind_kmh") else (row.get("wind_raw") or "—"))
        air    = str(row.get("temp_air")   or "—").strip()
        water  = str(row.get("temp_water") or "—").strip()
        wetsuit = str(row.get("wetsuit")   or "—").strip()
        weather = str(row.get("weather")   or "—").strip()
        humid  = f"{row['humidity']:.0f}%" if row.get("humidity") is not None else "—"
        precip = f"{row['precip']:.1f}" if row.get("precip") is not None else "—"
        if len(weather) > 80:
            weather = weather[:77] + "..."

        vals = [str(row["year"]), pd.to_datetime(row["date"]).strftime("%b %d"),
                air, humid, str(wind_val), precip, water, wetsuit, weather]
        for ci, (v, (_, _, align)) in enumerate(zip(vals, cols)):
            _set_cell(tbl.cell(ri, ci), v, font_size=10, align=align, bg_color=bg)

    _add_textbox(slide,
                 "Air temp, humidity, wind, and precipitation sourced from Open-Meteo historical archive "
                 "(race window 08:00–11:00 local) where WT API data is absent. "
                 "Values marked * are Open-Meteo estimates. "
                 "Water temp and wetsuit ruling are from the World Triathlon API per program.",
                 Inches(0.3), Inches(6.6), Inches(12.73), Inches(0.72),
                 font_size=9, color=MID_GRAY, italic=True)


def add_environmental_risk_slide(
    prs: Presentation, venue: str, unique_rows: list[dict]
):
    """Venue-level environmental risk: weather history + air quality + health placeholders."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Environmental Risk Profile", venue)

    # ── Top half: weather + AQ tables side by side ────────────────────────────
    # LEFT: weather table  (Year | Air Temp | Humidity | Wind | Precip)
    wx_cols = [
        ("Year",        0.65, PP_ALIGN.CENTER),
        ("Date",        0.85, PP_ALIGN.CENTER),
        ("Air Temp",    0.95, PP_ALIGN.CENTER),
        ("Humidity",    0.85, PP_ALIGN.CENTER),
        ("Wind km/h",   0.85, PP_ALIGN.CENTER),
        ("Precip mm",   0.85, PP_ALIGN.CENTER),
    ]
    n_rows = len(unique_rows) + 1
    row_h = min(0.44, 2.7 / n_rows)

    wx_shape = slide.shapes.add_table(
        n_rows, len(wx_cols),
        Inches(0.3), Inches(1.28),
        Inches(5.15), Inches(row_h * n_rows),
    )
    wx_tbl = wx_shape.table
    total_w = sum(c[1] for c in wx_cols)
    for ci, (_, w, _) in enumerate(wx_cols):
        wx_tbl.columns[ci].width = Inches(w / total_w * 5.15)
    for ci, (hdr, _, _) in enumerate(wx_cols):
        _set_cell(wx_tbl.cell(0, ci), hdr, bold=True, font_size=10,
                  color=WHITE, align=PP_ALIGN.CENTER, bg_color=NAVY)

    for ri, row in enumerate(unique_rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        air   = str(row.get("temp_air")  or "—").strip()
        humid = f"{row['humidity']:.0f}%" if row.get("humidity") is not None else "—"
        wind  = f"{row['wind_kmh']:.0f}" if row.get("wind_kmh") else (row.get("wind_raw") or "—")
        prec  = f"{row['precip']:.1f}"   if row.get("precip")   is not None else "—"
        vals  = [str(row["year"]), pd.to_datetime(row["date"]).strftime("%b %d"),
                 air, humid, str(wind), prec]
        for ci, (v, (_, _, align)) in enumerate(zip(vals, wx_cols)):
            _set_cell(wx_tbl.cell(ri, ci), v, font_size=10, align=align, bg_color=bg)

    # Weather section label
    wx_bar = slide.shapes.add_shape(1, Inches(0.3), Inches(1.18), Inches(5.15), Inches(0.1))
    wx_bar.fill.solid(); wx_bar.fill.fore_color.rgb = NAVY; wx_bar.line.fill.background()
    _add_textbox(slide, "RACE-DAY WEATHER (08:00–11:00 local)",
                 Inches(0.35), Inches(1.19), Inches(5.0), Inches(0.1),
                 font_size=9.5, bold=True, color=NAVY)

    # RIGHT: air quality table  (Year | PM2.5 | AQI | UV | Risk)
    def _aqi_label(aqi) -> str:
        if aqi is None: return "—"
        aqi = int(aqi)
        if aqi <= 20:  return "Good"
        if aqi <= 40:  return "Fair"
        if aqi <= 60:  return "Moderate"
        if aqi <= 80:  return "Poor"
        return "Very Poor"

    aq_cols = [
        ("Year",      0.65, PP_ALIGN.CENTER),
        ("PM2.5",     0.85, PP_ALIGN.CENTER),
        ("EU AQI",    0.75, PP_ALIGN.CENTER),
        ("UV Index",  0.85, PP_ALIGN.CENTER),
        ("Risk",      1.1,  PP_ALIGN.CENTER),
    ]
    aq_shape = slide.shapes.add_table(
        n_rows, len(aq_cols),
        Inches(5.65), Inches(1.28),
        Inches(4.45), Inches(row_h * n_rows),
    )
    aq_tbl = aq_shape.table
    total_w2 = sum(c[1] for c in aq_cols)
    for ci, (_, w, _) in enumerate(aq_cols):
        aq_tbl.columns[ci].width = Inches(w / total_w2 * 4.45)
    for ci, (hdr, _, _) in enumerate(aq_cols):
        _set_cell(aq_tbl.cell(0, ci), hdr, bold=True, font_size=10,
                  color=WHITE, align=PP_ALIGN.CENTER, bg_color=NAVY)

    for ri, row in enumerate(unique_rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        pm25 = f"{row['pm25']:.1f}" if row.get("pm25") is not None else "—"
        aqi  = f"{int(row['aqi'])}" if row.get("aqi") is not None else "—"
        uv   = f"{row['uv_index']:.1f}" if row.get("uv_index") is not None else "—"
        risk = _aqi_label(row.get("aqi"))
        vals = [str(row["year"]), pm25, aqi, uv, risk]
        for ci, (v, (_, _, align)) in enumerate(zip(vals, aq_cols)):
            _set_cell(aq_tbl.cell(ri, ci), v, font_size=10, align=align, bg_color=bg)

    _add_textbox(slide, "AIR QUALITY (Open-Meteo archive)",
                 Inches(5.7), Inches(1.19), Inches(4.3), Inches(0.1),
                 font_size=9.5, bold=True, color=NAVY)

    # ── Bottom: health / water quality risk section ───────────────────────────
    tbl_bottom = Inches(1.28 + row_h * n_rows + 0.25)

    risk_bar = slide.shapes.add_shape(1, Inches(0.3), tbl_bottom, Inches(12.73), Inches(0.35))
    risk_bar.fill.solid(); risk_bar.fill.fore_color.rgb = RED; risk_bar.line.fill.background()
    _add_textbox(slide, "HEALTH & WATER QUALITY RISK FACTORS",
                 Inches(0.35), tbl_bottom + Inches(0.02), Inches(12.2), Inches(0.3),
                 font_size=12, bold=True, color=WHITE)

    risk_items = [
        ("Water Quality",
         f"[PLACEHOLDER: Historical water quality data for {venue} — e.g. bacterial counts, "
         "EU Bathing Water Directive status, known algae / pollution events. "
         "Check local environmental agency reports for May race window.]"),
        ("Athlete Health Incidents",
         "[PLACEHOLDER: Any known GI illness, respiratory, or heat-related incidents at this "
         "or comparable venues. E.g. Sunderland 2023 — reported GI illness post-swim for multiple athletes.]"),
        ("Water Temperature",
         f"[PLACEHOLDER: Historical open-water temperature for {venue} in May "
         "(not available from Open-Meteo; check Copernicus Marine / NOAA or WT historical results).]"),
        ("Heat / Humidity Risk",
         "Auto-derived from above table — flag if humidity >70% or air temp >25°C at race time."),
    ]

    y_offset = tbl_bottom + Inches(0.42)
    item_h = (SLIDE_H - y_offset - Inches(0.2)) / len(risk_items)
    for label, text in risk_items:
        _add_textbox(slide, f"• {label}: {text}",
                     Inches(0.4), y_offset, Inches(12.5), item_h,
                     font_size=10, color=DARK_GRAY)
        y_offset += item_h

    _add_textbox(slide,
                 "Weather and AQ data: Open-Meteo historical archive (archive-api.open-meteo.com). "
                 "EU AQI: Good ≤20, Fair ≤40, Moderate ≤60, Poor ≤80, Very Poor >80.",
                 Inches(0.3), Inches(7.18), Inches(12.73), Inches(0.22),
                 font_size=8, color=MID_GRAY, italic=True)


def add_course_differentiators_slide(
    prs: Presentation, venue: str, all_rows: list[dict],
    events_df: pd.DataFrame, content: dict
):
    """Gender-neutral slide: venue overview, course features, quick links."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Key Course Differentiators", venue)

    # ── Left column: description + feature bullets ─────────────────────────────
    _add_textbox(slide, venue.upper(),
                 Inches(0.3), Inches(1.2), Inches(6.6), Inches(0.55),
                 font_size=20, bold=True, color=NAVY)

    venue_desc = (
        f"[PLACEHOLDER: 2-3 sentences on what makes {venue} unique as a race venue — "
        "geography, setting, and character. E.g. 'Set on the Coral Riviera of Sardinia, "
        "the WTCS Alghero course combines crystal-clear Mediterranean waters, a flat "
        "coastal bike circuit, and a multi-lap run through the ancient walled city.']"
    )
    _add_textbox(slide, venue_desc,
                 Inches(0.3), Inches(1.78), Inches(6.6), Inches(1.35),
                 font_size=11, color=DARK_GRAY)

    # Feature bullets header bar
    feat_bar = slide.shapes.add_shape(1, Inches(0.3), Inches(3.25), Inches(6.6), Inches(0.35))
    feat_bar.fill.solid()
    feat_bar.fill.fore_color.rgb = NAVY
    feat_bar.line.fill.background()
    _add_textbox(slide, "COURSE FEATURES",
                 Inches(0.35), Inches(3.27), Inches(6.2), Inches(0.3),
                 font_size=12, bold=True, color=WHITE)

    features = [
        f"[PLACEHOLDER: Feature 1 — e.g. 'Open-water swim in the Bay of {venue} — calm seas']",
        f"[PLACEHOLDER: Feature 2 — e.g. 'Flat coastal bike circuit — fast and tactical']",
        f"[PLACEHOLDER: Feature 3 — e.g. 'Multi-lap run through the historic city centre']",
        f"[PLACEHOLDER: Feature 4 — e.g. 'Hot Mediterranean conditions typical in late May']",
    ]
    for i, feat in enumerate(features):
        _add_textbox(slide, f"•  {feat}",
                     Inches(0.4), Inches(3.7 + i * 0.72), Inches(6.4), Inches(0.65),
                     font_size=11, color=DARK_GRAY)

    # ── Right column: course at a glance ───────────────────────────────────────
    glance_bar = slide.shapes.add_shape(1, Inches(7.2), Inches(1.2), Inches(5.8), Inches(0.35))
    glance_bar.fill.solid()
    glance_bar.fill.fore_color.rgb = NAVY
    glance_bar.line.fill.background()
    _add_textbox(slide, "COURSE AT A GLANCE",
                 Inches(7.25), Inches(1.22), Inches(5.4), Inches(0.3),
                 font_size=12, bold=True, color=WHITE)

    # Use most-recent complete row for course dimensions
    ref_row = next(
        (r for r in sorted(all_rows, key=lambda x: x["year"], reverse=True)
         if r.get("swim_km") or r.get("bike_km") or r.get("run_km")),
        None
    )

    def _fmt_d(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if v > 100:
            v = v / 1000.0
        return f"{v:.1f} km"

    def _fmt_laps(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return str(int(v))

    glance_items: list[tuple[str, str]] = []
    if ref_row:
        glance_items = [
            ("Swim",       _fmt_d(ref_row.get("swim_km"))),
            ("Bike",       _fmt_d(ref_row.get("bike_km"))),
            ("Run",        _fmt_d(ref_row.get("run_km"))),
            ("Format",     "Standard Triathlon"),
            ("Bike Laps",  _fmt_laps(ref_row.get("bike_laps"))),
            ("Run Laps",   _fmt_laps(ref_row.get("run_laps"))),
        ]
    else:
        glance_items = [
            ("Swim", "—"), ("Bike", "—"), ("Run", "—"),
            ("Format", "Standard Triathlon"), ("Bike Laps", "—"), ("Run Laps", "—"),
        ]

    for i, (label, val) in enumerate(glance_items):
        _add_textbox(slide, f"{label}:",
                     Inches(7.3), Inches(1.65 + i * 0.37), Inches(2.0), Inches(0.35),
                     font_size=12, bold=True, color=NAVY)
        _add_textbox(slide, val,
                     Inches(9.4), Inches(1.65 + i * 0.37), Inches(3.5), Inches(0.35),
                     font_size=12, color=DARK_GRAY)

    # ── Right column: quick links ──────────────────────────────────────────────
    links_y = 4.0
    links_bar = slide.shapes.add_shape(1, Inches(7.2), Inches(links_y), Inches(5.8), Inches(0.35))
    links_bar.fill.solid()
    links_bar.fill.fore_color.rgb = NAVY
    links_bar.line.fill.background()
    _add_textbox(slide, "QUICK LINKS",
                 Inches(7.25), Inches(links_y + 0.02), Inches(5.4), Inches(0.3),
                 font_size=12, bold=True, color=WHITE)

    upcoming_url = content.get("upcoming_event_url") or (
        f"[PLACEHOLDER: https://www.triathlon.org/events/event/"
        f"{date.today().year}_world_triathlon_championship_series_{venue.lower()}]"
    )
    last_url = (content.get("last_year_url") or "[PLACEHOLDER: World Triathlon results link]")

    links = [
        ("Current Race Info",   upcoming_url),
        ("Last Year's Results", last_url),
        ("Race Replay",         "[PLACEHOLDER: https://triathlonlive.tv/ (TriathlonLIVE replay)]"),
    ]
    for i, (label, url) in enumerate(links):
        _add_textbox(slide, label + ":",
                     Inches(7.3), Inches(links_y + 0.45 + i * 0.85), Inches(5.6), Inches(0.3),
                     font_size=11, bold=True, color=NAVY)
        _add_textbox(slide, url,
                     Inches(7.3), Inches(links_y + 0.78 + i * 0.85), Inches(5.6), Inches(0.42),
                     font_size=10, color=DARK_GRAY, italic=True)


def add_course_map_slide(
    prs: Presentation, venue: str, all_rows: list[dict], content: dict
):
    """Course profile and map slide with placeholder images + lap details table."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Course Profile & Map", venue)

    this_year = date.today().year

    for col_i, (label, yr_hint) in enumerate([
        (f"{this_year} Course", "upcoming course map or race photo"),
        (f"{this_year - 1} Course", "previous year course map or race photo"),
    ]):
        left = Inches(0.3 + col_i * 6.55)
        width = Inches(6.2)
        height = Inches(3.65)

        # Gray placeholder box
        ph = slide.shapes.add_shape(1, left, Inches(1.22), width, height)
        ph.fill.solid()
        ph.fill.fore_color.rgb = LIGHT_GRAY
        ph.line.color.rgb = MID_GRAY

        _add_textbox(slide,
                     f"[PLACEHOLDER: Insert {yr_hint} here]\n"
                     f"Right-click → Replace Image, or paste screenshot\n"
                     f"of the World Triathlon course map PDF.",
                     left + Inches(0.2), Inches(2.35), width - Inches(0.4), Inches(1.0),
                     font_size=10, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)

        _add_textbox(slide, label,
                     left, Inches(4.93), width, Inches(0.3),
                     font_size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

        src_url = (
            content.get("upcoming_event_url") if col_i == 0
            else content.get("last_year_url")
        ) or "[PLACEHOLDER: World Triathlon event page URL]"
        _add_textbox(slide, f"Source: {src_url}",
                     left, Inches(5.22), width, Inches(0.27),
                     font_size=9, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)

    # Bottom: course details table
    ref_row = next(
        (r for r in sorted(all_rows, key=lambda x: x["year"], reverse=True)
         if r.get("swim_km") or r.get("bike_km") or r.get("run_km")),
        None
    )

    def _fd(v, unit="") -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if v > 100:
            v = v / 1000.0
        return f"{v:.1f}{unit}"

    def _fl(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return str(int(v))

    r = ref_row or {}
    tbl_data = [
        ("Discipline", "Distance",           "Laps",              "Terrain / Notes"),
        ("Swim",       _fd(r.get("swim_km"), " km"), _fl(r.get("swim_laps")),
         f"[PLACEHOLDER: e.g. Open-water, Bay of {venue}]"),
        ("Bike",       _fd(r.get("bike_km"), " km"), _fl(r.get("bike_laps")),
         "[PLACEHOLDER: e.g. Flat coastal road circuit, closed course]"),
        ("Run",        _fd(r.get("run_km"),  " km"), _fl(r.get("run_laps")),
         "[PLACEHOLDER: e.g. Flat multi-lap, cobblestone city-centre finish]"),
    ]

    tbl_shape = slide.shapes.add_table(
        4, 4, Inches(0.3), Inches(5.42), Inches(12.73), Inches(1.8)
    )
    tbl = tbl_shape.table
    for ci, w in enumerate([Inches(1.6), Inches(1.7), Inches(1.0), Inches(8.43)]):
        tbl.columns[ci].width = w

    for ri, row_data in enumerate(tbl_data):
        for ci, val in enumerate(row_data):
            is_hdr = ri == 0
            bg = NAVY if is_hdr else (LIGHT_GRAY if ri % 2 == 0 else WHITE)
            clr = WHITE if is_hdr else DARK_GRAY
            align = PP_ALIGN.CENTER if ci < 3 else PP_ALIGN.LEFT
            _set_cell(tbl.cell(ri, ci), val, bold=is_hdr,
                      font_size=11 if is_hdr else 10, color=clr, bg_color=bg, align=align)


def add_race_overview_slide(
    prs: Presentation, gender: str, rows: list[dict], venue: str, content: dict
):
    """Race history paragraph + 3 Keys to Success for one gender."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Race Overview  —  Elite {gender.title()}", venue)

    # ── Left: Race history paragraph ───────────────────────────────────────────
    hist_bar = slide.shapes.add_shape(1, Inches(0.3), Inches(1.2), Inches(7.0), Inches(0.35))
    hist_bar.fill.solid()
    hist_bar.fill.fore_color.rgb = NAVY
    hist_bar.line.fill.background()
    _add_textbox(slide, f"RACE HISTORY AT {venue.upper()}",
                 Inches(0.35), Inches(1.22), Inches(6.6), Inches(0.3),
                 font_size=12, bold=True, color=WHITE)

    if rows:
        years = sorted(r["year"] for r in rows)
        year_range = f"{years[0]}–{years[-1]}"
        n = len(rows)
        winners = [r["winner_name"] for r in rows if r.get("winner_name")]
        winner_note = ""
        if winners:
            top = Counter(winners).most_common(1)[0]
            if top[1] > 1:
                winner_note = f" {top[0]} is the most successful athlete in this span with {top[1]} victories."
        cats = list(dict.fromkeys(
            r["cat_name"].replace("World Triathlon Championship Series", "WTCS")
            for r in rows if r.get("cat_name")
        ))
        cat_str = " / ".join(cats[:2])

        history_text = (
            f"The Elite {gender.title()} race at {venue} has been held {n} time{'s' if n > 1 else ''} "
            f"in this analysis window ({year_range}).{winner_note} "
            f"The event is staged as part of the {cat_str} calendar.\n\n"
            f"[PLACEHOLDER: Add 1–2 sentences on notable storylines, competitive history, "
            f"or what makes this event significant for Elite {gender.title()} athletes. "
            f"E.g. defending champions, record performances, dramatic finishes.]"
        )
    else:
        history_text = (
            f"[PLACEHOLDER: 3–4 sentences on the history of the Elite {gender.title()} race at {venue} — "
            "key moments, dominant athletes, course records, and the event's significance "
            "on the WTCS calendar.]"
        )

    _add_textbox(slide, history_text,
                 Inches(0.3), Inches(1.65), Inches(7.0), Inches(5.55),
                 font_size=12, color=DARK_GRAY)

    # ── Right: 3 Keys to Success ────────────────────────────────────────────────
    keys_bar = slide.shapes.add_shape(1, Inches(7.55), Inches(1.2), Inches(5.45), Inches(0.35))
    keys_bar.fill.solid()
    keys_bar.fill.fore_color.rgb = RED
    keys_bar.line.fill.background()
    _add_textbox(slide, "3 KEYS TO SUCCESS",
                 Inches(7.6), Inches(1.22), Inches(5.1), Inches(0.3),
                 font_size=12, bold=True, color=WHITE)

    keys = _derive_keys_to_success(rows)
    for i, key_text in enumerate(keys):
        top = Inches(1.7 + i * 1.85)
        # Numbered square badge
        badge = slide.shapes.add_shape(1, Inches(7.6), top, Inches(0.42), Inches(0.42))
        badge.fill.solid()
        badge.fill.fore_color.rgb = NAVY
        badge.line.fill.background()
        tf = badge.text_frame
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        run = tf.paragraphs[0].add_run()
        run.text = str(i + 1)
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = FONT

        _add_textbox(slide, key_text,
                     Inches(8.15), top - Inches(0.02), Inches(4.85), Inches(1.72),
                     font_size=11, color=DARK_GRAY)


# ── Deep-dive slides (single-race detailed analysis) ──────────────────────────

def add_pack_dynamics_slide(prs: Presentation, splits_df: pd.DataFrame,
                            gender: str, venue: str, year: int):
    """Three scatter plots: gap-to-leader vs placement at end of swim/bike/run."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Pack Dynamics — Elite {gender.title()} ({year})", venue)

    if splits_df is None or splits_df.empty:
        _add_textbox(slide, "Detailed split data unavailable.",
                     Inches(0.3), Inches(3.5), Inches(12.73), Inches(0.5),
                     font_size=14, color=DARK_GRAY, italic=True, align=PP_ALIGN.CENTER)
        return

    data = compute_pack_scatter(splits_df)
    leg_titles = {"Swim": "Swim Exit", "Bike": "T2 (Bike Exit)", "Run": "Finish"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.patch.set_facecolor("white")

    for ax, leg in zip(axes, ["Swim", "Bike", "Run"]):
        d = data.get(leg, {"points": []})
        pts = d["points"]
        if not pts:
            ax.set_title(f"{leg_titles[leg]}: no data", fontsize=12)
            ax.set_facecolor("white")
            continue
        lead_pts  = [p for p in pts if p["in_lead_swim"]]
        other_pts = [p for p in pts if not p["in_lead_swim"]]

        if other_pts:
            ax.scatter([p["gap"] for p in other_pts],
                       [p["placement"] for p in other_pts],
                       s=70, c=C_GRAY, alpha=0.55, edgecolor="white", linewidth=0.8,
                       label="Other")
        if lead_pts:
            ax.scatter([p["gap"] for p in lead_pts],
                       [p["placement"] for p in lead_pts],
                       s=100, c=C_RED, alpha=0.9, edgecolor="white", linewidth=1.2,
                       label="Swim lead pack (≤15s)")

        for p in pts[:3]:
            last_name = str(p["name"]).split()[-1]
            ax.annotate(f"{p['placement']}. {last_name}",
                        (p["gap"], p["placement"]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, va="center", color=C_NAVY, fontweight="bold")

        ax.set_title(f"At {leg_titles[leg]}", fontsize=12, pad=8, fontweight="bold")
        ax.set_xlabel("Gap to leader (s)", fontsize=10)
        ax.set_ylabel("Position", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("white")
        if lead_pts and other_pts:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

    plt.tight_layout(pad=1.5)
    slide.shapes.add_picture(fig_to_image(fig),
                             Inches(0.2), Inches(1.3), Inches(12.93), Inches(5.4))

    _add_textbox(
        slide,
        "Each point = one athlete; position derives from elapsed time at the checkpoint. "
        "Red points are athletes who exited the swim within 15s of the leader — "
        "trace them across the three plots to see whether the early swim group held to the line.",
        Inches(0.3), Inches(6.78), Inches(12.73), Inches(0.55),
        font_size=10, color=DARK_GRAY, italic=True, align=PP_ALIGN.CENTER,
    )


def add_top_splits_vs_season_slide(prs: Presentation, splits_df: pd.DataFrame,
                                   engine, gender: str, venue: str,
                                   year: int, prior_event_id: int,
                                   prior_prog_id: int, venue_date):
    """Top 3 splits per discipline at this race vs each athlete's WTCS 12-month best/avg."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Top Splits vs Season Norms — Elite {gender.title()} ({year})", venue)

    _add_textbox(
        slide,
        "Top 3 split times at this race vs each athlete's best and average split "
        "across WTCS Standard-distance races in the 12 months before the race.",
        Inches(0.3), Inches(1.18), Inches(12.73), Inches(0.4),
        font_size=10.5, color=DARK_GRAY, italic=True, align=PP_ALIGN.CENTER,
    )

    if splits_df is None or splits_df.empty:
        _add_textbox(slide, "Detailed split data unavailable.",
                     Inches(0.3), Inches(3.5), Inches(12.73), Inches(0.5),
                     font_size=14, color=DARK_GRAY, italic=True, align=PP_ALIGN.CENTER)
        return

    # Identify top 3 per discipline (finishers only — DNF rows excluded)
    finishers = splits_df[~splits_df["dnf"]].copy()
    top_by_leg: dict[str, list[dict]] = {}
    for leg_label, col in [("Swim", "swim_sec"), ("Bike", "bike_sec"), ("Run", "run_sec")]:
        valid = finishers[finishers[col].notna()].nsmallest(3, col)
        top_by_leg[leg_label] = [{"name": r["Name"], "split_sec": float(r[col])}
                                 for _, r in valid.iterrows()]

    all_names = list({p["name"] for plist in top_by_leg.values() for p in plist})
    name_to_aid = map_excel_names_to_athlete_ids(engine, prior_event_id, prior_prog_id, all_names)

    aids = list(name_to_aid.values())
    season_df = query_wtcs_season_splits(engine, aids, venue_date, gender)
    season_df = season_df[season_df.event_date < venue_date]

    norms_by_aid: dict[int, dict] = {}
    for aid, g in season_df.groupby("athlete_id"):
        norms_by_aid[int(aid)] = {
            "swim_best": g["swim_sec"].min(skipna=True) if g["swim_sec"].notna().any() else None,
            "swim_avg":  g["swim_sec"].mean(skipna=True) if g["swim_sec"].notna().any() else None,
            "bike_best": g["bike_sec"].min(skipna=True) if g["bike_sec"].notna().any() else None,
            "bike_avg":  g["bike_sec"].mean(skipna=True) if g["bike_sec"].notna().any() else None,
            "run_best":  g["run_sec"].min(skipna=True)  if g["run_sec"].notna().any()  else None,
            "run_avg":   g["run_sec"].mean(skipna=True) if g["run_sec"].notna().any()  else None,
            "n":         int(g.shape[0]),
        }

    panel_width = Inches(4.2)
    panel_left  = [Inches(0.3), Inches(4.55), Inches(8.8)]
    panel_top   = Inches(1.7)
    bar_color   = {"Swim": C_LIGHT_BLUE, "Bike": C_NAVY, "Run": C_RED}

    for left, leg_label in zip(panel_left, ["Swim", "Bike", "Run"]):
        bar = slide.shapes.add_shape(1, left, panel_top, panel_width, Inches(0.35))
        bar.fill.solid()
        bar.fill.fore_color.rgb = RGBColor.from_string(bar_color[leg_label].lstrip("#"))
        bar.line.fill.background()
        _add_textbox(slide, leg_label.upper(),
                     left + Inches(0.1), panel_top + Inches(0.04),
                     panel_width - Inches(0.2), Inches(0.28),
                     font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        tbl_top = panel_top + Inches(0.4)
        cols = [("Athlete", 1.65), ("Split", 0.72), ("Best", 0.72), ("Avg", 0.72), ("Δ Avg", 0.55)]
        n_rows = 1 + len(top_by_leg[leg_label])
        tbl_shape = slide.shapes.add_table(
            n_rows, len(cols), left, tbl_top, panel_width, Inches(0.4 * n_rows + 0.1)
        )
        tbl = tbl_shape.table
        for ci, (_, w) in enumerate(cols):
            tbl.columns[ci].width = Inches(w)
        for ci, (hdr, _) in enumerate(cols):
            _set_cell(tbl.cell(0, ci), hdr, bold=True, color=WHITE, bg_color=NAVY,
                      font_size=9)

        leg_key = leg_label.lower()
        for ri, entry in enumerate(top_by_leg[leg_label], start=1):
            bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
            aid = name_to_aid.get(entry["name"])
            norms = norms_by_aid.get(aid) if aid is not None else None
            best = norms.get(f"{leg_key}_best") if norms else None
            avg  = norms.get(f"{leg_key}_avg")  if norms else None
            split = entry["split_sec"]
            best_s = seconds_to_mmss(best) if best is not None and not pd.isna(best) else "—"
            avg_s  = seconds_to_mmss(avg)  if avg  is not None and not pd.isna(avg)  else "—"
            split_s = seconds_to_mmss(split)
            if avg is not None and not pd.isna(avg):
                delta = int(round(split - float(avg)))
                if delta < 0:
                    delta_s = f"{delta}s"
                    delta_color = RGBColor(0x1B, 0x7F, 0x3A)  # green
                elif delta > 0:
                    delta_s = f"+{delta}s"
                    delta_color = RGBColor(0xC0, 0x00, 0x00)  # red
                else:
                    delta_s = "0s"
                    delta_color = DARK_GRAY
            else:
                delta_s = "—"
                delta_color = DARK_GRAY
            last_first = str(entry["name"])
            parts = last_first.split()
            short_name = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else last_first
            _set_cell(tbl.cell(ri, 0), short_name, font_size=8.5, bg_color=bg, align=PP_ALIGN.LEFT)
            _set_cell(tbl.cell(ri, 1), split_s, font_size=8.5, bold=True, bg_color=bg)
            _set_cell(tbl.cell(ri, 2), best_s,  font_size=8.5, bg_color=bg)
            _set_cell(tbl.cell(ri, 3), avg_s,   font_size=8.5, bg_color=bg)
            _set_cell(tbl.cell(ri, 4), delta_s, font_size=8.5, bold=True, bg_color=bg,
                      color=delta_color)

    _add_textbox(
        slide,
        "Best / Avg = athlete's WTCS Standard-distance splits in the 12 months ending the day before this race. "
        "Δ Avg = this race's split minus the athlete's 12-month average; "
        "green = faster than average, red = slower.",
        Inches(0.3), Inches(6.4), Inches(12.73), Inches(0.55),
        font_size=9, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER,
    )


def add_who_to_watch_slide(prs: Presentation, engine, venue: str,
                           upcoming_event_id: int | None,
                           prior_men: dict | None,
                           prior_women: dict | None):
    """Single slide covering both genders: top 3 by ranking + returning prior-year podium."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Who to Watch — {venue} Race Week", "")

    panel_w = Inches(6.2)
    gap     = Inches(0.3)
    left_m  = Inches(0.3)
    left_w  = left_m + panel_w + gap
    panel_top = Inches(1.3)

    for left, gender, prior in [(left_m, "Men", prior_men),
                                (left_w, "Women", prior_women)]:
        header = slide.shapes.add_shape(1, left, panel_top, panel_w, Inches(0.4))
        header.fill.solid()
        header.fill.fore_color.rgb = NAVY
        header.line.fill.background()
        _add_textbox(slide, f"ELITE {gender.upper()}",
                     left + Inches(0.1), panel_top + Inches(0.05),
                     panel_w - Inches(0.2), Inches(0.32),
                     font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        rankings_df = query_top_by_world_ranking(engine, gender,
                                                 on_startlist_event_id=upcoming_event_id,
                                                 limit=3)
        rankings_label = "Top 3 Ranked on Startlist" if (upcoming_event_id and not rankings_df.empty) else "Top 3 by World Ranking"
        if rankings_df.empty and upcoming_event_id:
            rankings_df = query_top_by_world_ranking(engine, gender,
                                                     on_startlist_event_id=None, limit=3)
            rankings_label = "Top 3 by World Ranking (no startlist data yet)"

        sub_top = panel_top + Inches(0.5)
        sub_bar = slide.shapes.add_shape(1, left, sub_top, panel_w, Inches(0.3))
        sub_bar.fill.solid()
        sub_bar.fill.fore_color.rgb = RED
        sub_bar.line.fill.background()
        _add_textbox(slide, rankings_label,
                     left + Inches(0.1), sub_top + Inches(0.02),
                     panel_w - Inches(0.2), Inches(0.26),
                     font_size=11, bold=True, color=WHITE)

        tbl_top = sub_top + Inches(0.35)
        rank_cols = [("Rank", 0.8), ("Athlete", 3.6), ("Points", 1.6)]
        n_rows = 1 + max(len(rankings_df), 1)
        tbl_shape = slide.shapes.add_table(n_rows, len(rank_cols), left, tbl_top,
                                           panel_w, Inches(0.34 * n_rows + 0.1))
        tbl = tbl_shape.table
        for ci, (_, w) in enumerate(rank_cols):
            tbl.columns[ci].width = Inches(w)
        for ci, (hdr, _) in enumerate(rank_cols):
            _set_cell(tbl.cell(0, ci), hdr, bold=True, color=WHITE, bg_color=NAVY, font_size=10)
        if rankings_df.empty:
            _set_cell(tbl.cell(1, 0), "—", font_size=10, bg_color=WHITE)
            _set_cell(tbl.cell(1, 1), "No ranking data found", font_size=10, bg_color=WHITE,
                      align=PP_ALIGN.LEFT, italic=True)
            _set_cell(tbl.cell(1, 2), "—", font_size=10, bg_color=WHITE)
        else:
            for ri, r in enumerate(rankings_df.itertuples(index=False), start=1):
                bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
                pts = f"{int(r.total_points):,}" if r.total_points else "—"
                _set_cell(tbl.cell(ri, 0), str(int(r.rank_position)),
                          font_size=10, bold=True, bg_color=bg)
                _set_cell(tbl.cell(ri, 1), str(r.athlete_name),
                          font_size=10, bg_color=bg, align=PP_ALIGN.LEFT)
                _set_cell(tbl.cell(ri, 2), pts, font_size=10, bg_color=bg)

        # Prior-year podium block
        podium_top = tbl_top + Inches(0.34 * n_rows + 0.4)
        podium_bar = slide.shapes.add_shape(1, left, podium_top, panel_w, Inches(0.3))
        podium_bar.fill.solid()
        podium_bar.fill.fore_color.rgb = RED
        podium_bar.line.fill.background()
        podium_label = "Returning Podium Athletes on Startlist" if upcoming_event_id else f"Last Year's Podium at {venue}"
        _add_textbox(slide, podium_label,
                     left + Inches(0.1), podium_top + Inches(0.02),
                     panel_w - Inches(0.2), Inches(0.26),
                     font_size=11, bold=True, color=WHITE)

        body_top = podium_top + Inches(0.35)
        if upcoming_event_id and prior:
            podium_df = query_prior_podium_on_startlist(
                engine, prior["event_id"], prior["prog_id"], upcoming_event_id
            )
            if podium_df.empty:
                _add_textbox(slide,
                             "None of last year's podium are on the current startlist.",
                             left + Inches(0.1), body_top + Inches(0.1),
                             panel_w - Inches(0.2), Inches(0.5),
                             font_size=10.5, color=DARK_GRAY, italic=True)
            else:
                for ri, r in enumerate(podium_df.itertuples(index=False)):
                    txt = f"{int(r.position)}.  {r.athlete_full_name}   ({r.total_time})"
                    _add_textbox(slide, txt,
                                 left + Inches(0.15),
                                 body_top + Inches(0.05 + 0.32 * ri),
                                 panel_w - Inches(0.3), Inches(0.32),
                                 font_size=11, bold=(ri == 0), color=DARK_GRAY)
        elif prior:
            # No startlist available — just list last year's top 3
            sql = text("""
                SELECT position_sort AS position, athlete_full_name, total_time
                FROM race_results
                WHERE event_id = :eid AND prog_id = :pid
                  AND position_sort IS NOT NULL AND position_sort <= 3
                ORDER BY position_sort
            """)
            podium_df = pd.read_sql(sql, engine, params={"eid": prior["event_id"],
                                                         "pid": prior["prog_id"]})
            if podium_df.empty:
                _add_textbox(slide, "No prior-race podium data found.",
                             left + Inches(0.1), body_top + Inches(0.1),
                             panel_w - Inches(0.2), Inches(0.5),
                             font_size=10.5, color=DARK_GRAY, italic=True)
            else:
                for ri, r in enumerate(podium_df.itertuples(index=False)):
                    txt = f"{int(r.position)}.  {r.athlete_full_name}   ({r.total_time})"
                    _add_textbox(slide, txt,
                                 left + Inches(0.15),
                                 body_top + Inches(0.05 + 0.32 * ri),
                                 panel_w - Inches(0.3), Inches(0.32),
                                 font_size=11, bold=(ri == 0), color=DARK_GRAY)
        else:
            _add_textbox(slide, "No prior race history at this venue.",
                         left + Inches(0.1), body_top + Inches(0.1),
                         panel_w - Inches(0.2), Inches(0.5),
                         font_size=10.5, color=DARK_GRAY, italic=True)

    _add_textbox(
        slide,
        "Rankings = current World Triathlon rankings snapshot; startlist intersect uses program_entries when available.",
        Inches(0.3), Inches(7.05), Inches(12.73), Inches(0.25),
        font_size=9, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER,
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def _autodiscover_detailed_splits(venue: str) -> str | None:
    """Look in data/ for 'Detailed results <Venue> <YYYY>.xlsx' (most recent year)."""
    data_dir = os.path.join(REPO_ROOT, "data")
    if not os.path.isdir(data_dir):
        return None
    pattern = re.compile(rf"^Detailed results {re.escape(venue)} (\d{{4}})\.xlsx$",
                         re.IGNORECASE)
    matches = []
    for fn in os.listdir(data_dir):
        m = pattern.match(fn)
        if m:
            matches.append((int(m.group(1)), os.path.join(data_dir, fn)))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def main():
    parser = argparse.ArgumentParser(description="Generate venue historical race analysis PPT")
    parser.add_argument("--venue", required=True, help="Venue name (e.g. 'Yokohama')")
    parser.add_argument("--years", type=int, default=8, help="Years to look back (default: 8)")
    parser.add_argument("--gender", choices=["men", "women", "both"], default="both")
    parser.add_argument("--output", default=None, help="Output .pptx filename (optional)")
    parser.add_argument("--deep-dive", action="store_true",
                        help="Force the single-race deep-dive section (auto-enabled when N≤1 prior race)")
    parser.add_argument("--detailed-splits", default=None,
                        help="Path to detailed per-lap splits Excel (auto-discovered in data/ if omitted). Implies --deep-dive.")
    parser.add_argument("--upcoming-event-id", type=int, default=None,
                        help="Event ID of the upcoming race for 'Who to Watch' (auto-detected if omitted)")
    args = parser.parse_args()

    fname = args.output or f"{args.venue.replace(' ', '_')}_{date.today()}.pptx"
    if not os.path.isabs(fname):
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        fname = os.path.join(DEFAULT_OUTPUT_DIR, fname)

    print(f"Connecting to database...")
    engine = get_engine()

    print(f"Querying events at '{args.venue}' (past {args.years} years)...")
    events_df = query_venue_events(engine, args.venue, args.years, args.gender)
    if events_df.empty:
        print(f"No events found for venue '{args.venue}'.")
        sys.exit(1)

    print(f"Found {len(events_df)} program(s):")
    for _, r in events_df.iterrows():
        print(f"  {r.event_date}  {r.event_name}  ({r.prog_name})")

    men_df   = events_df[events_df.prog_name.str.contains("Elite Men",   case=False, na=False) &
                         ~events_df.prog_name.str.contains("Women",      case=False, na=False)]
    women_df = events_df[events_df.prog_name.str.contains("Elite Women", case=False, na=False)]

    print("\nCollecting race data...")
    men_data   = collect_race_data(engine, men_df)   if not men_df.empty   else []
    women_data = collect_race_data(engine, women_df) if not women_df.empty else []

    # Geocode venue and enrich all rows with Open-Meteo weather + AQ
    print(f"Geocoding '{args.venue}' and fetching Open-Meteo data...")
    coords = geocode_venue(args.venue)
    enrich_rows_with_openmeteo(men_data,   coords)
    enrich_rows_with_openmeteo(women_data, coords)

    print("Fetching venue content (WT API + URL construction)...")
    venue_content = fetch_venue_content(args.venue, events_df)

    # Build deduped venue-level rows for the environmental slide
    seen_eids: set = set()
    unique_env_rows: list[dict] = []
    for row in sorted(men_data + women_data, key=lambda r: r["year"]):
        if row["event_id"] not in seen_eids:
            seen_eids.add(row["event_id"])
            unique_env_rows.append(row)

    # ── Deep-dive trigger ─────────────────────────────────────────────────────
    n_men   = len(men_data)
    n_women = len(women_data)
    auto_trigger = max(n_men, n_women) <= 1
    deep_dive = args.deep_dive or bool(args.detailed_splits) or auto_trigger
    if deep_dive:
        reason = "explicit flag" if (args.deep_dive or args.detailed_splits) else "N<=1 prior race per gender"
        print(f"Deep-dive mode ON (reason: {reason})")

    detailed: dict[str, pd.DataFrame] = {}
    upcoming = None
    if deep_dive:
        xlsx_path = args.detailed_splits or _autodiscover_detailed_splits(args.venue)
        if xlsx_path and os.path.exists(xlsx_path):
            print(f"Loading detailed splits: {xlsx_path}")
            try:
                detailed = load_detailed_splits(xlsx_path)
            except Exception as exc:
                print(f"  Warning: failed to load detailed splits ({exc}); deep-dive split slides will be skipped")
                detailed = {}
        else:
            print("  No detailed-splits Excel found (looked in data/); split-based deep-dive slides will be skipped")

        if args.upcoming_event_id:
            upcoming = {"event_id": int(args.upcoming_event_id), "event_date": None,
                        "event_name": f"Upcoming event {args.upcoming_event_id}"}
        else:
            upcoming = find_upcoming_event(engine, args.venue)
            if upcoming:
                print(f"  Upcoming event detected: {upcoming['event_name']} ({upcoming['event_date']}) — event_id={upcoming['event_id']}")
            else:
                print("  No upcoming event found at this venue; 'Who to Watch' will use rankings-only mode")

    # Prior-race anchor per gender (for athlete_id lookup + podium queries)
    prior_men   = men_data[-1]   if men_data   else None
    prior_women = women_data[-1] if women_data else None

    print("Building PowerPoint...")
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    all_data = men_data + women_data
    add_title_slide(prs, args.venue, args.years)
    add_course_differentiators_slide(prs, args.venue, all_data, events_df, venue_content)
    add_course_map_slide(prs, args.venue, all_data, venue_content)
    add_environmental_risk_slide(prs, args.venue, unique_env_rows)

    if deep_dive and (prior_men or prior_women):
        add_who_to_watch_slide(
            prs, engine, args.venue,
            upcoming_event_id=upcoming["event_id"] if upcoming else None,
            prior_men={"event_id": prior_men["event_id"], "prog_id": prior_men["prog_id"]} if prior_men else None,
            prior_women={"event_id": prior_women["event_id"], "prog_id": prior_women["prog_id"]} if prior_women else None,
        )

    gender_sections = []
    if men_data   and args.gender in ("men",   "both"): gender_sections.append(("Men",   men_data))
    if women_data and args.gender in ("women", "both"): gender_sections.append(("Women", women_data))

    for gender_label, data in gender_sections:
        print(f"  Building {gender_label} slides ({len(data)} races)...")
        add_section_divider(prs, gender_label)
        add_race_overview_slide(prs, gender_label, data, args.venue, venue_content)
        add_overview_slide(prs, data, gender_label, args.venue)

        # Deep-dive slides (single-race detail) — insert after Results Summary
        if deep_dive:
            gkey = gender_label.lower()
            splits_df = detailed.get(gkey)
            prior = data[-1]
            if splits_df is not None and not splits_df.empty:
                add_pack_dynamics_slide(prs, splits_df, gender_label, args.venue, prior["year"])
                add_top_splits_vs_season_slide(
                    prs, splits_df, engine, gender_label, args.venue,
                    prior["year"], prior["event_id"], prior["prog_id"], prior["date"],
                )

        add_swim_slide(prs, data, gender_label, args.venue)
        add_bike_evolution_slide(prs, data, gender_label, args.venue)
        add_run_slide(prs, data, gender_label, args.venue)
        add_position_times_slide(prs, data, gender_label, args.venue)
        add_weather_slide(prs, data, gender_label, args.venue)

    prs.save(fname)
    print(f"\nDone! Saved to: {fname}")


if __name__ == "__main__":
    main()
