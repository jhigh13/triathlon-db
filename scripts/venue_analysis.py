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
    """Mark implausibly small split times as NaN so they don't poison min/mean.

    Three filters, applied in order:
      1. Non-positive values (timing-system gaps recorded as ``00:00:00``) → NaN.
      2. Values below 50% of the median split → NaN (clearly bad data — e.g.
         a 9-minute swim when the field ran 17 minutes).
      3. Iteratively, if the smallest remaining value is more than ``threshold``×
         faster than the second-smallest, drop it as a residual outlier.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0, np.nan)
    valid = s.dropna()
    if len(valid) < 2:
        return s
    median_val = float(valid.median())
    if median_val > 0:
        s = s.where(s.isna() | (s >= median_val * 0.5), np.nan)
    for _ in range(5):
        valid = s.dropna()
        if len(valid) < 2:
            break
        sorted_vals = valid.sort_values()
        if sorted_vals.iloc[0] * threshold < sorted_vals.iloc[1]:
            s = s.mask(s == sorted_vals.iloc[0], np.nan)
        else:
            break
    return s


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
        secs = adjust_outlier(secs, threshold=1.5)
        secs = secs.dropna()
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
        df[dst] = adjust_outlier(df[src].apply(parse_time_to_seconds))
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


# Hardcoded fallback for venues that have been geocoded historically. Avoids
# relying on Nominatim when local SSL certs / network are unavailable.
VENUE_COORDS_FALLBACK: dict[str, tuple[float, float]] = {
    "huatulco":  (15.831, -96.320),
    "alghero":   (40.564,   8.319),
    "yokohama":  (35.444, 139.638),
    "abu dhabi": (24.466,  54.367),
    "quiberon":  (47.485,  -3.114),
    "montreal":  (45.503, -73.534),   # Parc Jean-Drapeau, Notre-Dame Island
    "antofagasta": (-23.651, -70.398),  # Balneario Municipal, Av. República de Croacia
}


def geocode_venue(venue: str) -> tuple | None:
    """Geocode a venue string via Nominatim OSM. Returns (lat, lon) or None.
    Falls back to VENUE_COORDS_FALLBACK on network/SSL failure."""
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
        print(f"  [Geocode] Nominatim failed for '{venue}': {exc}")
    # Fallback to hardcoded coords for known venues
    fb = VENUE_COORDS_FALLBACK.get(venue.lower().strip())
    if fb:
        print(f"  [Geocode] Using fallback coords for '{venue}' -> {fb[0]:.3f}, {fb[1]:.3f}")
        _GEOCODE_CACHE[venue] = fb
        return fb
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


def fetch_sst_for_race_day(lat: float, lon: float, race_date: str,
                           years_back: int = 7) -> tuple[float | None, str]:
    """Best-effort sea-surface temperature for race_date.

    Tries the marine API for the actual race_date first (forecast window).
    Falls back to averaging the same calendar day across prior years.
    Returns (sst_celsius, source) where source is 'forecast' / 'climatology' / 'unavailable'.
    """
    direct = fetch_openmeteo_marine(lat, lon, race_date)
    if direct.get("sst") is not None:
        return float(direct["sst"]), "forecast"

    target = pd.to_datetime(race_date).date()
    current_year = date.today().year
    vals: list[float] = []
    for yr_off in range(1, years_back + 1):
        yr = current_year - yr_off
        d = date(yr, target.month, target.day).isoformat()
        result = fetch_openmeteo_marine(lat, lon, d)
        if result.get("sst") is not None:
            vals.append(float(result["sst"]))
    if vals:
        return sum(vals) / len(vals), "climatology"
    return None, "unavailable"


# ── Forward forecast + climatology (used by Race-Day Forecast slide) ─────────

def fetch_openmeteo_forecast(lat: float, lon: float, race_date: str) -> dict:
    """Forward forecast for race_date. Returns hourly arrays for 04:00–12:00 local,
    or empty dict if race_date is outside the 16-day forecast window.

    Result keys: 'time', 'temperature_2m', 'apparent_temperature',
    'relative_humidity_2m', 'wind_speed_10m', 'precipitation_probability', 'uv_index'.
    """
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": race_date, "end_date": race_date,
                "hourly": ",".join([
                    "temperature_2m", "apparent_temperature",
                    "relative_humidity_2m", "wind_speed_10m",
                    "precipitation_probability", "uv_index",
                ]),
                "wind_speed_unit": "kmh",
                "timezone": "auto",
                "forecast_days": 16,
            },
            timeout=15,
        )
        if not r.ok:
            return {}
        hrs = r.json().get("hourly", {})
        times = hrs.get("time", [])
        # Confirm race_date is actually present in the returned series
        if not any(t.startswith(race_date) for t in times):
            return {}
        keep_idx = [i for i, t in enumerate(times)
                    if t.startswith(race_date) and "04" <= t[11:13] <= "12"]
        if not keep_idx:
            return {}
        out: dict = {"time": [times[i][11:16] for i in keep_idx]}
        for col in ["temperature_2m", "apparent_temperature",
                    "relative_humidity_2m", "wind_speed_10m",
                    "precipitation_probability", "uv_index"]:
            vals = hrs.get(col, [])
            out[col] = [vals[i] if i < len(vals) else None for i in keep_idx]
        return out
    except Exception as exc:
        print(f"  [OpenMeteo forecast] {race_date}: {exc}")
        return {}


def fetch_openmeteo_climatology(lat: float, lon: float, race_date: str,
                                years_back: int = 7) -> dict:
    """Average each hour 04:00–12:00 across the same calendar day for the last
    `years_back` years. Used as a fallback when the race date is outside the
    forward-forecast window."""
    target = pd.to_datetime(race_date).date()
    current_year = date.today().year
    rows_per_hour: dict[str, dict[str, list]] = {}
    fields = ["temperature_2m", "apparent_temperature",
              "relative_humidity_2m", "wind_speed_10m",
              "precipitation", "uv_index"]
    for yr_off in range(1, years_back + 1):
        yr = current_year - yr_off
        d = date(yr, target.month, target.day).isoformat()
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": d, "end_date": d,
                    "hourly": ",".join(fields),
                    "wind_speed_unit": "kmh",
                    "timezone": "auto",
                },
                timeout=15,
            )
            if not r.ok:
                continue
            hrs = r.json().get("hourly", {})
            times = hrs.get("time", [])
            for i, t in enumerate(times):
                hour = t[11:16]
                if not ("04" <= t[11:13] <= "12"):
                    continue
                bucket = rows_per_hour.setdefault(hour, {f: [] for f in fields})
                for f in fields:
                    arr = hrs.get(f, [])
                    if i < len(arr) and arr[i] is not None:
                        bucket[f].append(arr[i])
        except Exception as exc:
            print(f"  [OpenMeteo climatology {d}]: {exc}")
            continue
    if not rows_per_hour:
        return {}
    times_sorted = sorted(rows_per_hour.keys())
    out: dict = {"time": times_sorted}
    out_field_map = {
        "temperature_2m": "temperature_2m",
        "apparent_temperature": "apparent_temperature",
        "relative_humidity_2m": "relative_humidity_2m",
        "wind_speed_10m": "wind_speed_10m",
        "precipitation": "precipitation",
        "uv_index": "uv_index",
    }
    for src, dst in out_field_map.items():
        out[dst] = [
            (sum(rows_per_hour[h][src]) / len(rows_per_hour[h][src]))
            if rows_per_hour[h][src] else None
            for h in times_sorted
        ]
    # Climatology has no precip-probability — synthesise from mean precipitation > 0.1mm
    out["precipitation_probability"] = [
        100.0 if (p is not None and p > 0.1) else 0.0 for p in out["precipitation"]
    ]
    return out


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
    _add_textbox(slide, f"{venue.upper()} RACE PREVIEW",
                 Inches(0.5), Inches(2.0), Inches(11.8), Inches(1.4),
                 font_size=54, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Subtitle — pull the upcoming race month from EVENT_SCHEDULES when available
    sched = EVENT_SCHEDULES.get(venue.lower().strip())
    when_str = sched.get("date_range", "").split(",")[-1].strip() if sched else ""
    if not when_str:
        when_str = date.today().strftime("%B %Y")
    elif when_str.isdigit():  # year-only
        when_str = sched.get("date_range", date.today().strftime("%B %Y"))
    _add_textbox(slide, f"Elite Analysis  ·  {when_str}",
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

    # Section headers (navy bars) sit at 1.20–1.50, tables below at 1.55
    wx_shape = slide.shapes.add_table(
        n_rows, len(wx_cols),
        Inches(0.3), Inches(1.55),
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

    # Weather section label (navy header bar, sits above the table)
    wx_bar = slide.shapes.add_shape(1, Inches(0.3), Inches(1.18), Inches(5.15), Inches(0.32))
    wx_bar.fill.solid(); wx_bar.fill.fore_color.rgb = NAVY; wx_bar.line.fill.background()
    _add_textbox(slide, "RACE-DAY WEATHER (08:00–11:00 local)",
                 Inches(0.35), Inches(1.22), Inches(5.0), Inches(0.26),
                 font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

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
        Inches(5.65), Inches(1.55),
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

    aq_bar = slide.shapes.add_shape(1, Inches(5.65), Inches(1.18), Inches(4.45), Inches(0.32))
    aq_bar.fill.solid(); aq_bar.fill.fore_color.rgb = NAVY; aq_bar.line.fill.background()
    _add_textbox(slide, "AIR QUALITY (Open-Meteo archive)",
                 Inches(5.7), Inches(1.22), Inches(4.3), Inches(0.26),
                 font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # ── Bottom: health / water quality risk section ───────────────────────────
    # Tables start at 1.55 + row_h*n_rows; add modest gap then short red bar.
    tbl_bottom = Inches(1.55 + row_h * n_rows + 0.15)

    risk_bar = slide.shapes.add_shape(1, Inches(0.3), tbl_bottom, Inches(12.73), Inches(0.28))
    risk_bar.fill.solid(); risk_bar.fill.fore_color.rgb = RED; risk_bar.line.fill.background()
    _add_textbox(slide, "HEALTH & WATER QUALITY RISK FACTORS",
                 Inches(0.35), tbl_bottom + Inches(0.02), Inches(12.2), Inches(0.24),
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

    y_offset = tbl_bottom + Inches(0.36)
    item_h = (SLIDE_H - y_offset - Inches(0.3)) / len(risk_items)
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
    """Gender-neutral venue intro: narrative, course features, quick links.

    Prefers structured data from VENUE_PREVIEW + SWIM/BIKE/RUN_COURSE_PROFILES.
    Falls back to historical race rows (`all_rows`) when no profile is defined.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Key Course Differentiators", venue)

    vkey = venue.lower().strip()
    preview = VENUE_PREVIEW.get(vkey, {})
    swim_p  = SWIM_COURSE_PROFILES.get(vkey, {})
    bike_p  = BIKE_COURSE_PROFILES.get(vkey, {})
    run_p   = RUN_COURSE_PROFILES.get(vkey, {})

    # ── Left column: venue name + narrative ───────────────────────────────────
    _add_textbox(slide, venue.upper(),
                 Inches(0.3), Inches(1.2), Inches(6.6), Inches(0.55),
                 font_size=20, bold=True, color=NAVY)

    narrative = preview.get("narrative") or (
        f"[PLACEHOLDER: 2-3 sentences on what makes {venue} unique as a race venue — "
        "geography, setting, and character.]"
    )
    _add_textbox(slide, narrative,
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

    features = preview.get("features") or [
        f"[PLACEHOLDER: Feature 1 — swim character]",
        f"[PLACEHOLDER: Feature 2 — bike character]",
        f"[PLACEHOLDER: Feature 3 — run character]",
        f"[PLACEHOLDER: Feature 4 — conditions]",
    ]
    for i, feat in enumerate(features[:4]):
        _add_textbox(slide, f"•  {feat}",
                     Inches(0.4), Inches(3.7 + i * 0.72), Inches(6.4), Inches(0.7),
                     font_size=11, color=DARK_GRAY)

    # ── Right column: course at a glance ───────────────────────────────────────
    glance_bar = slide.shapes.add_shape(1, Inches(7.2), Inches(1.2), Inches(5.8), Inches(0.35))
    glance_bar.fill.solid()
    glance_bar.fill.fore_color.rgb = NAVY
    glance_bar.line.fill.background()
    _add_textbox(slide, "COURSE AT A GLANCE",
                 Inches(7.25), Inches(1.22), Inches(5.4), Inches(0.3),
                 font_size=12, bold=True, color=WHITE)

    def _fmt_d(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if v > 100:
            v = v / 1000.0
        if v < 1:
            return f"{int(round(v * 1000))} m"
        return f"{v:g} km"

    def _fmt_swim_line(p: dict) -> str:
        d = p.get("total_km")
        laps = p.get("laps")
        if not d:
            return "—"
        if laps and laps > 1 and p.get("loop_km"):
            return f"{_fmt_d(d)} – {laps} laps × {_fmt_d(p['loop_km'])}"
        if laps == 1:
            return f"{_fmt_d(d)} – 1 lap"
        return _fmt_d(d)

    def _fmt_multi(p: dict) -> str:
        d = p.get("total_km")
        laps = p.get("loops") or p.get("laps")
        loop_km = p.get("loop_km")
        if not d:
            return "—"
        if laps and loop_km:
            return f"{_fmt_d(d)} – {laps} laps × {loop_km:g} km"
        if laps:
            return f"{_fmt_d(d)} – {laps} laps"
        return _fmt_d(d)

    # Prefer profile data; fall back to historical row for venues without profiles
    ref_row = next(
        (r for r in sorted(all_rows, key=lambda x: x["year"], reverse=True)
         if r.get("swim_km") or r.get("bike_km") or r.get("run_km")),
        None
    )
    swim_label = (_fmt_swim_line(swim_p) if swim_p
                  else (_fmt_d(ref_row.get("swim_km")) if ref_row else "—"))
    bike_label = (_fmt_multi(bike_p) if bike_p
                  else (_fmt_d(ref_row.get("bike_km")) if ref_row else "—"))
    run_label  = (_fmt_multi(run_p) if run_p
                  else (_fmt_d(ref_row.get("run_km")) if ref_row else "—"))

    format_label = preview.get("format_label") or "Sprint Triathlon"
    bike_laps_val = bike_p.get("loops") if bike_p else (
        int(ref_row["bike_laps"]) if ref_row and ref_row.get("bike_laps") else None)
    run_laps_val  = run_p.get("laps") if run_p else (
        int(ref_row["run_laps"]) if ref_row and ref_row.get("run_laps") else None)

    glance_items = [
        ("Swim",       swim_label),
        ("Bike",       bike_label),
        ("Run",        run_label),
        ("Format",     format_label),
        ("Bike Laps",  str(bike_laps_val) if bike_laps_val else "—"),
        ("Run Laps",   str(run_laps_val) if run_laps_val else "—"),
    ]

    for i, (label, val) in enumerate(glance_items):
        _add_textbox(slide, f"{label}:",
                     Inches(7.3), Inches(1.65 + i * 0.37), Inches(2.0), Inches(0.35),
                     font_size=12, bold=True, color=NAVY)
        _add_textbox(slide, val,
                     Inches(9.0), Inches(1.65 + i * 0.37), Inches(4.0), Inches(0.35),
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

    race_info_url = preview.get("race_info_url") or content.get("upcoming_event_url")
    race_info_text = preview.get("race_info_text") or (
        race_info_url or f"Race Info | {venue}")

    _add_textbox(slide, "Current Race Info:",
                 Inches(7.3), Inches(links_y + 0.45), Inches(5.6), Inches(0.3),
                 font_size=11, bold=True, color=NAVY)
    _add_textbox(slide, race_info_text or "—",
                 Inches(7.3), Inches(links_y + 0.78), Inches(5.6), Inches(0.42),
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


# ── Per-venue course profile data ─────────────────────────────────────────────
# Each dict is populated from official organiser course maps + race-info pages.
# All three discipline dicts share parallel structure so the slide builders can
# be lightly templated.

BIKE_COURSE_PROFILES: dict[str, dict] = {
    "antofagasta": {
        "source":        "Americas Triathlon NF Information Package — Antofagasta 4–5 Jul 2026",
        "loop_km":       5.414,
        "loops":         8,
        "total_km":      43.3,
        "gain_per_lap_m":  None,
        "loss_per_lap_m":  None,
        "max_grade_pos":   None,
        "max_grade_neg":   None,
        "avg_grade_pct":   None,
        "wind":           "Coastal Pacific (ocean breeze)",
        "surface":        "Closed coastal urban — Av República de Croacia ↔ Av Ejército",
        "key_features": [
            "Elite/U23 race is Standard distance — 8 × 5.414 km loops = 43.3 km (~3 km longer than a typical 40 km Olympic bike)",
            "Out-and-back beach corridor: southbound Av República de Croacia → Av Ejército to turnaround at Military Facility Access",
            "Northbound return on Av Ejército east side → Av República de Croacia to turnaround at Plaza Rotonda Croacia Grecia",
            "Pacific coastal exposure — Humboldt Current driving cool air + steady ocean breeze on the seaward sections",
            "Closed urban circuit on a wide multi-lane coastal avenue — fast, low-technical, hammer-friendly",
            "8 laps = 8 transit passes through the Plaza Rotonda turnaround — bike-handling premium on the lap point",
        ],
    },
    "montreal": {
        "source":        "World Triathlon Para Series Montréal Elite Athlete Guide — 27 Jun 2026",
        "loop_km":       4.1,
        "loops":         5,
        "total_km":      21.5,
        "gain_per_lap_m":  None,
        "loss_per_lap_m":  None,
        "max_grade_pos":   None,
        "max_grade_neg":   None,
        "avg_grade_pct":   None,
        "wind":           "Open island — variable",
        "surface":        "Closed F1 circuit — Circuit Gilles-Villeneuve, Parc Jean-Drapeau",
        "key_features": [
            "Bike held on the Circuit Gilles-Villeneuve (CGV) F1 track — clockwise 4.1 km loop × 5 = 21.5 km",
            "Pancake-flat, smooth tarmac throughout — no significant climbs; pure power/aero course",
            "Team wheel station at the CGV entrance; Neutral wheel station mid-lap with full disc/rim brake inventory",
            "Bike penalty box sits at the lap/transition split — athletes pass it 5 times across the race",
            "RaceRanger drafting devices are mandatory for PTS/PTVI classes — install at bike racking on Friday if not done at home",
            "Open island setting on Notre-Dame Island — winds off the St. Lawrence can swing crosswinds on the straights",
        ],
    },
    "quiberon": {
        "source":        "World Triathlon Elite Athlete Guide — WTCS Quiberon, 20 Jun 2026",
        "loop_km":       5.5,
        "loops":         4,
        "total_km":      22.0,
        "gain_per_lap_m":  None,
        "loss_per_lap_m":  None,
        "max_grade_pos":   None,
        "max_grade_neg":   None,
        "avg_grade_pct":   None,
        "wind":           "Coastal (~20 km/h)",
        "surface":        "Closed coastal/urban circuit — Boulevard René Cassin, Quiberon Peninsula",
        "key_features": [
            "Generally flat profile with no significant climbs — race typically decided on the run, not the bike",
            "Multiple urban changes of direction and turns in town sections — bike-handling premium on lap entries",
            "Coastal sections exposed to Atlantic wind — crosswinds can split the pack hard, especially near Pointe Riberen",
            "Two wheel stations: René Cassin Street (just out of T1) and Fort Neuf Street / Parking Nautique near Pointe Riberen",
            "Slightly longer than standard sprint — 22 km total over 4 × 5.5 km laps",
        ],
    },
    "huatulco": {
        "source":        "asdeporte 2026 Elite Cycling course map",
        "loop_km":       5.0,
        "loops":         4,
        "total_km":      20.0,
        "gain_per_lap_m":  84.4,
        "loss_per_lap_m":  84.6,
        "max_grade_pos":   18.4,
        "max_grade_neg":  -14.9,
        "avg_grade_pct":   3.2,
        "surface":        "Closed urban circuit (Bahía de Santa Cruz → Vialidad 5 → return)",
        "key_features": [
            "Out-and-back along Blvd Santa Cruz with a tight technical hairpin at the neutral wheel station",
            "Single sustained climb up Vialidad 5 — short but punchy (max 18.4%); strong riders can split the field every lap",
            "Fast technical descent back into Bahía de Santa Cruz; max -14.9% grade rewards confident bike-handling",
            "Tight start/finish chicane through transition — drafting packs reshuffle every lap on the climb and descent",
        ],
    },
}

SWIM_COURSE_PROFILES: dict[str, dict] = {
    "antofagasta": {
        "source":            "Americas Triathlon NF Information Package — Antofagasta 4–5 Jul 2026",
        "total_km":          1.5,
        "laps":              2,
        "loop_km":           0.75,
        "layout":            "Pacific Ocean — Balneario Municipal beach, buoy-marked, counter-clockwise",
        "format":            "Open ocean — Pacific (Humboldt Current)",
        "start_type":        "Beach start (with beach exit + re-entry between laps for Standard)",
        "water_temp_c":      None,
        "expected_water_temp_range_c": (14.0, 16.0),
        "wetsuit_note":      "Cold Pacific water (Humboldt Current) — expect 14–16 °C in July. Under WT rules wetsuit is mandatory < 16 °C and optional 16–20 °C — Antofagasta will almost certainly be a wetsuit race.",
        "key_features": [
            "Pacific Ocean swim directly off the Balneario Municipal beach — exposed open water",
            "Buoy-marked rectangular course, counter-clockwise direction",
            "Standard distance is 2 × 750 m laps with a BEACH EXIT AND RE-ENTRY between laps — Australian-exit format, rare on the WT circuit",
            "Humboldt Current keeps water cold year-round — July is winter in Southern Hemisphere, water typically 14–16 °C",
            "Swim cap + neoprene cap + booties / gloves all rules-legal at these temps; many will use them",
            "Pacific swell and surf entries possible — sighting on rolling surface waves and pack-position discipline at the beach turn",
        ],
        "missing": [
            "Day-of water temperature (Pacific SST will be filled from Open-Meteo Marine forecast / climatology)",
            "Exact buoy spacing and turn count (visible on race map; not numerically published in the NF package)",
        ],
    },
    "montreal": {
        "source":            "World Triathlon Para Series Montréal Elite Athlete Guide — 27 Jun 2026",
        "total_km":          0.75,
        "laps":              1,
        "loop_km":           0.75,
        "layout":            "Olympic Basin — Parc Jean-Drapeau (1976 Olympics rowing venue)",
        "format":            "Sheltered rowing basin — fresh water, calm conditions",
        "start_type":        "In-water start from pontoon",
        "water_temp_c":      None,
        "expected_water_temp_range_c": (22.0, 24.0),
        "wetsuit_note":      "Guide states June water ~23 °C — under WT rules wetsuit is forbidden > 22 °C; expect non-wetsuit race",
        "key_features": [
            "Sheltered Olympic Basin water — no chop, no current, no wave issues; navigation-only swim",
            "PTWC swim warm-up on course 06:15–06:45; PTS/PTVI use warm-up lane 07:00–08:15",
            "Lounge opens 05:30 (PTWC) / 06:15 (PTS/PTVI); athletes' intro 10 min before category start",
            "Water quality monitored by private lab — guide flags it consistently inside WT thresholds",
            "Warm water + warm air at 7 AM start — no cold-water acclimation concerns",
        ],
        "missing": [
            "Buoy layout for individual race (start gathers on pontoon — full layout TBD at familiarization)",
        ],
    },
    "quiberon": {
        "source":            "World Triathlon Elite Athlete Guide — WTCS Quiberon, 20 Jun 2026",
        "total_km":          0.75,
        "laps":              1,
        "loop_km":           0.75,
        "layout":            "Atlantic Ocean — beach loop off Boulevard René Cassin",
        "format":            "Open ocean — Atlantic / Bay of Quiberon",
        "start_type":        "Beach start (rue René Cassin, beach access from Place du Doued)",
        "water_temp_c":      None,
        "expected_water_temp_range_c": (14.9, 19.3),
        "wetsuit_note":      "Wetsuit territory — guide states mid-June Atlantic water 14.9–19.3 °C; under WT rules wetsuit is mandatory < 16 °C and optional 16–20 °C",
        "key_features": [
            "Single 750 m loop in the open Atlantic off the Quiberon peninsula — cold water by elite standards",
            "Beach start running entry from rue René Cassin — typical pack-formation chaos through the first 200 m",
            "Atlantic exposure brings chop and swell — sighting on bigger surface waves than typical WTCS venues",
            "Cold-water acclimation matters: 14.9–19.3 °C range means the first 100 m breath-control is a real risk",
            "Wetsuit decision driven by morning measurement — bring both options; men race at 10:00, women at 12:00",
            "Water quality testing site \"Lombardsbrücke\" referenced in guide (text appears templated from Hamburg) — actual Quiberon measuring points TBD pre-race",
        ],
        "missing": [
            "Buoy configuration and turn structure (not in race brief)",
            "Day-of water temperature (Atlantic SST will be filled from Open-Meteo Marine forecast)",
        ],
    },
    "huatulco": {
        "source":            "asdeporte 2026 Elite Swim course map (Natación Elite — Bahía de Santa Cruz)",
        "total_km":          0.75,
        "laps":              1,
        "loop_km":           0.75,
        "layout":            "U-shape rectangle — 300 m out, 150 m across, 300 m back",
        "format":            "Ocean — Bahía de Santa Cruz (protected marina)",
        "start_type":        "Beach start (running entry) at Plaza Santa Cruz",
        "water_temp_c":      None,
        "wetsuit_note":      "Non-wetsuit likely — June water temps in Bahía de Santa Cruz typically run 27–29 °C, above the 22 °C threshold",
        "key_features": [
            "Single 750 m loop — 300 m east leg, 150 m across south, 300 m west leg, beach exit by transition",
            "Tight rectangle inside the protected marina — minimal chop, current normally low at 06:30 start",
            "Two turn buoys at the south end — Chief Swim films from outside; angled kayak inside prevents course-cutting",
            "Sun rising due east at 06:30 — bright glare on the outbound 300 m east leg",
            "Extraction zone marked at north dock; short beach run into T1",
        ],
        "missing": [],
    },
}

RUN_COURSE_PROFILES: dict[str, dict] = {
    "antofagasta": {
        "source":            "Americas Triathlon NF Information Package — Antofagasta 4–5 Jul 2026",
        "total_km":          10.0,
        "laps":              4,
        "loop_km":           2.5,
        "surface":           "Asphalt — Av Grecia coastal road",
        "gain_per_lap_m":    None,
        "loss_per_lap_m":    None,
        "max_grade_pos":     None,
        "max_grade_neg":     None,
        "avg_grade_pct":     None,
        "heat_risk":         "LOW",
        "key_features": [
            "Standard distance: 4 × 2.5 km laps along Av Grecia — flat coastal road on the west side",
            "Out-and-back to turnaround at intersection with Antonio Toro Street; return southbound back to Balneario esplanade",
            "Cool winter temperatures (~13–18 °C) favour aggressive run pacing — no heat-stress concern",
            "Wide, smooth asphalt on a multi-lane road — fast surface, no technical sections",
            "4 laps of the same loop = 4 transitions through the same crowd-noise pocket — strong home-fan factor for Chilean athletes",
            "Cool ocean breeze along the run — light wind protection (arm sleeves) may help on the back half",
        ],
        "missing": [
            "Elevation profile (course is flat coastal road; no published grade data)",
        ],
    },
    "montreal": {
        "source":            "World Triathlon Para Series Montréal Elite Athlete Guide — 27 Jun 2026",
        "total_km":          5.0,
        "laps":              2,
        "loop_km":           2.5,
        "surface":           "Asphalt — Circuit Gilles-Villeneuve + Olympic Basin path",
        "gain_per_lap_m":    None,
        "loss_per_lap_m":    None,
        "max_grade_pos":     None,
        "max_grade_neg":     None,
        "avg_grade_pct":     None,
        "heat_risk":         "MODERATE",
        "key_features": [
            "Flat 2.5 km loop × 2 — exit transition north of basin, enter CGV at the hairpin, head west to pit lane",
            "Two left turns at the pit lane drop onto the north-side path along the Olympic Basin",
            "Aid stations / water at 370 m, 1.1 km, 2.3 km, 3.0 km, 4.1 km — five access points per athlete",
            "Run penalty box at ~2.6 km (loop turnaround); passed twice — drafting infractions compound across laps",
            "PTVI Free Leading zones: T2 exit→AS1, AS2 on CGV, paddocks turnaround, AS2 on Chemin Nord, Run Penalty Box turn",
            "Run team wheel station (for racing wheelchairs) co-located with Aid Station 2; both-side access",
        ],
        "missing": [
            "Elevation profile (course described as flat — no published grade data)",
        ],
    },
    "quiberon": {
        "source":            "World Triathlon Elite Athlete Guide — WTCS Quiberon, 20 Jun 2026",
        "total_km":          5.0,
        "laps":              2,
        "loop_km":           2.5,
        "surface":           "Asphalt out-and-back; small stabilized (gravel) section per lap",
        "gain_per_lap_m":    None,
        "loss_per_lap_m":    None,
        "max_grade_pos":     None,
        "max_grade_neg":     None,
        "avg_grade_pct":     None,
        "key_features": [
            "Flat out-and-back asphalt course with one short stabilized-surface section per lap",
            "Aid stations at 0.2 km (just after T2) and 1.25 km — pass each twice across the 2 laps",
            "Penalty Box located on the left, ~50 m before the Transition Area — easy to miss if drafting flagged",
            "Course measurement for coaches Friday 19:00 from the Finish Area (Bd René Cassin) — register at Friday package distribution",
            "Cool Atlantic conditions favour aggressive pacing; June daily high ~23 °C, low ~13 °C — no heat-stress concern",
            "Lap structure: 2 × 2.5 km out-and-back (inferred from aid station spacing in the brief)",
        ],
        "missing": [
            "Elevation profile and grade data (course described as \"flat\" — no numbers published)",
            "Exact turnaround coordinates (route map TBD in the guide PDF)",
        ],
    },
    "huatulco": {
        "source":            "asdeporte 2026 Elite Run course map (Carrera Elite — Boulevard Santa Cruz)",
        "total_km":          5.0,
        "laps":              2,
        "loop_km":           2.5,
        "surface":           "Paved urban — Blvd Santa Cruz / Camino a Santa Cruz",
        "gain_per_lap_m":    55.3,
        "loss_per_lap_m":    55.0,
        "max_grade_pos":     17.3,
        "max_grade_neg":    -15.3,
        "avg_grade_pct":     4.2,
        "heat_risk":         "HIGH",
        "elevation_range_m": (6, 32),
        "key_features": [
            "Hilly 2.5 km out-and-back loop — +55.3 m gain per lap (≈110 m total over 5 km)",
            "Max climb 17.3% on Camino a Santa Cruz — short steep pinch that bites harder on lap 2",
            "Steep -15.3% descent back into Bahía de Santa Cruz — light braking + high cadence preserves the legs",
            "08:30 men's start runs into peak surface temps with limited shade on the climb",
            "Aid stations at 1 km and 2 km plus start/finish; penalty box at start/finish — drafting tickets compound across both laps",
            "Course turnaround at the 2 km mark off Blvd Chahue — sharp 180° on a slight uphill",
        ],
        "missing": [],
    },
}

# ── Travel & arrival data (per venue, origin = Denver, CO USAT HQ) ───────────
TRAVEL_PROFILES: dict[str, dict] = {
    "antofagasta": {
        "origin":          "Denver, CO → Antofagasta race week",
        "core_read": (
            "No direct US → Antofagasta flights — every itinerary connects through Santiago de Chile "
            "(SCL). Cleanest route: Denver → SCL (overnight via DFW / IAH / MIA / ATL), then a 2-hour "
            "domestic hop SCL → ANF on LATAM, SKY, or JetSMART. Total door-to-door is typically 18–22 h "
            "with a layover in Santiago. Only +2 h time-zone shift from Denver to Chile (CLT, UTC-4), "
            "so jet-lag impact is minimal — but the long overnight leg + altitude (Antofagasta is "
            "sea-level after a high-altitude Santiago layover) warrants an extra recovery day. Plan to "
            "land by Wednesday July 1 for the Friday July 3 elite briefing."
        ),
        "stats": [
            ("Primary route",    "DEN → SCL → ANF", "2-stop itinerary"),
            ("Flight time",      "~14 – 16 h",       "Plus SCL layover"),
            ("Time zones",       "+2h",              "Chile ahead of Denver"),
            ("Airport Transfer", "ANF → ENJOY Hotel","LOC-recommended / taxi"),
            ("Transfer Distance","~25 min by car",   "Andrés Sabella → Av Angamos"),
        ],
        "hotel_title": "Host Hotel / Official Accommodation Read",
        "hotel_bullets": [
            "Official hotel: ENJOY ANTOFAGASTA, Av Angamos 1455. +56 5 5265 3000.",
            "Resort + casino property on the beach — pool, gym, full-service amenities; team-friendly for multi-night blocks.",
            "Alternates (all close to venue): Holiday Inn Express (Av Grecia / Antonio Poupin 1490), Hotel Florencia Suites, NH Antofagasta.",
            "ENJOY is also the official briefing venue (Elite, Para, Youth, AG) — staying on-property removes shuttle dependence for Fri/Sat briefings.",
            "Limited official-hotel capacity — book early via LOC contact gerente@fechitri.cl.",
        ],
        "food_title": "Food / Grocery Options Near Hotel + Venue",
        "food_bullets": [
            "Best stock-up: Líder / Jumbo / Unimarc supermarkets in central Antofagasta — staples, electrolytes, sealed water (do not drink tap water).",
            "Quick basics: pharmacies (Cruz Verde / Salcobrand) for sports nutrition, sunscreen, electrolyte powders.",
            "Athlete-friendly restaurants: seafood-heavy Pacific coast town; mariscos / ceviche are local specialties — be selective with raw seafood in race week.",
            "Hydration note: bottled / filtered water only — Antofagasta tap water is desalinated but variable; stick to sealed bottles for race week to remove GI risk.",
            "Cool-weather kit: pack warm layers + windbreaker — July is winter; mornings can be 12–13 °C with ocean breeze.",
        ],
    },
    "montreal": {
        "origin":          "Denver, CO → Montréal race week",
        "core_read": (
            "Denver → Montréal is a same-day continental hop — multiple nonstop options (Air Canada / "
            "United / WestJet seasonal) plus easy one-stop itineraries through ORD, YYZ, or EWR. "
            "Only +2 h time-zone shift (MDT → EDT), so jet-lag impact is minimal. LOC operates a "
            "scheduled airport shuttle YUL ↔ Alt Hotel Montréal between June 23–28 (60 USD mandatory "
            "per-person fee). Arrive at least Wednesday (Jun 24) to use the official Wed/Thu pool + gym "
            "training windows ahead of Friday's familiarization."
        ),
        "stats": [
            ("Primary route",    "DEN → YUL",       "Nonstop available"),
            ("Flight time",      "~3h 45m",         "Nonstop estimate"),
            ("Time zones",       "+2h",             "Montréal ahead of Denver"),
            ("Airport Transfer", "YUL → Alt Hotel", "LOC shuttle (60 USD)"),
            ("Transfer Distance","19 km / ~20 min", "By car/shuttle"),
        ],
        "hotel_title": "Host Hotel / Official Accommodation Read",
        "hotel_bullets": [
            "Official hotel: Alt Hotel Montréal (Germain Group), 120 Peel Street, H3C 0L8.",
            "Block code \"2606OBTRIA\" — book via the WTPS reservations link, email reservations.altmontreal@germainhotels.com, or call 514.375.0220.",
            "Distance to venue (Parc Jean-Drapeau): 6.3 km — 15 min by car or bike; LOC provides free venue transport on familiarization + race day.",
            "Underground parking on site (Indigo-managed); cash / Visa / Mastercard accepted.",
            "Hotel is downtown (Griffintown adjacent) — easy walk to Old Montréal, restaurants, and metro access.",
        ],
        "food_title": "Food / Grocery Options Near Hotel + Venue",
        "food_bullets": [
            "Best stock-up: Provigo / IGA / Métro grocery stores within walking distance of Alt Hotel (Peel St / Saint-Antoine corridor).",
            "Quick basics: Tim Hortons / Couche-Tard / Starbucks dotted around downtown; pharmacies (Jean Coutu / Pharmaprix) for sports nutrition + electrolytes.",
            "Athlete-friendly restaurants: Marcus, Burgundy Lion, and the Cours Mont-Royal food court are within 10–15 min walk for protein-forward options.",
            "Venue note: Parc Jean-Drapeau is on Notre-Dame Island — bring race-day food/fluid from the hotel; on-island vendor options are limited and seasonal.",
            "Heat consideration: Late June can hit 30 °C+ — pre-stock electrolytes / ice packs at hotel for daily training blocks.",
        ],
    },
    "quiberon": {
        "origin":          "Denver, CO → Quiberon race week",
        "core_read": (
            "There are no direct Denver → Nantes flights. The cleanest athlete route is a one-stop "
            "itinerary through CDG, Dublin, Amsterdam, or another European hub into Nantes, then use "
            "the LOC/Nirvana scheduled shuttle to Quiberon. Many flight options include an overnight "
            "long-haul travel as a part of the leg. Minimize the 8h time zone shift by shifting daily "
            "schedule earlier before travel."
        ),
        "stats": [
            ("Primary route",    "DEN → NTE",       "1 stop likely"),
            ("Flight time",      "~12h",            "Plus layover"),
            ("Time zones",       "+8h",             "France ahead of Denver"),
            ("Airport Transfer", "NTE → Quiberon",  "LOC Scheduled Transfer"),
            ("Transfer Distance","~2 hours",        ""),
        ],
        "hotel_title": "Host Hotel / Official Accommodation Read",
        "hotel_bullets": [
            "Official accommodation is 3 hotels: Sofitel Quiberon Thalassa Sea & Spa, Mercure Quiberon Hotel, and ENV-I2N.",
            "Sofitel = premium recovery/wellness choice, oceanfront thalassotherapy setting.",
            "Mercure = practical team option: renovated rooms, close to beach/thalasso area, free Wi-Fi, bike storage in package.",
            "ENV-I2N = budget-oriented accommodation within a short ride/cycle to the event site.",
        ],
        "food_title": "Food / Grocery Options Near Hotel + Venue",
        "food_bullets": [
            "Best stock-up target: Super U Quiberon (116 Rue du Port de Pêche) for groceries, water, snacks, and room food.",
            "Quick basics: town-center bakeries, small markets, pharmacies, and convenience food around Quiberon centre / beach area.",
            "Restaurant read: seafood/crêperie-heavy resort town; plan simple athlete meals if avoiding rich sauces or unfamiliar seafood.",
            "Logistics note: peninsula access is constrained by one road/rail link. Potential for heavy traffic if coming from outside town center.",
        ],
    },
}


# ── World Triathlon official points formula ──────────────────────────────────
# From the WT competition rules (Section 1.5–1.6):
#
#   points(rank) = base × (1 + QFF) × top5_bonus(rank) × decay(rank)
#
# Where:
#   • base — set by event tier (Continental Champs = 400, Continental Cup = 250,
#            WTCS = 1000, World Cup = 500, World Para Series = 500, etc.)
#   • QFF (Quality of Field Factor) — set annually per continent. Top-points
#            continent gets +30% on Continental Championships and +20% on
#            Continental Cups. Other continents get a proportional QFF.
#            Empirical 2025/26: Europe ≈ +30%, Asia ≈ +9%, Oceania ≈ +8%,
#            Africa ≈ +2%, Americas ≈ +12%.
#   • top5_bonus — extra points for top-5 finishers at Continental Champs only:
#            1st +25%, 2nd +20%, 3rd +15%, 4th +10%, 5th +5%.
#   • decay — universal: 0.925^(rank-1). Each finish position is 92.5% of the
#            previous one. Verified across WTCS, World Cup, and Continental Champs
#            data in our DB.
#
# Sanity-checks against our breakdown table:
#   Europe Champs winner: 400 × 1.30 × 1.25 × 1.000 = 650 ✓ (actual: 650)
#   Americas Cup winner:  250 × 1.12 × 1.00 × 1.000 = 280 ✓ (actual: 280)
#   WTCS winner:         1000 ×  1.0 × 1.00 × 1.000 = 1000 ✓ (actual: 1000)

POINTS_DECAY_FACTOR  = 0.925
TOP5_BONUS           = {1: 0.25, 2: 0.20, 3: 0.15, 4: 0.10, 5: 0.05}

TIER_BASE: dict[str, dict] = {
    # tier_name → {base, top5_bonus, qff_continent_field, default_qff}
    "World Championship Series": {"base": 1000, "top5": False, "qff_field": False, "default_qff": 0.0},
    "World Cup":                 {"base": 500,  "top5": False, "qff_field": False, "default_qff": 0.0},
    # default_qff is the Americas-region fallback; auto-derived per continent at runtime.
    "Continental Championships": {"base": 400,  "top5": True,  "qff_field": True,  "default_qff": 0.17},
    "Continental Cup":           {"base": 250,  "top5": False, "qff_field": True,  "default_qff": 0.12},
    "World Para Series":         {"base": 500,  "top5": False, "qff_field": False, "default_qff": 0.0},
    "World Para Cup":            {"base": 250,  "top5": False, "qff_field": False, "default_qff": 0.0},
}

CONTINENT_HINTS = {
    "americas": ["Americas", "Pan American", "Pan Am", "American", "Americas Triathlon"],
    "europe":   ["Europe Triathlon", "Europe ", "European"],
    "asia":     ["Asia Triathlon", "Asian"],
    "oceania":  ["Oceania Triathlon", "Oceania"],
    "africa":   ["Africa Triathlon", "Africa "],
}


def _detect_continent(event_name: str | None, venue: str | None = None) -> str | None:
    """Match an event/venue string to a continent code (americas/europe/asia/oceania/africa)."""
    hay = f"{event_name or ''}  {venue or ''}".lower()
    for code, hints in CONTINENT_HINTS.items():
        if any(h.lower() in hay for h in hints):
            return code
    return None


def derive_continental_qff(engine, continent: str | None,
                            tier_name: str = "Continental Championships") -> tuple[float | None, str]:
    """Empirically derive the QFF for (continent, tier_name) for the CURRENT season.

    Important: WT sets the QFF annually based on the prior year's points (rule 1.5.b).
    Continental Cup races happen frequently — they're the most reliable signal for
    the current QFF. Continental Championships happen ~once per continent per year,
    so for a brand-new event (like Antofagasta '26 before race day) we can only
    derive from current-year Cups and scale.

    Per rule 1.5.v, the top continent gets +20% on Cups and +30% on Championships
    (Champs = Cup × 1.5). For other continents, QFF scales proportionally, so the
    same Cup × 1.5 relationship is assumed when scaling Cup → Championships data.

    Algorithm:
        1. Query Continental Cup winners from the CURRENT calendar year, same continent.
        2. Un-discount period=2 rows (×3) to get face-value pts.
        3. Solve: face_pts = 250 × (1 + cup_qff)  →  cup_qff = face_pts/250 − 1
        4. If asked for Championships QFF: return cup_qff × 1.5.

    Returns (qff, source_label).
    """
    if not continent:
        return (None, "no continent")
    spec = TIER_BASE.get(tier_name)
    if not spec:
        return (None, f"unknown tier '{tier_name}'")

    hints = CONTINENT_HINTS.get(continent, [])
    if not hints:
        return (None, f"unknown continent '{continent}'")
    where_clauses = " OR ".join(f"ev.event_name ILIKE :h{i}" for i in range(len(hints)))
    params = {f"h{i}": f"%{h}%" for i, h in enumerate(hints)}
    current_year = date.today().year
    params["year_start"] = date(current_year, 1, 1)

    # Always derive from Continental Cup winners — base 250, no top-5 bonus,
    # decay 1.0 at pos 1. Restrict to current year to capture the current-season QFF.
    sql = text(f"""
        SELECT ev.event_id, ev.event_date,
               MIN(ev.event_name) AS event_name,
               MAX(b.points * CASE WHEN b.period=2 THEN 3 ELSE 1 END) AS face_pts
        FROM athlete_ranking_breakdown b
        JOIN (SELECT DISTINCT event_id, event_date, event_name, cat_name FROM events) ev
          USING (event_id)
        WHERE b.ranking_cat_id = 13
          AND b.position = 1
          AND ev.cat_name LIKE 'Continental Cup%%'
          AND ev.event_date >= :year_start
          AND ({where_clauses})
        GROUP BY ev.event_id, ev.event_date
        ORDER BY ev.event_date DESC
        LIMIT 15
    """)
    with engine.connect() as c:
        rows = c.execute(sql, params).fetchall()
    if not rows:
        return (None, f"no {current_year} {continent.title()} Continental Cup data")
    face_vals = sorted(r[3] for r in rows)
    median_face = face_vals[len(face_vals) // 2]
    cup_qff = round((median_face / 250.0) - 1.0, 3)

    if tier_name == "Continental Cup":
        return (cup_qff,
                f"derived from {len(rows)} {current_year} {continent.title()} Cup winners "
                f"(median {median_face:.0f} pts → 250 × (1 + {cup_qff:+.0%}))")
    elif tier_name == "Continental Championships":
        champs_qff = round(cup_qff * 1.5, 3)
        return (champs_qff,
                f"derived from {len(rows)} {current_year} {continent.title()} Cup winners "
                f"(Cup QFF {cup_qff:+.0%} × 1.5 per rule 1.5.v → Champs QFF {champs_qff:+.0%})")
    else:
        return (cup_qff, f"using Cup QFF {cup_qff:+.0%} for {tier_name}")


def compute_points_scheme(tier_name: str, qff: float = 0.0) -> dict[int, int]:
    """Apply the official WT formula to produce max points by finish position.
    Returns {1: ..., 3: ..., 5: ..., 10: ..., 15: ..., 20: ...}."""
    spec = TIER_BASE.get(tier_name)
    if not spec:
        return {}
    base = spec["base"]
    apply_top5 = spec["top5"]
    out: dict[int, int] = {}
    for rank in [1, 3, 5, 10, 15, 20]:
        bonus = (1 + TOP5_BONUS.get(rank, 0.0)) if (apply_top5 and rank <= 5) else 1.0
        decay = POINTS_DECAY_FACTOR ** (rank - 1)
        pts = base * (1 + qff) * bonus * decay
        out[rank] = int(round(pts))
    return out


def _resolve_points_scheme(cat_name: str | None,
                           continent: str | None = None,
                           engine=None,
                           qff_override: float | None = None) -> tuple[str | None, dict[int, int] | None, dict | None]:
    """Match a race's cat_name to a points scheme, deriving QFF from continent.

    Returns (tier_label, scheme_dict, derivation_meta).
    derivation_meta has keys: qff, qff_source, base, top5_bonus_applies.
    """
    if not cat_name:
        return (None, None, None)
    cat = cat_name.strip()
    # Resolve tier (exact then substring match)
    tier_name = None
    if cat in TIER_BASE:
        tier_name = cat
    else:
        for t in TIER_BASE:
            if t.lower() in cat.lower():
                tier_name = t
                break
    if not tier_name:
        return (None, None, None)
    spec = TIER_BASE[tier_name]
    # QFF: override > empirical > default
    qff_source = ""
    if qff_override is not None:
        qff = qff_override
        qff_source = f"CLI override ({qff:+.0%})"
    elif spec["qff_field"] and continent and engine is not None:
        derived, src = derive_continental_qff(engine, continent, tier_name)
        if derived is not None:
            qff = derived
            qff_source = src
        else:
            qff = spec["default_qff"]
            qff_source = f"default ({qff:+.0%}) — {src}"
    else:
        qff = spec["default_qff"] if spec["qff_field"] else 0.0
        qff_source = f"non-continental ({qff:+.0%})" if not spec["qff_field"] else f"default ({qff:+.0%})"
    scheme = compute_points_scheme(tier_name, qff)
    meta = {"qff": qff, "qff_source": qff_source,
            "base": spec["base"], "top5_bonus_applies": spec["top5"],
            "continent": continent}
    return (tier_name, scheme, meta)


def query_usa_ranking_snapshot(engine, ranking_cat_id: int, limit: int = 25) -> pd.DataFrame:
    """Latest USA ranking snapshot + rank delta vs the previous weekly snapshot.

    ranking_cat_id: 13 = World Men, 14 = World Women, 15 = WTCS Men, 16 = WTCS Women.
    Returns columns: rank_position, athlete_name, total_points, curr, prev, change.
    `change` is signed rank delta vs prior snapshot (negative = improved rank).
    """
    sql = text("""
        WITH snaps AS (
            SELECT retrieved_at,
                   ROW_NUMBER() OVER (ORDER BY retrieved_at DESC) AS rn
            FROM (SELECT DISTINCT retrieved_at
                  FROM athlete_rankings
                  WHERE ranking_cat_id = :cat) d
        ),
        latest AS (SELECT retrieved_at FROM snaps WHERE rn = 1),
        prior  AS (SELECT retrieved_at FROM snaps WHERE rn = 2)
        SELECT  ar.rank_position,
                ar.athlete_name,
                ar.total_points,
                ar.events_current_period       AS curr,
                ar.events_previous_period      AS prev,
                ar_prior.rank_position         AS prev_rank
        FROM    athlete_rankings ar
        JOIN    athlete a USING (athlete_id)
        LEFT JOIN athlete_rankings ar_prior
               ON ar_prior.athlete_id      = ar.athlete_id
              AND ar_prior.ranking_cat_id  = ar.ranking_cat_id
              AND ar_prior.retrieved_at    = (SELECT retrieved_at FROM prior)
        WHERE   ar.ranking_cat_id = :cat
          AND   a.country = 'United States'
          AND   ar.retrieved_at = (SELECT retrieved_at FROM latest)
        ORDER BY ar.rank_position
        LIMIT :limit
    """)
    df = pd.read_sql(sql, engine, params={"cat": ranking_cat_id, "limit": limit})
    if df.empty:
        return df
    df["change"] = df["prev_rank"] - df["rank_position"]
    return df


def query_latest_ranking_date(engine, ranking_cat_id: int = 13) -> date | None:
    with engine.connect() as c:
        r = c.execute(text(
            "SELECT MAX(retrieved_at) FROM athlete_rankings WHERE ranking_cat_id = :c"
        ), {"c": ranking_cat_id}).fetchone()
    return r[0] if r and r[0] else None


def _fmt_change(delta) -> tuple[str, RGBColor]:
    """Format a signed rank delta as '▲N' (green, improved) / '▼N' (red, dropped) / '—'."""
    if delta is None or pd.isna(delta) or delta == 0:
        return ("—", MID_GRAY)
    delta = int(delta)
    if delta > 0:
        return (f"▲{delta}", RGBColor(0x1B, 0x7F, 0x3A))   # rank went up (lower number)
    return (f"▼{abs(delta)}", RGBColor(0xC0, 0x00, 0x00))   # rank went down


def add_usa_rankings_snapshot_slide(prs: Presentation, engine, venue: str,
                                    race_cat_name: str | None = None,
                                    race_event_name: str | None = None,
                                    qff_override: float | None = None,
                                    limit_per_gender: int = 18,
                                    ranking_type: str = "world") -> bool:
    """Two side-by-side tables (USA Men | USA Women) for a ranking snapshot,
    plus a bottom card showing max points available at this race tier (world only).

    ranking_type:
      "world"   → cats 13/14 (current World Rankings). Shows max-points card.
      "olympic" → cats 11/12 (LA 2028 Olympic Qualification). No points card —
                  Olympic qualification ranking aggregates differently.
    """
    cat_men, cat_women = (13, 14) if ranking_type == "world" else (11, 12)
    title_label = "USA World Rankings Snapshot" if ranking_type == "world" \
                  else "USA Olympic Qualification Snapshot"
    type_label  = "World Triathlon Rankings" if ranking_type == "world" \
                  else "Olympic Qualification Rankings (LA 2028 cycle)"

    men_df   = query_usa_ranking_snapshot(engine, ranking_cat_id=cat_men,   limit=limit_per_gender)
    women_df = query_usa_ranking_snapshot(engine, ranking_cat_id=cat_women, limit=limit_per_gender)
    if men_df.empty and women_df.empty:
        print(f"  [{title_label}] no snapshot data — slide skipped")
        return False

    snap_date = query_latest_ranking_date(engine, cat_men)
    tier_label, scheme, meta = (None, None, None)
    if ranking_type == "world":
        continent = _detect_continent(race_event_name, venue)
        tier_label, scheme, meta = _resolve_points_scheme(
            race_cat_name, continent=continent, engine=engine, qff_override=qff_override
        )

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, title_label, venue)

    # Sub-header
    sub_top = Inches(1.2)
    sub_bar = slide.shapes.add_shape(1, Inches(0.3), sub_top, Inches(12.73), Inches(0.42))
    sub_bar.fill.solid()
    sub_bar.fill.fore_color.rgb = LIGHT_GRAY
    sub_bar.line.color.rgb = MID_GRAY
    snap_txt = f"As of {snap_date.isoformat()}" if snap_date else "Latest snapshot"
    if ranking_type == "world":
        tier_txt = f"   •   Race tier: {tier_label}" if tier_label else "   •   Race tier: unknown"
    else:
        tier_txt = ""
    _add_textbox(slide,
                 f"{snap_txt}   •   {type_label}   •   Heading into {venue}{tier_txt}",
                 Inches(0.4), sub_top + Inches(0.08),
                 Inches(12.5), Inches(0.28),
                 font_size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # ── Max Points Available card — MOVED ABOVE tables (was getting cut off) ──
    tables_top = Inches(1.85)  # default if no Max Points card
    if ranking_type == "world":
        pts_top = Inches(1.75)
        pts_bar = slide.shapes.add_shape(1, Inches(0.3), pts_top, Inches(12.73), Inches(0.32))
        pts_bar.fill.solid()
        pts_bar.fill.fore_color.rgb = RED
        pts_bar.line.fill.background()
        _add_textbox(slide, "Max Points Available at this Race",
                     Inches(0.4), pts_top + Inches(0.03),
                     Inches(12.5), Inches(0.27),
                     font_size=12, bold=True, color=WHITE)

        cards_top = pts_top + Inches(0.42)
        if scheme:
            positions = [(1, "Win"), (3, "Podium"), (5, "Top 5"), (10, "Top 10"),
                         (15, "Top 15"), (20, "Top 20")]
            n = len(positions)
            gap_in = 0.18
            card_w_in = (12.73 - gap_in * (n - 1)) / n
            for i, (pos, label) in enumerate(positions):
                cl = Inches(0.3 + (card_w_in + gap_in) * i)
                cw = Inches(card_w_in)
                bx = slide.shapes.add_shape(1, cl, cards_top, cw, Inches(0.55))
                bx.fill.solid()
                bx.fill.fore_color.rgb = WHITE
                bx.line.color.rgb = MID_GRAY
                _add_textbox(slide, label, cl + Inches(0.05), cards_top + Inches(0.03),
                             cw - Inches(0.1), Inches(0.2),
                             font_size=9, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
                _add_textbox(slide, f"{scheme[pos]:,} pts", cl + Inches(0.05), cards_top + Inches(0.22),
                             cw - Inches(0.1), Inches(0.32),
                             font_size=14, bold=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)
        else:
            _add_textbox(slide,
                         "Points scheme unknown for this race tier.",
                         Inches(0.4), cards_top, Inches(12.5), Inches(0.4),
                         font_size=10, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
        tables_top = pts_top + Inches(1.1)   # card spans ~1.0in; leave breathing room

    # ── Two side-by-side tables ──────────────────────────────────────────────
    col_specs = [
        ("Rank",    0.65, PP_ALIGN.CENTER),
        ("Athlete", 2.10, PP_ALIGN.LEFT),
        ("Points",  0.95, PP_ALIGN.CENTER),
        ("Curr",    0.55, PP_ALIGN.CENTER),
        ("Prev",    0.55, PP_ALIGN.CENTER),
        ("Change",  0.85, PP_ALIGN.CENTER),
    ]
    panel_w = Inches(6.35)
    panel_lefts = [Inches(0.3), Inches(0.3 + 6.35 + 0.05)]
    # How much vertical space remains for the tables (between tables_top and the footer)
    tables_top_in = float(tables_top) / 914400  # EMU → inches
    avail_h_in = 7.0 - tables_top_in - 0.45  # leave ~0.45in for footer + ribbon

    for panel_left, gender_label, df in [(panel_lefts[0], "ELITE MEN", men_df),
                                          (panel_lefts[1], "ELITE WOMEN", women_df)]:
        hdr = slide.shapes.add_shape(1, panel_left, tables_top, panel_w, Inches(0.32))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = NAVY
        hdr.line.fill.background()
        _add_textbox(slide, gender_label,
                     panel_left, tables_top + Inches(0.03),
                     panel_w, Inches(0.26),
                     font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        if df.empty:
            _add_textbox(slide, "No USA athletes in current snapshot.",
                         panel_left, tables_top + Inches(0.5),
                         panel_w, Inches(0.4),
                         font_size=11, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
            continue

        n_rows = len(df) + 1
        # Shrink rows hard to fit — tables now compete with Max Points card above
        row_h_in = max(0.20, min(0.28, (avail_h_in - 0.4) / n_rows))
        tbl_shape = slide.shapes.add_table(
            n_rows, len(col_specs),
            panel_left, tables_top + Inches(0.36),
            panel_w, Inches(row_h_in * n_rows),
        )
        tbl = tbl_shape.table
        total_w_units = sum(c[1] for c in col_specs)
        for ci, (_, w, _) in enumerate(col_specs):
            tbl.columns[ci].width = Inches(w / total_w_units * 6.35)
        for ci, (hdr_t, _, align) in enumerate(col_specs):
            _set_cell(tbl.cell(0, ci), hdr_t, bold=True, font_size=9,
                      color=WHITE, align=align, bg_color=NAVY)

        for ri, row in enumerate(df.itertuples(index=False), start=1):
            bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
            pts = f"{int(round(row.total_points)):,}"
            curr = str(int(row.curr)) if row.curr is not None and not pd.isna(row.curr) else "—"
            prev = str(int(row.prev)) if row.prev is not None and not pd.isna(row.prev) else "—"
            change_str, change_color = _fmt_change(row.change)
            cells = [
                (str(int(row.rank_position)), bg, DARK_GRAY, True,  PP_ALIGN.CENTER),
                (str(row.athlete_name),       bg, DARK_GRAY, False, PP_ALIGN.LEFT),
                (pts,                         bg, DARK_GRAY, False, PP_ALIGN.CENTER),
                (curr,                        bg, DARK_GRAY, False, PP_ALIGN.CENTER),
                (prev,                        bg, DARK_GRAY, False, PP_ALIGN.CENTER),
                (change_str,                  bg, change_color, True, PP_ALIGN.CENTER),
            ]
            for ci, (val, c_bg, c_color, c_bold, c_align) in enumerate(cells):
                _set_cell(tbl.cell(ri, ci), val,
                          font_size=8.5, bold=c_bold, color=c_color,
                          bg_color=c_bg, align=c_align)

    # ── Footer caption ───────────────────────────────────────────────────────
    if ranking_type == "world":
        if meta:
            base = meta["base"]
            qff = meta["qff"]
            t5 = "  •  Top-5 bonus: 1st +25% / 2nd +20% / 3rd +15% / 4th +10% / 5th +5%" \
                 if meta["top5_bonus_applies"] else ""
            footer = (
                f"Formula: base × (1+QFF) × top-5 bonus × 0.925^(rank−1).   "
                f"Base for {tier_label} = {base}.   QFF = {qff:+.0%} ({meta['qff_source']}).{t5}   "
                f"Rankings table: latest weekly snapshot (USA filter); Change = rank delta vs prior week."
            )
        else:
            footer = "Rankings: athlete_rankings (USA filter). Max points: scheme unknown for this race tier."
    else:
        footer = (
            "Rankings: athlete_rankings cat 11 (Male) / 12 (Female), USA filter, latest weekly snapshot.   "
            "Curr / Prev = events scoring in Year 2 vs Year 1 of the LA 2028 OQR cycle (results show as "
            "Period 'First' in the per-race breakdown). All current results are in Year 1 because we are "
            "still inside the first year of the qualifying window, so points appear under 'Prev'.   "
            "Change = rank delta vs prior weekly snapshot."
        )
    _add_textbox(slide, footer,
                 Inches(0.3), Inches(7.05), Inches(12.73), Inches(0.28),
                 font_size=8, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
    return True


# ── Olympic Qualification Pace constants ─────────────────────────────────────
# Paris 2024 OQR cutoff: rank 30 finished with 3,626.69 pts. Each athlete's
# OQR score is capped at their best 8 current-period + 4 previous-period
# events = 12 events max. So per-event pace target = 3626.69 / 12 ≈ 302 pts.
PARIS_OQR_CUTOFF_PTS = 3626.69
OQR_EVENT_CAP        = 12
OQR_PACE_TARGET      = PARIS_OQR_CUTOFF_PTS / OQR_EVENT_CAP   # ≈ 302.22


def _pace_position_for_tier(tier_name: str, qff: float = 0.0,
                            target_pts: float = OQR_PACE_TARGET) -> tuple[int | None, int | None]:
    """At what finish position at this tier does an athlete score >= target_pts?
    Returns (best_qualifying_rank, first_failing_rank). Returns (None, None) if
    even winning doesn't hit target."""
    spec = TIER_BASE.get(tier_name)
    if not spec:
        return (None, None)
    base = spec["base"]
    apply_top5 = spec["top5"]
    last_qualifying = None
    for rank in range(1, 31):
        bonus = (1 + TOP5_BONUS.get(rank, 0.0)) if (apply_top5 and rank <= 5) else 1.0
        decay = POINTS_DECAY_FACTOR ** (rank - 1)
        pts = base * (1 + qff) * bonus * decay
        if pts >= target_pts:
            last_qualifying = rank
        else:
            return (last_qualifying, rank)
    return (last_qualifying, None)


def query_usa_oqr_pace(engine, ranking_cat_id: int) -> pd.DataFrame:
    """Per-USA-athlete OQR pace: total pts, event count, on/off pace events, projection.
    pace = total_breakdown_pts / n_events; projected = pace × 12 cap."""
    sql = text("""
        WITH latest AS (
            SELECT MAX(retrieved_at) AS d FROM athlete_rankings WHERE ranking_cat_id = :cat
        )
        SELECT ar.rank_position,
               ar.athlete_name,
               ar.total_points,
               COUNT(b.event_id)::int                                  AS n_events,
               COUNT(*) FILTER (WHERE b.points >= :target)::int        AS on_pace,
               COUNT(*) FILTER (WHERE b.points <  :target)::int        AS off_pace,
               COALESCE(SUM(b.points), 0)::float                        AS sum_bd_pts
        FROM   athlete_rankings ar
        JOIN   athlete a USING (athlete_id)
        LEFT JOIN athlete_ranking_breakdown b
               ON b.athlete_id = ar.athlete_id
              AND b.ranking_cat_id = ar.ranking_cat_id
              AND b.included = TRUE
        WHERE  ar.ranking_cat_id = :cat
          AND  a.country = 'United States'
          AND  ar.retrieved_at = (SELECT d FROM latest)
        GROUP BY ar.rank_position, ar.athlete_name, ar.total_points
        ORDER BY ar.rank_position
    """)
    df = pd.read_sql(sql, engine, params={"cat": ranking_cat_id, "target": OQR_PACE_TARGET})
    if df.empty:
        return df
    # Project current pace across the 12-event cap
    df["pace_per_event"] = df.apply(
        lambda r: (r["sum_bd_pts"] / r["n_events"]) if r["n_events"] > 0 else None,
        axis=1
    )
    df["projected_at_12"] = df["pace_per_event"] * OQR_EVENT_CAP
    return df


def query_current_oqr_cutoff(engine, ranking_cat_id: int) -> dict | None:
    """Return the current rank-30 athlete (the bubble line at this snapshot)."""
    with engine.connect() as c:
        r = c.execute(text("""
            SELECT rank_position, athlete_name, total_points,
                   events_current_period, events_previous_period
            FROM athlete_rankings
            WHERE ranking_cat_id = :cat
              AND retrieved_at = (SELECT MAX(retrieved_at) FROM athlete_rankings WHERE ranking_cat_id = :cat)
              AND rank_position = 30
        """), {"cat": ranking_cat_id}).fetchone()
    if not r:
        return None
    return {"rank": int(r[0]), "name": r[1], "total": float(r[2]),
            "curr_events": r[3], "prev_events": r[4]}


def add_olympic_pace_slide(prs: Presentation, engine, venue: str,
                            race_continent: str | None = None) -> bool:
    """Olympic Qualification pace slide:
        - Header card with Paris 2024 benchmark + per-event pace target
        - Tier table: position at WTCS / WCup / Continental Champs / Continental Cup needed
        - Two USA athlete tables (Men | Women) showing pace + projection
        - Footer card: current rank-30 athlete (early-cycle bubble)
    """
    men_df   = query_usa_oqr_pace(engine, ranking_cat_id=11)
    women_df = query_usa_oqr_pace(engine, ranking_cat_id=12)
    if men_df.empty and women_df.empty:
        print("  [Olympic Pace] no OQR data — slide skipped")
        return False

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Olympic Qualification Pace", venue)

    # ── Top: benchmark + pace target card ───────────────────────────────────
    card_top = Inches(1.2)
    card_h   = Inches(0.95)
    card = slide.shapes.add_shape(1, Inches(0.3), card_top, Inches(12.73), card_h)
    card.fill.solid()
    card.fill.fore_color.rgb = NAVY
    card.line.fill.background()
    _add_textbox(slide,
                 f"Paris 2024 OQR cutoff (30th): {PARIS_OQR_CUTOFF_PTS:,.2f} pts   "
                 f"÷   {OQR_EVENT_CAP}-event cap (8 best current + 4 best previous)   "
                 f"=   {OQR_PACE_TARGET:.0f} pts / event on pace",
                 Inches(0.4), card_top + Inches(0.10),
                 Inches(12.5), Inches(0.40),
                 font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _add_textbox(slide,
                 "An athlete who averages ~302 pts per event across their best 12 events "
                 "is on pace to clear the LA 2028 qualifying threshold.",
                 Inches(0.4), card_top + Inches(0.52),
                 Inches(12.5), Inches(0.36),
                 font_size=11, color=RGBColor(0xAA, 0xBB, 0xDD),
                 align=PP_ALIGN.CENTER, italic=True)

    # ── Tier breakdown: what position at each tier scores 302+ ──────────────
    tier_top = card_top + card_h + Inches(0.2)
    tier_bar = slide.shapes.add_shape(1, Inches(0.3), tier_top, Inches(12.73), Inches(0.32))
    tier_bar.fill.solid()
    tier_bar.fill.fore_color.rgb = RED
    tier_bar.line.fill.background()
    _add_textbox(slide,
                 f"FINISH POSITION AT EACH TIER THAT HITS {OQR_PACE_TARGET:.0f}-PT PACE",
                 Inches(0.4), tier_top + Inches(0.03),
                 Inches(12.5), Inches(0.26),
                 font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Build the tier rows
    tier_specs = [
        ("World Championship Series",         0.0),
        ("World Cup",                          0.0),
        ("Continental Championships (Americas)", 0.18),
        ("Continental Championships (Europe)",  0.30),
        ("Continental Cup (Americas)",          0.12),
    ]
    tier_table_top = tier_top + Inches(0.42)
    tier_cols = [
        ("Tier",             4.8, PP_ALIGN.LEFT),
        ("Win pts",          1.5, PP_ALIGN.CENTER),
        ("Last on-pace pos", 2.4, PP_ALIGN.CENTER),
        ("First off-pace pos (pts)", 3.0, PP_ALIGN.CENTER),
        ("Verdict",          1.0, PP_ALIGN.CENTER),
    ]
    n_t_rows = len(tier_specs) + 1
    tt_shape = slide.shapes.add_table(
        n_t_rows, len(tier_cols),
        Inches(0.3), tier_table_top,
        Inches(12.73), Inches(0.28 * n_t_rows),
    )
    tt = tt_shape.table
    tot_w = sum(c[1] for c in tier_cols)
    for ci, (_, w, _) in enumerate(tier_cols):
        tt.columns[ci].width = Inches(w / tot_w * 12.73)
    for ci, (hdr, _, align) in enumerate(tier_cols):
        _set_cell(tt.cell(0, ci), hdr, bold=True, font_size=9.5,
                  color=WHITE, align=align, bg_color=NAVY)

    for ri, (tier_label, qff) in enumerate(tier_specs, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        # Strip "(Americas)"/"(Europe)" suffix for tier resolution
        tier_key = tier_label.split(" (")[0]
        spec = TIER_BASE[tier_key]
        win_pts = spec["base"] * (1 + qff) * ((1.25) if spec["top5"] else 1.0)
        last_ok, first_fail = _pace_position_for_tier(tier_key, qff)
        if last_ok is None:
            verdict = "Not achievable"
            v_color = RGBColor(0xC0, 0x00, 0x00)
            last_str = f"None (win = {win_pts:.0f})"
            fail_str = "—"
        else:
            verdict = f"Top {last_ok}"
            v_color = RGBColor(0x1B, 0x7F, 0x3A)
            last_str = f"#{last_ok}"
            # Compute first-fail pts for context
            spec_ = TIER_BASE[tier_key]
            bonus = (1 + TOP5_BONUS.get(first_fail, 0.0)) if (spec_["top5"] and first_fail <= 5) else 1.0
            fail_pts = spec_["base"] * (1 + qff) * bonus * (POINTS_DECAY_FACTOR ** (first_fail - 1)) if first_fail else None
            fail_str = f"#{first_fail} ({fail_pts:.0f} pts)" if first_fail else "—"
        row_cells = [
            (tier_label,      bg, DARK_GRAY, False, PP_ALIGN.LEFT),
            (f"{win_pts:.0f}", bg, DARK_GRAY, True,  PP_ALIGN.CENTER),
            (last_str,        bg, DARK_GRAY, True,  PP_ALIGN.CENTER),
            (fail_str,        bg, DARK_GRAY, False, PP_ALIGN.CENTER),
            (verdict,         bg, v_color,   True,  PP_ALIGN.CENTER),
        ]
        for ci, (val, c_bg, c_color, c_bold, c_align) in enumerate(row_cells):
            _set_cell(tt.cell(ri, ci), val, font_size=9.5, bold=c_bold,
                      color=c_color, bg_color=c_bg, align=c_align)

    # ── USA pace tables ──────────────────────────────────────────────────────
    usa_top = tier_table_top + Inches(0.28 * n_t_rows + 0.25)
    panel_w_in = 6.35
    panel_w    = Inches(panel_w_in)
    panel_lefts = [Inches(0.3), Inches(0.3 + panel_w_in + 0.05)]

    pace_cols = [
        ("Rk",        0.45, PP_ALIGN.CENTER),
        ("Athlete",   2.10, PP_ALIGN.LEFT),
        ("Total",     0.80, PP_ALIGN.CENTER),
        ("Races",     0.55, PP_ALIGN.CENTER),
        ("On",        0.45, PP_ALIGN.CENTER),
        ("Off",       0.45, PP_ALIGN.CENTER),
        ("Pace",      0.75, PP_ALIGN.CENTER),
        ("Proj @12",  0.80, PP_ALIGN.CENTER),
    ]

    for panel_left, gender_label, df in [(panel_lefts[0], "USA MEN",   men_df),
                                          (panel_lefts[1], "USA WOMEN", women_df)]:
        hdr = slide.shapes.add_shape(1, panel_left, usa_top, panel_w, Inches(0.30))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = NAVY
        hdr.line.fill.background()
        _add_textbox(slide, gender_label,
                     panel_left, usa_top + Inches(0.03),
                     panel_w, Inches(0.24),
                     font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        if df.empty:
            _add_textbox(slide, "No USA athletes in OQR snapshot.",
                         panel_left, usa_top + Inches(0.45),
                         panel_w, Inches(0.4),
                         font_size=10, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
            continue

        n_rows = len(df) + 1
        row_h_in = 0.23
        tbl_shape = slide.shapes.add_table(
            n_rows, len(pace_cols),
            panel_left, usa_top + Inches(0.35),
            panel_w, Inches(row_h_in * n_rows),
        )
        tbl = tbl_shape.table
        tot = sum(c[1] for c in pace_cols)
        for ci, (_, w, _) in enumerate(pace_cols):
            tbl.columns[ci].width = Inches(w / tot * panel_w_in)
        for ci, (hdr_t, _, align) in enumerate(pace_cols):
            _set_cell(tbl.cell(0, ci), hdr_t, bold=True, font_size=8.5,
                      color=WHITE, align=align, bg_color=NAVY)

        for ri, row in enumerate(df.itertuples(index=False), start=1):
            bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
            pace  = row.pace_per_event
            proj  = row.projected_at_12
            on_pace_proj = (proj is not None and not pd.isna(proj) and proj >= PARIS_OQR_CUTOFF_PTS)
            proj_color = RGBColor(0x1B, 0x7F, 0x3A) if on_pace_proj else DARK_GRAY
            cells = [
                (str(int(row.rank_position)),      bg, DARK_GRAY,  True,  PP_ALIGN.CENTER),
                (str(row.athlete_name),            bg, DARK_GRAY,  False, PP_ALIGN.LEFT),
                (f"{row.total_points:,.0f}",       bg, DARK_GRAY,  True,  PP_ALIGN.CENTER),
                (str(int(row.n_events)),           bg, DARK_GRAY,  False, PP_ALIGN.CENTER),
                (str(int(row.on_pace)),            bg, RGBColor(0x1B, 0x7F, 0x3A), True, PP_ALIGN.CENTER),
                (str(int(row.off_pace)),           bg, RGBColor(0xC0, 0x00, 0x00), True, PP_ALIGN.CENTER),
                (f"{pace:.0f}" if pace is not None and not pd.isna(pace) else "—",
                 bg, DARK_GRAY, False, PP_ALIGN.CENTER),
                (f"{proj:,.0f}" if proj is not None and not pd.isna(proj) else "—",
                 bg, proj_color, True, PP_ALIGN.CENTER),
            ]
            for ci, (val, c_bg, c_color, c_bold, c_align) in enumerate(cells):
                _set_cell(tbl.cell(ri, ci), val, font_size=8.5, bold=c_bold,
                          color=c_color, bg_color=c_bg, align=c_align)

    # ── Footer: current OQR cutoff line (early-cycle context) + caption ─────
    men_30 = query_current_oqr_cutoff(engine, 11)
    wom_30 = query_current_oqr_cutoff(engine, 12)
    cutoff_strs = []
    if men_30:
        cutoff_strs.append(f"Men #{men_30['rank']}: {men_30['name']} = {men_30['total']:.0f} pts")
    if wom_30:
        cutoff_strs.append(f"Women #{wom_30['rank']}: {wom_30['name']} = {wom_30['total']:.0f} pts")
    cutoff_line = (
        "Current early-cycle rank-30 line:  " + "   •   ".join(cutoff_strs)
        + "   (low because LA 2028 cycle is just opening — Paris cutoff 3,627 pts is the 3-year benchmark)"
    ) if cutoff_strs else ""

    _add_textbox(slide,
                 cutoff_line + "\n"
                 "On-pace = events scoring ≥302 pts. Off-pace = events <302 pts. "
                 "Pace = total breakdown pts ÷ events raced. Proj @12 = pace × 12-event cap (green if ≥ Paris cutoff).",
                 Inches(0.3), Inches(7.00), Inches(12.73), Inches(0.35),
                 font_size=8, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
    return True


def add_top_swim_threats_slide(prs: Presentation, engine, venue: str,
                                upcoming_event_id: int | None,
                                limit_per_panel: int = 7,
                                distance_override: str | None = None) -> bool:
    """Top swim threats on the startlist, ranked by the prediction model's
    EWMA swim features. Four-panel grid: Elite M | Elite W on top row,
    U23 M | U23 W on bottom row (so USA U23 athletes on dual-category startlists
    surface alongside the elite field).

    Uses tri_analysis.prediction.features.build_features_for_program (~30s per
    program). distance_override forces the same distance category across all
    programs (useful when DB rows are stale, e.g. Antofagasta Elite is tagged
    'sprint' in the DB but the LOC's NF guide says Standard).
    """
    if not upcoming_event_id:
        print("  [Swim Threats] no upcoming event id — slide skipped")
        return False

    # Resolve Elite + U23 prog_ids for this event (Men/Women)
    with engine.connect() as c:
        progs = c.execute(text("""
            SELECT prog_id, prog_name FROM events
            WHERE event_id = :eid
              AND (prog_name ILIKE 'Elite%' OR prog_name ILIKE 'U23%')
            ORDER BY prog_id
        """), {"eid": upcoming_event_id}).fetchall()
    if not progs:
        print(f"  [Swim Threats] no Elite / U23 programs for event_id={upcoming_event_id}")
        return False

    def _pick(prefix: str, women: bool):
        for p in progs:
            n = p[1]
            if women:
                if n.startswith(prefix) and "Women" in n:
                    return p
            else:
                if n.startswith(prefix) and "Men" in n and "Women" not in n:
                    return p
        return None

    panel_specs = [
        ("ELITE MEN",   _pick("Elite", women=False)),
        ("ELITE WOMEN", _pick("Elite", women=True)),
        ("U23 MEN",     _pick("U23",   women=False)),
        ("U23 WOMEN",   _pick("U23",   women=True)),
    ]

    try:
        from tri_analysis.prediction.features import build_features_for_program
        from tri_analysis.prediction.sql import ProgramKey
    except Exception as exc:
        print(f"  [Swim Threats] prediction package not importable ({exc}) — slide skipped")
        return False

    override_meta = ({"prog_distance_category": distance_override}
                     if distance_override else None)

    def _swim_rank(prog_row) -> pd.DataFrame:
        if prog_row is None:
            return pd.DataFrame()
        try:
            df = build_features_for_program(
                engine, ProgramKey(event_id=upcoming_event_id, prog_id=prog_row[0]),
                use_start_list=True, match_distance=True, elite_only=True,
                event_meta_override=override_meta,
            )
        except Exception as exc:
            print(f"  [Swim Threats] feature build failed for prog_id={prog_row[0]}: {exc}")
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        # Need at least 1 prior swim result in same-distance history. The Races
        # column makes small-sample athletes visible so the reader can discount
        # accordingly (e.g. an athlete with 1 race + 67% front pack vs 20 races + 50%).
        if "n_swim_results" in df.columns:
            df = df[df["n_swim_results"].fillna(0) >= 1].copy()
        sort_cols, ascending = [], []
        if "front_pack_rate" in df.columns:
            sort_cols.append("front_pack_rate"); ascending.append(False)
        if "avg_swim_gap_leader" in df.columns:
            sort_cols.append("avg_swim_gap_leader"); ascending.append(True)
        if "ema_swim_sec_5" in df.columns:
            sort_cols.append("ema_swim_sec_5"); ascending.append(True)
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
        return df.head(limit_per_panel).reset_index(drop=True)

    print(f"  [Swim Threats] building features for {sum(1 for _, p in panel_specs if p)} programs "
          f"at event_id={upcoming_event_id} (distance override: {distance_override or 'none'})")
    panel_dfs = []
    for label, prog in panel_specs:
        df = _swim_rank(prog) if prog else pd.DataFrame()
        panel_dfs.append((label, prog, df))

    if all(df.empty for _, _, df in panel_dfs):
        print("  [Swim Threats] no startlist features computed — slide skipped")
        return False

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Top Swim Threats", venue)

    # Sub-header
    sub_top = Inches(1.18)
    sub_bar = slide.shapes.add_shape(1, Inches(0.3), sub_top, Inches(12.73), Inches(0.35))
    sub_bar.fill.solid()
    sub_bar.fill.fore_color.rgb = LIGHT_GRAY
    sub_bar.line.color.rgb = MID_GRAY
    dist_note = f"   •   Distance: {distance_override.title()}" if distance_override else ""
    _add_textbox(slide,
                 f"Ranked by EWMA swim features from the elite prediction model   •   "
                 f"Front Pack % → Avg Gap → EWMA split{dist_note}",
                 Inches(0.4), sub_top + Inches(0.04),
                 Inches(12.5), Inches(0.27),
                 font_size=10.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # ── Four-panel grid (2 cols × 2 rows) ─────────────────────────────────────
    col_specs = [
        ("#",            0.35, PP_ALIGN.CENTER),
        ("Athlete",      1.95, PP_ALIGN.LEFT),
        ("Lead Pack %",  1.05, PP_ALIGN.CENTER),
        ("Avg Gap (s)",  1.05, PP_ALIGN.CENTER),
        ("EWMA Swim",    1.00, PP_ALIGN.CENTER),
        ("Races",        0.55, PP_ALIGN.CENTER),
    ]
    panel_w_in   = 6.35
    panel_w      = Inches(panel_w_in)
    panel_gap_in = 0.05
    panel_lefts  = [Inches(0.3), Inches(0.3 + panel_w_in + panel_gap_in)]
    row_tops     = [Inches(1.65), Inches(4.45)]
    row_h_in     = 0.28
    hdr_h        = Inches(0.30)

    for idx, (label, _prog, df) in enumerate(panel_dfs):
        col = idx % 2
        row = idx // 2
        panel_left = panel_lefts[col]
        panel_top  = row_tops[row]
        # Header bar
        hdr = slide.shapes.add_shape(1, panel_left, panel_top, panel_w, hdr_h)
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = NAVY
        hdr.line.fill.background()
        _add_textbox(slide, label,
                     panel_left, panel_top + Inches(0.03),
                     panel_w, Inches(0.24),
                     font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        if df.empty:
            _add_textbox(slide, "No startlist / insufficient swim history.",
                         panel_left, panel_top + Inches(0.4),
                         panel_w, Inches(0.4),
                         font_size=10, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
            continue

        n_rows = len(df) + 1
        tbl_top = panel_top + hdr_h + Inches(0.05)
        tbl_shape = slide.shapes.add_table(
            n_rows, len(col_specs),
            panel_left, tbl_top,
            panel_w, Inches(row_h_in * n_rows),
        )
        tbl = tbl_shape.table
        total_w_units = sum(c[1] for c in col_specs)
        for ci, (_, w, _) in enumerate(col_specs):
            tbl.columns[ci].width = Inches(w / total_w_units * panel_w_in)
        for ci, (hdr_t, _, align) in enumerate(col_specs):
            _set_cell(tbl.cell(0, ci), hdr_t, bold=True, font_size=9,
                      color=WHITE, align=align, bg_color=NAVY)

        for ri, brow in enumerate(df.itertuples(index=False), start=1):
            bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
            fpr = getattr(brow, "front_pack_rate", None)
            gap = getattr(brow, "avg_swim_gap_leader", None)
            ema = getattr(brow, "ema_swim_sec_5", None)
            nrace = getattr(brow, "n_swim_results", None)
            name = getattr(brow, "athlete_full_name", None) or getattr(brow, "athlete_name", "—")
            # Highlight USA athletes (the team's focus)
            country = (getattr(brow, "athlete_country_name", None) or "").lower()
            is_usa = country in ("united states", "usa")
            name_color = RED if is_usa else DARK_GRAY
            fpr_str  = f"{fpr * 100:.0f}%" if (fpr is not None and not pd.isna(fpr)) else "—"
            gap_str  = f"{gap:+.1f}" if (gap is not None and not pd.isna(gap)) else "—"
            ema_str  = seconds_to_mmss(ema) if (ema is not None and not pd.isna(ema)) else "—"
            nrace_str = f"{int(nrace)}" if (nrace is not None and not pd.isna(nrace)) else "—"
            cells = [
                (str(ri),   bg, NAVY,       True,  PP_ALIGN.CENTER),
                (str(name), bg, name_color, is_usa, PP_ALIGN.LEFT),
                (fpr_str,   bg, DARK_GRAY,  True,  PP_ALIGN.CENTER),
                (gap_str,   bg, DARK_GRAY,  False, PP_ALIGN.CENTER),
                (ema_str,   bg, DARK_GRAY,  False, PP_ALIGN.CENTER),
                (nrace_str, bg, MID_GRAY,   False, PP_ALIGN.CENTER),
            ]
            for ci, (val, c_bg, c_color, c_bold, c_align) in enumerate(cells):
                _set_cell(tbl.cell(ri, ci), val,
                          font_size=8.5, bold=c_bold, color=c_color,
                          bg_color=c_bg, align=c_align)

    # ── Footer caption ───────────────────────────────────────────────────────
    _add_textbox(slide,
                 "Lead Pack % = fraction of recent races finishing the swim in pack 1.   "
                 "Avg Gap = mean seconds behind the swim leader.   "
                 "EWMA Swim = exponentially-weighted moving average of the last 5 swim splits.   "
                 "USA athletes highlighted in red.   "
                 "Races column = sample size (small samples warrant caution).   "
                 "Features from tri_analysis.prediction.features.build_features_for_program.",
                 Inches(0.3), Inches(7.0), Inches(12.73), Inches(0.35),
                 font_size=8, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
    return True


def add_travel_load_slide(prs: Presentation, venue: str) -> bool:
    """Travel & arrival logistics — DEN→venue routing, hotel, food."""
    travel = TRAVEL_PROFILES.get(venue.lower().strip())
    if not travel:
        return False

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Travel & Arrival Load", travel.get("origin", venue))

    # ── Core travel read (rounded card top) ──────────────────────────────────
    read_top = Inches(1.2)
    read_h   = Inches(1.05)
    read_bx = slide.shapes.add_shape(1, Inches(0.3), read_top, Inches(12.73), read_h)
    read_bx.fill.solid()
    read_bx.fill.fore_color.rgb = RGBColor(0xE9, 0xEF, 0xF9)
    read_bx.line.color.rgb = MID_GRAY
    _add_textbox(slide, "Core travel read",
                 Inches(0.4), read_top + Inches(0.07),
                 Inches(12.5), Inches(0.3),
                 font_size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _add_textbox(slide, travel.get("core_read", ""),
                 Inches(0.5), read_top + Inches(0.38),
                 Inches(12.3), Inches(0.62),
                 font_size=10.5, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    # ── Stat cards row ───────────────────────────────────────────────────────
    stats = travel.get("stats", [])
    if stats:
        cards_top = read_top + read_h + Inches(0.2)
        n = min(len(stats), 5)
        card_total_w_in = 12.73
        gap_in = 0.15
        card_w_in = (card_total_w_in - gap_in * (n - 1)) / n
        for i, (label, main, sub) in enumerate(stats[:n]):
            cl = Inches(0.3 + (card_w_in + gap_in) * i)
            cw = Inches(card_w_in)
            bx = slide.shapes.add_shape(1, cl, cards_top, cw, Inches(0.95))
            bx.fill.solid()
            bx.fill.fore_color.rgb = WHITE
            bx.line.color.rgb = MID_GRAY
            _add_textbox(slide, label, cl + Inches(0.08), cards_top + Inches(0.05),
                         cw - Inches(0.16), Inches(0.22),
                         font_size=10, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
            _add_textbox(slide, main, cl + Inches(0.08), cards_top + Inches(0.28),
                         cw - Inches(0.16), Inches(0.32),
                         font_size=14, bold=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)
            if sub:
                _add_textbox(slide, sub, cl + Inches(0.08), cards_top + Inches(0.62),
                             cw - Inches(0.16), Inches(0.28),
                             font_size=9, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)

    # ── Two panels (hotel + food) ────────────────────────────────────────────
    panels_top = read_top + read_h + Inches(1.35)
    panel_h    = Inches(2.6)
    panel_w    = Inches(6.2)
    panels = [
        (Inches(0.3),              travel.get("hotel_title", ""),  travel.get("hotel_bullets", [])),
        (Inches(0.3) + panel_w + Inches(0.3), travel.get("food_title", ""), travel.get("food_bullets", [])),
    ]
    for left, title, bullets in panels:
        bx = slide.shapes.add_shape(1, left, panels_top, panel_w, panel_h)
        bx.fill.solid()
        bx.fill.fore_color.rgb = WHITE
        bx.line.color.rgb = MID_GRAY
        _add_textbox(slide, title,
                     left + Inches(0.15), panels_top + Inches(0.12),
                     panel_w - Inches(0.3), Inches(0.32),
                     font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        for i, b in enumerate(bullets[:6]):
            _add_textbox(slide, f"•  {b}",
                         left + Inches(0.18), panels_top + Inches(0.5 + i * 0.34),
                         panel_w - Inches(0.36), Inches(0.36),
                         font_size=10, color=DARK_GRAY)
    return True


# ── Venue preview narrative + feature bullets (Key Course Differentiators) ──
# Used to populate the venue intro slide. When a venue has no historical race
# results, this is the primary way an athlete gets a feel for the course.
VENUE_PREVIEW: dict[str, dict] = {
    "antofagasta": {
        "narrative": (
            "First-time Americas Triathlon & Para Championships host. Antofagasta sits on Chile's "
            "northern Pacific coast where the cold Humboldt Current meets the Atacama Desert — "
            "expect a cool, dry, light-wind race in Southern Hemisphere winter (avg high 18 °C). "
            "Elite/U23 athletes contest a Standard race on Saturday with a bike circuit slightly "
            "longer than typical (8 × 5.414 km = 43.3 km), and the swim uses a distinctive "
            "beach-exit-and-re-entry between laps. Expect a wetsuit race in 14–16 °C ocean water."
        ),
        "features": [
            "Pacific Ocean swim — Humboldt-cold (~14–16 °C) — Standard format swims 2 laps with a beach exit + re-entry between them.",
            "Bike: 8 × 5.414 km laps (43.3 km — ~3 km longer than standard 40 km) along Av República de Croacia ↔ Av Ejército. Flat, fast, exposed to ocean breeze.",
            "Run: 4 × 2.5 km laps on Av Grecia coastal road — flat asphalt out-and-back, fast splits in cool conditions.",
            "First-ever AT Championships in Antofagasta — no prior race-day data; weather climatology + Pacific SST will drive prep.",
        ],
        "format_label":   "Standard Triathlon (Elite)",
        "race_info_url":  "https://events.triathlon.org/2026-americas-triathlon-championships-antofagasta-/race-info",
        "race_info_text": "Race Info | 2026 Americas Triathlon & Para Championships Antofagasta",
    },
    "montreal": {
        "narrative": (
            "World Triathlon Para Series Montréal returns to Parc Jean-Drapeau for the 2026 "
            "edition. The race uses the venerable Olympic Basin for a 750 m swim, then heads to "
            "the Circuit Gilles-Villeneuve — the F1 track — for a flat, fast 21.5 km bike on a "
            "closed circuit. The run mixes the CGV and the path alongside the rowing basin. "
            "Late June in Montréal can run hot and humid; with 23 °C water, expect a non-wetsuit "
            "race and a power-driven bike."
        ),
        "features": [
            "Olympic Basin swim — sheltered, ~23 °C fresh water, in-water pontoon start. Non-wetsuit.",
            "Bike on Circuit Gilles-Villeneuve (F1 track): pancake-flat 4.1 km clockwise loop × 5 = 21.5 km.",
            "Run on CGV + Olympic Basin path: flat 2.5 km loop × 2 with five aid-station access points.",
            "12 distinct Para sport-class starts (PTWC, PTVI, PTS2–5) staged 07:00 → 08:30 across both genders.",
        ],
        "format_label":   "Sprint Para Triathlon",
        "race_info_url":  "https://events.triathlon.org/2026-world-triathlon-para-series-montreal",
        "race_info_text": "Race Info | 2026 World Triathlon Para Series Montréal",
    },
    "quiberon": {
        "narrative": (
            "Set on the southern tip of Brittany's Quiberon peninsula, this will be the first "
            "edition of WTCS Quiberon. Expect a cool-water, likely wetsuit sprint race with a "
            "decently long beach start. The bike is largely flat but has multiple sections "
            "along the coast with potential for exposure to coastal winds. The run is also a "
            "simple out-and-back on asphalt that should lead to fast splits."
        ),
        "features": [
            "Beach Start 750 m Atlantic Swim. Cool water, likely wetsuit-legal.",
            "Flat bike course. 4 laps of 5.5 km will make the distance slightly longer than standard. "
            "No real elevation but potential for exposed wind could come into play.",
            "Mostly asphalt out-and-back run. Simple 2-lap course.",
            "First race at this venue.",
        ],
        "format_label":   "Sprint Triathlon",
        "race_info_url":  "https://events.triathlon.org/2026-world-triathlon-championship-series-quiberon",
        "race_info_text": "Race Info | 2026 World Triathlon Championship Series Quiberon",
    },
    "huatulco": {
        "narrative": (
            "Huatulco's World Cup returns to the Bahía de Santa Cruz on Mexico's Pacific coast. "
            "A staple Tier-2 World Cup with strong historical depth in the database, the course "
            "rewards swimmers off the beach start, riders who can survive the short Vialidad 5 "
            "climb, and runners who can hold pace in tropical heat at the 08:30 men's start."
        ),
        "features": [
            "Bahía de Santa Cruz swim — warm water (~30 °C), beach start, 750 m single loop.",
            "5 km bike loop × 4 (20 km) with a punchy Vialidad 5 climb (max 18.4%) every lap.",
            "Hilly 2.5 km run loop × 2 — +55 m gain/lap, max 17.3% on Camino a Santa Cruz.",
            "Tropical heat is the dominant tactical variable. Pre-cool, sponge plan mandatory.",
        ],
        "format_label":   "Sprint Triathlon",
        "race_info_url":  "https://events.triathlon.org/2026-world-triathlon-cup-huatulco",
        "race_info_text": "Race Info | 2026 World Triathlon Cup Huatulco",
    },
}


# ── Race-week schedule data ───────────────────────────────────────────────────
# Each row: (time_label, activity_text, is_highlighted). Highlighted rows render
# in red (used for start times and mandatory meetings).
EVENT_SCHEDULES: dict[str, dict] = {
    "antofagasta": {
        "title":      "2026 Americas Triathlon & Para Championships — Antofagasta",
        "date_range": "July 3 – 5, 2026",
        "venue_note": "Balneario Municipal, Av República de Croacia, Antofagasta (CLT, UTC-4)",
        "race_starts": [("Elite race window", "10:00"), ("Mid-day", "12:00")],
        "days": [
            ("Fri • July 3", "Familiarization & Briefing", [
                ("All day",       "Elite / Junior Cycling familiarization", False),
                ("All day",       "Elite / Junior Swimming familiarization", False),
                ("All day",       "Para Triathlon Cycling + Swim familiarizations", False),
                ("13:00 – 18:00", "Age-Group Registration / Kit delivery — ENJOY Antofagasta", False),
                ("16:00 – 16:45", "★ Elite / Junior briefing — ENJOY Antofagasta", True),
                ("16:45 – 18:15", "Elite / Junior Registration / Kit delivery", False),
            ]),
            ("Sat • July 4", "CHAMPIONSHIP RACE DAY", [
                ("AM",            "★ Junior Women Championship — Sprint", True),
                ("AM",            "★ Junior Men Championship — Sprint", True),
                ("Late AM",       "★ Elite / U23 Women Championship — STANDARD", True),
                ("Mid-day",       "★ Elite / U23 Men Championship — STANDARD", True),
                ("15:00 – 18:00", "Age-Group Registration / Kit delivery", False),
                ("16:00 – 16:45", "Para Triathlon Championship Briefing", False),
                ("16:45",         "Youth Championship Briefing", False),
                ("18:00",         "Age-Group Championship Briefing", False),
            ]),
            ("Sun • July 5", "Para + AG + Relay", [
                ("AM",            "★ Para Triathlon Championship — Sprint", True),
                ("AM",            "Youth Women Championship — Super Sprint", False),
                ("AM",            "Youth Men Championship — Super Sprint", False),
                ("Mid-day",       "Age-Group Championship — Standard", False),
                ("PM",            "2×2 Mixed Relay — Elite + Juniors", False),
            ]),
        ],
    },
    "montreal": {
        "title":      "2026 World Triathlon Para Series Montréal — Race Week",
        "date_range": "June 24 – 27, 2026",
        "venue_note": "Parc Jean-Drapeau, Notre-Dame Island, Montréal (EDT, UTC-4)",
        "race_starts": [("PTWC start", "07:00"), ("PTS/PTVI window", "08:00")],
        "days": [
            ("Thu • June 25", "Open Training", [
                ("11:00 – 15:00", "Swim training — Aquatic Complex (50 m pool, 4 lanes)", False),
                ("14:00 – 17:00", "Gym training — Athletes' Quarter gym", False),
                ("(Wed Jun 24)",   "Pool training 14:00–18:00 + gym 14:00–17:00", False),
            ]),
            ("Fri • June 26", "Familiarization & Briefing", [
                ("07:45",         "Departures from Host Hotel", False),
                ("08:45 – 09:15", "PTS/PTVI Bike Fam  •  PTWC Run Fam — Transition", False),
                ("09:30 – 10:00", "PTS/PTVI Run Fam  •  PTWC Bike Fam — Transition", False),
                ("10:15 – 11:00", "Swim Familiarisation (ALL) — Swim Start", False),
                ("10:15 – 10:45", "Coaches Onsite Meeting — Transition Area", False),
                ("10:30 – 12:00", "Equipment verification (ALL)", False),
                ("12:00 – 12:45", "★ WTPS Pre-race briefing (ALL) — Athletes' Quarter", True),
                ("12:45 – 13:30", "Race Package Distribution", False),
                ("13:45",         "Departures from venue", False),
            ]),
            ("Sat • June 27", "PARA RACE DAY", [
                ("05:00 / 05:30", "Hotel departures — PTWC 05:00 • PTS/PTVI 05:30", False),
                ("05:30 / 06:15", "Athletes' Lounge opens — PTWC 05:30 • PTS/PTVI 06:15", False),
                ("06:15 – 06:45", "PTWC swim warm-up (on course) + transition check-in", False),
                ("06:53",         "PTWC Athletes' Introduction", False),
                ("07:00 / 07:03", "★ PTWC1 Men 07:00  •  PTWC2 Men 07:03", True),
                ("07:04 / 07:08", "★ PTWC1 Women 07:04  •  PTWC2 Women 07:08", True),
                ("07:45",         "★ PTS5 Men start", True),
                ("07:50 / 07:53", "★ PTVI1 Men 07:50  •  PTVI2/3 Men 07:53", True),
                ("07:54 / 07:57", "★ PTVI1 Women 07:54  •  PTVI2/3 Women 07:57", True),
                ("08:10 / 08:15", "★ PTS5 Women 08:10  •  PTS2/3/4 Women 08:15", True),
                ("08:25 / 08:30", "★ PTS4 Men 08:25  •  PTS2/3 Men 08:30", True),
                ("10:15 / 12:00", "Venue departures — 1st bus 10:15  •  2nd 12:00", False),
                ("10:30 – 11:30", "WTPS Medal Ceremony — Podium", False),
            ]),
        ],
    },
    "quiberon": {
        "title":      "2026 WTCS Quiberon — Race Week",
        "date_range": "June 18 – 21, 2026",
        "venue_note": "Espace Louison Bobet, Bd René Cassin, Quiberon (CEST, UTC+2)",
        "race_starts": [("Men start", "10:00"), ("Women start", "12:00")],
        "days": [
            ("Friday • June 19", "Familiarization & Briefing", [
                ("07:00 – 10:00", "Pool training — Neptune Swimming Pool", False),
                ("09:00 – 09:30", "Bike familiarization — 2 laps from transition (escorted)", False),
                ("10:30",         "Swim familiarization — beach", False),
                ("14:00 – 17:00", "Pool training — Neptune", False),
                ("16:00 – 16:30", "★ Mandatory Elite Athletes' briefing — Espace Louison Bobet", True),
                ("16:30 – 17:00", "Elite team medical meeting — LOC Office", False),
                ("Post-briefing", "Race package distribution", False),
                ("19:00",         "Run course measurement (coaches) — Finish Area, Bd René Cassin", False),
            ]),
            ("Saturday • June 20", "WTCS SPRINT RACE DAY", [
                ("08:30 – 09:30", "WTCS Men — Athletes' lounge check-in", False),
                ("09:00 – 09:45", "WTCS Men — Transition + swim warm-up", False),
                ("09:50",         "WTCS Men — Athletes' introduction", False),
                ("10:00",         "★ WTCS MEN START", True),
                ("11:05",         "WTCS Men medals ceremony", False),
                ("10:30 – 11:30", "WTCS Women — Athletes' lounge check-in", False),
                ("11:00 – 11:45", "WTCS Women — Transition + swim warm-up", False),
                ("11:50",         "WTCS Women — Athletes' introduction", False),
                ("12:00",         "★ WTCS WOMEN START", True),
                ("13:10",         "WTCS Women medals ceremony", False),
                ("13:30 – 14:00", "Mixed Relay — Team Declaration", False),
                ("20:00 – 20:30", "Mixed Relay — Team Managers' Meeting (Elite Athletes Area)", False),
            ]),
            ("Sunday • June 21", "Mixed Relay Day", [
                ("07:00 – 10:00", "Pool training — Neptune", False),
                ("12:30 – 12:45", "Mixed Relay — Final Team Declaration", False),
                ("15:30 – 16:30", "Mixed Relay — Athletes' lounge check-in", False),
                ("16:00 – 16:45", "Mixed Relay — Transition + swim warm-up", False),
                ("16:50",         "Mixed Relay — Athletes' introduction", False),
                ("17:00",         "★ MIXED RELAY START", True),
                ("18:30",         "Mixed Relay medals ceremony", False),
            ]),
        ],
    },
    "huatulco": {
        "title":      "2026 World Triathlon Cup Huatulco — Race Week",
        "date_range": "June 12 – 14, 2026",
        "venue_note": "Plaza Santa Cruz Huatulco, Mexico",
        "race_starts": [("Female start", "06:30"), ("Male start", "08:30")],
        "days": [
            ("Friday • June 12", "Familiarization Day", [
                ("10:00",         "Bike familiarization — 3 laps from Bahía Santa Cruz", False),
                ("10:30",         "Run familiarization", False),
                ("11:00 – 11:45", "Swim familiarization — Santa Cruz", False),
            ]),
            ("Saturday • June 13", "Briefing & Packet Pickup", [
                ("16:00 – 16:30", "★ Mandatory Elite athlete meeting — Hotel Biniguenda", True),
                ("16:30 – 17:00", "Packet pickup — Hotel Biniguenda", False),
            ]),
            ("Sunday • June 14", "RACE DAY", [
                ("05:30", "Elite Female lounge opens", False),
                ("06:30", "★ ELITE FEMALE START", True),
                ("07:45", "Elite Male lounge opens", False),
                ("08:30", "★ ELITE MALE START", True),
                ("10:00", "Awards ceremony", False),
            ]),
        ],
    },
}


def build_bike_profile_chart(profile: dict) -> io.BytesIO:
    """Synthesize a stylised elevation profile over the full bike distance.

    The chart is a representative single-loop wave repeated `loops` times,
    scaled to match the published gain_per_lap_m. It is intentionally schematic
    — exact contours require a GPS-recorded FIT/GPX. The shape uses one big
    climb + descent + short rolling section per loop, matching the bike course
    map for Huatulco (and most short-circuit World Cups).
    """
    import numpy as _np

    loops      = int(profile.get("loops", 1))
    loop_km    = float(profile.get("loop_km", 5.0))
    gain_lap   = float(profile.get("gain_per_lap_m", 80.0))
    loss_lap   = float(profile.get("loss_per_lap_m", gain_lap))

    pts_per_loop = 400
    xs_loop = _np.linspace(0, loop_km, pts_per_loop, endpoint=False)
    # Asymmetric wave: short steep climb (first 40%) then a longer descent (40-90%)
    # then a small bump back to baseline (90-100%). Scale to gain_lap.
    def _loop_shape(x):
        t = x / loop_km
        y = _np.where(
            t < 0.4,
            (t / 0.4) ** 1.2,
            _np.where(
                t < 0.9,
                1.0 - ((t - 0.4) / 0.5) ** 1.1,
                0.15 * _np.sin((t - 0.9) / 0.1 * _np.pi),
            ),
        )
        return y
    base = _loop_shape(xs_loop)
    base = base - base.min()
    base = base / max(base.max(), 1e-6) * gain_lap

    xs_full = _np.concatenate([xs_loop + i * loop_km for i in range(loops)])
    ys_full = _np.tile(base, loops)

    fig, ax = plt.subplots(figsize=(11.0, 2.4), dpi=150)
    ax.fill_between(xs_full, ys_full, 0, color="#4472C4", alpha=0.35, linewidth=0)
    ax.plot(xs_full, ys_full, color="#002060", linewidth=1.4)

    for i in range(1, loops):
        ax.axvline(i * loop_km, color="#C00000", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(i * loop_km, ys_full.max() * 1.02, f"Lap {i + 1}",
                ha="center", va="bottom", fontsize=8, color="#C00000")
    ax.text(loop_km / 2, ys_full.max() * 1.02, "Lap 1",
            ha="center", va="bottom", fontsize=8, color="#C00000")

    ax.set_xlim(0, loops * loop_km)
    ax.set_ylim(0, ys_full.max() * 1.18)
    ax.set_xlabel("Distance (km)", fontsize=9)
    ax.set_ylabel("Relative elevation (m)", fontsize=9)
    ax.set_title(f"Bike Course — {loops} × {loop_km:.0f} km loop ({loops * loop_km:.0f} km total)",
                 fontsize=11, fontweight="bold", color="#002060", pad=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig_to_image(fig)


def add_bike_course_profile_slide(prs: Presentation, venue: str, all_rows: list[dict]):
    """Dedicated bike course slide — published course specs + synthesized profile + tactical notes.

    Only added when BIKE_COURSE_PROFILES has data for this venue. Optionally embeds
    a course-map image from ppt files/support/{venue_lc}_bike_course.png if present.
    """
    profile = BIKE_COURSE_PROFILES.get(venue.lower().strip())
    if not profile:
        return False

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Bike Course Profile", venue)

    # ── Left: course map image (if present at expected path) ──────────────────
    map_path = os.path.join(REPO_ROOT, "ppt files", "support",
                            f"{venue.lower().strip()}_bike_course.png")
    map_left  = Inches(0.3)
    map_top   = Inches(1.25)
    map_w     = Inches(5.6)
    map_h     = Inches(3.6)

    if os.path.exists(map_path):
        slide.shapes.add_picture(map_path, map_left, map_top, map_w, map_h)
    else:
        ph = slide.shapes.add_shape(1, map_left, map_top, map_w, map_h)
        ph.fill.solid()
        ph.fill.fore_color.rgb = LIGHT_GRAY
        ph.line.color.rgb = MID_GRAY
        _add_textbox(slide,
                     f"[Drop course map here]\n\n"
                     f"Save the organiser course map as:\n"
                     f"ppt files/support/{venue.lower().strip()}_bike_course.png\n\n"
                     f"It will be embedded automatically on the next run.",
                     map_left + Inches(0.2), map_top + Inches(1.2),
                     map_w - Inches(0.4), Inches(1.4),
                     font_size=10, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)

    # Source caption under map
    _add_textbox(slide, f"Source: {profile.get('source', '—')}",
                 map_left, map_top + map_h + Inches(0.05),
                 map_w, Inches(0.25),
                 font_size=9, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)

    # ── Right: stats card ─────────────────────────────────────────────────────
    stats_left = Inches(6.1)
    stats_top  = Inches(1.25)
    stats_w    = Inches(6.9)

    # Big distance card
    hero_h = Inches(0.9)
    hero = slide.shapes.add_shape(1, stats_left, stats_top, stats_w, hero_h)
    hero.fill.solid()
    hero.fill.fore_color.rgb = NAVY
    hero.line.fill.background()
    _add_textbox(slide,
                 f"{profile['total_km']:g} km   •   {profile['loops']} × {profile['loop_km']:g} km loops",
                 stats_left, stats_top + Inches(0.12),
                 stats_w, Inches(0.4),
                 font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _add_textbox(slide, profile.get("surface", ""),
                 stats_left, stats_top + Inches(0.54),
                 stats_w, Inches(0.32),
                 font_size=11, color=RGBColor(0xAA, 0xBB, 0xDD),
                 align=PP_ALIGN.CENTER, italic=True)

    # Stats grid (3 cols × 2 rows). Tolerate None values for venues with no
    # published elevation data (e.g. flat coastal courses).
    grid_top = stats_top + hero_h + Inches(0.15)
    grid_h   = Inches(1.5)
    cell_w   = stats_w / 3
    g  = profile.get("gain_per_lap_m")
    l_ = profile.get("loss_per_lap_m")
    avg_g  = profile.get("avg_grade_pct")
    max_p  = profile.get("max_grade_pos")
    max_n  = profile.get("max_grade_neg")
    wind   = profile.get("wind")           # optional — e.g. "Coastal (~20 km/h)"
    loops_ = profile.get("loops") or 1
    loop_km_val = profile.get("loop_km")
    # Loops cell prefers exact lap distance ("4 × 5.5 km") over loose rounding
    loops_str = (f"{loops_} × {loop_km_val:g} km"
                 if loop_km_val is not None else f"{loops_}")
    # If wind is set, it bumps Max Descent out of the grid (more actionable for
    # flat coastal courses like Quiberon/Montreal).
    middle_row_cell = (("Wind", wind) if wind
                       else ("Max Descent",
                             f"{max_n:.1f}%" if max_n is not None else "—"))
    grid_cells = [
        ("Elevation / Lap", f"+{g:.1f} m / {l_:.1f} m" if (g is not None and l_ is not None) else "Flat (no data)"),
        ("Total Climb",     f"~{g * loops_:.0f} m" if g is not None else "—"),
        ("Avg Grade",       f"{avg_g:.1f}%" if avg_g is not None else "—"),
        ("Max Climb Grade", f"+{max_p:.1f}%" if max_p is not None else "—"),
        middle_row_cell,
        ("Loops",           loops_str),
    ]
    for idx, (label, value) in enumerate(grid_cells):
        col = idx % 3
        row = idx // 3
        cl = stats_left + cell_w * col + Inches(0.05)
        cw = cell_w - Inches(0.1)
        ct = grid_top + (Inches(0.78) * row)
        bx = slide.shapes.add_shape(1, cl, ct, cw, Inches(0.7))
        bx.fill.solid()
        bx.fill.fore_color.rgb = LIGHT_GRAY
        bx.line.color.rgb = MID_GRAY
        _add_textbox(slide, label, cl + Inches(0.05), ct + Inches(0.04),
                     cw - Inches(0.1), Inches(0.22),
                     font_size=9, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        _add_textbox(slide, value, cl + Inches(0.05), ct + Inches(0.26),
                     cw - Inches(0.1), Inches(0.38),
                     font_size=15, bold=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    # Tactical notes
    notes_top = grid_top + grid_h + Inches(0.25)
    notes_bar = slide.shapes.add_shape(1, stats_left, notes_top, stats_w, Inches(0.32))
    notes_bar.fill.solid()
    notes_bar.fill.fore_color.rgb = RED
    notes_bar.line.fill.background()
    _add_textbox(slide, "Tactical Implications",
                 stats_left + Inches(0.1), notes_top + Inches(0.03),
                 stats_w - Inches(0.2), Inches(0.26),
                 font_size=12, bold=True, color=WHITE)

    body_top = notes_top + Inches(0.4)
    for i, note in enumerate(profile.get("key_features", [])):
        _add_textbox(slide, f"•  {note}",
                     stats_left + Inches(0.1),
                     body_top + Inches(0.32 * i),
                     stats_w - Inches(0.2), Inches(0.34),
                     font_size=10.5, color=DARK_GRAY)

    # ── Bottom: synthesized elevation profile chart (skip for flat courses) ───
    has_elev = profile.get("gain_per_lap_m") is not None
    if not has_elev:
        chart_left   = Inches(0.3)
        chart_top    = Inches(5.6)
        chart_width  = Inches(12.73)
        chart_height = Inches(1.55)
        ph = slide.shapes.add_shape(1, chart_left, chart_top, chart_width, chart_height)
        ph.fill.solid()
        ph.fill.fore_color.rgb = LIGHT_GRAY
        ph.line.color.rgb = MID_GRAY
        _add_textbox(slide,
                     "Flat course — published profile shows no significant climbs.\n"
                     "No elevation chart synthesized; if a race GPS FIT becomes available, "
                     "real contours will replace this block.",
                     chart_left, chart_top + Inches(0.5),
                     chart_width, Inches(0.6),
                     font_size=11, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
    else:
        try:
            chart_buf = build_bike_profile_chart(profile)
            chart_left   = Inches(0.3)
            chart_top    = Inches(5.45)
            chart_width  = Inches(12.73)
            chart_height = Inches(1.85)
            slide.shapes.add_picture(chart_buf, chart_left, chart_top, chart_width, chart_height)
            _add_textbox(slide,
                         "Stylised — exact contours from a GPS-recorded race FIT would refine this.",
                         chart_left, chart_top + chart_height - Inches(0.05),
                         chart_width, Inches(0.22),
                         font_size=8.5, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
        except Exception as exc:
            _add_textbox(slide, f"[Profile chart could not be rendered: {exc}]",
                         Inches(0.3), Inches(5.5), Inches(12.73), Inches(0.4),
                         font_size=10, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)

    return True


def _add_discipline_profile_slide(prs: Presentation, venue: str, discipline: str,
                                  profile: dict, accent_color: RGBColor,
                                  map_filename_suffix: str) -> bool:
    """Shared layout for Swim / Run course profile slides.

    Left panel: course map image (if dropped at ppt files/support/{venue}_{suffix}.png)
                or a styled placeholder.
    Right panel: hero card (distance / laps / format), 2×3 stat grid, key features,
                and a 'Missing from race brief' note when relevant.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"{discipline} Course Profile", venue)

    # ── Left: course map slot ────────────────────────────────────────────────
    map_path = os.path.join(REPO_ROOT, "ppt files", "support",
                            f"{venue.lower().strip()}_{map_filename_suffix}.png")
    map_left = Inches(0.3)
    map_top  = Inches(1.25)
    map_w    = Inches(5.6)
    map_h    = Inches(4.2)

    if os.path.exists(map_path):
        slide.shapes.add_picture(map_path, map_left, map_top, map_w, map_h)
    else:
        ph = slide.shapes.add_shape(1, map_left, map_top, map_w, map_h)
        ph.fill.solid()
        ph.fill.fore_color.rgb = LIGHT_GRAY
        ph.line.color.rgb = MID_GRAY
        _add_textbox(slide,
                     f"[Drop {discipline.lower()} course map here]\n\n"
                     f"Save the organiser map as:\n"
                     f"ppt files/support/{venue.lower().strip()}_{map_filename_suffix}.png\n\n"
                     f"It will be embedded automatically on the next run.",
                     map_left + Inches(0.2), map_top + Inches(1.5),
                     map_w - Inches(0.4), Inches(1.4),
                     font_size=10, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)

    _add_textbox(slide, f"Source: {profile.get('source', '—')}",
                 map_left, map_top + map_h + Inches(0.05),
                 map_w, Inches(0.25),
                 font_size=9, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)

    # ── Right: hero + stats grid + features ──────────────────────────────────
    right_left = Inches(6.1)
    right_top  = Inches(1.25)
    right_w    = Inches(6.9)

    hero_h = Inches(0.9)
    hero = slide.shapes.add_shape(1, right_left, right_top, right_w, hero_h)
    hero.fill.solid()
    hero.fill.fore_color.rgb = NAVY
    hero.line.fill.background()

    total_km = profile.get("total_km")
    laps     = profile.get("laps")
    loop_km  = profile.get("loop_km")

    def _fmt_distance(km: float | None) -> str:
        if not km:
            return "TBD"
        if km < 1:
            return f"{int(round(km * 1000))} m"
        return f"{km:.1f} km"

    hero_main = _fmt_distance(total_km) if total_km else "Distance TBD"
    if laps and laps > 1:
        per_lap = _fmt_distance(loop_km) if loop_km else _fmt_distance(total_km / laps if total_km else None)
        hero_main += f"   •   {laps} × {per_lap}"
    elif laps == 1:
        hero_main += "   •   single lap"
    _add_textbox(slide, hero_main,
                 right_left, right_top + Inches(0.12),
                 right_w, Inches(0.4),
                 font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    subtitle = profile.get("format") or profile.get("surface") or ""
    _add_textbox(slide, subtitle,
                 right_left, right_top + Inches(0.54),
                 right_w, Inches(0.32),
                 font_size=11, color=RGBColor(0xAA, 0xBB, 0xDD),
                 align=PP_ALIGN.CENTER, italic=True)

    # Stat grid — 3 cols × 2 rows. Cells unknown to this discipline render '—'.
    grid_top = right_top + hero_h + Inches(0.15)
    cell_w   = right_w / 3
    if discipline.lower() == "swim":
        water_temp = profile.get("water_temp_c")
        water_src  = profile.get("water_temp_source", "unavailable")
        # Wetsuit decision: use live/climatology SST if available, else fall back
        # to the published expected range from the profile, else "TBD".
        # Elite rules: mandatory <16 °C, optional 16–20 °C, forbidden >22 °C.
        expected_range = profile.get("expected_water_temp_range_c")
        ref_temp = water_temp
        if ref_temp is None and expected_range:
            ref_temp = sum(expected_range) / 2
        if ref_temp is None:
            wetsuit = "TBD — depends on day-of water temp"
        elif ref_temp >= 22:
            wetsuit = "Forbidden (> 22 °C)"
        elif ref_temp >= 20:
            wetsuit = f"Likely non-wetsuit ({ref_temp:.0f} °C)"
        elif ref_temp >= 16:
            wetsuit = f"Wetsuit optional ({ref_temp:.0f} °C)"
        else:
            wetsuit = f"Wetsuit mandatory (< 16 °C)"

        if water_temp is not None:
            tag = {"forecast": "live", "climatology": "7-yr avg"}.get(water_src, "")
            water_label = f"{water_temp:.1f} °C ({tag})" if tag else f"{water_temp:.1f} °C"
        elif expected_range:
            water_label = f"{expected_range[0]:.1f}–{expected_range[1]:.1f} °C (expected)"
        else:
            water_label = "Awaiting marine data"
        layout     = profile.get("layout") or profile.get("format", "—")
        grid_cells = [
            ("Distance",      _fmt_distance(total_km)),
            ("Layout",        layout.split(" — ")[0] if " — " in (layout or "") else (layout or "—")),
            ("Start",         (profile.get("start_type") or "—").split(" at ")[0]),
            ("Water Temp",    water_label),
            ("Wetsuit",       wetsuit),
            ("Lap Count",     f"{laps}" if laps else "—"),
        ]
    else:  # run
        gain_lap = profile.get("gain_per_lap_m")
        max_pos  = profile.get("max_grade_pos")
        max_neg  = profile.get("max_grade_neg")
        avg_g    = profile.get("avg_grade_pct")
        # Heat risk: prefer explicit profile field; default to "—" so it isn't
        # falsely flagged HIGH for cool venues. Forecast slide is the
        # authoritative heat-stress source.
        heat_risk = profile.get("heat_risk", "—")
        grid_cells = [
            ("Distance",      _fmt_distance(total_km)),
            ("Lap Structure", f"{laps} × {_fmt_distance(loop_km)}" if (laps and loop_km) else (f"{laps} laps" if laps else "Not published")),
            ("Total Climb",   f"~{gain_lap * laps:.0f} m" if (gain_lap and laps) else (f"+{gain_lap:.0f} m / lap" if gain_lap else "Not published")),
            ("Avg Grade",     f"{avg_g:.1f}%" if avg_g else "—"),
            ("Max ▲ / ▼",    f"+{max_pos:.1f}% / {max_neg:.1f}%" if (max_pos and max_neg) else (f"+{max_pos:.1f}%" if max_pos else "Not published")),
            ("Heat Risk",     heat_risk),
        ]

    for idx, (label, value) in enumerate(grid_cells):
        col = idx % 3
        row = idx // 3
        cl = right_left + cell_w * col + Inches(0.05)
        cw = cell_w - Inches(0.1)
        ct = grid_top + (Inches(0.78) * row)
        bx = slide.shapes.add_shape(1, cl, ct, cw, Inches(0.7))
        bx.fill.solid()
        bx.fill.fore_color.rgb = LIGHT_GRAY
        bx.line.color.rgb = MID_GRAY
        _add_textbox(slide, label, cl + Inches(0.05), ct + Inches(0.04),
                     cw - Inches(0.1), Inches(0.22),
                     font_size=9, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        _add_textbox(slide, value, cl + Inches(0.05), ct + Inches(0.26),
                     cw - Inches(0.1), Inches(0.38),
                     font_size=14, bold=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    # Key features panel
    notes_top = grid_top + Inches(1.65)
    notes_bar = slide.shapes.add_shape(1, right_left, notes_top, right_w, Inches(0.32))
    notes_bar.fill.solid()
    notes_bar.fill.fore_color.rgb = accent_color
    notes_bar.line.fill.background()
    _add_textbox(slide, "Tactical Notes",
                 right_left + Inches(0.1), notes_top + Inches(0.03),
                 right_w - Inches(0.2), Inches(0.26),
                 font_size=12, bold=True, color=WHITE)

    body_top = notes_top + Inches(0.4)
    for i, note in enumerate(profile.get("key_features", [])):
        _add_textbox(slide, f"•  {note}",
                     right_left + Inches(0.1),
                     body_top + Inches(0.32 * i),
                     right_w - Inches(0.2), Inches(0.34),
                     font_size=10.5, color=DARK_GRAY)

    # ── Bottom: 'Missing from race brief' caveat strip ───────────────────────
    missing = profile.get("missing") or []
    if missing:
        miss_top = Inches(6.6)
        miss_bar = slide.shapes.add_shape(1, Inches(0.3), miss_top, Inches(12.73), Inches(0.32))
        miss_bar.fill.solid()
        miss_bar.fill.fore_color.rgb = LIGHT_GRAY
        miss_bar.line.color.rgb = MID_GRAY
        text = "Not in race brief: " + "  •  ".join(missing)
        _add_textbox(slide, text,
                     Inches(0.4), miss_top + Inches(0.05),
                     Inches(12.5), Inches(0.25),
                     font_size=9.5, italic=True, color=DARK_GRAY, align=PP_ALIGN.LEFT)

    return True


def add_swim_course_profile_slide(prs: Presentation, venue: str,
                                  water_temp_c: float | None = None,
                                  water_temp_source: str = "unavailable") -> bool:
    profile = SWIM_COURSE_PROFILES.get(venue.lower().strip())
    if not profile:
        return False
    profile = dict(profile)
    if water_temp_c is not None:
        profile["water_temp_c"] = water_temp_c
        profile["water_temp_source"] = water_temp_source
    return _add_discipline_profile_slide(
        prs, venue, "Swim", profile, RED, "swim_course"
    )


def add_run_course_profile_slide(prs: Presentation, venue: str) -> bool:
    profile = RUN_COURSE_PROFILES.get(venue.lower().strip())
    if not profile:
        return False
    return _add_discipline_profile_slide(
        prs, venue, "Run", profile, RED, "run_course"
    )


def add_race_week_schedule_slide(prs: Presentation, venue: str) -> bool:
    """Race-week timetable. Three day-columns side by side."""
    schedule = EVENT_SCHEDULES.get(venue.lower().strip())
    if not schedule:
        return False

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Race Week Schedule", venue)

    # Sub-header bar with date range + venue
    sub_top = Inches(1.2)
    sub_bar = slide.shapes.add_shape(1, Inches(0.3), sub_top, Inches(12.73), Inches(0.42))
    sub_bar.fill.solid()
    sub_bar.fill.fore_color.rgb = LIGHT_GRAY
    sub_bar.line.color.rgb = MID_GRAY
    _add_textbox(slide,
                 f"{schedule['date_range']}   •   {schedule.get('venue_note', '')}",
                 Inches(0.4), sub_top + Inches(0.07),
                 Inches(12.5), Inches(0.3),
                 font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    days = schedule["days"]
    n_days = len(days)
    col_top    = Inches(1.85)
    gap_in     = 0.25
    total_w_in = 12.73 - gap_in * (n_days - 1)
    col_w_in   = total_w_in / n_days
    col_w      = Inches(col_w_in)
    col_gap    = Inches(gap_in)
    col_left0  = Inches(0.3)

    for di, (day_label, day_subtitle, items) in enumerate(days):
        cl = col_left0 + (col_w + col_gap) * di

        # Day header
        header_h = Inches(0.5)
        header = slide.shapes.add_shape(1, cl, col_top, col_w, header_h)
        header.fill.solid()
        header.fill.fore_color.rgb = NAVY
        header.line.fill.background()
        _add_textbox(slide, day_label,
                     cl + Inches(0.1), col_top + Inches(0.04),
                     col_w - Inches(0.2), Inches(0.22),
                     font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _add_textbox(slide, day_subtitle,
                     cl + Inches(0.1), col_top + Inches(0.26),
                     col_w - Inches(0.2), Inches(0.22),
                     font_size=9.5, color=RGBColor(0xAA, 0xBB, 0xDD),
                     italic=True, align=PP_ALIGN.CENTER)

        # Items table (time | activity)
        items_top = col_top + header_h + Inches(0.1)
        n_rows = len(items)
        tbl_h = Inches(0.55 * n_rows)
        tbl_shape = slide.shapes.add_table(n_rows, 2, cl, items_top, col_w, tbl_h)
        tbl = tbl_shape.table
        tbl.columns[0].width = Inches(col_w_in * 0.36)
        tbl.columns[1].width = Inches(col_w_in * 0.64)
        for ri, (tm, act, is_hi) in enumerate(items):
            bg     = RGBColor(0xFD, 0xEC, 0xEC) if is_hi else (LIGHT_GRAY if ri % 2 == 0 else WHITE)
            fg     = RED if is_hi else DARK_GRAY
            _set_cell(tbl.cell(ri, 0), tm,
                      font_size=10, bold=is_hi, color=fg, bg_color=bg, align=PP_ALIGN.CENTER)
            _set_cell(tbl.cell(ri, 1), act,
                      font_size=10, bold=is_hi, color=fg, bg_color=bg, align=PP_ALIGN.LEFT)

    # Footer caption — venue/timezone pulled from schedule.venue_note when present
    tz_note = schedule.get("venue_note") or f"local to {venue}"
    _add_textbox(slide,
                 f"Times {tz_note}. Schedule confirmed from World Triathlon race-info page / "
                 f"athlete guide; subject to organiser updates.",
                 Inches(0.3), Inches(6.95), Inches(12.73), Inches(0.3),
                 font_size=9, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
    return True


def _heat_band(apparent_c: float | None) -> tuple[str, RGBColor]:
    """Heat-stress band based on apparent temperature."""
    if apparent_c is None:
        return ("UNKNOWN", MID_GRAY)
    if apparent_c < 24:    return ("LOW",      RGBColor(0x1B, 0x7F, 0x3A))
    if apparent_c < 28:    return ("MODERATE", RGBColor(0xE6, 0xA8, 0x17))
    if apparent_c < 32:    return ("HIGH",     RGBColor(0xE6, 0x6A, 0x00))
    return                  ("EXTREME",        RGBColor(0xC0, 0x00, 0x00))


def add_race_day_forecast_slide(prs: Presentation, venue: str,
                                coords: tuple | None,
                                race_date: str | None,
                                race_starts: list[tuple[str, str]] | None = None) -> bool:
    """Race-day forecast slide. Pulls the forward forecast for race_date if it is
    inside Open-Meteo's 16-day window; otherwise falls back to a 7-year climatology
    averaged across the same calendar day."""
    if coords is None or race_date is None:
        return False
    lat, lon = coords

    forecast = fetch_openmeteo_forecast(lat, lon, race_date)
    mode = "forecast"
    if not forecast:
        forecast = fetch_openmeteo_climatology(lat, lon, race_date)
        mode = "climatology"
    if not forecast:
        return False

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, "Race-Day Weather Outlook", venue)

    mode_label = ("Forward forecast — Open-Meteo (≤16 days)"
                  if mode == "forecast"
                  else "Historical climatology — same calendar day, 7-year mean")
    sub_top = Inches(1.2)
    sub_bar = slide.shapes.add_shape(1, Inches(0.3), sub_top, Inches(12.73), Inches(0.42))
    sub_bar.fill.solid()
    sub_bar.fill.fore_color.rgb = LIGHT_GRAY
    sub_bar.line.color.rgb = MID_GRAY
    _add_textbox(slide,
                 f"{race_date}   •   {mode_label}",
                 Inches(0.4), sub_top + Inches(0.07),
                 Inches(12.5), Inches(0.3),
                 font_size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # ── Hourly chart (top-left) ───────────────────────────────────────────────
    times = forecast["time"]
    temps = forecast["temperature_2m"]
    apparent = forecast.get("apparent_temperature") or [None] * len(times)
    humidity = forecast["relative_humidity_2m"]
    wind     = forecast["wind_speed_10m"]
    uv       = forecast.get("uv_index") or [None] * len(times)

    fig, ax1 = plt.subplots(figsize=(8.4, 3.6), dpi=150)
    x = list(range(len(times)))
    ax1.plot(x, temps,    color="#C00000", linewidth=2.2, marker="o", markersize=4, label="Air temp (°C)")
    ax1.plot(x, apparent, color="#E66A00", linewidth=1.6, linestyle="--", marker="x",
             markersize=4, label="Apparent (°C)", alpha=0.85)
    ax1.set_xlabel("Hour (local)", fontsize=10)
    ax1.set_ylabel("Temperature (°C)", color="#C00000", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#C00000")
    ax1.set_xticks(x)
    ax1.set_xticklabels(times, rotation=45, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, humidity, color="#002060", linewidth=2.0, marker="s", markersize=4, label="Humidity (%)")
    ax2.set_ylabel("Humidity (%)", color="#002060", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#002060")
    ax2.set_ylim(0, 100)

    # Race-start vertical markers
    if race_starts:
        for label, hhmm in race_starts:
            if hhmm in times:
                xi = times.index(hhmm)
                ax1.axvline(xi, color="black", linestyle=":", linewidth=1.5, alpha=0.7)
                ax1.text(xi, ax1.get_ylim()[1] * 0.98, label,
                         rotation=90, va="top", ha="right", fontsize=8.5,
                         color="black", fontweight="bold")

    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.set_title("Hourly outlook — race window", fontsize=11, fontweight="bold", color="#002060")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    img_buf = fig_to_image(fig)
    slide.shapes.add_picture(img_buf, Inches(0.3), Inches(1.75), Inches(7.7), Inches(3.4))

    # ── Race-window summary stat cards (right side) ───────────────────────────
    right_left = Inches(8.2)
    right_w    = Inches(4.83)

    def _hour_idx(hhmm: str) -> int | None:
        """Find the time index closest to hhmm. Snaps half-hour starts (e.g.
        '06:30') to the hour bucket that contains them ('06:00')."""
        if hhmm in times:
            return times.index(hhmm)
        try:
            target_min = int(hhmm[:2]) * 60 + int(hhmm[3:5])
        except (ValueError, IndexError):
            return None
        best_i, best_d = None, None
        for i, t in enumerate(times):
            try:
                tm = int(t[:2]) * 60 + int(t[3:5])
            except (ValueError, IndexError):
                continue
            d = abs(tm - target_min)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        return best_i if best_d is not None and best_d <= 60 else None

    def _race_window_avg(values: list, start_idx: int, end_idx: int) -> float | None:
        vals = [v for v in values[start_idx:end_idx + 1] if v is not None]
        return sum(vals) / len(vals) if vals else None

    cards: list[tuple[str, list[tuple[str, str, RGBColor]]]] = []
    for label, hhmm in (race_starts or []):
        idx = _hour_idx(hhmm)
        if idx is None:
            continue
        end = min(idx + 2, len(times) - 1)
        avg_temp = _race_window_avg(temps, idx, end)
        avg_app  = _race_window_avg(apparent, idx, end) if any(a is not None for a in apparent) else None
        avg_hum  = _race_window_avg(humidity, idx, end)
        avg_wind = _race_window_avg(wind, idx, end)
        peak_uv  = max((v for v in uv[idx:end + 1] if v is not None), default=None)
        heat_label, heat_color = _heat_band(avg_app if avg_app is not None else avg_temp)
        cards.append((label + f" — {hhmm}", [
            ("Temperature",    f"{avg_temp:.1f} °C" if avg_temp is not None else "—", DARK_GRAY),
            ("Apparent / Heat", f"{avg_app:.1f} °C" if avg_app is not None else "—", DARK_GRAY),
            ("Humidity",       f"{avg_hum:.0f}%" if avg_hum is not None else "—", DARK_GRAY),
            ("Wind",           f"{avg_wind:.1f} km/h" if avg_wind is not None else "—", DARK_GRAY),
            ("UV peak",        f"{peak_uv:.1f}" if peak_uv is not None else "—", DARK_GRAY),
            ("Heat Stress",    heat_label, heat_color),
        ]))

    card_top = Inches(1.75)
    for ci, (card_title, rows_) in enumerate(cards[:2]):
        ct = card_top + Inches(0.05 + ci * 1.75)
        hdr = slide.shapes.add_shape(1, right_left, ct, right_w, Inches(0.35))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = NAVY
        hdr.line.fill.background()
        _add_textbox(slide, card_title,
                     right_left + Inches(0.1), ct + Inches(0.04),
                     right_w - Inches(0.2), Inches(0.27),
                     font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        n_cols = 3
        cell_w = right_w / n_cols
        for ri, (k, v, color) in enumerate(rows_):
            col = ri % n_cols
            row = ri // n_cols
            cl = right_left + cell_w * col
            cw = cell_w - Inches(0.04)
            cyt = ct + Inches(0.4 + row * 0.62)
            bx = slide.shapes.add_shape(1, cl + Inches(0.02), cyt, cw, Inches(0.58))
            bx.fill.solid()
            bx.fill.fore_color.rgb = LIGHT_GRAY
            bx.line.color.rgb = MID_GRAY
            _add_textbox(slide, k, cl + Inches(0.04), cyt + Inches(0.03),
                         cw - Inches(0.04), Inches(0.2),
                         font_size=8, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
            _add_textbox(slide, v, cl + Inches(0.04), cyt + Inches(0.22),
                         cw - Inches(0.04), Inches(0.34),
                         font_size=12, bold=True, color=color, align=PP_ALIGN.CENTER)

    # ── Bottom strip: tactical recommendations ────────────────────────────────
    rec_top = Inches(5.3)
    rec_bar = slide.shapes.add_shape(1, Inches(0.3), rec_top, Inches(12.73), Inches(0.32))
    rec_bar.fill.solid()
    rec_bar.fill.fore_color.rgb = RED
    rec_bar.line.fill.background()
    _add_textbox(slide, "Hydration & Cooling Cues",
                 Inches(0.4), rec_top + Inches(0.03),
                 Inches(12.5), Inches(0.26),
                 font_size=12, bold=True, color=WHITE)

    # Build cues from the data
    cues: list[str] = []
    avg_window = _race_window_avg(apparent, 0, len(times) - 1) if any(a is not None for a in apparent) else None
    if avg_window is None:
        avg_window = _race_window_avg(temps, 0, len(times) - 1)
    if avg_window and avg_window >= 30:
        cues.append("Pre-cool 20 min before start (ice slurry, ice towel on neck); fluid 6–8 ml/kg in the hour pre-race")
        cues.append("Plan 2 ice-sock handoffs on the run — every other lap; sponges at every aid station on bike & run")
    elif avg_window and avg_window >= 25:
        cues.append("Standard heat protocol — pre-cool 10 min, sponges/ice at every aid station, target 600–800 ml/h on bike")
    else:
        cues.append("Cool to mild outlook — standard fluid plan (~500 ml/h), no special pre-cooling required")

    # Female vs male delta cue
    if len(cards) == 2:
        f_temp = next((float(v.split()[0]) for k, v, _ in cards[0][1] if k == "Temperature" and v != "—"), None)
        m_temp = next((float(v.split()[0]) for k, v, _ in cards[1][1] if k == "Temperature" and v != "—"), None)
        if f_temp is not None and m_temp is not None:
            delta = m_temp - f_temp
            if delta >= 2:
                cues.append(f"Men's race runs {delta:.1f} °C hotter than women's — heat-acclim work + heavier cooling kit for the 08:30 start")
            elif delta >= 1:
                cues.append(f"Men's race ~{delta:.1f} °C warmer — add one extra cooling touchpoint vs. women's plan")
    # UV
    peak_uv_all = max((v for v in uv if v is not None), default=None)
    if peak_uv_all and peak_uv_all >= 8:
        cues.append(f"UV peaks at {peak_uv_all:.1f} (very high) — long-sleeve race kit + sunscreen reapply 30 min pre-start")

    # Wind — significant for coastal/exposed bike courses
    peak_wind = max((v for v in wind if v is not None), default=None)
    if peak_wind and peak_wind >= 25:
        cues.append(f"Wind peaks at {peak_wind:.0f} km/h — pack will split on exposed coastal sections; "
                    f"shallow front wheel + position discipline through crosswind segments")
    elif peak_wind and peak_wind >= 15:
        cues.append(f"Notable wind ({peak_wind:.0f} km/h) — drafting protection matters on exposed coastal bike sections; "
                    f"sight more frequently in chop")

    # Cold-water swim cue
    avg_temp_full = _race_window_avg(temps, 0, len(times) - 1)
    if avg_temp_full is not None and avg_temp_full < 16:
        cues.append("Cold air pre-race — extended warm-up + warm clothing to lounge; "
                    "wetsuit decision likely tilts mandatory if water tracks the same direction")

    body_top = rec_top + Inches(0.4)
    for i, cue in enumerate(cues[:4]):
        _add_textbox(slide, f"•  {cue}",
                     Inches(0.4), body_top + Inches(0.32 * i),
                     Inches(12.5), Inches(0.34),
                     font_size=10.5, color=DARK_GRAY)

    # Footnote
    _add_textbox(slide,
                 ("Data source: Open-Meteo. "
                  + ("Forecast refreshes every run. "
                     if mode == "forecast"
                     else "Forecast will replace climatology once race day is within 16 days. "))
                 + "Historical Race Conditions slide shows what actually happened in prior years.",
                 Inches(0.3), Inches(7.05), Inches(12.73), Inches(0.25),
                 font_size=9, italic=True, color=MID_GRAY, align=PP_ALIGN.CENTER)
    return True


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
        col_clean = adjust_outlier(finishers[col].dropna()).reindex(finishers.index)
        valid = finishers[col_clean.notna()].copy()
        valid[col] = col_clean[col_clean.notna()]
        valid = valid.nsmallest(3, col)
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

        WHO_TO_WATCH_LIMIT = 10
        rankings_df = query_top_by_world_ranking(engine, gender,
                                                 on_startlist_event_id=upcoming_event_id,
                                                 limit=WHO_TO_WATCH_LIMIT)
        rankings_label = f"Top {WHO_TO_WATCH_LIMIT} Ranked on Startlist" if (upcoming_event_id and not rankings_df.empty) else f"Top {WHO_TO_WATCH_LIMIT} by World Ranking"
        if rankings_df.empty and upcoming_event_id:
            rankings_df = query_top_by_world_ranking(engine, gender,
                                                     on_startlist_event_id=None,
                                                     limit=WHO_TO_WATCH_LIMIT)
            rankings_label = f"Top {WHO_TO_WATCH_LIMIT} by World Ranking (no startlist data yet)"

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
        row_h_in = 0.26  # tighter than the old 0.34 so 11 rows still fit
        tbl_shape = slide.shapes.add_table(n_rows, len(rank_cols), left, tbl_top,
                                           panel_w, Inches(row_h_in * n_rows + 0.05))
        tbl = tbl_shape.table
        for ci, (_, w) in enumerate(rank_cols):
            tbl.columns[ci].width = Inches(w)
        for ci, (hdr, _) in enumerate(rank_cols):
            _set_cell(tbl.cell(0, ci), hdr, bold=True, color=WHITE, bg_color=NAVY, font_size=10)
        if rankings_df.empty:
            _set_cell(tbl.cell(1, 0), "—", font_size=9.5, bg_color=WHITE)
            _set_cell(tbl.cell(1, 1), "No ranking data found", font_size=9.5, bg_color=WHITE,
                      align=PP_ALIGN.LEFT, italic=True)
            _set_cell(tbl.cell(1, 2), "—", font_size=9.5, bg_color=WHITE)
        else:
            for ri, r in enumerate(rankings_df.itertuples(index=False), start=1):
                bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
                pts = f"{int(r.total_points):,}" if r.total_points else "—"
                _set_cell(tbl.cell(ri, 0), str(int(r.rank_position)),
                          font_size=9.5, bold=True, bg_color=bg)
                _set_cell(tbl.cell(ri, 1), str(r.athlete_name),
                          font_size=9.5, bg_color=bg, align=PP_ALIGN.LEFT)
                _set_cell(tbl.cell(ri, 2), pts, font_size=9.5, bg_color=bg)

        # Prior-year podium block
        podium_top = tbl_top + Inches(row_h_in * n_rows + 0.25)
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
    parser.add_argument("--preview-only", action="store_true",
                        help="Suppress Elite gender-section slides (e.g. for Para Series venues where "
                             "historical Elite race data is not relevant to the upcoming race).")
    parser.add_argument("--race-date", default=None,
                        help="Override the race date (YYYY-MM-DD) used for forecast + SST lookups. "
                             "Useful when the DB event_date is stale or when no event row exists yet.")
    parser.add_argument("--race-tier", default=None,
                        help="Override the race tier label used by the USA Rankings Snapshot slide "
                             "for max-points lookup. One of: 'World Championship Series', "
                             "'Continental Championships', 'World Cup', 'Continental Cup', "
                             "'World Para Series', 'World Para Cup'.")
    parser.add_argument("--continental-qff", type=float, default=None,
                        help="Override the Continental Quality-of-Field Factor (e.g. 0.12 for "
                             "Americas, 0.30 for Europe). If omitted, derived from recent same-region "
                             "Continental Cup winners.")
    parser.add_argument("--race-distance", default=None,
                        choices=["sprint", "standard", "super_sprint", "long"],
                        help="Override the race's distance category (used by feature build + "
                             "swim-history matching). Useful when the DB row is stale.")
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
        # No prior race history. Continue in preview-only mode iff we have
        # static profile data + an upcoming event at this venue; otherwise abort.
        venue_key = args.venue.lower().strip()
        has_profile = (venue_key in BIKE_COURSE_PROFILES
                       or venue_key in SWIM_COURSE_PROFILES
                       or venue_key in RUN_COURSE_PROFILES
                       or venue_key in EVENT_SCHEDULES)
        if not has_profile:
            print(f"No events found for venue '{args.venue}' and no preview profile data.")
            sys.exit(1)
        print(f"No prior race history for '{args.venue}' — building preview-only deck "
              f"from course profile data.")
    else:
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

    # Upcoming-event lookup runs regardless of deep-dive — Who to Watch is useful for any venue
    # where we have prior race history.
    upcoming = None
    if args.upcoming_event_id:
        # Look up the real event row for the supplied id so cat_name + event_name
        # are populated (needed for tier / continent detection downstream).
        eid = int(args.upcoming_event_id)
        with engine.connect() as _c:
            _r = _c.execute(text(
                "SELECT event_id, MIN(event_date) AS event_date, "
                "MIN(event_name) AS event_name, MIN(cat_name) AS cat_name "
                "FROM events WHERE event_id = :eid GROUP BY event_id"
            ), {"eid": eid}).fetchone()
        if _r:
            upcoming = {"event_id": eid, "event_date": _r[1],
                        "event_name": _r[2], "cat_name": _r[3]}
            print(f"  Upcoming event (CLI override): {upcoming['event_name']} ({upcoming['event_date']})")
        else:
            upcoming = {"event_id": eid, "event_date": None,
                        "event_name": f"Upcoming event {eid}", "cat_name": None}
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
    if add_race_week_schedule_slide(prs, args.venue):
        print(f"  Race-week schedule slide added (venue: {args.venue})")
    add_course_differentiators_slide(prs, args.venue, all_data, events_df, venue_content)
    add_course_map_slide(prs, args.venue, all_data, venue_content)

    # Sea-surface temp for race day (forecast if in window, else prior-years avg)
    # CLI --race-date takes precedence over the DB upcoming-event date.
    if args.race_date:
        race_date_str = args.race_date
        print(f"  Race date override (CLI): {race_date_str}")
    else:
        race_date_str = (str(pd.to_datetime(upcoming["event_date"]).date())
                         if upcoming and upcoming.get("event_date") else None)
    sst_c, sst_src = (None, "unavailable")
    if coords and race_date_str:
        sst_c, sst_src = fetch_sst_for_race_day(coords[0], coords[1], race_date_str)
        if sst_c is not None:
            print(f"  Sea-surface temp for {race_date_str}: {sst_c:.1f} °C ({sst_src})")

    if add_swim_course_profile_slide(prs, args.venue, water_temp_c=sst_c,
                                     water_temp_source=sst_src):
        print(f"  Swim course profile slide added (venue: {args.venue})")
    if add_bike_course_profile_slide(prs, args.venue, all_data):
        print(f"  Bike course profile slide added (venue: {args.venue})")
    if add_run_course_profile_slide(prs, args.venue):
        print(f"  Run course profile slide added (venue: {args.venue})")

    # Race-day forecast — race-start times pulled from EVENT_SCHEDULES per venue.
    venue_schedule = EVENT_SCHEDULES.get(args.venue.lower().strip())
    race_starts = venue_schedule.get("race_starts") if venue_schedule else None
    if race_date_str and add_race_day_forecast_slide(prs, args.venue, coords,
                                                    race_date_str, race_starts):
        print(f"  Race-day forecast slide added (race_date={race_date_str})")

    if add_travel_load_slide(prs, args.venue):
        print(f"  Travel & arrival slide added (venue: {args.venue})")

    add_environmental_risk_slide(prs, args.venue, unique_env_rows)

    if prior_men or prior_women or upcoming:
        add_who_to_watch_slide(
            prs, engine, args.venue,
            upcoming_event_id=upcoming["event_id"] if upcoming else None,
            prior_men={"event_id": prior_men["event_id"], "prog_id": prior_men["prog_id"]} if prior_men else None,
            prior_women={"event_id": prior_women["event_id"], "prog_id": prior_women["prog_id"]} if prior_women else None,
        )

    # Top Swim Threats — uses the model's EWMA swim features on the startlist
    if upcoming and add_top_swim_threats_slide(
        prs, engine, args.venue,
        upcoming_event_id=upcoming["event_id"],
        distance_override=args.race_distance,
    ):
        print(f"  Top Swim Threats slide added")

    # USA Rankings Snapshot — current World Triathlon rankings + max points at this race tier
    race_cat_name = (args.race_tier
                     or (upcoming.get("cat_name") if upcoming else None)
                     or (events_df.iloc[0]["cat_name"] if not events_df.empty else None))
    race_event_name = (upcoming.get("event_name") if upcoming
                       else (events_df.iloc[0]["event_name"] if not events_df.empty else None))
    if add_usa_rankings_snapshot_slide(prs, engine, args.venue,
                                       race_cat_name=race_cat_name,
                                       race_event_name=race_event_name,
                                       qff_override=args.continental_qff,
                                       ranking_type="world"):
        print(f"  USA World Rankings Snapshot slide added (tier: {race_cat_name or 'unknown'})")

    # USA Olympic Qualification Snapshot — separate slide using cats 11/12
    if add_usa_rankings_snapshot_slide(prs, engine, args.venue,
                                       race_cat_name=race_cat_name,
                                       race_event_name=race_event_name,
                                       ranking_type="olympic"):
        print(f"  USA Olympic Qualification Snapshot slide added")

    # Olympic Qualification Pace — Paris benchmark + per-tier finish targets + USA pace
    if add_olympic_pace_slide(prs, engine, args.venue):
        print(f"  Olympic Qualification Pace slide added")

    gender_sections = []
    if args.preview_only:
        print("  Preview-only mode — skipping Elite gender-section slides")
    else:
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
