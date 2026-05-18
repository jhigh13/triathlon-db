from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import plotly.io as pio
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from tri_analysis.database import get_engine
from tri_analysis.time_utils import (
    pace_sec_per_100m,
    pace_sec_per_km,
    seconds_to_hms,
    time_to_seconds,
)

USA_COUNTRY_ALIASES = ["USA", "United States", "United States of America", "U.S.A", "US"]

DEFAULT_MAJOR_CAT_PATTERNS = [
    "%Major Games%",  # includes Paralympic Games; program filter keeps this para-only
    "%Para Cup%",
    "%Para Series%",
    # Keep champs/worlds included even if category ids differ in the feed
    "%Para Championship%",
    "%Para Championships%",
    "%World Triathlon Para%",
    "%World Championships%",  # Covers para world championships (e.g., Wollongong 2025)
]

# Additional top contenders (beyond benchmark medalists) to include in benchmarks.
# These are athletes who didn't medal at the benchmark event but are considered top-tier competitors.
# Note: PTS3 Women uses Pontevedra 2023 as benchmark (not contested at Paris 2024).
ADDITIONAL_BENCHMARK_ATHLETES = {
    "PTS2 Men": ["Maurits Morsink", "Lionel Morales", "Wim De Paepe"],
    "PTS3 Men": ["Henry Urand", "Cedric Denuziere"],  # Paris 2024 benchmark
    "PTS4 Men": ["Pierre-Antoine Baele", "Gregoire Berthon"],
    "PTS5 Men": ["Stefan Daniel", "Bence Mocsari"],
    "PTVI Men": ["Owen Cravens", "Héctor Catalá Laparra"],
    "PTWC Women": ["Jessica Ferreira"],
    "PTS2 Women": ["Anu Francis"],
    "PTS3 Women": [],  # Pontevedra 2023 top 3 are already benchmark medalists (Elise Marc, Kenia Villalobos, Sanne Koopman)
    "PTS4 Women": ["Kelly Elmlinger", "Grace Brimelow"],
    "PTS5 Women": ["Kamylle Frenette"],
    "PTVI Women": ["Leticia Freitas", "Alison Peasgood", "Chloe MacCombe", "Judith MacCombe"],
}


@dataclass(frozen=True)
class BenchmarkEvent:
    event_id: int
    prog_id: int
    event_name: str
    event_date: str
    prog_name: str
    prog_distance_category: Optional[str]


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)


def _build_cat_filter_sql(pattern_param_names: Iterable[str]) -> str:
    # (e.cat_name ILIKE :cat0 OR e.cat_name ILIKE :cat1 ...)
    parts = [f"COALESCE(e.cat_name, '') ILIKE :{p}" for p in pattern_param_names]
    return "(" + " OR ".join(parts) + ")"


def resolve_usa_athlete_by_name(engine: Engine, name: str) -> tuple[int, str]:
    # Pull a small candidate set; disambiguate interactively if needed.
    like_pat = f"%{name.strip()}%"

    query = text(
        """
        SELECT athlete_id, full_name, country
        FROM athlete
        WHERE full_name ILIKE :pat
        ORDER BY full_name
        LIMIT 25
        """
    )

    with engine.begin() as conn:
        candidates = pd.read_sql(query, conn, params={"pat": like_pat})

    if candidates.empty:
        raise SystemExit(
            f"No athlete found matching name pattern '{name}'. Try a different spelling."  # noqa: EM102
        )

    # Prefer USA matches
    usa = candidates[candidates["country"].isin(USA_COUNTRY_ALIASES)].copy()
    pool = usa if not usa.empty else candidates

    # If only one candidate, take it.
    if len(pool) == 1:
        row = pool.iloc[0]
        if row.get("country") not in USA_COUNTRY_ALIASES:
            raise SystemExit(
                f"Found '{row['full_name']}' but country='{row.get('country')}'. "
                "Please select a USA athlete name."  # noqa: EM102
            )
        return int(row["athlete_id"]), str(row["full_name"])

    print("Multiple athlete matches found:")
    for i, (_, r) in enumerate(pool.iterrows(), start=1):
        print(f"  {i}. {r['full_name']} (athlete_id={r['athlete_id']}, country={r.get('country')})")

    while True:
        try:
            sel = input("Select athlete number: ").strip()
            idx = int(sel)
            if idx < 1 or idx > len(pool):
                raise ValueError
        except Exception:
            print("Invalid selection. Enter a number from the list.")
            continue

        chosen = pool.iloc[idx - 1]
        if chosen.get("country") not in USA_COUNTRY_ALIASES:
            raise SystemExit(
                f"Selected athlete '{chosen['full_name']}' is not USA (country={chosen.get('country')})."  # noqa: EM102
            )
        return int(chosen["athlete_id"]), str(chosen["full_name"])


def list_usa_athletes_for_category(
    engine: Engine,
    category: str,
    since: str,
    major_cat_patterns: list[str],
    limit: int = 50,
) -> pd.DataFrame:
    """List USA athletes who have raced in the selected para category (major events) since a date."""

    cat_params = {f"cat{i}": pat for i, pat in enumerate(major_cat_patterns)}
    cat_clause = _build_cat_filter_sql(cat_params.keys())

    country_params = {f"c{i}": c for i, c in enumerate(USA_COUNTRY_ALIASES)}
    country_clause = "(" + " OR ".join([f"a.country = :{k}" for k in country_params.keys()]) + ")"

    query = text(
        f"""
        SELECT
            rr.athlete_id,
            a.full_name,
            a.country,
            COUNT(DISTINCT rr.event_id) AS events,
            COUNT(*) AS starts
        FROM race_results rr
        JOIN athlete a ON a.athlete_id = rr.athlete_id
        JOIN events e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        WHERE e.prog_name = :category
          AND e.event_date >= :since
          AND {cat_clause}
          AND {country_clause}
        GROUP BY rr.athlete_id, a.full_name, a.country
        ORDER BY events DESC, starts DESC, a.full_name
        LIMIT :limit
        """
    )

    params = {
        "category": category,
        "since": since,
        "limit": limit,
        **cat_params,
        **country_params,
    }

    with engine.begin() as conn:
        return pd.read_sql(query, conn, params=params)


def select_usa_athlete_from_category(
    engine: Engine,
    category: str,
    since: str,
    major_cat_patterns: list[str],
    typed_name: Optional[str] = None,
    auto_select_first: bool = False,
) -> tuple[int, str]:
    """Resolve a USA athlete either by name, or by prompting from category participants.

    - If typed_name matches athlete table → use existing resolver.
    - Otherwise, list USA athletes that appear in the selected category (major events since date)
      and prompt for selection.
    """

    if typed_name:
        try:
            return resolve_usa_athlete_by_name(engine, typed_name)
        except SystemExit as e:
            print(str(e))
            print("Falling back to listing USA athletes found in this category...")

    options = list_usa_athletes_for_category(
        engine,
        category=category,
        since=since,
        major_cat_patterns=major_cat_patterns,
        limit=50,
    )
    if options.empty:
        raise SystemExit(
            "No USA athletes found for this category after applying major-event filters and date window. "
            "Either broaden the date range or verify the DB import includes these events."  # noqa: EM102
        )

    print(f"USA athletes found for '{category}' since {since} (major events):")
    for i, (_, r) in enumerate(options.iterrows(), start=1):
        print(f"  {i}. {r['full_name']} (athlete_id={r['athlete_id']}, events={r['events']}, starts={r['starts']})")

    if auto_select_first:
        chosen = options.iloc[0]
        print(f"Auto-selected: {chosen['full_name']} (athlete_id={int(chosen['athlete_id'])})")
        return int(chosen["athlete_id"]), str(chosen["full_name"])

    while True:
        try:
            sel = input("Select athlete number: ").strip()
            idx = int(sel)
            if idx < 1 or idx > len(options):
                raise ValueError
        except Exception:
            print("Invalid selection. Enter a number from the list.")
            continue
        chosen = options.iloc[idx - 1]
        return int(chosen["athlete_id"]), str(chosen["full_name"])


def find_paris_benchmark_event(
    engine: Engine,
    category: str,
    benchmark_year: int,
    benchmark_city: str,
    major_cat_patterns: list[str],
) -> BenchmarkEvent:
    # Special handling for PTS3 Women: use Pontevedra 2023 instead of Paris 2024
    # (PTS3 Women was not a medal event at Paris 2024)
    if category == "PTS3 Women":
        benchmark_year = 2023
        benchmark_city = "Pontevedra"
        print(f"Note: Using Pontevedra 2023 as benchmark for {category} (PTS3 Women not contested at Paris 2024)")
    
    year_start = f"{benchmark_year}-01-01"
    year_end = f"{benchmark_year + 1}-01-01"

    # Build cat filters
    cat_params = {f"cat{i}": pat for i, pat in enumerate(major_cat_patterns)}
    cat_clause = _build_cat_filter_sql(cat_params.keys())

    query = text(
        f"""
        SELECT
            e.event_id,
            e.prog_id,
            e.event_name,
            e.event_date,
            e.prog_name,
            e.prog_distance_category
        FROM events e
        WHERE e.prog_name = :category
          AND e.event_date >= :year_start
          AND e.event_date < :year_end
          AND {cat_clause}
          AND COALESCE(e.event_name, '') ILIKE :city
        ORDER BY e.event_date DESC
        """
    )

    params = {
        "category": category,
        "year_start": year_start,
        "year_end": year_end,
        "city": f"%{benchmark_city}%",
        **cat_params,
    }

    with engine.begin() as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        raise SystemExit(
            f"Could not find a {benchmark_city} {benchmark_year} benchmark race in the DB for the selected category. "
            f"Ensure {benchmark_city} {benchmark_year} para events are imported for this program name."  # noqa: EM102
        )

    if len(df) > 1:
        print(f"Warning: multiple {benchmark_city} {benchmark_year} benchmark events found; using the latest by date:")
        for _, r in df.head(5).iterrows():
            print(f"  - {r['event_date']}: {r['event_name']} (event_id={r['event_id']}, prog_id={r['prog_id']})")

    row = df.iloc[0]
    return BenchmarkEvent(
        event_id=int(row["event_id"]),
        prog_id=int(row["prog_id"]),
        event_name=str(row.get("event_name") or ""),
        event_date=str(row.get("event_date") or ""),
        prog_name=str(row.get("prog_name") or category),
        prog_distance_category=(str(row.get("prog_distance_category")) if pd.notna(row.get("prog_distance_category")) else None),
    )


def fetch_paris_medalists(engine: Engine, bench: BenchmarkEvent) -> pd.DataFrame:
    query = text(
        """
        SELECT
            rr.athlete_id,
            a.full_name,
            rr.finish_position,
            rr.finish_status
        FROM race_results rr
        JOIN athlete a ON a.athlete_id = rr.athlete_id
        WHERE rr.event_id = :event_id
          AND rr.prog_id = :prog_id
          AND rr.finish_status = 'FINISH'
          AND rr.finish_position IN (1, 2, 3)
        ORDER BY rr.finish_position ASC
        """
    )

    with engine.begin() as conn:
        df = pd.read_sql(query, conn, params={"event_id": bench.event_id, "prog_id": bench.prog_id})

    if len(df) != 3:
        raise SystemExit(
            f"Expected 3 medalists for benchmark race, found {len(df)}. "
            "Verify finish_position/finish_status are populated for that event."  # noqa: EM102
        )

    return df


def fetch_additional_benchmark_athletes(engine: Engine, category: str) -> pd.DataFrame:
    """Fetch additional top contenders for the category to include in benchmarks.
    
    Returns a dataframe with athlete_id, full_name, and a synthetic finish_position (starting at 4).
    """
    athlete_names = ADDITIONAL_BENCHMARK_ATHLETES.get(category, [])
    if not athlete_names:
        return pd.DataFrame(columns=["athlete_id", "full_name", "finish_position"])
    
    results = []
    for name in athlete_names:
        query = text(
            """
            SELECT athlete_id, full_name
            FROM athlete
            WHERE full_name ILIKE :pattern
            LIMIT 1
            """
        )
        with engine.begin() as conn:
            df = pd.read_sql(query, conn, params={"pattern": f"%{name}%"})
        
        if not df.empty:
            results.append(df.iloc[0])
        else:
            print(f"Warning: Could not find athlete '{name}' in database for category '{category}'")
    
    if not results:
        return pd.DataFrame(columns=["athlete_id", "full_name", "finish_position"])
    
    additional = pd.DataFrame(results)
    # Assign synthetic positions starting at 4 (after the 3 medalists)
    additional["finish_position"] = range(4, 4 + len(additional))
    return additional


def fetch_major_history(
    engine: Engine,
    category: str,
    since: str,
    athlete_ids: list[int],
    major_cat_patterns: list[str],
) -> pd.DataFrame:
    cat_params = {f"cat{i}": pat for i, pat in enumerate(major_cat_patterns)}
    cat_clause = _build_cat_filter_sql(cat_params.keys())

    query = (
        text(
            f"""
            SELECT
                rr.event_id,
                rr.prog_id,
                rr.athlete_id,
                a.full_name,
                a.country,
                rr.finish_status,
                rr.finish_position,
                rr.swimtime,
                rr.t1time,
                rr.biketime,
                rr.t2time,
                rr.runtime,
                rr.total_time,
                e.prog_name,
                e.prog_distance_category,
                e.event_name,
                e.event_date,
                e.cat_name,
                e.swim_distance,
                e.bike_distance,
                e.run_distance
            FROM race_results rr
            JOIN athlete a ON a.athlete_id = rr.athlete_id
            JOIN events e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
            WHERE e.prog_name = :category
              AND e.event_date >= :since
              AND rr.athlete_id IN :athlete_ids
              AND {cat_clause}
            ORDER BY e.event_date ASC
            """
        )
        .bindparams(bindparam("athlete_ids", expanding=True))
    )

    params = {"category": category, "since": since, "athlete_ids": athlete_ids, **cat_params}

    with engine.begin() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["event_date"])  # type: ignore

    return df


def _panelize_distance_category(val: object) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "Unknown"
    s = str(val).strip().lower()
    if "sprint" in s:
        return "Sprint"
    if "standard" in s or "olympic" in s:
        return "Standard"
    return "Other"


def _infer_distance_panel_from_distances(
    swim_m: object,
    bike_m: object,
    run_m: object,
) -> str:
    """Infer Sprint/Standard when prog_distance_category is missing.

    Uses broad thresholds since para courses vary (e.g., bike 20–23km).
    """

    def _to_float(x: object) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, float) and x != x:  # NaN
            return None
        try:
            return float(x)
        except Exception:
            return None

    swim = _to_float(swim_m)
    bike = _to_float(bike_m)
    run = _to_float(run_m)

    if swim is None or bike is None or run is None:
        return "Unknown"
    if swim <= 0 or bike <= 0 or run <= 0:
        return "Unknown"

    # Sprint-ish (very common in para): ~750m swim, ~20–23km bike, ~5km run
    if swim <= 1000 and bike <= 26000 and run <= 6500:
        return "Sprint"

    # Standard-ish: ~1500m swim, ~40km bike, ~10km run
    if 1000 < swim <= 2000 and 26000 < bike <= 50000 and 6500 < run <= 13000:
        return "Standard"

    return "Other"


# ------------------------------------------------------------------
# Outlier Detection
# ------------------------------------------------------------------
# Para triathlon has a wide range of times due to different impairments.
# However, extreme outliers (e.g., data errors, equipment issues) can
# distort charts and analysis. These thresholds are intentionally high
# to avoid removing legitimate variation while flagging likely errors.

# Relative thresholds: flag if value > global_median * threshold
OUTLIER_THRESHOLDS = {
    # Swim/Bike/Run: flag if value > median * 3.0 (200% above median)
    "swimtime_sec": 3.0,
    "biketime_sec": 3.0,
    "runtime_sec": 3.0,
    "swim_pace_s_per_100m": 3.0,
    "bike_pace_s_per_km": 3.0,
    "run_pace_s_per_km": 3.0,
    "bike_speed_kmh": 0.33,  # For speed, outlier is < median * 0.33 (inverted)
    # T1/T2: flag if value > median * 5.0 (400% above median)
    "t1time_sec": 5.0,
    "t2time_sec": 5.0,
}

# Absolute bounds: hard limits for clearly impossible values (safety net)
# These catch data errors even when median-based detection fails
ABSOLUTE_BOUNDS = {
    # Paces: upper bounds (seconds) - anything slower is clearly wrong
    "swim_pace_s_per_100m": (None, 300),      # Max 5:00/100m (very slow but possible)
    "run_pace_s_per_km": (None, 900),          # Max 15:00/km (very slow but possible for some para classes)
    "bike_pace_s_per_km": (None, 300),         # Max 5:00/km = 12 km/h (very slow)
    # Speeds: lower bounds (km/h)
    "bike_speed_kmh": (8, None),               # Min 8 km/h (anything slower is likely error)
    # Split times: upper bounds (seconds)
    "swimtime_sec": (None, 3600),              # Max 1 hour swim
    "biketime_sec": (None, 7200),              # Max 2 hour bike
    "runtime_sec": (None, 5400),               # Max 1.5 hour run
    "t1time_sec": (None, 600),                 # Max 10 min T1
    "t2time_sec": (None, 600),                 # Max 10 min T2
}


def flag_outliers(
    df: pd.DataFrame,
    thresholds: dict[str, float] = OUTLIER_THRESHOLDS,
    absolute_bounds: dict[str, tuple] = ABSOLUTE_BOUNDS,
    nullify: bool = True,
) -> pd.DataFrame:
    """
    Flag and optionally nullify outlier values using two methods:
    1. Relative: values outside median * threshold (computed globally)
    2. Absolute: values outside hard bounds (catches clear data errors)
    
    For 'lower is better' metrics (times/paces), an outlier is a value > median * threshold.
    For 'higher is better' metrics (speed), an outlier is a value < median * threshold.
    
    Creates a new column '{col}_outlier' (bool) for each flagged column.
    If nullify=True, also sets the original column to NaN for outliers.
    """
    if df.empty:
        return df
    
    out = df.copy()
    outlier_counts = {}
    
    # Combine all columns to check
    all_cols = set(thresholds.keys()) | set(absolute_bounds.keys())
    
    for col in all_cols:
        if col not in out.columns:
            continue
        
        out[f"{col}_outlier"] = False
        values = pd.to_numeric(out[col], errors="coerce")
        
        if values.isna().all():
            continue
        
        # For bike speed, lower values are outliers (inverted logic)
        is_speed_metric = col in ["bike_speed_kmh", "bike_speed_mph"]
        
        # Step 1: Check absolute bounds first (catches clear errors)
        if col in absolute_bounds:
            lower, upper = absolute_bounds[col]
            if lower is not None:
                out.loc[values < lower, f"{col}_outlier"] = True
            if upper is not None:
                out.loc[values > upper, f"{col}_outlier"] = True
        
        # Step 2: Check relative threshold using GLOBAL median (more robust than per-event)
        if col in thresholds:
            threshold = thresholds[col]
            # Use global median across all non-outlier values
            non_outlier_values = values[~out[f"{col}_outlier"]]
            median = non_outlier_values.median()
            
            if pd.notna(median) and median > 0:
                if is_speed_metric:
                    # For speed: outlier if value < median * threshold (very slow)
                    relative_outliers = values < (median * threshold)
                else:
                    # For time/pace: outlier if value > median * threshold (very slow)
                    relative_outliers = values > (median * threshold)
                
                out.loc[relative_outliers, f"{col}_outlier"] = True
        
        # Count and optionally nullify
        n_outliers = out[f"{col}_outlier"].sum()
        if n_outliers > 0:
            outlier_counts[col] = n_outliers
            if nullify:
                out.loc[out[f"{col}_outlier"], col] = pd.NA
    
    if outlier_counts:
        print(f"  Outliers flagged (nullified): {outlier_counts}")
    
    return out


def compute_metrics(df: pd.DataFrame, normalize_time_factor: bool = True) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # Some unit tests / ad-hoc uses pass a minimal dataframe.
    # Default missing status fields to FINISH so metric computation works.
    if "finish_status" not in out.columns:
        out["finish_status"] = "FINISH"
    if "event_name" not in out.columns:
        out["event_name"] = ""
    if "event_date" not in out.columns:
        out["event_date"] = pd.NaT
    if "full_name" not in out.columns:
        out["full_name"] = ""
    if "prog_name" not in out.columns:
        out["prog_name"] = ""

    for col in ["swimtime", "t1time", "biketime", "t2time", "runtime", "total_time"]:
        out[col + "_sec"] = out[col].apply(time_to_seconds)

    # Treat 0-second split placeholders as missing (prevents misleading drops to zero).
    # This also naturally removes most non-finish placeholder values (often 00:00:00).
    for col in ["swimtime_sec", "t1time_sec", "biketime_sec", "t2time_sec", "runtime_sec", "total_time_sec"]:
        out.loc[out[col] == 0, col] = pd.NA

    # Never plot split/pace lines for non-finishers; keep them only for markers.
    nonfinish = out["finish_status"].fillna("").str.upper().ne("FINISH")
    for col in ["swimtime_sec", "t1time_sec", "biketime_sec", "t2time_sec", "runtime_sec", "total_time_sec"]:
        out.loc[nonfinish, col] = pd.NA

    # --- Para time-factor normalization (PTVI/PTWC) ---
    # Some para categories apply a fixed start offset ("time factor") to certain subclasses.
    # In the results feed this often appears as extra time embedded in the swim split (and thus total_time).
    # For standards/pace analysis we normalize by removing that factor so swim/total reflect comparable effort.
    # We preserve the original values for transparency.

    def _parse_gender_from_prog_name(prog_name: object) -> Optional[str]:
        s = str(prog_name or "").lower()
        if "women" in s:
            return "Women"
        if "men" in s:
            return "Men"
        return None

    def _parse_division_from_prog_name(prog_name: object) -> Optional[str]:
        s = str(prog_name or "").upper()
        if "PTVI" in s:
            return "PTVI"
        if "PTWC" in s:
            return "PTWC"
        return None

    def _extract_para_subclass(full_name: object) -> Optional[str]:
        s = str(full_name or "").strip()
        if not s:
            return None
        m = re.search(r"\b(B[123]|H[12])\b\s*$", s)
        if not m:
            return None
        return m.group(1)

    # Known seasonal factors (seconds).
    # Only apply factors for matching years. For future years beyond our table, carry forward the latest year.
    _PARA_TIME_FACTOR_SECONDS: dict[int, dict[tuple[str, str], int]] = {
        2024: {
            ("PTWC", "Women"): 3 * 60 + 38,
            ("PTWC", "Men"): 3 * 60 + 0,
            ("PTVI", "Women"): 3 * 60 + 11,
            ("PTVI", "Men"): 2 * 60 + 41,
        },
        2025: {
            ("PTWC", "Women"): 3 * 60 + 38,
            ("PTWC", "Men"): 3 * 60 + 8,
            ("PTVI", "Women"): 3 * 60 + 11,
            ("PTVI", "Men"): 2 * 60 + 41,
        },
    }

    def _factor_year(event_year: Optional[int]) -> Optional[int]:
        if not event_year:
            return None
        years = sorted(_PARA_TIME_FACTOR_SECONDS.keys())
        if not years:
            return None
        if event_year in _PARA_TIME_FACTOR_SECONDS:
            return event_year
        # If we have future seasons not yet encoded, assume latest known year
        if event_year > years[-1]:
            return years[-1]
        # For years earlier than our factor table, do not guess.
        return None

    def _infer_time_factor_seconds(row: pd.Series) -> tuple[Optional[int], Optional[int]]:
        division = _parse_division_from_prog_name(row.get("prog_name"))
        gender = _parse_gender_from_prog_name(row.get("prog_name"))
        if not division or not gender:
            return None, None

        dt = pd.to_datetime(row.get("event_date"), errors="coerce")
        year = int(dt.year) if pd.notna(dt) else None
        factor_year = _factor_year(year)
        if factor_year is None:
            return None, None

        sec = _PARA_TIME_FACTOR_SECONDS.get(factor_year, {}).get((division, gender))
        if not sec:
            return None, factor_year
        return int(sec), factor_year

    out["para_subclass"] = out["full_name"].apply(_extract_para_subclass)
    factor_info = out.apply(_infer_time_factor_seconds, axis=1)
    out["time_factor_sec"] = factor_info.apply(lambda t: t[0])
    out["time_factor_year"] = factor_info.apply(lambda t: t[1])

    division = out["prog_name"].apply(_parse_division_from_prog_name)
    disadvantaged = (
        ((division == "PTVI") & out["para_subclass"].isin(["B2", "B3"]))
        | ((division == "PTWC") & (out["para_subclass"] == "H2"))
    )

    factor_seconds = pd.to_numeric(out["time_factor_sec"], errors="coerce")
    out["time_factor_applied"] = disadvantaged & factor_seconds.fillna(0).gt(0)

    out["swimtime_sec_adjusted"] = out["swimtime_sec"]
    out["total_time_sec_adjusted"] = out["total_time_sec"]

    # Important: In our DB/feeds, the time factor is reliably reflected in TOTAL time, but not always embedded in the swim split.
    # So we do NOT try to adjust swim split times by the factor (it can double-count and exaggerate differences).
    # Instead, treat the factor as a separate offset that bridges effort total -> adjusted-for-placing total.

    split_cols = ["swimtime_sec", "t1time_sec", "biketime_sec", "t2time_sec", "runtime_sec"]
    splits_num = out[split_cols].apply(pd.to_numeric, errors="coerce")
    out["sum_splits_sec"] = splits_num.sum(axis=1, min_count=len(split_cols))

    # Effort total: prefer sum of splits; if unavailable, fall back to subtracting expected factor when applicable.
    out["total_time_sec_raw"] = out["sum_splits_sec"]
    fallback_effort = out["total_time_sec_raw"].isna() & out["time_factor_applied"]
    out.loc[fallback_effort, "total_time_sec_raw"] = (
        pd.to_numeric(out.loc[fallback_effort, "total_time_sec_adjusted"], errors="coerce")
        - factor_seconds.loc[fallback_effort]
    )
    out.loc[pd.to_numeric(out["total_time_sec_raw"], errors="coerce") < 0, "total_time_sec_raw"] = 0

    # Keep swim "raw" equal to as-reported swim split (typically already effort-based).
    out["swimtime_sec_raw"] = out["swimtime_sec_adjusted"]

    # Observed factor segment (if we can compute effort total). This should usually equal the expected factor.
    out["time_factor_segment_sec"] = pd.to_numeric(out["total_time_sec_adjusted"], errors="coerce") - pd.to_numeric(
        out["total_time_sec_raw"], errors="coerce"
    )
    out.loc[out["time_factor_segment_sec"].isna(), "time_factor_segment_sec"] = pd.NA

    # Primary: use prog_distance_category text when present.
    # Fallback: infer from distances when missing/blank.
    primary = out["prog_distance_category"].apply(_panelize_distance_category)
    inferred = out.apply(
        lambda r: _infer_distance_panel_from_distances(
            r.get("swim_distance"), r.get("bike_distance"), r.get("run_distance")
        ),
        axis=1,
    )
    out["distance_panel"] = primary
    needs_fallback = primary.isin(["Unknown", "Other"])
    out.loc[needs_fallback, "distance_panel"] = inferred.loc[needs_fallback]

    out["swim_pace_s_per_100m"] = out.apply(
        lambda r: pace_sec_per_100m(r["swimtime_sec"], r.get("swim_distance")), axis=1
    )
    out["bike_pace_s_per_km"] = out.apply(
        lambda r: pace_sec_per_km(r["biketime_sec"], r.get("bike_distance")), axis=1
    )
    out["run_pace_s_per_km"] = out.apply(
        lambda r: pace_sec_per_km(r["runtime_sec"], r.get("run_distance")), axis=1
    )

    # Bike speed (often more intuitive for coaches/athletes than sec/km pace)
    bike_km = pd.to_numeric(out.get("bike_distance"), errors="coerce") / 1000.0
    bike_hours = pd.to_numeric(out.get("biketime_sec"), errors="coerce") / 3600.0
    out["bike_speed_kmh"] = bike_km / bike_hours
    out.loc[(bike_km <= 0) | (bike_hours <= 0), "bike_speed_kmh"] = pd.NA
    out["bike_speed_mph"] = pd.to_numeric(out["bike_speed_kmh"], errors="coerce") * 0.621371

    # Format overall time for hover
    out["total_time_fmt"] = out["total_time_sec"].apply(seconds_to_hms)

    # X-axis label: show event names, but keep chronological ordering via event_date.
    out["event_label"] = out["event_name"].fillna("").astype(str)

    # Flag and nullify extreme outliers (likely data errors)
    out = flag_outliers(out, nullify=True)

    return out


def _ordered_event_labels(df: pd.DataFrame) -> list[str]:
    """Return event labels ordered chronologically (stable across ties)."""
    if df.empty:
        return []
    cols = ["event_label", "event_date"]
    if "event_id" in df.columns:
        cols.append("event_id")
    if "prog_id" in df.columns:
        cols.append("prog_id")

    work = df[cols].drop_duplicates().copy()
    work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce")
    sort_cols = ["event_date"]
    if "event_id" in work.columns:
        sort_cols.append("event_id")
    if "prog_id" in work.columns:
        sort_cols.append("prog_id")
    sort_cols.append("event_label")
    work = work.sort_values(sort_cols, ascending=True)
    return work["event_label"].tolist()


def _summary_table_html(title: str, summary: pd.DataFrame) -> str:
    if summary.empty:
        return f"<p><b>{title}:</b> (no comparable FINISH rows)</p>"
    
    # Calculate total races per opponent and create beat_rate column
    safe = summary.copy()
    safe["total_races"] = safe["usa_faster"] + safe["usa_slower"]
    safe["beat_rate"] = safe.apply(
        lambda r: f"{int(r['usa_faster'])}/{int(r['total_races'])}" if r['total_races'] > 0 else "0/0",
        axis=1,
    )
    
    cols = ["opponent", "beat_rate", "avg_diff", "min_diff", "max_diff"]
    safe = safe[cols].rename(
        columns={
            "opponent": "Opponent",
            "beat_rate": "# of Races Target Athlete beat Opponent",
            "avg_diff": "Average Time Difference between Target Athlete and Opponent",
            "min_diff": "Best Time Difference between Target Athlete and Opponent",
            "max_diff": "Worst Time Difference between Target Athlete and Opponent",
        }
    )
    
    # Generate custom HTML table with color-coded beat_rate column
    table_html = "<table><thead><tr>"
    for col in safe.columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr></thead><tbody>"
    
    for _, row in safe.iterrows():
        table_html += "<tr>"
        for col in safe.columns:
            cell_value = row[col]
            if col == "# of Races Target Athlete beat Opponent":
                # Apply color-coding to beat rate column
                cell_value = _color_code_beat_rate(str(cell_value))
                table_html += f"<td>{cell_value}</td>"
            else:
                # Escape other cells for safety
                import html
                table_html += f"<td>{html.escape(str(cell_value))}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"
    
    return f"<div class='summary-block'><p><b>{title}</b></p>{table_html}</div>"


def _factor_audit_html(df: pd.DataFrame, usa_athlete_id: int, medalists: pd.DataFrame) -> str:
    """Render a compact audit of para time-factor handling.

    Uses:
    - total_time_sec_adjusted: placing / official total
    - total_time_sec_raw: effort total (sum of splits or adjusted minus expected factor)
    - time_factor_sec: expected seasonal factor
    - time_factor_segment_sec: observed (adjusted - raw)
    """

    required = {"time_factor_applied", "time_factor_sec", "time_factor_segment_sec", "total_time_sec_raw", "total_time_sec_adjusted"}
    if not required.issubset(set(df.columns)):
        return ""

    try:
        medalist_ids = set(pd.to_numeric(medalists.get("athlete_id"), errors="coerce").dropna().astype(int).tolist())
    except Exception:
        medalist_ids = set()

    focus_ids = medalist_ids | {int(usa_athlete_id)}
    work = df[df.get("athlete_id").isin(list(focus_ids))].copy() if "athlete_id" in df.columns else df.copy()

    work["finish_status"] = work.get("finish_status", "").fillna("").astype(str).str.upper()
    work = work[work["finish_status"].eq("FINISH")]
    if work.empty:
        return ""

    work["time_factor_applied"] = work["time_factor_applied"].fillna(False).astype(bool)
    if not bool(work["time_factor_applied"].any()):
        return ""

    work["expected_factor_sec"] = pd.to_numeric(work["time_factor_sec"], errors="coerce")
    work["observed_segment_sec"] = pd.to_numeric(work["time_factor_segment_sec"], errors="coerce")
    work["factor_diff_sec"] = work["observed_segment_sec"] - work["expected_factor_sec"]
    work["abs_factor_diff_sec"] = work["factor_diff_sec"].abs()

    group_cols = [c for c in ["full_name", "para_subclass"] if c in work.columns]
    if not group_cols:
        group_cols = ["athlete_id"] if "athlete_id" in work.columns else []

    applied = work[work["time_factor_applied"]].copy()
    if applied.empty:
        return ""

    summary = (
        applied.groupby(group_cols, dropna=False)
        .agg(
            events_with_factor=("time_factor_applied", "sum"),
            expected_factor_sec=("expected_factor_sec", "median"),
            observed_segment_sec=("observed_segment_sec", "median"),
            median_abs_error_sec=("abs_factor_diff_sec", "median"),
            max_abs_error_sec=("abs_factor_diff_sec", "max"),
        )
        .reset_index()
    )

    for col in ["expected_factor_sec", "observed_segment_sec", "median_abs_error_sec", "max_abs_error_sec"]:
        summary[col] = summary[col].apply(_format_seconds_mmss)

    # USA-only per-event detail (kept compact)
    usa_detail = applied[applied.get("athlete_id") == int(usa_athlete_id)].copy() if "athlete_id" in applied.columns else pd.DataFrame()
    detail_html = ""
    if not usa_detail.empty:
        usa_detail["event_date"] = pd.to_datetime(usa_detail.get("event_date"), errors="coerce")
        usa_detail = usa_detail.sort_values(["event_date", "event_name"], ascending=[True, True])
        cols = [c for c in ["event_date", "event_name", "prog_name"] if c in usa_detail.columns]
        cols += ["total_time_sec_adjusted", "total_time_sec_raw", "expected_factor_sec", "observed_segment_sec", "factor_diff_sec"]
        d = usa_detail[cols].copy()
        for c in ["total_time_sec_adjusted", "total_time_sec_raw", "expected_factor_sec", "observed_segment_sec", "factor_diff_sec"]:
            d[c] = d[c].apply(_format_seconds_mmss)
        if "event_date" in d.columns:
            d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce").dt.date.astype(str)
        d = d.rename(
            columns={
                "event_date": "Date",
                "event_name": "Event",
                "prog_name": "Program",
                "total_time_sec_adjusted": "Placing total",
                "total_time_sec_raw": "Effort total",
                "expected_factor_sec": "Expected factor",
                "observed_segment_sec": "Observed segment",
                "factor_diff_sec": "Observed - expected",
            }
        )
        detail_html = (
            "<h4>USA athlete event-level audit (rows with factor applied)</h4>"
            + d.to_html(index=False, escape=True)
        )

    return (
        "<div class='summary-block'>"
        "<h2>Time-Factor Audit</h2>"
        "<p>Benchmarks use <b>placing totals</b>. This section audits whether the observed factor segment (placing total − effort total) matches the expected seasonal factor.</p>"
        + summary.to_html(index=False, escape=True)
        + detail_html
        + "</div><hr/>"
    )


def _format_seconds_mmss(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        seconds = float(value)
    except Exception:
        return str(value)

    sign = "-" if seconds < 0 else ""
    seconds = abs(int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{sign}{h}:{m:02d}:{s:02d}"
    return f"{sign}{m:02d}:{s:02d}"


def _format_summary_time_columns(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    for col in ["avg_diff", "min_diff", "max_diff"]:
        if col in out.columns:
            out[col] = out[col].apply(_format_seconds_mmss)
    return out


def _format_summary_numeric_columns(summary: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    for col in ["avg_diff", "min_diff", "max_diff"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(decimals)
    return out


def _default_medalist_weights(medalists: pd.DataFrame) -> dict[int, float]:
    """Weights by finish position: (1,2,3) -> (0.5,0.3,0.2), positions 4+ -> 0.15.

    Positions 4+ represent additional top contenders who didn't medal at Paris
    but are considered competitive benchmarks.

    Returned dict is athlete_id -> weight.
    """
    pos_weights = {1: 0.5, 2: 0.3, 3: 0.2}
    weights: dict[int, float] = {}
    if medalists is None or medalists.empty:
        return weights
    for _, row in medalists.iterrows():
        try:
            athlete_id = int(row["athlete_id"])
            pos = int(row["finish_position"])
        except Exception:
            continue
        if pos in pos_weights:
            weights[athlete_id] = float(pos_weights[pos])
        elif pos >= 4:
            # Additional benchmark contenders (not benchmark medalists but top-tier)
            weights[athlete_id] = 0.15
    return weights


def _winner_only_weights(medalists: pd.DataFrame) -> dict[int, float]:
    """Return athlete_id -> 1.0 for the benchmark winner (finish_position==1)."""
    if medalists is None or medalists.empty:
        return {}
    try:
        winner = medalists.loc[medalists["finish_position"] == 1].iloc[0]
        return {int(winner["athlete_id"]): 1.0}
    except Exception:
        return {}


def _weighted_average(values: pd.Series, weights: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna()
    if not mask.any():
        return None
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return None
    return float((v * w).sum() / w_sum)


def compute_weighted_benchmark_by_event(
    df: pd.DataFrame,
    athlete_weights: dict[int, float],
    value_col: str,
) -> pd.DataFrame:
    """Per (event_id, prog_id) weighted benchmark from medalist athletes.

    Uses only FINISH rows and available athletes in athlete_weights.
    Weights are re-normalized automatically based on which athletes have values for that event.
    """
    if df.empty or not athlete_weights:
        return pd.DataFrame(columns=["event_id", "prog_id", "bench_value"])

    work = df[["event_id", "prog_id", "athlete_id", "finish_status", value_col]].copy()
    work["finish_status"] = work["finish_status"].fillna("").astype(str).str.upper()
    work = work[work["finish_status"].eq("FINISH")]
    work = work[work["athlete_id"].isin(list(athlete_weights.keys()))]
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(columns=["event_id", "prog_id", "bench_value"])

    work["w"] = work["athlete_id"].map(athlete_weights).astype(float)

    work["vw"] = work[value_col].astype(float) * work["w"].astype(float)
    grouped = work.groupby(["event_id", "prog_id"], dropna=False).agg(
        vw_sum=("vw", "sum"),
        w_sum=("w", "sum"),
    ).reset_index()
    grouped = grouped[grouped["w_sum"] > 0]
    grouped["bench_value"] = grouped["vw_sum"] / grouped["w_sum"]
    return grouped[["event_id", "prog_id", "bench_value"]]


def compute_usa_vs_benchmark_summary(
    df: pd.DataFrame,
    usa_athlete_id: int,
    athlete_weights: dict[int, float],
    value_col: str,
    fmt: str,
    better_is_lower: bool = True,
) -> dict[str, object]:
    """Summarize USA vs weighted benchmark for a metric.

    delta is computed as usa_value - bench_value.
    For lower-is-better metrics (time/pace): delta <= 0 means USA meets/beats benchmark.
    For higher-is-better metrics (speed): delta >= 0 means USA meets/beats benchmark.
    fmt: 'time' formats deltas/values as mm:ss, 'float' leaves numeric.
    """
    if df.empty:
        return {
            "events": 0,
            "beats": 0,
            "usa_avg": "",
            "bench_avg": "",
            "avg_delta": "",
            "min_delta": "",
            "max_delta": "",
        }

    bench = compute_weighted_benchmark_by_event(df, athlete_weights, value_col=value_col)
    if bench.empty:
        return {
            "events": 0,
            "beats": 0,
            "usa_avg": "",
            "bench_avg": "",
            "avg_delta": "",
            "min_delta": "",
            "max_delta": "",
        }

    usa = df[["event_id", "prog_id", "athlete_id", "finish_status", value_col]].copy()
    usa["finish_status"] = usa["finish_status"].fillna("").astype(str).str.upper()
    usa = usa[(usa["finish_status"].eq("FINISH")) & (usa["athlete_id"] == usa_athlete_id)]
    usa[value_col] = pd.to_numeric(usa[value_col], errors="coerce")
    usa = usa.dropna(subset=[value_col])
    if usa.empty:
        return {
            "events": 0,
            "beats": 0,
            "usa_avg": "",
            "bench_avg": "",
            "avg_delta": "",
            "min_delta": "",
            "max_delta": "",
        }

    merged = usa.merge(bench, on=["event_id", "prog_id"], how="inner")
    if merged.empty:
        return {
            "events": 0,
            "beats": 0,
            "usa_avg": "",
            "bench_avg": "",
            "avg_delta": "",
            "min_delta": "",
            "max_delta": "",
        }

    merged = merged.assign(delta=merged[value_col] - merged["bench_value"])
    events = int(len(merged))
    if better_is_lower:
        beats = int((merged["delta"] <= 0).sum())
    else:
        beats = int((merged["delta"] >= 0).sum())
    usa_avg = float(pd.to_numeric(merged[value_col], errors="coerce").mean())
    bench_avg = float(pd.to_numeric(merged["bench_value"], errors="coerce").mean())
    avg_delta = float(pd.to_numeric(merged["delta"], errors="coerce").mean())
    min_delta = float(pd.to_numeric(merged["delta"], errors="coerce").min())
    max_delta = float(pd.to_numeric(merged["delta"], errors="coerce").max())

    if fmt == "time":
        return {
            "events": events,
            "beats": beats,
            "usa_avg": _format_seconds_mmss(usa_avg),
            "bench_avg": _format_seconds_mmss(bench_avg),
            "avg_delta": _format_seconds_mmss(avg_delta),
            "min_delta": _format_seconds_mmss(min_delta),
            "max_delta": _format_seconds_mmss(max_delta),
        }
    return {
        "events": events,
        "beats": beats,
        "usa_avg": round(usa_avg, 3),
        "bench_avg": round(bench_avg, 3),
        "avg_delta": round(avg_delta, 3),
        "min_delta": round(min_delta, 3),
        "max_delta": round(max_delta, 3),
    }


def compute_usa_vs_benchmark_events(
    df: pd.DataFrame,
    usa_athlete_id: int,
    athlete_weights: dict[int, float],
    value_col: str,
) -> pd.DataFrame:
    """Per-event merged USA vs benchmark values.

    Returns columns: event_id, prog_id, event_label, event_date, usa_value, bench_value, delta
    where delta = usa_value - bench_value.
    """
    if df.empty or not athlete_weights:
        return pd.DataFrame()

    bench = compute_weighted_benchmark_by_event(df, athlete_weights, value_col=value_col)
    if bench.empty:
        return pd.DataFrame()

    cols = ["event_id", "prog_id", "event_label", "event_date", "athlete_id", "finish_status", value_col]
    usa = df[cols].copy()
    usa["finish_status"] = usa["finish_status"].fillna("").astype(str).str.upper()
    usa = usa[(usa["finish_status"].eq("FINISH")) & (usa["athlete_id"] == usa_athlete_id)]
    usa[value_col] = pd.to_numeric(usa[value_col], errors="coerce")
    usa = usa.dropna(subset=[value_col])
    if usa.empty:
        return pd.DataFrame()

    merged = usa.merge(bench, on=["event_id", "prog_id"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged = merged.rename(columns={value_col: "usa_value"})
    merged["delta"] = pd.to_numeric(merged["usa_value"], errors="coerce") - pd.to_numeric(merged["bench_value"], errors="coerce")
    return merged[["event_id", "prog_id", "event_label", "event_date", "usa_value", "bench_value", "delta"]]


def _robust_scale(series: pd.Series) -> float:
    """Robust scale estimate for normalization (prefers IQR)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 1.0
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr and iqr > 0:
        return float(iqr)
    std = float(s.std())
    if std and std > 0:
        return float(std)
    return 1.0


def compute_composite_scores(
    df: pd.DataFrame,
    usa_athlete_id: int,
    athlete_weights: dict[int, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute per-event composite score vs benchmark.

    Composite score is a weighted, unitless normalization of per-metric deltas.
    For each metric we compute delta = (USA - benchmark). We convert to an 'advantage'
    where positive means USA is better than benchmark:
      - lower-is-better metrics: advantage = -(delta / scale)
      - higher-is-better metrics (speed): advantage = +(delta / scale)
    Scale is the IQR of deltas across events (fallback to std, then 1).

    Returns:
      - per-event dataframe with columns: event_label, event_date, score
      - scales dict by metric key
    """
    metric_specs = [
        # key, value_col, better_is_lower, weight
        ("swim_pace", "swim_pace_s_per_100m", True, 0.25),
        ("bike_speed", "bike_speed_kmh", False, 0.25),
        ("run_pace", "run_pace_s_per_km", True, 0.35),
        ("t1", "t1time_sec", True, 0.075),
        ("t2", "t2time_sec", True, 0.075),
    ]

    per_metric: dict[str, pd.DataFrame] = {}
    scales: dict[str, float] = {}

    for key, value_col, _better_is_lower, _weight in metric_specs:
        merged = compute_usa_vs_benchmark_events(df, usa_athlete_id, athlete_weights, value_col=value_col)
        if merged.empty:
            continue
        per_metric[key] = merged
        scales[key] = _robust_scale(merged["delta"])

    if not per_metric:
        return pd.DataFrame(columns=["event_label", "event_date", "score"]), scales

    # Start from the union of events present in any metric
    events = None
    for merged in per_metric.values():
        cols = ["event_id", "prog_id", "event_label", "event_date"]
        uniq = merged[cols].drop_duplicates()
        events = uniq if events is None else pd.concat([events, uniq], ignore_index=True)
    events = events.drop_duplicates(subset=["event_id", "prog_id"]).copy()

    # Join per-metric deltas
    for key, value_col, better_is_lower, _weight in metric_specs:
        if key not in per_metric:
            continue
        m = per_metric[key][["event_id", "prog_id", "delta"]].rename(columns={"delta": f"delta_{key}"})
        events = events.merge(m, on=["event_id", "prog_id"], how="left")
        scale = float(scales.get(key, 1.0) or 1.0)
        if better_is_lower:
            # delta < 0 means USA better => advantage positive
            events[f"adv_{key}"] = -(pd.to_numeric(events[f"delta_{key}"], errors="coerce") / scale)
        else:
            events[f"adv_{key}"] = pd.to_numeric(events[f"delta_{key}"], errors="coerce") / scale

    # Weighted average of available advantages per event (re-normalize weights when missing)
    weight_map = {k: w for k, _c, _b, w in metric_specs}
    score = pd.Series([float("nan")] * len(events), index=events.index, dtype="float64")

    for idx, row in events.iterrows():
        numer = 0.0
        denom = 0.0
        for key in weight_map.keys():
            adv = row.get(f"adv_{key}")
            if adv is None or pd.isna(adv):
                continue
            w = float(weight_map[key])
            numer += w * float(adv)
            denom += w
        if denom > 0:
            score.loc[idx] = numer / denom

    out = events[["event_label", "event_date"]].copy()
    out["score"] = score
    out = out.dropna(subset=["score"])
    return out, scales


def _projection_table_html(
    title: str,
    df: pd.DataFrame,
    usa_athlete_id: int,
    athlete_weights: dict[int, float],
) -> str:
    """Projection view as a compact table.

    For each metric, we compute the median (USA - Bench) across comparable events.
    If the median already meets the benchmark, show a ✅ in the Needed column.
    Otherwise show the magnitude of improvement needed (in the metric's units).
    """

    total_col = "total_time_sec_adjusted" if "total_time_sec_adjusted" in df.columns else "total_time_sec"

    metric_specs = [
        # name, col, unit_label, fmt, better_is_lower
        ("Swim pace", "swim_pace_s_per_100m", "mm:ss/100m", "time", True),
        ("Bike speed", "bike_speed_kmh", "km/h", "float", False),
        ("Run pace", "run_pace_s_per_km", "mm:ss/km", "time", True),
        ("T1", "t1time_sec", "mm:ss", "time", True),
        ("T2", "t2time_sec", "mm:ss", "time", True),
        ("Total (placing)", total_col, "mm:ss", "time", True),
    ]

    rows: list[dict[str, object]] = []
    for name, value_col, unit, fmt, better_is_lower in metric_specs:
        merged = compute_usa_vs_benchmark_events(df, usa_athlete_id, athlete_weights, value_col=value_col)
        if merged.empty:
            rows.append({"Metric": f"{name} ({unit})", "Median (USA - Bench)": "", "Needed to meet": ""})
            continue

        delta = pd.to_numeric(merged["delta"], errors="coerce").dropna()
        if delta.empty:
            rows.append({"Metric": f"{name} ({unit})", "Median (USA - Bench)": "", "Needed to meet": ""})
            continue

        median_delta = float(delta.median())

        if better_is_lower:
            meets = median_delta <= 0
            needed_mag = 0.0 if meets else median_delta
        else:
            meets = median_delta >= 0
            needed_mag = 0.0 if meets else -median_delta

        if fmt == "time":
            median_txt = _format_seconds_mmss(median_delta)
            needed_txt = "✅" if meets else _format_seconds_mmss(needed_mag)
        else:
            median_txt = round(median_delta, 2)
            needed_txt = "✅" if meets else round(needed_mag, 2)

        rows.append(
            {
                "Metric": f"{name} ({unit})",
                "Median (USA - Bench)": median_txt,
                "Needed to meet": needed_txt,
            }
        )

    table = pd.DataFrame(rows)
    return (
        "<div class='summary-block'>"
        f"<h3>{title} (Projection)</h3>"
        "<p>Median across comparable events. ✅ means the median already meets the benchmark.</p>"
        + table.to_html(index=False, escape=True)
        + "</div>"
    )


def _color_code_beat_rate(beat_rate_str: str) -> str:
    """
    Apply color-coding to beat_rate cell based on percentage.
    100% = light green bg / dark green text
    50% = light yellow bg / dark yellow-brown text
    0% = light red bg / dark red text
    Interpolates linearly between these points.
    """
    if not beat_rate_str or "/" not in beat_rate_str:
        return beat_rate_str
    
    try:
        beats, events = beat_rate_str.split("/")
        beats = int(beats)
        events = int(events)
        if events == 0:
            return beat_rate_str
        
        percentage = beats / events
        
        # Color interpolation: 0% (red) -> 50% (yellow) -> 100% (green)
        if percentage >= 0.5:
            # Interpolate between yellow (50%) and green (100%)
            t = (percentage - 0.5) / 0.5  # 0 at 50%, 1 at 100%
            # Background: #fff3cd (yellow) -> #d4edda (green)
            bg_r = int(255 + (212 - 255) * t)
            bg_g = int(243 + (237 - 243) * t)
            bg_b = int(205 + (218 - 205) * t)
            # Text: #856404 (yellow-brown) -> #155724 (green)
            fg_r = int(133 + (21 - 133) * t)
            fg_g = int(100 + (87 - 100) * t)
            fg_b = int(4 + (36 - 4) * t)
        else:
            # Interpolate between red (0%) and yellow (50%)
            t = percentage / 0.5  # 0 at 0%, 1 at 50%
            # Background: #f8d7da (red) -> #fff3cd (yellow)
            bg_r = int(248 + (255 - 248) * t)
            bg_g = int(215 + (243 - 215) * t)
            bg_b = int(218 + (205 - 218) * t)
            # Text: #721c24 (red) -> #856404 (yellow-brown)
            fg_r = int(114 + (133 - 114) * t)
            fg_g = int(28 + (100 - 28) * t)
            fg_b = int(36 + (4 - 36) * t)
        
        bg_color = f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}"
        fg_color = f"#{fg_r:02x}{fg_g:02x}{fg_b:02x}"
        
        return f'<span style="background-color: {bg_color}; color: {fg_color}; padding: 4px 8px; display: inline-block; border-radius: 3px; font-weight: 600;">{beat_rate_str}</span>'
    except (ValueError, ZeroDivisionError):
        return beat_rate_str


def _color_code_composite_score(score_value: float) -> str:
    """
    Apply color-coding to composite score based on value.
    Negative (better) = green shades
    Positive (worse) = red shades
    Zero = neutral yellow
    Color intensity increases with absolute value magnitude.
    """
    try:
        score = float(score_value)
        
        # Clamp to reasonable range for color scaling (-5 to +5)
        clamped = max(-5, min(5, score))
        
        if score < 0:
            # Negative = better = green
            # Scale from 0 (yellow) to -5 (bright green)
            intensity = abs(clamped) / 5.0  # 0 to 1
            # Background: #fff3cd (yellow) -> #d4edda (green)
            bg_r = int(255 - (255 - 212) * intensity)
            bg_g = int(243 - (243 - 237) * intensity)
            bg_b = int(205 + (218 - 205) * intensity)
            # Text: #856404 (yellow-brown) -> #155724 (green)
            fg_r = int(133 - (133 - 21) * intensity)
            fg_g = int(100 - (100 - 87) * intensity)
            fg_b = int(4 + (36 - 4) * intensity)
        elif score > 0:
            # Positive = worse = red
            # Scale from 0 (yellow) to +5 (bright red)
            intensity = clamped / 5.0  # 0 to 1
            # Background: #fff3cd (yellow) -> #f8d7da (red)
            bg_r = int(255 - (255 - 248) * intensity)
            bg_g = int(243 - (243 - 215) * intensity)
            bg_b = int(205 + (218 - 205) * intensity)
            # Text: #856404 (yellow-brown) -> #721c24 (red)
            fg_r = int(133 - (133 - 114) * intensity)
            fg_g = int(100 - (100 - 28) * intensity)
            fg_b = int(4 - (4 - 36) * intensity)
        else:
            # Zero = neutral yellow
            bg_r, bg_g, bg_b = 255, 243, 205
            fg_r, fg_g, fg_b = 133, 100, 4
        
        bg_color = f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}"
        fg_color = f"#{fg_r:02x}{fg_g:02x}{fg_b:02x}"
        
        return f'<span style="background-color: {bg_color}; color: {fg_color}; padding: 4px 8px; display: inline-block; border-radius: 3px; font-weight: 600;">{score_value:.3f}</span>'
    except (ValueError, TypeError):
        return str(score_value)


def _benchmark_scoreboard_html(
    df: pd.DataFrame,
    usa_athlete_id: int,
    medalists: pd.DataFrame,
) -> str:
    def _build_block(title: str, description: str, weights: dict[int, float]) -> str:
        if not weights:
            return ""

        composite_df, composite_scales = compute_composite_scores(df, usa_athlete_id=usa_athlete_id, athlete_weights=weights)
        if not composite_df.empty:
            comp_events = int(len(composite_df))
            comp_beats = int((pd.to_numeric(composite_df["score"], errors="coerce") >= 0).sum())
            comp_avg = float(pd.to_numeric(composite_df["score"], errors="coerce").mean())
            comp_min = float(pd.to_numeric(composite_df["score"], errors="coerce").min())
            comp_max = float(pd.to_numeric(composite_df["score"], errors="coerce").max())
        else:
            comp_events = 0
            comp_beats = 0
            comp_avg = float("nan")
            comp_min = float("nan")
            comp_max = float("nan")

        rows = []
        rows.append(
            {
                "metric": "Swim pace (mm:ss/100m)",
                **compute_usa_vs_benchmark_summary(
                    df,
                    usa_athlete_id,
                    weights,
                    "swim_pace_s_per_100m",
                    fmt="time",
                    better_is_lower=True,
                ),
            }
        )
        rows.append(
            {
                "metric": "Bike speed (km/h)",
                **compute_usa_vs_benchmark_summary(
                    df,
                    usa_athlete_id,
                    weights,
                    "bike_speed_kmh",
                    fmt="float",
                    better_is_lower=False,
                ),
            }
        )
        rows.append(
            {
                "metric": "Run pace (mm:ss/km)",
                **compute_usa_vs_benchmark_summary(
                    df,
                    usa_athlete_id,
                    weights,
                    "run_pace_s_per_km",
                    fmt="time",
                    better_is_lower=True,
                ),
            }
        )
        rows.append(
            {
                "metric": "Swim split (mm:ss)",
                **compute_usa_vs_benchmark_summary(df, usa_athlete_id, weights, "swimtime_sec", fmt="time"),
            }
        )
        rows.append(
            {
                "metric": "Bike split (mm:ss)",
                **compute_usa_vs_benchmark_summary(df, usa_athlete_id, weights, "biketime_sec", fmt="time"),
            }
        )
        rows.append(
            {
                "metric": "Run split (mm:ss)",
                **compute_usa_vs_benchmark_summary(df, usa_athlete_id, weights, "runtime_sec", fmt="time"),
            }
        )
        rows.append(
            {
                "metric": "T1 (mm:ss)",
                **compute_usa_vs_benchmark_summary(df, usa_athlete_id, weights, "t1time_sec", fmt="time"),
            }
        )
        rows.append(
            {
                "metric": "T2 (mm:ss)",
                **compute_usa_vs_benchmark_summary(df, usa_athlete_id, weights, "t2time_sec", fmt="time"),
            }
        )
        rows.append(
            {
                "metric": "Total time (placing, mm:ss)",
                **compute_usa_vs_benchmark_summary(
                    df,
                    usa_athlete_id,
                    weights,
                    "total_time_sec_adjusted" if "total_time_sec_adjusted" in df.columns else "total_time_sec",
                    fmt="time",
                ),
            }
        )

        # Dual-frame transparency (does not change benchmark frame): show effort total + factor segment when present.
        if "total_time_sec_raw" in df.columns:
            rows.append(
                {
                    "metric": "Effort total (mm:ss)",
                    **compute_usa_vs_benchmark_summary(df, usa_athlete_id, weights, "total_time_sec_raw", fmt="time"),
                }
            )
        if "time_factor_segment_sec" in df.columns:
            rows.append(
                {
                    "metric": "Factor segment (mm:ss)",
                    **compute_usa_vs_benchmark_summary(
                        df,
                        usa_athlete_id,
                        weights,
                        "time_factor_segment_sec",
                        fmt="time",
                        better_is_lower=True,
                    ),
                }
            )

        # Composite score row (unitless, positive = USA better than benchmark)
        if comp_events:
            rows.append(
                {
                    "metric": "Composite score (unitless)",
                    "events": comp_events,
                    "beats": comp_beats,
                    "usa_avg": "",
                    "bench_avg": "",
                    "avg_delta": round(comp_avg, 3),
                    "min_delta": round(comp_min, 3),
                    "max_delta": round(comp_max, 3),
                }
            )

        scoreboard = pd.DataFrame(rows)
        scoreboard["beat_rate"] = scoreboard.apply(
            lambda r: f"{int(r['beats'])}/{int(r['events'])}" if r.get("events") else "",
            axis=1,
        )
        cols = ["metric", "beat_rate", "usa_avg", "bench_avg", "avg_delta"]
        scoreboard = scoreboard[cols].rename(
            columns={
                "metric": "Metric",
                "beat_rate": "# of races Athlete faster than Benchmark",
                "usa_avg": "Athlete Avg.",
                "bench_avg": "Benchmark Avg.",
                "avg_delta": "Athlete Average Time Relative To Benchmark",
            }
        )

        # Generate custom HTML table with color-coded beat_rate column
        table_html = "<table><thead><tr>"
        for col in scoreboard.columns:
            table_html += f"<th>{col}</th>"
        table_html += "</tr></thead><tbody>"
        
        for _, row in scoreboard.iterrows():
            table_html += "<tr>"
            for col in scoreboard.columns:
                cell_value = row[col]
                if col == "# of races Athlete faster than Benchmark":
                    # Apply color-coding to beat rate column
                    cell_value = _color_code_beat_rate(str(cell_value))
                    table_html += f"<td>{cell_value}</td>"
                elif col == "Athlete Average Time Relative To Benchmark" and row.get("Metric") == "Composite score (unitless)":
                    # Apply color-coding to composite score value (negative = better = green)
                    if cell_value != "" and pd.notna(cell_value):
                        cell_value = _color_code_composite_score(float(cell_value))
                        table_html += f"<td>{cell_value}</td>"
                    else:
                        import html
                        table_html += f"<td>{html.escape(str(cell_value))}</td>"
                else:
                    # Escape other cells for safety
                    import html
                    table_html += f"<td>{html.escape(str(cell_value))}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

        block_html = (
            "<div class='summary-block'>"
            f"<h2>{title}</h2>"
            f"<p>{description}</p>"
            "<p><b>Composite score:</b> weighted, IQR-normalized advantage across swim pace, bike speed, run pace, T1, T2. Negative = better than benchmark.</p>"
            + table_html
            + "</div>"
        )

        projection_html = _projection_table_html(title, df, usa_athlete_id=usa_athlete_id, athlete_weights=weights)
        return block_html + projection_html + "<hr/>"

    win_weights = _winner_only_weights(medalists)
    contend_weights = _default_medalist_weights(medalists)

    html = ""
    html += _build_block(
        "Win Gold Benchmark",
        "Benchmark = benchmark gold medalist only (top finisher).",
        win_weights,
    )
    html += _build_block(
        "Gold Contention Benchmark",
        "Benchmark = weighted average of benchmark medalists (1/2/3 weights = 0.5/0.3/0.2) plus additional top contenders (weight = 0.15 each), computed per event and summarized across events.",
        contend_weights,
    )
    return html


def _apply_mmss_yaxis(fig, series: pd.Series) -> None:
    """Apply an mm:ss tick label mapping to a numeric y-axis."""
    try:
        values = pd.to_numeric(series, errors="coerce").dropna()
    except Exception:
        return
    if values.empty:
        return

    y_min = float(values.min())
    y_max = float(values.max())
    tickvals = [y_min, y_max]
    if y_min <= 0.0 <= y_max:
        tickvals.append(0.0)
    tickvals = sorted(set(tickvals))
    # Add a few intermediate ticks for readability
    if y_min != y_max:
        for t in [0.25, 0.5, 0.75]:
            tickvals.append(y_min + (y_max - y_min) * t)
    tickvals = sorted(set([float(v) for v in tickvals]))
    ticktext = [_format_seconds_mmss(v) for v in tickvals]
    fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)


def summarize_differences(
    df: pd.DataFrame,
    usa_athlete_id: int,
    value_col: str,
    label: str,
    better_is_lower: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Compute per-opponent summary stats for (opponent - USA) differences.

    Only uses rows where both athletes are FINISH and value_col is non-null.
    """
    if df.empty:
        return pd.DataFrame(), label

    base_cols = ["event_id", "prog_id", "athlete_id", "full_name", "finish_status", value_col]
    work = df[base_cols].copy()
    work["finish_status"] = work["finish_status"].fillna("").astype(str).str.upper()
    work = work[work["finish_status"].eq("FINISH")]
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(), label

    usa = work[work["athlete_id"] == usa_athlete_id][["event_id", "prog_id", value_col]].rename(
        columns={value_col: "usa_value"}
    )
    opp = work[work["athlete_id"] != usa_athlete_id].copy()
    merged = opp.merge(usa, on=["event_id", "prog_id"], how="inner")
    if merged.empty:
        return pd.DataFrame(), label

    merged["diff"] = merged[value_col] - merged["usa_value"]
    # For 'lower is better' metrics (time/pace): diff > 0 => opponent slower => USA better.
    # For 'higher is better' metrics (speed): diff < 0 => opponent slower => USA better.
    if better_is_lower:
        usa_better = merged["diff"] > 0
        usa_worse = merged["diff"] < 0
    else:
        usa_better = merged["diff"] < 0
        usa_worse = merged["diff"] > 0

    merged = merged.assign(
        _usa_faster=usa_better.astype(int),
        _usa_slower=usa_worse.astype(int),
    )
    grp = merged.groupby("full_name", dropna=False)
    summary = grp.agg(
        avg_diff=("diff", "mean"),
        min_diff=("diff", "min"),
        max_diff=("diff", "max"),
        usa_faster=("_usa_faster", "sum"),
        usa_slower=("_usa_slower", "sum"),
    ).reset_index()
    summary = summary.rename(columns={"full_name": "opponent"})
    return summary.sort_values(["usa_faster", "opponent"], ascending=[False, True]), label


def compute_gap_long_df(
    df: pd.DataFrame,
    usa_athlete_id: int,
    value_col: str,
    gap_name: str,
) -> pd.DataFrame:
    """Return long dataframe for plotting opponent-vs-USA gaps by event.

    Produces columns: event_label, event_date, opponent, diff
    where diff = opponent_value - usa_value.
    """
    if df.empty:
        return pd.DataFrame()

    base = df[[
        "event_id",
        "prog_id",
        "event_label",
        "event_date",
        "distance_panel",
        "athlete_id",
        "full_name",
        "finish_status",
        value_col,
    ]].copy()
    base["finish_status"] = base["finish_status"].fillna("").astype(str).str.upper()
    base = base[base["finish_status"].eq("FINISH")]
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")
    base = base.dropna(subset=[value_col])
    if base.empty:
        return pd.DataFrame()

    usa = base[base["athlete_id"] == usa_athlete_id][["event_id", "prog_id", value_col]].rename(
        columns={value_col: "usa_value"}
    )
    opp = base[base["athlete_id"] != usa_athlete_id].copy()
    merged = opp.merge(usa, on=["event_id", "prog_id"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["diff"] = merged[value_col] - merged["usa_value"]
    merged = merged.rename(columns={"full_name": "opponent"})
    merged["gap_metric"] = gap_name
    return merged[["event_label", "event_date", "opponent", "diff", "gap_metric"]]


def _add_nonfinish_markers(fig, df: pd.DataFrame, y_col: str) -> None:
    nonfinish = df[df["finish_status"].fillna("").str.upper().ne("FINISH")].copy()
    if nonfinish.empty:
        return

    finishers = df[df["finish_status"].fillna("").str.upper().eq("FINISH")]
    y_max = None
    if not finishers.empty and y_col in finishers.columns:
        try:
            y_max = pd.to_numeric(finishers[y_col], errors="coerce").max()
        except Exception:
            y_max = None

    if y_max is None or pd.isna(y_max):
        y_marker = 1.0
    else:
        y_marker = float(y_max) * 1.05

    nonfinish = nonfinish.assign(_nonfinish_y=y_marker)

    fig.add_scatter(
        x=nonfinish["event_label"],
        y=nonfinish["_nonfinish_y"],
        mode="markers+text",
        text=nonfinish["finish_status"],
        textposition="top center",
        marker=dict(symbol="x", size=10),
        name="Non-finish",
        showlegend=True,
    )


def make_panel_figure(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    invert_y: bool = False,
    mmss_axis: bool = False,
    hover_mmss: bool = False,
    extra_hover_cols: list[str] | None = None,
):
    if df.empty:
        return None

    work = df.copy()
    # Use only rows where metric exists for line chart (FINISH-only; non-finish shown separately)
    finish_mask = work["finish_status"].fillna("").astype(str).str.upper().eq("FINISH")
    line_df = work[finish_mask & work[y_col].notna()].copy()

    event_order = _ordered_event_labels(work)

    plot_df = line_df.copy()
    if hover_mmss:
        plot_df["_y_mmss"] = plot_df[y_col].apply(_format_seconds_mmss)

    hover_data = {
        "event_name": True,
        "event_date": True,
        "finish_status": True,
        "finish_position": True,
        "total_time_fmt": True,
    }
    if hover_mmss:
        hover_data["_y_mmss"] = True
        hover_data[y_col] = False
    if extra_hover_cols:
        for c in extra_hover_cols:
            if c in plot_df.columns:
                hover_data[c] = True

    fig = px.line(
        plot_df,
        x="event_label",
        y=y_col,
        color="full_name",
        markers=True,
        hover_data=hover_data,
        category_orders={"event_label": event_order},
        title=title,
        labels={y_col: y_label, "event_label": "Event"},
        height=600,  # Taller chart for better y-axis visibility
    )

    # Remove trendline/averages (explicitly not requested)
    fig.update_traces(connectgaps=False)

    # Add non-finish markers (outside the main range) per whole figure
    _add_nonfinish_markers(fig, work, y_col)

    fig.update_layout(legend_title_text="Athlete")
    fig.update_xaxes(tickangle=45, automargin=True)
    if invert_y:
        fig.update_yaxes(autorange="reversed")
    if mmss_axis:
        _apply_mmss_yaxis(fig, plot_df[y_col])
    return fig


def make_gap_figure(gap_df: pd.DataFrame, title: str, y_label: str, mmss: bool = False):
    if gap_df.empty:
        return None
    event_order = _ordered_event_labels(gap_df)

    plot_df = gap_df.copy()
    if mmss:
        plot_df["diff_mmss"] = plot_df["diff"].apply(_format_seconds_mmss)
    plot_df["event_date_str"] = plot_df["event_date"].astype(str)

    fig = px.line(
        plot_df,
        x="event_label",
        y="diff",
        color="opponent",
        markers=True,
        hover_data=(
            {"event_date_str": True, "diff_mmss": True, "diff": False}
            if mmss
            else {"event_date_str": True, "diff": True}
        ),
        category_orders={"event_label": event_order},
        title=title,
        labels={
            "diff": y_label,
            "event_label": "Event",
            "event_date_str": "Date",
            "diff_mmss": "Gap",
        },
        height=600,  # Taller chart for better y-axis visibility
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(legend_title_text="Opponent")
    fig.update_xaxes(tickangle=45, automargin=True)

    if mmss:
        _apply_mmss_yaxis(fig, plot_df["diff"])
    return fig


def write_outputs(
    outdir: Path,
    category: str,
    usa_athlete_name: str,
    usa_athlete_id: int,
    bench: BenchmarkEvent,
    medalists: pd.DataFrame,
    df: pd.DataFrame,
    write_png: bool,
) -> None:
    _ensure_outdir(outdir)

    # Data export
    df.to_csv(outdir / "dataset.csv", index=False)
    print(f"Wrote dataset: {outdir / 'dataset.csv'}")

    figures = []
    plotly_js_included = False

    # Write a simple HTML report
    report_path = outdir / "report.html"
    blocks = []

    # Separate benchmark medalists (positions 1-3) from additional contenders (4+)
    paris_medalists = medalists[medalists["finish_position"] <= 3].copy()
    additional_contenders = medalists[medalists["finish_position"] > 3].copy()

    medal_lines = "".join(
        f"<li>{r['full_name']} (athlete_id={int(r['athlete_id'])})</li>"
        for _, r in paris_medalists.sort_values("finish_position").iterrows()
    )

    additional_lines = ""
    if not additional_contenders.empty:
        additional_lines = (
            "<p><b>Additional benchmark athletes:</b></p><ul>"
            + "".join(
                f"<li>{r['full_name']} (athlete_id={int(r['athlete_id'])})</li>"
                for _, r in additional_contenders.iterrows()
            )
            + "</ul>"
        )

    header = f"""
    <h1>Para Standards Report</h1>
    <p><b>Category:</b> {category}</p>
    <p><b>Selected USA athlete:</b> {usa_athlete_name}</p>
    <p><b>Benchmark event:</b> {bench.event_name} ({bench.event_date})</p>
    <p><b>Benchmark medalists:</b></p>
    <ul>{medal_lines}</ul>
    {additional_lines}
    <hr/>
    """

    blocks.append(header)

    # Top-level benchmark scoreboard
    blocks.append(_benchmark_scoreboard_html(df, usa_athlete_id=usa_athlete_id, medalists=medalists))

    # --- Pace charts + summaries -------------------------------------------------------
    pace_sections = [
        ("swim_pace", "swim_pace_s_per_100m", "Swim pace (mm:ss/100m)", "Swim pace differences (opponent - USA)", True, True, True, None),
        ("bike_speed", "bike_speed_kmh", "Bike speed (km/h)", "Bike speed differences (opponent - USA)", False, False, False, ["bike_speed_mph"]),
        ("run_pace", "run_pace_s_per_km", "Run pace (mm:ss/km)", "Run pace differences (opponent - USA)", True, True, True, None),
    ]

    for key, col, y_label, summary_title, invert_y, mmss_axis, hover_mmss, extra_hover in pace_sections:
        better_is_lower = col != "bike_speed_kmh"
        summary_df, _ = summarize_differences(
            df,
            usa_athlete_id=usa_athlete_id,
            value_col=col,
            label=summary_title,
            better_is_lower=better_is_lower,
        )
        if col in ["swim_pace_s_per_100m", "run_pace_s_per_km"]:
            summary_df = _format_summary_time_columns(summary_df)
        elif col == "bike_speed_kmh":
            summary_df = _format_summary_numeric_columns(summary_df, decimals=2)
        blocks.append(_summary_table_html(summary_title, summary_df))

        fig = make_panel_figure(
            df,
            col,
            f"{category}: {y_label} vs benchmark (since 2021)",
            y_label,
            invert_y=invert_y,
            mmss_axis=mmss_axis,
            hover_mmss=hover_mmss,
            extra_hover_cols=extra_hover,
        )
        if fig:
            include_plotly_js = "inline" if not plotly_js_included else False
            blocks.append(pio.to_html(fig, include_plotlyjs=include_plotly_js, full_html=False))
            plotly_js_included = True
            figures.append((key, fig))

    # --- Raw split gap charts (seconds) + summaries -----------------------------------
    gap_sections = [
        ("swim_gap", "swimtime_sec", "Swim split gap vs USA (sec)", "Swim split time gaps (sec, opponent - USA)"),
        ("bike_gap", "biketime_sec", "Bike split gap vs USA (sec)", "Bike split time gaps (sec, opponent - USA)"),
        ("run_gap", "runtime_sec", "Run split gap vs USA (sec)", "Run split time gaps (sec, opponent - USA)"),
    ]

    for key, col, y_label, summary_title in gap_sections:
        summary_df, _ = summarize_differences(df, usa_athlete_id=usa_athlete_id, value_col=col, label=summary_title)
        summary_df = _format_summary_time_columns(summary_df)
        blocks.append(_summary_table_html(summary_title, summary_df))

        gap_df = compute_gap_long_df(df, usa_athlete_id=usa_athlete_id, value_col=col, gap_name=key)
        fig = make_gap_figure(
            gap_df,
            title=f"{category}: {y_label} (since 2021)",
            y_label=y_label,
            mmss=True,
        )
        if fig:
            include_plotly_js = "inline" if not plotly_js_included else False
            blocks.append(pio.to_html(fig, include_plotlyjs=include_plotly_js, full_html=False))
            plotly_js_included = True
            figures.append((key, fig))

    # --- Transition/overall charts (raw seconds) + summaries ---------------------------
    other_sections = [
        ("t1", "t1time_sec", "T1 (sec)", "T1 differences (sec, opponent - USA)"),
        ("t2", "t2time_sec", "T2 (sec)", "T2 differences (sec, opponent - USA)"),
        ("total_time", "total_time_sec", "Total time (sec)", "Overall total time gaps (sec, opponent - USA)"),
    ]

    for key, col, y_label, summary_title in other_sections:
        summary_df, _ = summarize_differences(df, usa_athlete_id=usa_athlete_id, value_col=col, label=summary_title)
        blocks.append(_summary_table_html(summary_title, summary_df))

        fig = make_panel_figure(
            df,
            col,
            f"{category}: {y_label} vs benchmark (since 2021)",
            y_label,
        )
        if fig:
            include_plotly_js = "inline" if not plotly_js_included else False
            blocks.append(pio.to_html(fig, include_plotlyjs=include_plotly_js, full_html=False))
            plotly_js_included = True
            figures.append((key, fig))

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Para Standards</title>"
        "<style>"
        ".summary-block{ text-align:center; margin: 18px 0; }"
        ".summary-block table{ margin-left:auto; margin-right:auto; border-collapse:collapse; }"
        ".summary-block th{ padding: 6px 10px; border: 1px solid #ddd; background-color: #d3d3d3; font-weight: bold; text-align: center; color: #000; }"
        ".summary-block td{ padding: 6px 10px; border: 1px solid #ddd; }"
        "</style>"
        "</head><body>"
        + "\n".join(blocks)
        + "</body></html>"
    )
    report_path.write_text(html, encoding="utf-8")

    print(f"Wrote report: {report_path}")

    if write_png:
        # PNG export can hang on some Windows setups (kaleido/spawn issues).
        # Write the HTML report first (above), then attempt PNGs; failures are non-fatal.
        print("Exporting PNG figures...")
        png_dir = outdir / "figures"
        exported = 0
        for key, fig in figures:
            target = png_dir / f"{key}.png"
            print(f"  - {target.name}")
            try:
                fig.write_image(target, width=1400, height=650, scale=2)
                exported += 1
            except Exception as e:
                print(
                    f"PNG export failed for {target.name}: {e}\n"
                    "Tip: re-run with --no-png to skip image export (HTML+CSV will still be produced)."
                )
        if exported:
            print(f"Wrote PNGs: {png_dir}")


def run(
    category: str,
    usa_athlete_name: str,
    since: str,
    benchmark_year: int,
    benchmark_city: str,
    outdir: Path,
    write_png: bool,
    normalize_time_factor: bool = True,
    auto_select_first: bool = False,
    major_cat_patterns: Optional[list[str]] = None,
) -> None:
    engine = get_engine()

    patterns = major_cat_patterns or DEFAULT_MAJOR_CAT_PATTERNS

    print("Resolving selected USA athlete...")
    usa_id, usa_name_resolved = select_usa_athlete_from_category(
        engine,
        category=category,
        since=since,
        major_cat_patterns=patterns,
        typed_name=usa_athlete_name,
        auto_select_first=auto_select_first,
    )
    print(f"Selected USA athlete: {usa_name_resolved} (athlete_id={usa_id})")

    print("Finding benchmark event...")
    bench = find_paris_benchmark_event(
        engine,
        category=category,
        benchmark_year=benchmark_year,
        benchmark_city=benchmark_city,
        major_cat_patterns=patterns,
    )
    print(f"Benchmark event: {bench.event_name} ({bench.event_date}) event_id={bench.event_id} prog_id={bench.prog_id}")

    print("Deriving benchmark medalists (top 3)...")
    medalists = fetch_paris_medalists(engine, bench)
    print(
        "Benchmark medalists: "
        + ", ".join(
            f"{int(r['finish_position'])}. {r['full_name']}" for _, r in medalists.sort_values("finish_position").iterrows()
        )
    )

    # Include additional top contenders beyond benchmark medalists
    additional = fetch_additional_benchmark_athletes(engine, category)
    if not additional.empty:
        print(f"Including {len(additional)} additional benchmark athletes: {', '.join(additional['full_name'].tolist())}")
        medalists = pd.concat([medalists, additional], ignore_index=True)

    cohort_ids = [usa_id] + [int(x) for x in medalists["athlete_id"].tolist()]
    cohort_ids = sorted(set(cohort_ids))

    print("Fetching major-event history since start date...")
    history = fetch_major_history(
        engine,
        category=category,
        since=since,
        athlete_ids=cohort_ids,
        major_cat_patterns=patterns,
    )
    print(f"Fetched {len(history)} result rows.")

    if history.empty:
        raise SystemExit(
            "No historical rows found after filtering. "
            "Check that the DB contains major para events for this category since the requested start date."  # noqa: EM102
        )

    print("Computing pace/time metrics...")
    metrics = compute_metrics(history, normalize_time_factor=normalize_time_factor)

    print("Writing outputs (HTML+CSV" + ("+PNG" if write_png else "") + ")...")
    write_outputs(
        outdir=outdir,
        category=category,
        usa_athlete_name=usa_name_resolved,
        usa_athlete_id=usa_id,
        bench=bench,
        medalists=medalists,
        df=metrics,
        write_png=write_png,
    )

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Para standards report: compare USA athlete vs benchmark medalists (Paris 2024 or Wollongong 2025 for PTS3)")
    parser.add_argument("--category", required=True, help='Program name, e.g. "PTWC Women"')
    parser.add_argument(
        "--usa-athlete-name",
        required=False,
        default=None,
        help=(
            "USA athlete name (partial match OK). If omitted or not found, the script lists USA athletes "
            "with results in this category and prompts you to select one."
        ),
    )
    parser.add_argument("--since", default="2021-01-01", help="Start date (YYYY-MM-DD). Default: 2021-01-01")
    parser.add_argument("--benchmark-year", type=int, default=2024, help="Benchmark year. Default: 2024 (auto-overridden to 2025 for PTS3)")
    parser.add_argument("--benchmark-city", default="Paris", help="City keyword to locate benchmark race. Default: Paris (auto-overridden to Wollongong for PTS3)")
    parser.add_argument(
        "--outdir",
        default=str(Path("tri_analysis") / "outputs" / "para_standards"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-png",
        "--no.png",
        action="store_true",
        help="Skip PNG export (HTML+CSV only). Useful if kaleido is not installed.",
    )
    parser.add_argument(
        "--no-factor-normalization",
        action="store_true",
        help=(
            "Disable para time-factor handling (PTVI/PTWC). When enabled (default), the report computes factor audit columns and an "
            "effort total (<code>total_time_sec_raw</code>) from the sum of splits / expected factor, while keeping total time for benchmarks "
            "as adjusted-for-placing (<code>total_time_sec_adjusted</code>)."
        ),
    )
    parser.add_argument(
        "--auto-select-first",
        action="store_true",
        help="Automatically select the first USA athlete without prompting (useful for batch processing).",
    )

    args = parser.parse_args()

    run(
        category=args.category,
        usa_athlete_name=args.usa_athlete_name,
        since=args.since,
        benchmark_year=args.benchmark_year,
        benchmark_city=args.benchmark_city,
        outdir=Path(args.outdir),
        write_png=(not args.no_png),
        normalize_time_factor=(not args.no_factor_normalization),
        auto_select_first=args.auto_select_first,
    )


if __name__ == "__main__":
    main()
