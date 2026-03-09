"""
Project Podium Analysis — Reusable Template
============================================
Queries triathlon DB for a set of athletes at target events,
tallies podiums/wins by category, and optionally generates charts.

Usage:
    1. Set DATABASE_URL in .env (or pass directly)
    2. Edit PP_ATHLETES and TARGET_EVENTS lists
    3. Run: python project_podium_analysis.py
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ── Configuration ──────────────────────────────────────────────

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URI")
OUT_DIR = Path(__file__).parent
PROG_FILTER = "Elite Men"
YEAR = 2025

PP_ATHLETES = [
    "Reese Vannerson", "Sullivan Middaugh", "Braxton Legg",
    "Blake Bullard", "Porter Middaugh", "Mathis Beaulieu",
    "Luke Anthony", "Keller Norland", "Blake Harris",
]

TARGET_EVENTS = [
    "2025 Americas Triathlon Cup La Habana",
    "2025 Americas Triathlon Cup Magog",
    "2025 Americas Triathlon Cup Miami",
    "2025 Americas Triathlon Cup Montreal",
    "2025 Americas Triathlon Cup Salinas",
    "2025 Americas Triathlon Cup Kelowna",
    "2025 Americas Triathlon Cup La Paz",
]

COUNTRY_MAP = {
    "united states": "USA", "usa": "USA", "us": "USA",
    "canada": "CAN", "can": "CAN",
    "mexico": "MEX", "mex": "MEX",
}


# ── Helpers ────────────────────────────────────────────────────

def categorize(row: pd.Series, pp_set: set[str]) -> str:
    """Assign category: Project Podium / Other USA / Canada / Mexico / Other."""
    name = str(row.get("athlete_name", "")).title()
    country = str(row.get("country_norm", "")).upper()
    if name in pp_set:
        return "Project Podium"
    if country == "USA":
        return "Other USA"
    if country == "CAN":
        return "Canada"
    if country == "MEX":
        return "Mexico"
    return "Other Countries"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean strings, normalize countries, coerce position."""
    for col in ["event_name", "athlete_name", "country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "athlete_name" in df.columns:
        df["athlete_name"] = df["athlete_name"].str.title()
    if "country" in df.columns:
        df["country_norm"] = (
            df["country"].str.lower().map(COUNTRY_MAP)
            .fillna(df["country"].str.upper())
        )
    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
    return df


# ── Data Loading ───────────────────────────────────────────────

def fetch_event_results(engine, events: list[str]) -> pd.DataFrame:
    """Pull results for target events from DB."""
    frames = []
    with engine.connect() as conn:
        for ev in events:
            q = text("""
                SELECT e.event_name, e.event_date,
                       a.full_name AS athlete_name, a.country,
                       rr.position, e.prog_name AS program
                FROM race_results rr
                JOIN events e  ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
                JOIN athlete a ON rr.athlete_id = a.athlete_id
                WHERE e.event_name = :event_name
                  AND e.prog_name  = :prog
                  AND e.event_date >= :start AND e.event_date < :end
            """)
            df = pd.read_sql(q, conn, params={
                "event_name": ev, "prog": PROG_FILTER,
                "start": f"{YEAR}-01-01", "end": f"{YEAR + 1}-01-01",
            })
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_athlete_results(engine, names: list[str]) -> pd.DataFrame:
    """Pull all results for specific athletes from DB."""
    with engine.connect() as conn:
        q = text("""
            SELECT e.event_name, e.event_date, e.prog_name AS program,
                   a.full_name AS athlete_name, a.country, rr.position
            FROM race_results rr
            JOIN athlete a ON rr.athlete_id = a.athlete_id
            JOIN events  e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
            WHERE a.full_name = ANY(:names)
              AND e.prog_name = :prog
              AND e.event_date >= :start AND e.event_date < :end
        """)
        return pd.read_sql(q, conn, params={
            "names": [n.title() for n in names], "prog": PROG_FILTER,
            "start": f"{YEAR}-01-01", "end": f"{YEAR + 1}-01-01",
        })


# ── Analysis ───────────────────────────────────────────────────

def compute_podium_tallies(df: pd.DataFrame, pp_set: set[str]):
    """Return (overall_counts, by_event_counts) DataFrames."""
    podium = df[df["position"].notna() & (df["position"] <= 3)].copy()
    podium["category"] = podium.apply(categorize, axis=1, pp_set=pp_set)

    order = ["Project Podium", "Other USA", "Canada", "Mexico", "Other Countries"]
    overall = (
        podium["category"].value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("category").reset_index(name="count")
    )
    by_event = (
        podium.pivot_table(
            index="event_name", columns="category",
            values="athlete_name", aggfunc="count", fill_value=0,
        )
        .reindex(columns=order, fill_value=0)
        .reset_index()
    )
    return overall, by_event


def compute_athlete_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Wins, podiums, races per athlete."""
    df = df.copy()
    df["is_win"] = df["position"].notna() & (df["position"] == 1)
    df["is_podium"] = df["position"].notna() & (df["position"] <= 3)
    return (
        df.groupby("athlete_name")
        .agg(wins=("is_win", "sum"), podiums=("is_podium", "sum"), races=("position", "count"))
        .reset_index()
        .sort_values(["wins", "podiums"], ascending=False)
    )


# ── Main ───────────────────────────────────────────────────────

def main():
    if not DATABASE_URL:
        print("Set DATABASE_URL in .env to connect to the database.")
        return

    engine = create_engine(DATABASE_URL)
    pp_set = {n.title() for n in PP_ATHLETES}

    # Event results → podium tallies
    event_df = normalize(fetch_event_results(engine, TARGET_EVENTS))
    if not event_df.empty:
        overall, by_event = compute_podium_tallies(event_df, pp_set)
        print("\n=== Overall Podium Counts ===")
        print(overall.to_string(index=False))
        print("\n=== By Event ===")
        print(by_event.to_string(index=False))

    # All athlete results → win/podium totals
    all_df = normalize(fetch_athlete_results(engine, PP_ATHLETES))
    if not all_df.empty:
        totals = compute_athlete_totals(all_df)
        totals.to_csv(OUT_DIR / "pp_wins_podiums_totals.csv", index=False)
        print("\n=== Athlete Totals ===")
        print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
