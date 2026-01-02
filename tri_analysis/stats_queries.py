"""
Quick-start stats queries for triathlon_results PostgreSQL DB.
Usage (PowerShell):
  python -m tri_analysis.stats_queries

Relies on DB_URI env var or default in config.DB_URI.
"""
import os
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from tri_analysis.config import DB_URI, ATHLETE_TABLE_NAME, EVENTS_TABLE_NAME, RACE_RESULTS_TABLE_NAME


# Normalize USA country filter (covers common label variants)
USA_FILTER = "LOWER(COALESCE(a.country, '')) IN ('usa','united states','united states of america','u.s.a','us')"


def date_filter(year: int | None):
    """Return SQL date filter for events table; None means no filter."""
    if year is None:
        return "1=1"
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return f"e.event_date >= '{start}' AND e.event_date < '{end}'"


def get_engine():
    """Create SQLAlchemy engine using env override if provided."""
    uri = os.environ.get("DB_URI", DB_URI)
    return create_engine(uri)


def fetch_df(sql: str):
    """Run SQL and return DataFrame (empty if no rows)."""
    engine = get_engine()
    with engine.begin() as conn:
        return pd.read_sql(sql, conn)


def usa_medals_detail(year: int | None):
        return fetch_df(
                f"""
                SELECT
                        rr.athlete_id,
                        a.full_name AS athlete_name,
                        a.country,
                        rr.finish_position AS position,
                        CASE rr.finish_position
                                WHEN 1 THEN 'Gold'
                                WHEN 2 THEN 'Silver'
                                WHEN 3 THEN 'Bronze'
                        END AS medal,
                        e.event_name,
                        e.event_country,
                        e.event_venue,
                        e.event_date,
                        COALESCE(e.cat_name, e.prog_name) AS race_category
                FROM {RACE_RESULTS_TABLE_NAME} rr
                JOIN {ATHLETE_TABLE_NAME} a
                    ON a.athlete_id = rr.athlete_id
                JOIN {EVENTS_TABLE_NAME} e
                    ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
                WHERE {USA_FILTER}
                    AND rr.finish_position IN (1, 2, 3)
                    AND {date_filter(year)}
                ORDER BY e.event_date DESC, rr.finish_position;
                """
        )


def usa_medals_by_category(year: int | None):
    return fetch_df(
        f"""
        SELECT
            COALESCE(e.cat_name, e.prog_name) AS race_category,
            SUM(CASE WHEN rr.finish_position = 1 THEN 1 ELSE 0 END) AS golds,
            SUM(CASE WHEN rr.finish_position = 2 THEN 1 ELSE 0 END) AS silvers,
            SUM(CASE WHEN rr.finish_position = 3 THEN 1 ELSE 0 END) AS bronzes,
            COUNT(*) AS total_medals
        FROM {RACE_RESULTS_TABLE_NAME} rr
        JOIN {ATHLETE_TABLE_NAME} a ON a.athlete_id = rr.athlete_id
        JOIN {EVENTS_TABLE_NAME} e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        WHERE {USA_FILTER}
          AND rr.finish_position IN (1, 2, 3)
          AND {date_filter(year)}
        GROUP BY COALESCE(e.cat_name, e.prog_name)
        ORDER BY total_medals DESC;
        """
    )


def usa_medals_by_host_country(year: int | None):
    return fetch_df(
        f"""
        SELECT
            e.event_country,
            SUM(CASE WHEN rr.finish_position = 1 THEN 1 ELSE 0 END) AS golds,
            SUM(CASE WHEN rr.finish_position = 2 THEN 1 ELSE 0 END) AS silvers,
            SUM(CASE WHEN rr.finish_position = 3 THEN 1 ELSE 0 END) AS bronzes,
            COUNT(*) AS total_medals
        FROM {RACE_RESULTS_TABLE_NAME} rr
        JOIN {ATHLETE_TABLE_NAME} a ON a.athlete_id = rr.athlete_id
        JOIN {EVENTS_TABLE_NAME} e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        WHERE {USA_FILTER}
          AND rr.finish_position IN (1, 2, 3)
          AND {date_filter(year)}
        GROUP BY e.event_country
        ORDER BY total_medals DESC;
        """
    )


def medal_tally_per_athlete(usa_only: bool = True, year: int | None = None):
    country_clause = f"WHERE {USA_FILTER}" if usa_only else "WHERE 1=1"
    return fetch_df(
        f"""
        SELECT
            rr.athlete_id,
            a.full_name AS athlete_name,
            a.country,
            SUM(CASE WHEN rr.finish_position = 1 THEN 1 ELSE 0 END) AS golds,
            SUM(CASE WHEN rr.finish_position = 2 THEN 1 ELSE 0 END) AS silvers,
            SUM(CASE WHEN rr.finish_position = 3 THEN 1 ELSE 0 END) AS bronzes,
            SUM(CASE WHEN rr.finish_position BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS podiums,
            COUNT(*) AS starts
        FROM {RACE_RESULTS_TABLE_NAME} rr
        JOIN {ATHLETE_TABLE_NAME} a ON a.athlete_id = rr.athlete_id
        JOIN {EVENTS_TABLE_NAME} e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        {country_clause}
          AND {date_filter(year)}
        GROUP BY rr.athlete_id, a.full_name, a.country
        ORDER BY podiums DESC, starts DESC, athlete_name
        """
    )


def podium_conversion_usa(min_starts: int = 1, year: int | None = None):
    df = medal_tally_per_athlete(usa_only=True, year=year)
    if df.empty:
        return df
    df = df.copy()
    df["podium_rate"] = df["podiums"] / df["starts"].replace(0, pd.NA)
    df = df[df["starts"] >= min_starts]
    return df.sort_values(["podium_rate", "podiums", "starts"], ascending=[False, False, False])


def countries_raced_per_athlete(usa_only: bool = True, year: int | None = None):
    country_clause = f"WHERE {USA_FILTER}" if usa_only else "WHERE 1=1"
    return fetch_df(
        f"""
        SELECT
            rr.athlete_id,
            a.full_name AS athlete_name,
            a.country AS athlete_country,
            COUNT(DISTINCT e.event_country) AS countries_raced
        FROM {RACE_RESULTS_TABLE_NAME} rr
        JOIN {ATHLETE_TABLE_NAME} a ON a.athlete_id = rr.athlete_id
        JOIN {EVENTS_TABLE_NAME} e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        {country_clause}
          AND {date_filter(year)}
        GROUP BY rr.athlete_id, a.full_name, a.country
        ORDER BY countries_raced DESC, athlete_name;
        """
    )


def countries_raced_overall(usa_only: bool = True, year: int | None = None):
    country_clause = f"WHERE {USA_FILTER}" if usa_only else "WHERE 1=1"
    athlete_join = f"JOIN {ATHLETE_TABLE_NAME} a ON a.athlete_id = rr.athlete_id" if usa_only else ""
    return fetch_df(
        f"""
        SELECT COUNT(DISTINCT e.event_country) AS countries_raced
        FROM {RACE_RESULTS_TABLE_NAME} rr
        JOIN {EVENTS_TABLE_NAME} e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        {athlete_join}
        {country_clause}
          AND {date_filter(year)};
        """
    )


def print_section(title: str, df: pd.DataFrame, max_rows: int = 20):
    print(f"\n=== {title} ===")
    if df.empty:
        print("(no rows)")
        return
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", None):
        print(df)


def export_excel(dfs: dict[str, pd.DataFrame], filename: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    with pd.ExcelWriter(path) as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"saved workbook -> {path}")


def main():
    parser = argparse.ArgumentParser(description="USA medal and country stats")
    parser.add_argument("--year", type=int, default=2025, help="Filter by event year (default: 2025). Use --year 0 for all years.")
    parser.add_argument("--min-starts", type=int, default=3, help="Minimum starts for podium conversion table")
    args = parser.parse_args()

    year_filter = None if args.year == 0 else args.year

    print("Using DB_URI from environment if set, else config.DB_URI")
    print(f"Year filter: {'ALL' if year_filter is None else year_filter}")

    outdir = Path(__file__).resolve().parent / "outputs"

    df_medal_detail = usa_medals_detail(year_filter)
    df_medal_cat = usa_medals_by_category(year_filter)
    df_medal_host = usa_medals_by_host_country(year_filter)
    df_countries_usa = countries_raced_per_athlete(usa_only=True, year=year_filter)
    df_countries_usa_total = countries_raced_overall(usa_only=True, year=year_filter)
    df_countries_all_total = countries_raced_overall(usa_only=False, year=year_filter)
    df_medal_tally_usa = medal_tally_per_athlete(usa_only=True, year=year_filter)
    df_podium_rate_usa = podium_conversion_usa(min_starts=args.min_starts, year=year_filter)

    print_section("USA Medal Details (top 50)", df_medal_detail.head(50))
    print_section("USA Medal Counts by Category", df_medal_cat)
    print_section("USA Medal Counts by Host Country", df_medal_host)
    print_section("Countries Raced per USA Athlete", df_countries_usa)
    print_section("Distinct Countries USA Athletes Raced In", df_countries_usa_total)
    print_section("Distinct Countries (All Athletes)", df_countries_all_total)
    print_section("USA Medal Tally per Athlete", df_medal_tally_usa.head(50))
    print_section(f"USA Podium Conversion (starts >={args.min_starts})", df_podium_rate_usa.head(50))

    export_excel(
        {
            "usa_medal_detail": df_medal_detail,
            "usa_medal_by_category": df_medal_cat,
            "usa_medal_by_host": df_medal_host,
            "countries_per_usa_athlete": df_countries_usa,
            "countries_total_usa": df_countries_usa_total,
            "countries_total_all": df_countries_all_total,
            "usa_medal_tally": df_medal_tally_usa,
            "usa_podium_conversion": df_podium_rate_usa,
        },
        "stats_summary.xlsx",
        outdir,
    )


if __name__ == "__main__":
    main()
