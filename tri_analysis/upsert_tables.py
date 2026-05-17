from sqlalchemy import bindparam, text
from sqlalchemy.exc import DataError
import pandas as pd
try:
    from tri_analysis.database import get_engine
except ImportError:  # pragma: no cover
    from database import get_engine  # type: ignore

def upsert_dataframe(df, table_name, conflict_cols, update_cols, engine=None):
    """
    Upsert a DataFrame into a PostgreSQL table using ON CONFLICT.
    - df: pandas DataFrame
    - table_name: str, table name
    - conflict_cols: list of str, columns for ON CONFLICT
    - update_cols: list of str, columns to update on conflict
    - engine: SQLAlchemy engine (optional)
    """
    if df.empty:
        print(f"No data to upsert for {table_name}.")
        return
    if engine is None:
        engine = get_engine()
    # Build column lists for SQL
    cols = list(df.columns)
    insert_cols = ",\n    ".join(cols)
    insert_vals = ",\n    ".join([f":{col}" for col in cols])
    conflict_target = ", ".join(conflict_cols)
    update_stmt = ",\n    ".join([f"{col}=EXCLUDED.{col}" for col in update_cols])
    sql = f"""
        INSERT INTO {table_name} (
            {insert_cols}
        )
        VALUES (
            {insert_vals}
        )
        ON CONFLICT ({conflict_target}) DO UPDATE SET
            {update_stmt}
    """
    def _sanitize_value(v):
        # Convert pandas Timestamp to python datetime
        try:
            if isinstance(v, pd.Timestamp):
                return None if pd.isna(v) else v.to_pydatetime()
        except Exception:
            pass

        # Convert pandas NA / numpy NaN to None (critical for integer columns)
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        return v

    records = df.to_dict(orient="records")
    # Ensure DB driver never sees NaN / NA for integer columns
    records = [{k: _sanitize_value(v) for k, v in rec.items()} for rec in records]
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), records)
    except DataError as e:
        msg = str(e).lower()
        if "out of range" in msg or "numericvalueoutofrange" in msg:
            # Fallback: try row-by-row with a fresh transaction to identify offenders
            print(f"Bulk upsert failed for {table_name} with out of range; scanning rows to locate offending record(s)...")
            for idx, rec in enumerate(records):
                try:
                    with get_engine().begin() as conn2:
                        conn2.execute(text(sql), [rec])
                except DataError as erow:
                    print(f"Offending row index {idx} in {table_name}: {rec}")
                    raise
            # If nothing raised inside loop, re-raise original
            raise
        else:
            raise

def upsert_athlete(df, engine):
    """Upsert athlete information into the database."""
    upsert_dataframe(
        df,
        "athlete",
        ["athlete_id"],
        [
            "full_name",
            "gender",
            "country",
            "age",
            "category_to",
            "category_coach",
            "category_athlete",
            "category_medical",
            "category_paratriathlete"
        ],
        engine
    )

def upsert_events(df, engine):
    """Upsert event information into the database."""
    upsert_dataframe(
        df,
        "events",
        ["event_id", "prog_id"],
        [
            "prog_name",
            "prog_distance_category",
            "is_para",
            "swim_laps",
            "swim_distance",
            "bike_laps",
            "bike_distance",
            "run_laps",
            "run_distance",
            "event_name",
            "event_venue",
            "event_date",
            "event_country",
            "event_latitude",
            "event_longitude",
            "cat_name",
            "temperature_water",
            "temperature_air",
            "humidity",
            "wbgt",
            "wind",
            "weather",
            "wetsuit",
            "technical_delegates",
            "drafting",
            "wind_speed_kmh",
            "wind_gust_kmh",
            "apparent_temp",
            "precipitation_mm",
            "cloud_cover_pct",
            "wet_bulb_temp",
            "weather_source",
            "weather_fetched_at",
        ],
        engine
    )

def upsert_race_results(df, engine):
    """Upsert race results information into the database."""
    upsert_dataframe(
        df,
        "race_results",
        ["athlete_id", "prog_id", "total_time"],
        [
            "athlete_full_name",
            "swimtime",
            "t1time",
            "biketime",
            "t2time",
            "runtime",
            "position",
            "finish_status",
            "finish_position",
            "position_sort",
            "start_num"
        ],
        engine
    )


def upsert_program_entries(df: pd.DataFrame, engine):
    """Upsert program entry (startlist) rows into the database."""
    upsert_dataframe(
        df,
        "program_entries",
        ["event_id", "prog_id", "entry_id"],
        [
            "entry_type",
            "approved",
            "start_num",
            "api_updated_at",
            "last_fetched_at",
            "is_active",
            "removed_at",
            "athlete_id",
            "athlete_full_name",
            "athlete_first",
            "athlete_last",
            "athlete_gender",
            "athlete_country_id",
            "athlete_country_name",
            "athlete_noc",
            "athlete_yob",
            "athlete_slug",
            "validated",
        ],
        engine,
    )


def deactivate_missing_program_entries(
    engine,
    event_id: int,
    prog_id: int,
    entry_type: str,
    active_entry_ids: list[int] | list[str],
    removed_at,
):
    """Mark previously-active entries as inactive when they disappear from the latest pull.

    This assumes the caller successfully fetched a complete start list for (event_id, prog_id, entry_type).
    """
    with engine.begin() as conn:
        if active_entry_ids:
            stmt = text(
                """
                UPDATE program_entries
                SET
                    is_active = FALSE,
                    removed_at = :removed_at
                WHERE event_id = :event_id
                  AND prog_id = :prog_id
                  AND entry_type = :entry_type
                  AND is_active = TRUE
                  AND entry_id NOT IN :entry_ids
                """
            ).bindparams(bindparam("entry_ids", expanding=True))
            conn.execute(
                stmt,
                {
                    "event_id": event_id,
                    "prog_id": prog_id,
                    "entry_type": entry_type,
                    "removed_at": removed_at,
                    "entry_ids": list(active_entry_ids),
                },
            )
        else:
            # If the API returns an empty list successfully, deactivate all currently-active entries.
            conn.execute(
                text(
                    """
                    UPDATE program_entries
                    SET
                        is_active = FALSE,
                        removed_at = :removed_at
                    WHERE event_id = :event_id
                      AND prog_id = :prog_id
                      AND entry_type = :entry_type
                      AND is_active = TRUE
                    """
                ),
                {
                    "event_id": event_id,
                    "prog_id": prog_id,
                    "entry_type": entry_type,
                    "removed_at": removed_at,
                },
            )