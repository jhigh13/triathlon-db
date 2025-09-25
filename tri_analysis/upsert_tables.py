from sqlalchemy import text
from sqlalchemy.exc import DataError
import pandas as pd
from database import get_engine

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
    records = df.to_dict(orient="records")
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
            "wetsuit"
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