import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text

try:
    from tri_analysis.database import get_engine
except ImportError:  # pragma: no cover
    from database import get_engine  # type: ignore

def adjust_outlier(series, threshold=2):
    # For a given series of split times, if the minimum positive value is more than
    # 200% lower than the second smallest, mark it as an outlier (set to NA).
    valid = series[series > 0]
    if len(valid) < 2:
        return series
    sorted_vals = valid.sort_values()
    if sorted_vals.iloc[0] * threshold < sorted_vals.iloc[1]:
        return series.mask(series == sorted_vals.iloc[0], pd.NA)
    return series


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_latest_event_ids(events_file: str | Path = "latest_events.txt") -> list[int]:
    """Load event IDs from a newline-delimited file.

    Lines may contain extra whitespace; non-integer lines are ignored.
    """
    events_path = Path(events_file)
    if not events_path.is_absolute():
        events_path = _project_root() / events_path

    event_ids: list[int] = []
    if not events_path.exists():
        raise FileNotFoundError(f"Latest events file not found: {events_path}")

    for line in events_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            event_ids.append(int(s))
        except ValueError:
            continue

    # De-dupe while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for eid in event_ids:
        if eid not in seen:
            seen.add(eid)
            deduped.append(eid)
    return deduped

# Load in race results and calculate position metrics.
def calculate_position_metrics(event_ids: list[int] | None = None, engine=None) -> pd.DataFrame:
    engine = engine or get_engine()

    base_sql = """
        SELECT
            athlete_id,
            event_id,
            prog_id,
            swimtime,
            t1time,
            biketime,
            t2time,
            runtime,
            position,
            total_time,
            start_num,
            athlete_full_name
        FROM race_results
    """

    if event_ids:
        stmt = text(base_sql + " WHERE event_id IN :event_ids").bindparams(
            bindparam("event_ids", expanding=True)
        )
        df = pd.read_sql(stmt, engine, params={"event_ids": event_ids})
    else:
        df = pd.read_sql(text(base_sql), engine)
    
    def parse_time_to_secs(t):
        if pd.isna(t) or t == "":
            return 0
        parts = str(t).split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + int(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + int(s)
        except ValueError:
            return 0
        return 0

    # Calculate elapsed times at each checkpoint (cumulative)
    df['elapsedswim'] = df['swimtime'].apply(parse_time_to_secs)
    df['elapsedt1'] = df['elapsedswim'] + df['t1time'].apply(parse_time_to_secs)
    df['elapsedbike'] = df['elapsedt1'] + df['biketime'].apply(parse_time_to_secs)
    df['elapsedt2'] = df['elapsedbike'] + df['t2time'].apply(parse_time_to_secs)
    df['elapsedrun'] = df['elapsedt2'] + df['runtime'].apply(parse_time_to_secs)

    # Ensure all elapsed columns are int64 and no NaN/inf
    for col in ['elapsedswim','elapsedt1','elapsedbike','elapsedt2','elapsedrun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Parse individual split seconds for filtering
    df['swimsecs'] = df['swimtime'].apply(parse_time_to_secs)
    df['t1secs'] = df['t1time'].apply(parse_time_to_secs)
    df['bikesecs'] = df['biketime'].apply(parse_time_to_secs)
    df['t2secs'] = df['t2time'].apply(parse_time_to_secs)
    df['runsecs'] = df['runtime'].apply(parse_time_to_secs)
    
    # Global exclusion: if any split is zero, exclude from minima/ranking calculations
    valid_all_splits = (df['swimsecs'] > 0) & (df['t1secs'] > 0) & (df['bikesecs'] > 0) & (df['t2secs'] > 0) & (df['runsecs'] > 0)
    
    # Apply outlier check per group for each split segment
    for col in ['swimsecs', 't1secs', 'bikesecs', 't2secs', 'runsecs', 'elapsedswim', 'elapsedt1', 'elapsedbike', 'elapsedt2', 'elapsedrun']:
        if col in ['t1secs', 't2secs']:  # T1 and T2 are transitions, not splits
            threshold = 3
        else:
            threshold = 2
        df[col] = df.groupby(['event_id','prog_id'])[col].transform(lambda s: adjust_outlier(s, threshold))

    # Compute seconds behind leader per event_id + prog_id
    min_swim = df[valid_all_splits].groupby(['event_id','prog_id'])['elapsedswim'].transform('min').fillna(0)
    df['behindswim'] = df['elapsedswim'] - min_swim
    min_t1 = df[valid_all_splits].groupby(['event_id','prog_id'])['elapsedt1'].transform('min').fillna(0)
    df['behindt1'] = df['elapsedt1'] - min_t1
    min_bike = df[valid_all_splits].groupby(['event_id','prog_id'])['elapsedbike'].transform('min').fillna(0)
    df['behindbike'] = df['elapsedbike'] - min_bike
    min_t2 = df[valid_all_splits].groupby(['event_id','prog_id'])['elapsedt2'].transform('min').fillna(0)
    df['behindt2'] = df['elapsedt2'] - min_t2
    min_run = df[valid_all_splits].groupby(['event_id','prog_id'])['elapsedrun'].transform('min').fillna(0)
    df['behindrun'] = df['elapsedrun'] - min_run

    # Calculate positions at each checkpoint (only for athletes with non-zero times)
    mask_swim = (df['swimsecs'] > 0) & valid_all_splits
    df.loc[mask_swim, 'position_at_swim'] = df.loc[mask_swim].groupby(['event_id', 'prog_id'])['elapsedswim'].rank(method='min')
    mask_t1 = (df['t1secs'] > 0) & valid_all_splits
    df.loc[mask_t1, 'position_at_t1'] = df.loc[mask_t1].groupby(['event_id', 'prog_id'])['elapsedt1'].rank(method='min')
    mask_bike = (df['bikesecs'] > 0) & valid_all_splits
    df.loc[mask_bike, 'position_at_bike'] = df.loc[mask_bike].groupby(['event_id', 'prog_id'])['elapsedbike'].rank(method='min')
    mask_t2 = (df['t2secs'] > 0) & valid_all_splits
    df.loc[mask_t2, 'position_at_t2'] = df.loc[mask_t2].groupby(['event_id', 'prog_id'])['elapsedt2'].rank(method='min')
    mask_run = (df['runsecs'] > 0) & valid_all_splits
    df.loc[mask_run, 'position_at_run'] = df.loc[mask_run].groupby(['event_id', 'prog_id'])['elapsedrun'].rank(method='min')

    # Calculate position changes between checkpoints (negative = gained positions)
    df['swim_to_t1_pos_change'] = df['position_at_t1'] - df['position_at_swim']
    df['t1_to_bike_pos_change'] = df['position_at_bike'] - df['position_at_t1']
    df['bike_to_t2_pos_change'] = df['position_at_t2'] - df['position_at_bike']
    df['t2_to_run_pos_change'] = df['position_at_run'] - df['position_at_t2']

    # Calculate individual split rankings (rank by split time, not cumulative time)
    print("Calculating individual split rankings...")

    mask_swim_rank = (df['swimsecs'] > 0) & valid_all_splits
    df.loc[mask_swim_rank, 'swimrank'] = df.loc[mask_swim_rank].groupby(['event_id', 'prog_id'])['swimsecs'].rank(method='min')
    mask_t1_rank = (df['t1secs'] > 0) & valid_all_splits
    df.loc[mask_t1_rank, 't1rank'] = df.loc[mask_t1_rank].groupby(['event_id', 'prog_id'])['t1secs'].rank(method='min')
    mask_bike_rank = (df['bikesecs'] > 0) & valid_all_splits
    df.loc[mask_bike_rank, 'bikerank'] = df.loc[mask_bike_rank].groupby(['event_id', 'prog_id'])['bikesecs'].rank(method='min')
    mask_t2_rank = (df['t2secs'] > 0) & valid_all_splits
    df.loc[mask_t2_rank, 't2rank'] = df.loc[mask_t2_rank].groupby(['event_id', 'prog_id'])['t2secs'].rank(method='min')
    mask_run_rank = (df['runsecs'] > 0) & valid_all_splits
    df.loc[mask_run_rank, 'runrank'] = df.loc[mask_run_rank].groupby(['event_id', 'prog_id'])['runsecs'].rank(method='min')

    # Race-level aggregates (for fast feature building)
    # n_finishers: count of valid finishers per race
    df['n_finishers'] = df[valid_all_splits].groupby(['event_id', 'prog_id'])['athlete_id'].transform('count')
    # median_total_sec: median total elapsed time per race (distance-agnostic baseline)
    df['median_total_sec'] = df[valid_all_splits].groupby(['event_id', 'prog_id'])['elapsedrun'].transform('median')

    position_cols = [
        'position_at_swim', 'position_at_t1', 'position_at_bike', 'position_at_t2', 'position_at_run',
        'swim_to_t1_pos_change', 't1_to_bike_pos_change', 'bike_to_t2_pos_change', 't2_to_run_pos_change',
        'swimrank', 't1rank', 'bikerank', 't2rank', 'runrank'
    ]

    for col in position_cols:
        df[col] = df[col].astype('Int64')  # Nullable integer type

    for col in ['behindswim','behindt1','behindbike','behindt2','behindrun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Drop temporary split-second columns and any columns not in the new schema
    df.drop(
        columns=[
            'swimsecs','t1secs','bikesecs','t2secs','runsecs',
            # Remove old/extra columns if present
            "position","total_time","start_num",
            "swimtime", "t1time", "biketime", "t2time", "runtime", "athlete_full_name"
        ],
        errors='ignore',
        inplace=True
    )
    return df


def refresh_position_metrics(metrics_df: pd.DataFrame, engine=None, chunksize: int = 5000) -> None:
    """Refresh metrics for the events present in metrics_df.

    This keeps historical metrics for other events intact.
    Strategy:
    1) DELETE existing position_metrics rows for these event_id(s)
    2) INSERT (append) the freshly computed rows

    This avoids requiring a UNIQUE constraint for ON CONFLICT upserts.
    """
    if metrics_df.empty:
        print("No metrics to write.")
        return

    engine = engine or get_engine()
    event_ids = sorted({int(x) for x in metrics_df["event_id"].dropna().unique().tolist()})
    if not event_ids:
        raise ValueError("metrics_df has no event_id values")

    delete_stmt = text("DELETE FROM position_metrics WHERE event_id IN :event_ids").bindparams(
        bindparam("event_ids", expanding=True)
    )

    with engine.begin() as conn:
        conn.execute(delete_stmt, {"event_ids": event_ids})

    metrics_df.to_sql(
        "position_metrics",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize,
    )

# Save the calculated metrics to a database or file
def main():
    parser = argparse.ArgumentParser(description="Compute and persist position_metrics.")
    parser.add_argument(
        "--latest-events",
        action="store_true",
        help="Only compute metrics for event_ids listed in latest_events.txt",
    )
    parser.add_argument(
        "--events-file",
        default="latest_events.txt",
        help="Path to latest events file (default: latest_events.txt)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Insert chunksize for writing position_metrics",
    )
    args = parser.parse_args()

    engine = get_engine()
    event_ids = None
    if args.latest_events:
        event_ids = load_latest_event_ids(args.events_file)
        print(f"Loaded {len(event_ids)} event_id(s) from {args.events_file}.")
        if not event_ids:
            print("No event IDs found; nothing to do.")
            return

    df = calculate_position_metrics(event_ids=event_ids, engine=engine)
    if df.empty:
        print("No valid data available for metrics calculation.")
        return

    print(f"Computed {len(df)} position_metrics rows; refreshing table for {df['event_id'].nunique()} event(s)...")
    refresh_position_metrics(df, engine=engine, chunksize=args.chunksize)
    print("position_metrics refresh complete.")

if __name__ == "__main__":
    main()

