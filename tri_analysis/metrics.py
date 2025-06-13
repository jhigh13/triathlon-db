from database import get_engine
import pandas as pd

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

# Load in all race results and calculate position metrics. Save to the position_metrics table.
def calculate_position_metrics():

    df = pd.read_sql_table('race_results', get_engine(), schema='public')
    
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
    
    # Apply outlier check per group for each split segment
    for col in ['swimsecs', 't1secs', 'bikesecs', 't2secs', 'runsecs']:
        df[col] = df.groupby(['event_id','prog_id'])[col].transform(lambda s: adjust_outlier(s, 2))

    # Compute seconds behind leader per event_id + prog_id
    min_swim = df[df['swimsecs'] > 0].groupby(['event_id','prog_id'])['elapsedswim'].transform('min').fillna(0)
    df['behindswim'] = df['elapsedswim'] - min_swim
    min_t1 = df[df['t1secs'] > 0].groupby(['event_id','prog_id'])['elapsedt1'].transform('min').fillna(0)
    df['behindt1'] = df['elapsedt1'] - min_t1
    min_bike = df[df['bikesecs'] > 0].groupby(['event_id','prog_id'])['elapsedbike'].transform('min').fillna(0)
    df['behindbike'] = df['elapsedbike'] - min_bike
    min_t2 = df[df['t2secs'] > 0].groupby(['event_id','prog_id'])['elapsedt2'].transform('min').fillna(0)
    df['behindt2'] = df['elapsedt2'] - min_t2
    min_run = df[df['runsecs'] > 0].groupby(['event_id','prog_id'])['elapsedrun'].transform('min').fillna(0)
    df['behindrun'] = df['elapsedrun'] - min_run

    # Calculate positions at each checkpoint (only for athletes with non-zero times)
    mask_swim = df['swimsecs'] > 0
    df.loc[mask_swim, 'position_at_swim'] = df.loc[mask_swim].groupby(['event_id', 'prog_id'])['elapsedswim'].rank(method='min')
    mask_t1 = df['t1secs'] > 0
    df.loc[mask_t1, 'position_at_t1'] = df.loc[mask_t1].groupby(['event_id', 'prog_id'])['elapsedt1'].rank(method='min')
    mask_bike = df['bikesecs'] > 0
    df.loc[mask_bike, 'position_at_bike'] = df.loc[mask_bike].groupby(['event_id', 'prog_id'])['elapsedbike'].rank(method='min')
    mask_t2 = df['t2secs'] > 0
    df.loc[mask_t2, 'position_at_t2'] = df.loc[mask_t2].groupby(['event_id', 'prog_id'])['elapsedt2'].rank(method='min')
    mask_run = df['runsecs'] > 0
    df.loc[mask_run, 'position_at_run'] = df.loc[mask_run].groupby(['event_id', 'prog_id'])['elapsedrun'].rank(method='min')

    # Calculate position changes between checkpoints (negative = gained positions)
    df['swim_to_t1_pos_change'] = df['position_at_t1'] - df['position_at_swim']
    df['t1_to_bike_pos_change'] = df['position_at_bike'] - df['position_at_t1']
    df['bike_to_t2_pos_change'] = df['position_at_t2'] - df['position_at_bike']
    df['t2_to_run_pos_change'] = df['position_at_run'] - df['position_at_t2']

    # Calculate individual split rankings (rank by split time, not cumulative time)
    print("Calculating individual split rankings...")

    mask_swim_rank = df['swimsecs'] > 0
    df.loc[mask_swim_rank, 'swimrank'] = df.loc[mask_swim_rank].groupby(['event_id', 'prog_id'])['swimsecs'].rank(method='min')
    mask_t1_rank = df['t1secs'] > 0
    df.loc[mask_t1_rank, 't1rank'] = df.loc[mask_t1_rank].groupby(['event_id', 'prog_id'])['t1secs'].rank(method='min')
    mask_bike_rank = df['bikesecs'] > 0
    df.loc[mask_bike_rank, 'bikerank'] = df.loc[mask_bike_rank].groupby(['event_id', 'prog_id'])['bikesecs'].rank(method='min')
    mask_t2_rank = df['t2secs'] > 0
    df.loc[mask_t2_rank, 't2rank'] = df.loc[mask_t2_rank].groupby(['event_id', 'prog_id'])['t2secs'].rank(method='min')
    mask_run_rank = df['runsecs'] > 0
    df.loc[mask_run_rank, 'runrank'] = df.loc[mask_run_rank].groupby(['event_id', 'prog_id'])['runsecs'].rank(method='min')

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

# Save the calculated metrics to a database or file
def main():
    df = calculate_position_metrics()
    if df.empty:
        print("No valid data available for metrics calculation.")
        return

    engine = get_engine()
    df.to_sql('position_metrics', engine, if_exists='replace', index=False, method='multi')

if __name__ == "__main__":
    main()

