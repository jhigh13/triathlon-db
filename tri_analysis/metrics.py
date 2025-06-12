from database import get_engine
import pandas as pd

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
    df['ElapsedSwim'] = df['SwimTime'].apply(parse_time_to_secs)
    df['ElapsedT1'] = df['ElapsedSwim'] + df['T1Time'].apply(parse_time_to_secs)
    df['ElapsedBike'] = df['ElapsedT1'] + df['BikeTime'].apply(parse_time_to_secs)
    df['ElapsedT2'] = df['ElapsedBike'] + df['T2Time'].apply(parse_time_to_secs)
    df['ElapsedRun'] = df['ElapsedT2'] + df['RunTime'].apply(parse_time_to_secs)

    # Ensure all elapsed columns are int64 and no NaN/inf
    for col in ['ElapsedSwim','ElapsedT1','ElapsedBike','ElapsedT2','ElapsedRun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Parse individual split seconds for filtering
    df['SwimSecs'] = df['SwimTime'].apply(parse_time_to_secs)
    df['T1Secs'] = df['T1Time'].apply(parse_time_to_secs)
    df['BikeSecs'] = df['BikeTime'].apply(parse_time_to_secs)
    df['T2Secs'] = df['T2Time'].apply(parse_time_to_secs)
    df['RunSecs'] = df['RunTime'].apply(parse_time_to_secs)

    # Compute seconds behind leader per event_id + prog_id
    min_swim = df[df['SwimSecs'] > 0].groupby(['event_id','prog_id'])['ElapsedSwim'].transform('min').fillna(0)
    df['BehindSwim'] = df['ElapsedSwim'] - min_swim
    min_t1 = df[df['T1Secs'] > 0].groupby(['event_id','prog_id'])['ElapsedT1'].transform('min').fillna(0)
    df['BehindT1'] = df['ElapsedT1'] - min_t1
    min_bike = df[df['BikeSecs'] > 0].groupby(['event_id','prog_id'])['ElapsedBike'].transform('min').fillna(0)
    df['BehindBike'] = df['ElapsedBike'] - min_bike
    min_t2 = df[df['T2Secs'] > 0].groupby(['event_id','prog_id'])['ElapsedT2'].transform('min').fillna(0)
    df['BehindT2'] = df['ElapsedT2'] - min_t2
    min_run = df[df['RunSecs'] > 0].groupby(['event_id','prog_id'])['ElapsedRun'].transform('min').fillna(0)
    df['BehindRun'] = df['ElapsedRun'] - min_run

    # Calculate positions at each checkpoint (only for athletes with non-zero times)
    mask_swim = df['SwimSecs'] > 0
    df.loc[mask_swim, 'Position_at_Swim'] = df.loc[mask_swim].groupby(['event_id', 'prog_id'])['ElapsedSwim'].rank(method='min')
    mask_t1 = df['T1Secs'] > 0
    df.loc[mask_t1, 'Position_at_T1'] = df.loc[mask_t1].groupby(['event_id', 'prog_id'])['ElapsedT1'].rank(method='min')
    mask_bike = df['BikeSecs'] > 0
    df.loc[mask_bike, 'Position_at_Bike'] = df.loc[mask_bike].groupby(['event_id', 'prog_id'])['ElapsedBike'].rank(method='min')
    mask_t2 = df['T2Secs'] > 0
    df.loc[mask_t2, 'Position_at_T2'] = df.loc[mask_t2].groupby(['event_id', 'prog_id'])['ElapsedT2'].rank(method='min')
    mask_run = df['RunSecs'] > 0
    df.loc[mask_run, 'Position_at_Run'] = df.loc[mask_run].groupby(['event_id', 'prog_id'])['ElapsedRun'].rank(method='min')

    # Calculate position changes between checkpoints (negative = gained positions)
    df['Swim_to_T1_pos_change'] = df['Position_at_T1'] - df['Position_at_Swim']
    df['T1_to_Bike_pos_change'] = df['Position_at_Bike'] - df['Position_at_T1']
    df['Bike_to_T2_pos_change'] = df['Position_at_T2'] - df['Position_at_Bike']
    df['T2_to_Run_pos_change'] = df['Position_at_Run'] - df['Position_at_T2']

    # Calculate individual split rankings (rank by split time, not cumulative time)
    print("Calculating individual split rankings...")

    mask_swim_rank = df['SwimSecs'] > 0
    df.loc[mask_swim_rank, 'SwimRank'] = df.loc[mask_swim_rank].groupby(['event_id', 'prog_id'])['SwimSecs'].rank(method='min')
    mask_t1_rank = df['T1Secs'] > 0
    df.loc[mask_t1_rank, 'T1Rank'] = df.loc[mask_t1_rank].groupby(['event_id', 'prog_id'])['T1Secs'].rank(method='min')
    mask_bike_rank = df['BikeSecs'] > 0
    df.loc[mask_bike_rank, 'BikeRank'] = df.loc[mask_bike_rank].groupby(['event_id', 'prog_id'])['BikeSecs'].rank(method='min')
    mask_t2_rank = df['T2Secs'] > 0
    df.loc[mask_t2_rank, 'T2Rank'] = df.loc[mask_t2_rank].groupby(['event_id', 'prog_id'])['T2Secs'].rank(method='min')
    mask_run_rank = df['RunSecs'] > 0
    df.loc[mask_run_rank, 'RunRank'] = df.loc[mask_run_rank].groupby(['event_id', 'prog_id'])['RunSecs'].rank(method='min')

    position_cols = ['Position_at_Swim', 'Position_at_T1', 'Position_at_Bike', 'Position_at_T2', 'Position_at_Run',
                     'Swim_to_T1_pos_change', 'T1_to_Bike_pos_change', 'Bike_to_T2_pos_change', 'T2_to_Run_pos_change',
                     'SwimRank', 'T1Rank', 'BikeRank', 'T2Rank', 'RunRank']

    for col in position_cols:
        df[col] = df[col].astype('Int64')  # Nullable integer type

    for col in ['BehindSwim','BehindT1','BehindBike','BehindT2','BehindRun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Drop temporary split-second columns and any columns not in the new schema
    df.drop(
        columns=[
            'SwimSecs','T1Secs','BikeSecs','T2Secs','RunSecs',
            # Remove old/extra columns if present
            "position","total_time","start_num",
            "SwimTime", "T1Time", "BikeTime", "T2Time", "RunTime", "athlete_full_name"
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

    # Upload only the first row to position_metrics table for debugging
    engine = get_engine()
    df.to_sql('position_metrics', engine, if_exists='append', index=False, method='multi')

if __name__ == "__main__":
    main()

