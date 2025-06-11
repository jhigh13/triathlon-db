from Data_Import.database import get_engine
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
    df['ElapsedT1'] = df['ElapsedSwim'] + df['T1'].apply(parse_time_to_secs)
    df['ElapsedBike'] = df['ElapsedT1'] + df['BikeTime'].apply(parse_time_to_secs)
    df['ElapsedT2'] = df['ElapsedBike'] + df['T2'].apply(parse_time_to_secs)
    df['ElapsedRun'] = df['ElapsedT2'] + df['RunTime'].apply(parse_time_to_secs)

    # Ensure all elapsed columns are int64 and no NaN/inf
    for col in ['ElapsedSwim','ElapsedT1','ElapsedBike','ElapsedT2','ElapsedRun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Parse individual split seconds for filtering
    df['SwimSecs'] = df['SwimTime'].apply(parse_time_to_secs)
    df['T1Secs'] = df['T1'].apply(parse_time_to_secs)
    df['BikeSecs'] = df['BikeTime'].apply(parse_time_to_secs)
    df['T2Secs'] = df['T2'].apply(parse_time_to_secs)
    df['RunSecs'] = df['RunTime'].apply(parse_time_to_secs)

    # Compute seconds behind leader per EventID + Program
    # Swim
    min_swim = df[df['SwimSecs'] > 0].groupby(['EventID','Program'])['ElapsedSwim'].transform('min').fillna(0)
    df['BehindSwim'] = df['ElapsedSwim'] - min_swim
    # T1
    min_t1 = df[df['T1Secs'] > 0].groupby(['EventID','Program'])['ElapsedT1'].transform('min').fillna(0)
    df['BehindT1'] = df['ElapsedT1'] - min_t1
    # Bike
    min_bike = df[df['BikeSecs'] > 0].groupby(['EventID','Program'])['ElapsedBike'].transform('min').fillna(0)
    df['BehindBike'] = df['ElapsedBike'] - min_bike
    # T2
    min_t2 = df[df['T2Secs'] > 0].groupby(['EventID','Program'])['ElapsedT2'].transform('min').fillna(0)
    df['BehindT2'] = df['ElapsedT2'] - min_t2
    # Run
    min_run = df[df['RunSecs'] > 0].groupby(['EventID','Program'])['ElapsedRun'].transform('min').fillna(0)
    df['BehindRun'] = df['ElapsedRun'] - min_run

    # Calculate positions at each checkpoint (only for athletes with non-zero times)
    # Position at Swim
    mask_swim = df['SwimSecs'] > 0
    df.loc[mask_swim, 'Position_at_Swim'] = df.loc[mask_swim].groupby(['EventID', 'Program'])['ElapsedSwim'].rank(method='min')
    
    # Position at T1
    mask_t1 = df['T1Secs'] > 0
    df.loc[mask_t1, 'Position_at_T1'] = df.loc[mask_t1].groupby(['EventID', 'Program'])['ElapsedT1'].rank(method='min')
    
    # Position at Bike
    mask_bike = df['BikeSecs'] > 0
    df.loc[mask_bike, 'Position_at_Bike'] = df.loc[mask_bike].groupby(['EventID', 'Program'])['ElapsedBike'].rank(method='min')
    
    # Position at T2
    mask_t2 = df['T2Secs'] > 0
    df.loc[mask_t2, 'Position_at_T2'] = df.loc[mask_t2].groupby(['EventID', 'Program'])['ElapsedT2'].rank(method='min')
    
    # Position at Run/Finish
    mask_run = df['RunSecs'] > 0
    df.loc[mask_run, 'Position_at_Run'] = df.loc[mask_run].groupby(['EventID', 'Program'])['ElapsedRun'].rank(method='min')
    
    # Calculate position changes between checkpoints (negative = gained positions)
    df['Swim_to_T1_pos_change'] = df['Position_at_T1'] - df['Position_at_Swim']
    df['T1_to_Bike_pos_change'] = df['Position_at_Bike'] - df['Position_at_T1']
    df['Bike_to_T2_pos_change'] = df['Position_at_T2'] - df['Position_at_Bike']
    df['T2_to_Run_pos_change'] = df['Position_at_Run'] - df['Position_at_T2']
    
    # Calculate individual split rankings (rank by split time, not cumulative time)
    print("Calculating individual split rankings...")
    
    # Swim ranking (rank by SwimSecs, only athletes with SwimSecs > 0)
    mask_swim_rank = df['SwimSecs'] > 0
    df.loc[mask_swim_rank, 'SwimRank'] = df.loc[mask_swim_rank].groupby(['EventID', 'Program'])['SwimSecs'].rank(method='min')
    
    # T1 ranking (rank by T1Secs, only athletes with T1Secs > 0)
    mask_t1_rank = df['T1Secs'] > 0
    df.loc[mask_t1_rank, 'T1Rank'] = df.loc[mask_t1_rank].groupby(['EventID', 'Program'])['T1Secs'].rank(method='min')
    
    # Bike ranking (rank by BikeSecs, only athletes with BikeSecs > 0)
    mask_bike_rank = df['BikeSecs'] > 0
    df.loc[mask_bike_rank, 'BikeRank'] = df.loc[mask_bike_rank].groupby(['EventID', 'Program'])['BikeSecs'].rank(method='min')
    
    # T2 ranking (rank by T2Secs, only athletes with T2Secs > 0)
    mask_t2_rank = df['T2Secs'] > 0
    df.loc[mask_t2_rank, 'T2Rank'] = df.loc[mask_t2_rank].groupby(['EventID', 'Program'])['T2Secs'].rank(method='min')
    
    # Run ranking (rank by RunSecs, only athletes with RunSecs > 0)
    mask_run_rank = df['RunSecs'] > 0
    df.loc[mask_run_rank, 'RunRank'] = df.loc[mask_run_rank].groupby(['EventID', 'Program'])['RunSecs'].rank(method='min')
    
    # Convert position columns to nullable integers (Int64) to handle NaN values properly
    position_cols = ['Position_at_Swim', 'Position_at_T1', 'Position_at_Bike', 'Position_at_T2', 'Position_at_Run',
                     'Swim_to_T1_pos_change', 'T1_to_Bike_pos_change', 'Bike_to_T2_pos_change', 'T2_to_Run_pos_change',
                     'SwimRank', 'T1Rank', 'BikeRank', 'T2Rank', 'RunRank']
    
    for col in position_cols:
        df[col] = df[col].astype('Int64')  # Nullable integer type

    # Ensure behind columns are int64
    for col in ['BehindSwim','BehindT1','BehindBike','BehindT2','BehindRun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    
    # Drop temporary split-second columns
    df.drop(columns=['ProgID','Program','CategoryName','EventSpecifications',"Position","TotalTime","SwimTime",
             "T1", "BikeTime", "T2", "RunTime",'SwimSecs','T1Secs','BikeSecs','T2Secs','RunSecs'], inplace=True)
    return df

# Save the calculated metrics to a database or file
def main():
    df = calculate_position_metrics()
    
    if df.empty:
        print("No valid data available for metrics calculation.")
        return

    # Save the DataFrame to a database or file
    save_metrics_to_database(df)
