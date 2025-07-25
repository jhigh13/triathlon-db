import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import openpyxl

# ===== SHARED UTILITY FUNCTIONS =====

@st.cache_data(ttl=600)
def load_athlete_names(_engine):
    """Cached load of athlete full names from database"""
    return pd.read_sql("SELECT full_name FROM athlete ORDER BY full_name", _engine)

@st.cache_data(ttl=600)
def load_event_names(_engine):
    """Cached load of distinct event names from database"""
    return pd.read_sql("SELECT DISTINCT event_name FROM events ORDER BY event_name", _engine)

def time_to_seconds(timestr):
    """Convert a time string (HH:MM:SS or MM:SS or SS) to seconds (int). Handles None/NaN."""
    if pd.isna(timestr) or timestr is None:
        return None
    if isinstance(timestr, (int, float)):
        return int(timestr)
    parts = str(timestr).split(":")
    try:
        if len(parts) == 3:
            h, m, s = [int(float(p)) for p in parts]
            return h*3600 + m*60 + s
        elif len(parts) == 2:
            m, s = [int(float(p)) for p in parts]
            return m*60 + s
        elif len(parts) == 1:
            return int(float(parts[0]))
    except Exception:
        return None
    return None

def seconds_to_hms(seconds):
    """Convert seconds (int) to MM:SS string, preserving sign for negatives."""
    if pd.isna(seconds):
        return None
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(round(seconds)))
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{sign}{m:02}:{s:02}"

# Cache H2H summary to avoid repeated DB calls for same selections
@st.cache_data(show_spinner=False, persist=False)
def h2h_summary(athletes, events, _engine) -> pd.DataFrame: 
  
  # Load athlete and event lookup for filtering
  df_athletes = pd.read_sql("SELECT athlete_id, full_name FROM athlete", _engine)
  df_events = pd.read_sql("SELECT event_id, event_name, prog_id, prog_name FROM events", _engine)
  df_events = df_events[df_events['prog_name'] != "Mixed Relay"]
  athlete_ids = df_athletes[df_athletes['full_name'].isin(athletes)]['athlete_id'].tolist()
  event_ids = df_events[df_events['event_name'].isin(events)]['event_id'].tolist()

  # Load only relevant race_results and metrics
  df = pd.read_sql(
      f"""
      SELECT
          rr.event_id,
          rr.athlete_id,
          rr.prog_id,
          rr.position,
          rr.swimtime,
          rr.t1time,
          rr.biketime,
          rr.t2time,
          rr.runtime,
          pm.elapsedrun
      FROM race_results rr
      LEFT JOIN position_metrics pm
          ON rr.event_id = pm.event_id AND rr.athlete_id = pm.athlete_id AND rr.prog_id = pm.prog_id
      WHERE rr.athlete_id IN ({','.join(map(str, athlete_ids))})
        AND rr.event_id IN ({','.join(map(str, event_ids))})
      """ , _engine
  )
  
  # Load split rankings filtered by selection
  df_metrics = pd.read_sql(
      f"""
      SELECT
          event_id,
          athlete_id,
          prog_id,
          swimrank,
          t1rank,
          bikerank,
          t2rank,
          runrank
      FROM position_metrics
      WHERE athlete_id IN ({','.join(map(str, athlete_ids))})
        AND event_id IN ({','.join(map(str, event_ids))})
      """ , _engine
  )

  # Convert segment times to seconds
  for col in ['swimtime','t1time','biketime','t2time','runtime']:
      df[col + '_sec'] = df[col].apply(time_to_seconds)

  # Merge split rankings into main race results DataFrame
  df = df.merge(df_metrics, on=["event_id", "athlete_id", "prog_id"], how="left")

  # Merge athlete names
  df = df.merge(df_athletes, on="athlete_id", how="left")
  df = df.merge(df_events[['event_id', 'prog_id', 'prog_name']], on=['event_id', 'prog_id'], how='left')
  df = df[df['prog_name'] != "Mixed Relay"]

  # ----- FILTERING ------  
  athlete_ids = df_athletes[df_athletes['full_name'].isin(athletes)]['athlete_id'].tolist()
  df = df[df['athlete_id'].isin(athlete_ids)]
  event_ids = df_events[df_events['event_name'].isin(events)]['event_id'].tolist()
  df = df[df['event_id'].isin(event_ids)]

  # create every A vs B in the same event/program
  df_pairs = (
    df.merge(df, on=["prog_id"], suffixes=("_a", "_b"))
      .query("athlete_id_a < athlete_id_b")  # unordered pairs only once
)

  # Win flags for overall and each segment (renamed to match h2h_summary table)
  df_pairs["wins_a"]         = df_pairs["position_a"]   < df_pairs["position_b"]
  df_pairs["swim_wins_a"]    = df_pairs["swimrank_a"]    < df_pairs["swimrank_b"]
  df_pairs["t1_wins_a"]      = df_pairs["t1rank_a"]      < df_pairs["t1rank_b"]
  df_pairs["bike_wins_a"]    = df_pairs["bikerank_a"]    < df_pairs["bikerank_b"]
  df_pairs["t2_wins_a"]      = df_pairs["t2rank_a"]      < df_pairs["t2rank_b"]
  df_pairs["run_wins_a"]     = df_pairs["runrank_a"]     < df_pairs["runrank_b"]

  # --- Average Time Gap (elapsedrun) ---
  df_pairs["time_diff_sec"] = df_pairs["elapsedrun_a"] - df_pairs["elapsedrun_b"]

  # --- Segment time differences ---
  for seg in ['swimtime_sec','t1time_sec','biketime_sec','t2time_sec','runtime_sec']:
      df_pairs[f'{seg}_diff'] = df_pairs[f'{seg}_a'] - df_pairs[f'{seg}_b']

  # Aggregate H2H metrics by athlete pair
  agg_dict = {
      "matches": ("prog_id", "count"),
      "wins_a": ("wins_a", "sum"),
      "swim_wins_a": ("swim_wins_a", "sum"),
      "t1_wins_a": ("t1_wins_a", "sum"),
      "bike_wins_a": ("bike_wins_a", "sum"),
      "t2_wins_a": ("t2_wins_a", "sum"),
      "run_wins_a": ("run_wins_a", "sum"),
      "avg_time_diff_sec": ("time_diff_sec", "mean"),
      "min_time_diff_sec": ("time_diff_sec", "min"),
      "max_time_diff_sec": ("time_diff_sec", "max"),
  }
  # Add segment avg diff columns
  for seg in ['swimtime_sec','t1time_sec','biketime_sec','t2time_sec','runtime_sec']:
      agg_dict[f'avg_{seg}_diff'] = (f'{seg}_diff', 'mean')
      agg_dict[f'min_{seg}_diff'] = (f'{seg}_diff', 'min')
      agg_dict[f'max_{seg}_diff'] = (f'{seg}_diff', 'max')

  h2h = (
    df_pairs
      .groupby(["athlete_id_a", "athlete_id_b"])
      .agg(**agg_dict)
      .reset_index()
)

  # Calculate win percentages (renamed to match h2h_summary)
  h2h["win_pct_a"]        = h2h["wins_a"]      / h2h["matches"]
  h2h["swim_win_pct_a"]  = h2h["swim_wins_a"] / h2h["matches"]
  h2h["t1_win_pct_a"]    = h2h["t1_wins_a"]   / h2h["matches"]
  h2h["bike_win_pct_a"]  = h2h["bike_wins_a"] / h2h["matches"]
  h2h["t2_win_pct_a"]    = h2h["t2_wins_a"]   / h2h["matches"]
  h2h["run_win_pct_a"]   = h2h["run_wins_a"]  / h2h["matches"]

  # Compute wins and win percentages for athlete B (renamed to match h2h_summary)
  h2h['wins_b']        = h2h['matches'] - h2h['wins_a']
  h2h['swim_wins_b']   = h2h['matches'] - h2h['swim_wins_a']
  h2h['t1_wins_b']     = h2h['matches'] - h2h['t1_wins_a']
  h2h['bike_wins_b']   = h2h['matches'] - h2h['bike_wins_a']
  h2h['t2_wins_b']     = h2h['matches'] - h2h['t2_wins_a']
  h2h['run_wins_b']    = h2h['matches'] - h2h['run_wins_a']

  # Compute win percentage for athlete B (renamed to match h2h_summary)
  h2h['win_pct_b']        = h2h['wins_b']      / h2h['matches']
  h2h['swim_win_pct_b']   = h2h['swim_wins_b'] / h2h['matches']
  h2h['t1_win_pct_b']     = h2h['t1_wins_b']   / h2h['matches']
  h2h['bike_win_pct_b']   = h2h['bike_wins_b'] / h2h['matches']
  h2h['t2_win_pct_b']     = h2h['t2_wins_b']   / h2h['matches']
  h2h['run_win_pct_b']    = h2h['run_wins_b']  / h2h['matches']

  # Map athlete_id to full_name for both athlete_id_a and athlete_id_b
  id_to_name = df_athletes.set_index('athlete_id')['full_name'].to_dict()
  h2h['athlete_name_a'] = h2h['athlete_id_a'].map(id_to_name)
  h2h['athlete_name_b'] = h2h['athlete_id_b'].map(id_to_name)

  # Add formatted time gap (HH:MM:SS) for overall and segments
  h2h['avg_time_diff'] = h2h['avg_time_diff_sec'].apply(seconds_to_hms)
  h2h['min_time_diff'] = h2h['min_time_diff_sec'].apply(seconds_to_hms)
  h2h['max_time_diff'] = h2h['max_time_diff_sec'].apply(seconds_to_hms)
  for seg in ['swimtime_sec','t1time_sec','biketime_sec','t2time_sec','runtime_sec']:
      h2h[f'avg_{seg}_diff_fmt'] = h2h[f'avg_{seg}_diff'].apply(seconds_to_hms)
      h2h[f'min_{seg}_diff_fmt'] = h2h[f'min_{seg}_diff'].apply(seconds_to_hms)
      h2h[f'max_{seg}_diff_fmt'] = h2h[f'max_{seg}_diff'].apply(seconds_to_hms)

  return h2h

# Utility to color heatmap cells
def color_code(val):
    if val == '-':
        return 0
    if not isinstance(val, str) or '-' not in val:
        return 0
    wins, losses = val.split('-')[0:2]
    if not (wins.isdigit() and losses.isdigit()):
        return 0
    w, l = int(wins), int(losses)
    return 1 if w > l else (-1 if w < l else 0)

# --- Chart Functions ---
def build_overall_matrix(h2h_df):
    athletes = sorted(set(h2h_df['athlete_name_a']).union(h2h_df['athlete_name_b']))
    matrix = pd.DataFrame('-', index=athletes, columns=athletes)
    annot_matrix = pd.DataFrame('-', index=athletes, columns=athletes)
    for _, row in h2h_df.iterrows():
        a = row['athlete_name_a']
        b = row['athlete_name_b']
        matches = int(row['matches'])
        wins_a = int(row['wins_a'])
        losses_a = matches - wins_a
        wins_b = int(row['wins_b'])
        losses_b = matches - wins_b
        time_gap = row['avg_time_diff'] if pd.notna(row['avg_time_diff']) else ''
        time_gap_b = '-' if pd.isna(row['avg_time_diff_sec']) else seconds_to_hms(-row['avg_time_diff_sec'])
        min_time_gap = row['min_time_diff'] if pd.notna(row['min_time_diff']) else ''
        min_time_gap_b = '-' if pd.isna(row['min_time_diff_sec']) else seconds_to_hms(-row['min_time_diff_sec'])
        max_time_gap = row['max_time_diff'] if pd.notna(row['max_time_diff']) else ''
        max_time_gap_b = '-' if pd.isna(row['max_time_diff_sec']) else seconds_to_hms(-row['max_time_diff_sec'])    
        if matches > 0:
            matrix.loc[a, b] = f"{wins_a}-{losses_a}"
            matrix.loc[b, a] = f"{wins_b}-{losses_b}"
            annot_matrix.loc[a, b] = f"Record: {wins_a}-{losses_a}\nAvg. Gap: {time_gap}"
            annot_matrix.loc[b, a] = f"Record: {wins_b}-{losses_b}\nAvg. Gap: {time_gap_b}"
            annot_matrix.loc[a, b] += f"\nMin Gap: {min_time_gap}\nMax Gap: {max_time_gap}"
            annot_matrix.loc[b, a] += f"\nMin Gap: {min_time_gap_b}\nMax Gap: {max_time_gap_b}"
    for name in athletes:
        matrix.loc[name, name] = '-'
        annot_matrix.loc[name, name] = '-'
    return matrix, annot_matrix

def build_segment_matrix(h2h_df, segment):
    # segment: one of 'swim', 't1', 'bike', 't2', 'run'
    seg_map = {
        'swim': 'swimtime_sec',
        't1': 't1time_sec',
        'bike': 'biketime_sec',
        't2': 't2time_sec',
        'run': 'runtime_sec',
    }
    win_col = f'{segment}_wins_a'
    avg_gap_col = f'avg_{seg_map[segment]}_diff_fmt'
    min_gap_col = f'min_{seg_map[segment]}_diff_fmt'
    max_gap_col = f'max_{seg_map[segment]}_diff_fmt'
    athletes = sorted(set(h2h_df['athlete_name_a']).union(h2h_df['athlete_name_b']))
    matrix = pd.DataFrame('-', index=athletes, columns=athletes)
    annot_matrix = pd.DataFrame('-', index=athletes, columns=athletes)
    for _, row in h2h_df.iterrows():
        a = row['athlete_name_a']
        b = row['athlete_name_b']
        matches = int(row['matches'])
        wins_a = int(row[win_col]) if win_col in row else 0
        losses_a = matches - wins_a
        wins_b = matches - wins_a
        losses_b = wins_a
        avg_gap = row[avg_gap_col] if pd.notna(row[avg_gap_col]) else ''
        avg_gap_b = '-' if pd.isna(row[avg_gap_col]) else seconds_to_hms(-time_to_seconds(row[avg_gap_col]))
        min_gap = row[min_gap_col] if pd.notna(row[min_gap_col]) else ''
        min_gap_b = '-' if pd.isna(row[min_gap_col]) else seconds_to_hms(-time_to_seconds(row[min_gap_col]))
        max_gap = row[max_gap_col] if pd.notna(row[max_gap_col]) else ''
        max_gap_b = '-' if pd.isna(row[max_gap_col]) else seconds_to_hms(-time_to_seconds(row[max_gap_col]))
        if matches > 0:
            matrix.loc[a, b] = f"{wins_a}-{losses_a}"
            matrix.loc[b, a] = f"{wins_b}-{losses_b}"
            annot_matrix.loc[a, b] = f"Record: {wins_a}-{losses_a}\nAvg. Gap: {avg_gap}"
            annot_matrix.loc[b, a] = f"Record: {wins_b}-{losses_b}\nAvg. Gap: {avg_gap_b}"
            annot_matrix.loc[a, b] += f"\nMin Gap: {min_gap}\nMax Gap: {max_gap}"
            annot_matrix.loc[b, a] += f"\nMin Gap: {min_gap_b}\nMax Gap: {max_gap_b}"
    for name in athletes:
        matrix.loc[name, name] = '-'
        annot_matrix.loc[name, name] = '-'
    return matrix, annot_matrix

def plot_heatmap(mat, annot, title):
    z = mat.map(color_code).values
    n = len(mat)
    # Make heatmap smaller and more reasonable for web display
    size = max(6, min(8, n * 1.5))  # Between 6-10 inches, much smaller scaling
    fig, ax = plt.subplots(figsize=(size, size))
    cmap = sns.color_palette(["#dc3545", "white", "#28a745"])
    sns.heatmap(
        z,
        annot=annot.values,
        fmt='',
        cmap=cmap,
        cbar=False,
        xticklabels=mat.index,
        yticklabels=mat.index,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": max(8, 11 - n * 0.5)},  # Smaller text for larger matrices
        ax=ax
    )
    ax.set_xlabel('Athlete B')
    ax.set_ylabel('Athlete A')
    ax.set_title(title)
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)

# ===== PACK DYNAMICS FUNCTIONS =====

def convert_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS or MM:SS) to total seconds"""
    if pd.isna(time_str):
        return np.nan
    
    try:
        # Handle different time formats
        if isinstance(time_str, str):
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 1:  # SS
                return int(parts[0])
        elif isinstance(time_str, (int, float)):
            return float(time_str)
    except:
        return np.nan
    return np.nan

def identify_packs(times, max_gap_to_leader=2, max_gap_within_pack=1):
    """
    Identify packs based on time gaps between consecutive athletes
    
    Parameters:
    times: array of elapsed times (sorted)
    gap_threshold: maximum gap in seconds to be considered same pack
    
    Returns: array of pack IDs (starting from 0), -1 for DNF/lapped (NaN)
    """
    if len(times) == 0:
        return np.array([])

    pack_ids = np.full(len(times), -1, dtype=int)
    current_pack = 0

    # Get indices of valid (non-NaN) times
    valid_indices = [i for i, t in enumerate(times) if pd.notna(t)]
    if not valid_indices:
        return pack_ids

    i = 0
    while i < len(valid_indices):
        leader_idx = valid_indices[i]
        pack_ids[leader_idx] = current_pack
        pack_leader_time = times[leader_idx]
        j = i + 1
        while j < len(valid_indices):
            curr_idx = valid_indices[j]
            prev_idx = valid_indices[j-1]
            gap_to_prev = times[curr_idx] - times[prev_idx]
            gap_to_leader = times[curr_idx] - pack_leader_time
            if gap_to_prev <= max_gap_within_pack or gap_to_leader <= max_gap_to_leader:
                pack_ids[curr_idx] = current_pack
                j += 1
            else:
                break
        current_pack += 1
        i = j

    return pack_ids

def calculate_elapsed_times(df):
    """Calculate cumulative elapsed times with DNF handling"""
    df_elapsed = df.copy()
    
    # Convert all time columns to seconds
    time_cols = ['S1', 'T1', 'B1T1', 'B1T2', 'BL1', 'B2T1', 'B2T2', 'BL2', 
                 'B3T1', 'B3T2', 'BL3', 'B4T1', 'B4T2', 'BL4', 'B5T1', 'B5T2', 
                 'BL5', 'B6T1', 'B6T2', 'BL6', 'T2', 'RT1', 'RL1', 'RT2', 'RL2']
    
    # Convert times to seconds
    for col in time_cols:
        if col in df_elapsed.columns:
            df_elapsed[f'{col}_sec'] = df_elapsed[col].apply(convert_time_to_seconds)
    
    # Calculate key elapsed time checkpoints
    checkpoints = {}
    
    # After swim
    if 'S1_sec' in df_elapsed.columns:
        checkpoints['Elapsed_After_Swim'] = df_elapsed['S1_sec'].copy()
    
    # After T1
    if 'T1_sec' in df_elapsed.columns and 'S1_sec' in df_elapsed.columns:
        elapsed_after_t1 = df_elapsed['S1_sec'] + df_elapsed['T1_sec']
        elapsed_after_t1 = elapsed_after_t1.where(
            df_elapsed['S1_sec'].notna() & df_elapsed['T1_sec'].notna(),
            np.nan
        )
        checkpoints['Elapsed_After_T1'] = elapsed_after_t1
    elif 'Elapsed_After_Swim' in checkpoints:
        checkpoints['Elapsed_After_T1'] = checkpoints['Elapsed_After_Swim'].copy()
    
    # Add bike laps (simplified for Hamburg data)
    elapsed_so_far = checkpoints.get('Elapsed_After_T1', checkpoints.get('Elapsed_After_Swim'))
    if elapsed_so_far is not None:
        for lap_num in range(1, 7):  # Bike laps 1-6
            bike_segments = [f'B{lap_num}T1_sec', f'B{lap_num}T2_sec', f'BL{lap_num}_sec']
            
            lap_elapsed_time = pd.Series(0.0, index=df_elapsed.index)
            lap_has_data = pd.Series(True, index=df_elapsed.index)
            
            for seg in bike_segments:
                if seg in df_elapsed.columns:
                    segment_valid = df_elapsed[seg].notna()
                    lap_has_data = lap_has_data & segment_valid
                    lap_elapsed_time = lap_elapsed_time + df_elapsed[seg].fillna(0)
            
            elapsed_so_far = elapsed_so_far + lap_elapsed_time
            elapsed_so_far = elapsed_so_far.where(
                elapsed_so_far.isna() | lap_has_data,
                np.nan
            )
            
            checkpoints[f'Elapsed_After_Bike_Lap_{lap_num}'] = elapsed_so_far.copy()
    
    # After T2
    if 'T2_sec' in df_elapsed.columns and elapsed_so_far is not None:
        elapsed_after_t2 = elapsed_so_far + df_elapsed['T2_sec']
        elapsed_after_t2 = elapsed_after_t2.where(
            elapsed_so_far.isna() | df_elapsed['T2_sec'].notna(),
            np.nan
        )
        elapsed_so_far = elapsed_after_t2
        checkpoints['Elapsed_After_T2'] = elapsed_so_far.copy()
    
    # Add run segments
    run_segments = ['RT1_sec', 'RL1_sec', 'RT2_sec', 'RL2_sec']
    for i, seg in enumerate(run_segments):
        if seg in df_elapsed.columns and elapsed_so_far is not None:
            has_run_data = df_elapsed[seg].notna()
            elapsed_so_far = elapsed_so_far + df_elapsed[seg].fillna(0)
            elapsed_so_far = elapsed_so_far.where(
                elapsed_so_far.isna() | has_run_data,
                np.nan
            )
            checkpoints[f'Elapsed_After_Run_Seg_{i+1}'] = elapsed_so_far.copy()
    
    # Add checkpoint columns to dataframe
    for checkpoint, times in checkpoints.items():
        df_elapsed[checkpoint] = times
    
    return df_elapsed, checkpoints

def analyze_packs_at_checkpoint(df, checkpoint_col, max_gap_to_leader=2, max_gap_within_pack=1):
    """Analyze pack composition at a specific checkpoint"""
    if checkpoint_col not in df.columns:
        return pd.DataFrame(), {}
    
    # Get valid data and sort by elapsed time
    valid_data = df[df[checkpoint_col].notna()].copy()
    if len(valid_data) == 0:
        return pd.DataFrame(), {}
    
    valid_data = valid_data.sort_values(checkpoint_col).reset_index(drop=True)
    
    # Identify packs
    pack_ids = identify_packs(valid_data[checkpoint_col].values, max_gap_to_leader, max_gap_within_pack)
    valid_data['Pack_ID'] = pack_ids
    
    # Create pack statistics
    pack_stats = {}
    for pack_id in sorted(valid_data['Pack_ID'].unique()):
        if pack_id == -1:  # DNF athletes
            continue
            
        pack_athletes = valid_data[valid_data['Pack_ID'] == pack_id]
        fastest_time = pack_athletes[checkpoint_col].min()
        slowest_time = pack_athletes[checkpoint_col].max()
        time_spread = slowest_time - fastest_time
        
        pack_stats[f'Pack {pack_id + 1}'] = {
            'Size': len(pack_athletes),
            'Fastest_Time': fastest_time,
            'Slowest_Time': slowest_time,
            'Time_Spread': time_spread,
            'Athletes': pack_athletes['Name'].tolist() if 'Name' in pack_athletes.columns else []
        }
    
    return valid_data, pack_stats

def create_pack_composition_table(pack_data, pack_stats):
    """Create a formatted table showing pack composition"""
    if pack_data.empty:
        return pd.DataFrame()
    
    # Helper function to convert seconds to MM:SS format
    def seconds_to_time_format(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    # Create summary table
    summary_data = []
    for pack_name, stats in pack_stats.items():
        summary_data.append({
            'Pack': pack_name,
            'Size': stats['Size'],
            'Spread': f"{stats['Time_Spread']:.1f}s",
            'Fastest Time': seconds_to_time_format(stats['Fastest_Time']),
            'Athletes': ', '.join(stats['Athletes'])  # Show all athletes, no truncation
        })
    
    df = pd.DataFrame(summary_data)
    # Reset index to remove the default 0, 1, 2, 3... index column
    return df.reset_index(drop=True)

def create_position_tracking_chart(df, checkpoints, selected_athletes=None):
    """Create position tracking chart for selected athletes"""
    if not checkpoints or df.empty:
        return None
    
    # Filter to selected athletes if provided
    if selected_athletes:
        df_filtered = df[df['Name'].isin(selected_athletes)].copy()
    else:
        df_filtered = df.head(10).copy()  # Top 10 by default
    
    fig = go.Figure()
    
    checkpoint_names = list(checkpoints.keys())
    
    for _, athlete in df_filtered.iterrows():
        times = []
        positions = []
        checkpoint_labels = []
        
        for i, checkpoint in enumerate(checkpoint_names):
            if checkpoint in df.columns and pd.notna(athlete[checkpoint]):
                times.append(athlete[checkpoint])
                
                # Calculate position at this checkpoint
                valid_times = df[df[checkpoint].notna()][checkpoint]
                position = (valid_times <= athlete[checkpoint]).sum()
                positions.append(position)
                checkpoint_labels.append(checkpoint.replace('Elapsed_After_', ''))
        
        if times:
            fig.add_trace(go.Scatter(
                x=checkpoint_labels,
                y=positions,
                mode='lines+markers',
                name=athlete.get('Name', f"Athlete {athlete.get('Bib', 'Unknown')}"),
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Position Tracking Throughout Race",
        xaxis_title="Checkpoint",
        yaxis_title="Position",
        yaxis=dict(autorange='reversed'),  # Lower position numbers at top
        hovermode='x unified',
        height=500
    )
    
    return fig

# ===== PHASE 3 PACK DYNAMICS FUNCTIONS =====

def create_pack_evolution_timeline(df, checkpoints, max_gap_to_leader=2, max_gap_within_pack=1):
    """Create interactive timeline showing pack evolution throughout race"""
    if not checkpoints or df.empty:
        return None
    
    # Calculate pack assignments at each checkpoint
    pack_evolution_data = []
    checkpoint_names = list(checkpoints.keys())
    
    for checkpoint in checkpoint_names:
        if checkpoint not in df.columns:
            continue
            
        valid_data = df[df[checkpoint].notna()].copy()
        if len(valid_data) == 0:
            continue
            
        valid_data = valid_data.sort_values(checkpoint).reset_index(drop=True)
        pack_ids = identify_packs(valid_data[checkpoint].values, max_gap_to_leader, max_gap_within_pack)
        
        for idx, (_, athlete) in enumerate(valid_data.iterrows()):
            pack_evolution_data.append({
                'Checkpoint': checkpoint.replace('Elapsed_After_', ''),
                'Athlete': athlete.get('Name', f"Athlete {athlete.get('Bib', 'Unknown')}"),
                'Pack_ID': pack_ids[idx] if idx < len(pack_ids) else -1,
                'Elapsed_Time': athlete[checkpoint],
                'Position': idx + 1
            })
    
    if not pack_evolution_data:
        return None
    
    evolution_df = pd.DataFrame(pack_evolution_data)
    
    # Create timeline chart
    fig = go.Figure()
    
    # Get unique athletes and assign colors
    unique_athletes = evolution_df['Athlete'].unique()[:15]  # Limit to top 15 for readability
    colors = px.colors.qualitative.Set3[:len(unique_athletes)]
    
    for i, athlete in enumerate(unique_athletes):
        athlete_data = evolution_df[evolution_df['Athlete'] == athlete]
        
        fig.add_trace(go.Scatter(
            x=athlete_data['Checkpoint'],
            y=athlete_data['Pack_ID'],
            mode='lines+markers',
            name=athlete,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate=(
                f"<b>{athlete}</b><br>" +
                "Checkpoint: %{x}<br>" +
                "Pack: %{y}<br>" +
                "Position: %{customdata[0]}<br>" +
                "Time: %{customdata[1]:.1f}s<br>" +
                "<extra></extra>"
            ),
            customdata=athlete_data[['Position', 'Elapsed_Time']].values
        ))
    
    fig.update_layout(
        title="Pack Membership Evolution Throughout Race",
        xaxis_title="Race Checkpoint",
        yaxis_title="Pack Number",
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def create_advanced_gap_analysis(df, checkpoints):
    """Create advanced gap analysis charts showing race dynamics"""
    if not checkpoints or df.empty:
        return None
    
    checkpoint_names = list(checkpoints.keys())
    gap_data = []
    
    # Calculate gaps to leader at each checkpoint
    for checkpoint in checkpoint_names:
        if checkpoint not in df.columns:
            continue
            
        valid_data = df[df[checkpoint].notna()].copy()
        if len(valid_data) == 0:
            continue
            
        valid_data = valid_data.sort_values(checkpoint).reset_index(drop=True)
        leader_time = valid_data[checkpoint].iloc[0] if len(valid_data) > 0 else 0
        
        for idx, (_, athlete) in enumerate(valid_data.iterrows()):
            gap_to_leader = athlete[checkpoint] - leader_time
            gap_data.append({
                'Checkpoint': checkpoint.replace('Elapsed_After_', ''),
                'Athlete': athlete.get('Name', f"Athlete {athlete.get('Bib', 'Unknown')}"),
                'Gap_to_Leader': gap_to_leader,
                'Position': idx + 1
            })
    
    if not gap_data:
        return None
    
    gap_df = pd.DataFrame(gap_data)
    
    # Gap evolution chart (top 10 athletes)
    top_10_athletes = gap_df[gap_df['Checkpoint'] == gap_df['Checkpoint'].iloc[0]].head(10)['Athlete'].tolist()
    gap_evolution_fig = go.Figure()
    
    for athlete in top_10_athletes:
        athlete_gaps = gap_df[gap_df['Athlete'] == athlete]
        gap_evolution_fig.add_trace(go.Scatter(
            x=athlete_gaps['Checkpoint'],
            y=athlete_gaps['Gap_to_Leader'],
            mode='lines+markers',
            name=athlete,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    gap_evolution_fig.update_layout(
        title="Gap to Leader Evolution (Top 10)",
        xaxis_title="Checkpoint",
        yaxis_title="Gap to Leader (seconds)",
        hovermode='x unified',
        height=400
    )
    
    # Gap distribution at final checkpoint
    final_checkpoint = checkpoint_names[-1] if checkpoint_names else None
    if final_checkpoint and final_checkpoint in df.columns:
        final_gaps = gap_df[gap_df['Checkpoint'] == final_checkpoint.replace('Elapsed_After_', '')]
        
        gap_distribution_fig = px.histogram(
            final_gaps,
            x='Gap_to_Leader',
            nbins=20,
            title="Final Gap Distribution",
            labels={'Gap_to_Leader': 'Gap to Leader (seconds)', 'count': 'Number of Athletes'}
        )
        gap_distribution_fig.update_layout(height=400)
    else:
        gap_distribution_fig = None
    
    return {
        'gap_evolution': gap_evolution_fig,
        'gap_distribution': gap_distribution_fig
    }

def create_individual_athlete_analysis(df, checkpoints, athlete_name, max_gap_to_leader=2, max_gap_within_pack=1):
    """Create detailed analysis for individual athlete"""
    if not checkpoints or df.empty or athlete_name not in df['Name'].values:
        return None
    
    athlete_data = df[df['Name'] == athlete_name].iloc[0]
    checkpoint_names = list(checkpoints.keys())
    
    # Track pack membership and performance
    athlete_analysis_data = []
    
    for checkpoint in checkpoint_names:
        if checkpoint not in df.columns:
            continue
            
        # Get all athletes at this checkpoint
        valid_data = df[df[checkpoint].notna()].copy()
        if len(valid_data) == 0 or pd.isna(athlete_data[checkpoint]):
            continue
            
        valid_data = valid_data.sort_values(checkpoint).reset_index(drop=True)
        pack_ids = identify_packs(valid_data[checkpoint].values, max_gap_to_leader, max_gap_within_pack)
        
        # Find athlete's position and pack
        athlete_idx = valid_data[valid_data['Name'] == athlete_name].index
        if len(athlete_idx) > 0:
            athlete_idx = athlete_idx[0]
            athlete_pack = pack_ids[athlete_idx] if athlete_idx < len(pack_ids) else -1
            athlete_position = athlete_idx + 1
            leader_time = valid_data[checkpoint].iloc[0]
            gap_to_leader = athlete_data[checkpoint] - leader_time
            
            athlete_analysis_data.append({
                'Checkpoint': checkpoint.replace('Elapsed_After_', ''),
                'Pack_ID': athlete_pack,
                'Position': athlete_position,
                'Gap_to_Leader': gap_to_leader,
                'Elapsed_Time': athlete_data[checkpoint]
            })
    
    if not athlete_analysis_data:
        return None
    
    analysis_df = pd.DataFrame(athlete_analysis_data)
    
    # Pack membership chart
    pack_membership_fig = go.Figure()
    pack_membership_fig.add_trace(go.Scatter(
        x=analysis_df['Checkpoint'],
        y=analysis_df['Pack_ID'],
        mode='lines+markers',
        name=f"{athlete_name} Pack Membership",
        line=dict(width=3, color='blue'),
        marker=dict(size=10, color='red', symbol='diamond')
    ))
    pack_membership_fig.update_layout(
        title=f"{athlete_name} - Pack Membership Throughout Race",
        xaxis_title="Checkpoint",
        yaxis_title="Pack Number",
        height=300
    )
    
    # Gap to leader chart
    gap_to_leader_fig = go.Figure()
    gap_to_leader_fig.add_trace(go.Scatter(
        x=analysis_df['Checkpoint'],
        y=analysis_df['Gap_to_Leader'],
        mode='lines+markers',
        name=f"{athlete_name} Gap to Leader",
        line=dict(width=3, color='orange'),
        marker=dict(size=8, color='red')
    ))
    gap_to_leader_fig.update_layout(
        title=f"{athlete_name} - Gap to Leader Throughout Race",
        xaxis_title="Checkpoint",
        yaxis_title="Gap to Leader (seconds)",
        height=300
    )
    
    # Performance summary
    summary_data = {
        'Metric': [
            'Best Position',
            'Worst Position',
            'Position Change',
            'Smallest Gap to Leader',
            'Largest Gap to Leader',
            'Most Common Pack',
            'Pack Changes'
        ],
        'Value': [
            analysis_df['Position'].min(),
            analysis_df['Position'].max(),
            analysis_df['Position'].iloc[-1] - analysis_df['Position'].iloc[0] if len(analysis_df) > 1 else 0,
            f"{analysis_df['Gap_to_Leader'].min():.1f}s",
            f"{analysis_df['Gap_to_Leader'].max():.1f}s",
            analysis_df['Pack_ID'].mode().iloc[0] if len(analysis_df['Pack_ID'].mode()) > 0 else 'N/A',
            len(analysis_df['Pack_ID'].unique())
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    return {
        'pack_membership': pack_membership_fig,
        'gap_to_leader': gap_to_leader_fig,
        'summary': summary_df
    }

def create_gap_position_scatter(df, checkpoint_col, max_gap_to_leader=2, max_gap_within_pack=1):
    """Create scatter plot showing gap to leader vs position at checkpoint"""
    if checkpoint_col not in df.columns:
        return None
    
    # Get valid data and sort by elapsed time
    valid_data = df[df[checkpoint_col].notna()].copy()
    if len(valid_data) == 0:
        return None
    
    valid_data = valid_data.sort_values(checkpoint_col).reset_index(drop=True)
    
    # Calculate positions and gaps to leader
    positions = list(range(1, len(valid_data) + 1))
    leader_time = valid_data[checkpoint_col].iloc[0]
    gaps_to_leader = valid_data[checkpoint_col] - leader_time
    
    # Identify packs
    pack_ids = identify_packs(valid_data[checkpoint_col].values, max_gap_to_leader, max_gap_within_pack)
    valid_data['Pack_ID'] = pack_ids
    
    # Identify lead pack (pack 0)
    lead_pack_athletes = valid_data[valid_data['Pack_ID'] == 0]
    
    fig = go.Figure()
    
    # All athletes scatter plot
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=gaps_to_leader,
            mode='markers',
            marker=dict(size=12, color='#1f77b4', opacity=0.8, line=dict(width=2, color='darkblue')),
            text=valid_data['Name'].tolist() if 'Name' in valid_data.columns else [f"Athlete {i+1}" for i in range(len(valid_data))],
            hovertemplate='<b>%{text}</b><br>Position: %{x}<br>Gap to Leader: %{y:.1f}s<br><extra></extra>',
            name='All Athletes'
        )
    )
    
    # Highlight the lead pack if it has more than 1 athlete
    if len(lead_pack_athletes) > 1:
        lead_pack_positions = [i+1 for i in lead_pack_athletes.index]
        lead_pack_gaps = gaps_to_leader.iloc[lead_pack_athletes.index]
        
        fig.add_trace(
            go.Scatter(
                x=lead_pack_positions,
                y=lead_pack_gaps,
                mode='markers',
                marker=dict(size=15, color='red', opacity=0.6, symbol='circle-open', line=dict(width=3, color='red')),
                text=lead_pack_athletes['Name'].tolist() if 'Name' in lead_pack_athletes.columns else [f"Leader {i+1}" for i in range(len(lead_pack_athletes))],
                hovertemplate='<b>%{text}</b> (Lead Pack)<br>Position: %{x}<br>Gap to Leader: %{y:.1f}s<br><extra></extra>',
                name='Lead Pack'
            )
        )
    
    checkpoint_name = checkpoint_col.replace('Elapsed_After_', '').replace('_', ' ')
    
    fig.update_layout(
        title=f'Gap to Leader vs Position - {checkpoint_name}',
        xaxis_title='Race Position',
        yaxis_title='Gap to Leader (seconds)',
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=2, range=[0.5, max(positions) + 0.5])
    # Set y-axis ticks every 10 seconds
    max_gap = max(gaps_to_leader) if len(gaps_to_leader) > 0 else 10
    fig.update_yaxes(tickmode='linear', dtick=10, range=[0, max_gap * 1.05])
    
    return fig

# ===== EVENT ANALYSIS FUNCTIONS =====

def process_uploaded_excel(file_contents, file_name):
    """Process uploaded Excel file with detailed timing splits"""
    try:
        excel_file = pd.ExcelFile(file_contents)
        return excel_file
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

def validate_excel_format(df):
    """Validate uploaded Excel has required columns"""
    required_columns = ["Name", "Rank", "Total"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def process_excel_sheet(excel_file, sheet_name):
    """Process a specific sheet from the Excel file"""
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        if not validate_excel_format(df):
            return None
            
        # Basic data cleaning
        df = df.dropna(subset=['Name', 'Rank'])
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error processing sheet {sheet_name}: {e}")
        return None

def create_race_overview_metrics(df):
    """Create overview metrics for race"""
    if df is None or df.empty:
        return None
    
    total_racers = len(df)
    if 'Total' in df.columns:
        dnf_count = df['Total'].str.contains('DNF|LAP', na=False).sum()
    dnf_rate = (dnf_count / len(df)) * 100 if len(df) > 0 else 0
    
    winning_time = df.loc[df['Rank'] == 1, 'Total'].iloc[0] if 'Total' in df.columns and not df[df['Rank'] == 1].empty else "N/A"
    
    top_10 = df.head(10)
    if 'Total' in df.columns and len(top_10) >= 10:
        # Calculate time spread for top 10 (placeholder logic)
        time_spread = "Calculate based on time format"
    else:
        time_spread = "N/A"
    
    return {
        'total_racers': total_racers,
        'dnf_rate': dnf_rate,
        'winning_time': winning_time,
        'time_spread': time_spread
    }

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Triathlon Analysis Dashboard", 
    page_icon="🏊‍♂️", 
    layout="wide"
)

# ===== DATABASE CONNECTION =====
# Get database URI from environment or Streamlit secrets
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    st.error("Database URI not set. Please configure DB_URI in your environment or Streamlit secrets.")
    st.stop()
engine = create_engine(DB_URI, echo=False)

# ===== MAIN APPLICATION =====

def main():
    """Main application with page navigation"""
    
    # Navigation in sidebar
    st.sidebar.title("🏊‍♂️ Triathlon Analysis")
    page = st.sidebar.selectbox(
        "Choose Analysis Type", 
        ["H2H Analysis", "Event Analysis"]
    )
    
    if page == "H2H Analysis":
        show_h2h_page()
    elif page == "Event Analysis":
        show_event_analysis_page()

def show_h2h_page():
    """Head-to-Head Analysis Page"""
    st.title("Triathlon H2H Dashboard")
    
    # Load options with caching, spinner, and error handling
    try:
        with st.spinner("Loading athlete and event options..."):
            ath_df = load_athlete_names(engine)
            ev_df = load_event_names(engine)
    except SQLAlchemyError as e:
        st.error(f"Database error while loading options: {e}")
        st.stop()

    athletes = ath_df['full_name'].tolist()
    events = ev_df['event_name'].tolist()

    # --- USA Men/Women Defaults ---
    USA_MEN = [
        "John Reed", "Chase McQueen", "Morgan Pearson", "Seth Rider", "Darr Smith"
    ]
    USA_WOMEN = [
        "Gwen Jorgensen",
        "Taylor Spivey",
        "Gina Sereno",
        "Erika Ackerlund",
        "Summer Rappaport"
    ]
    DEFAULT_EVENTS = [
        "2025 World Triathlon Championship Series Abu Dhabi",
        "2025 World Triathlon Championship Series Yokohama",
        "2025 World Triathlon Championship Series Alghero",
    ]

    # --- Load event dates for date range filtering ---
    @st.cache_data(ttl=600)
    def load_event_dates(_engine):
        return pd.read_sql("SELECT event_name, event_date FROM events ORDER BY event_date", _engine)

    event_dates_df = load_event_dates(engine)
    event_dates_df['event_date'] = pd.to_datetime(event_dates_df['event_date'])
    min_event_date = event_dates_df['event_date'].min()
    max_event_date = event_dates_df['event_date'].max()

    # Sidebar selectors with event/date mode
    with st.sidebar:
        st.markdown("### H2H Selection")
        st.markdown("#### Select Default U.S. Mens or Womens Athletes/Events or Choose Your Own")
        if 'selected_athletes' not in st.session_state:
            st.session_state.selected_athletes = []
        if 'selected_events' not in st.session_state:
            st.session_state.selected_events = []
        if 'date_range' not in st.session_state:
            st.session_state.date_range = (min_event_date, max_event_date)
        if 'event_mode' not in st.session_state:
            st.session_state.event_mode = 'By Event'

        # Place USA Men and USA Women buttons side by side
        col1, col2 = st.columns(2)
        with col1:
            if st.button("USA Men"):
                st.session_state.selected_athletes = [a for a in USA_MEN if a in athletes]
                if st.session_state.event_mode == "By Event":
                    st.session_state.selected_events = [e for e in DEFAULT_EVENTS if e in events]
                else:
                    st.session_state.selected_events = []
                st.rerun()
        with col2:
            if st.button("USA Women"):
                st.session_state.selected_athletes = [a for a in USA_WOMEN if a in athletes]
                if st.session_state.event_mode == "By Event":
                    st.session_state.selected_events = [e for e in DEFAULT_EVENTS if e in events]
                else:
                    st.session_state.selected_events = []
                st.rerun()

        selected_athletes = st.multiselect(
            "Select Athletes to Compare",
            athletes,
            default=st.session_state.selected_athletes,
            key="athlete_multiselect"
        )

        # Move event selection mode radio right above event selection/date
        event_mode = st.radio(
            "Event Selection Mode",
            ["By Event", "By Date Range"],
            index=0 if st.session_state.event_mode == 'By Event' else 1,
            key="event_mode_radio"
        )
        st.session_state.event_mode = event_mode

        if event_mode == "By Event":
            selected_events = st.multiselect(
                "Select Events to Include",
                events,
                default=st.session_state.selected_events,
                key="event_multiselect"
            )
            date_range = st.session_state.date_range
        else:
            date_range = st.date_input(
                "Select event date range",
                value=st.session_state.date_range,
                min_value=min_event_date,
                max_value=max_event_date,
                key="date_range_picker"
            )
            st.session_state.date_range = date_range
            selected_events = []  # Not used in this mode

        # Update session state
        st.session_state.selected_athletes = selected_athletes
        st.session_state.selected_events = selected_events
        if event_mode == "By Date Range":
            st.session_state.date_range = date_range

        # Show current selections
        st.markdown("### Current Selections")
        st.write(f"**Athletes:** {len(st.session_state.selected_athletes)}")
        for ath in st.session_state.selected_athletes:
            st.write(ath)
        if event_mode == "By Event":
            st.write(f"**Events:** {len(st.session_state.selected_events)}")
            for ev in st.session_state.selected_events:
                st.write(ev)
        if event_mode == "By Date Range":
            st.markdown(f"---\\n**Date Range:** {date_range[0]} to {date_range[1]}")

    # --- H2H Analysis Logic ---
    run_h2h = False
    selected_events_for_h2h = []
    if st.session_state.selected_athletes:
        if st.session_state.event_mode == "By Event" and st.session_state.selected_events:
            selected_events_for_h2h = st.session_state.selected_events
            run_h2h = True
        elif st.session_state.event_mode == "By Date Range":
            # Use all events in the selected date range
            date_range = st.session_state.date_range
            selected_events_for_h2h = event_dates_df[
                (event_dates_df['event_date'] >= pd.to_datetime(date_range[0])) &
                (event_dates_df['event_date'] <= pd.to_datetime(date_range[1]))
            ]['event_name'].tolist()
            if selected_events_for_h2h:
                run_h2h = True
            else:
                st.warning("No events found in the selected date range.")

    if run_h2h:
        st.markdown(f"**Analyzing {len(selected_events_for_h2h)} events with {len(st.session_state.selected_athletes)} athletes**")
        
        # Load H2H data
        try:
            with st.spinner("Loading H2H analysis..."):
                df = h2h_summary(st.session_state.selected_athletes, selected_events_for_h2h, engine)
        except SQLAlchemyError as e:
            st.error(f"Database error during H2H analysis: {e}")
            st.stop()

        if not df.empty:
            # Create tabs for different analysis types
            tabs = st.tabs(["Overall", "Swim", "T1", "Bike", "T2", "Run"])
            
            with tabs[0]:
                mat, annot = build_overall_matrix(df)
                if mat.empty or mat.shape[0] == 0 or mat.shape[1] == 0:
                    st.warning("No valid H2H matrix could be generated for this selection.")
                else:
                    plot_heatmap(mat, annot, "Overall H2H Record & Time Gaps")
            
            # Segments
            segments = ["swim", "t1", "bike", "t2", "run"]
            for i, seg in enumerate(segments, start=1):
                with tabs[i]:
                    mat, annot = build_segment_matrix(df, seg)
                    if mat.empty or mat.shape[0] == 0 or mat.shape[1] == 0:
                        st.warning(f"No valid {seg.capitalize()} segment H2H matrix for this selection.")
                    else:
                        plot_heatmap(mat, annot, f"{seg.capitalize()} Segment H2H Record & Time Gaps")

            # --- Show missing athletes at the bottom of the page ---
            athletes_in_results = set(df['athlete_name_a'].unique()).union(set(df['athlete_name_b'].unique()))
            missing_athletes = [a for a in st.session_state.selected_athletes if a not in athletes_in_results]
            if missing_athletes:
                st.markdown("---")
                st.info(
                    "The following athlete(s) were not found in the H2H results (they did not race in the selected event(s)):")
                for athlete in missing_athletes:
                    st.write(f"- {athlete}")
        else:
            st.warning("No H2H data found for the selected athletes and events.")
    else:
        st.write("Please select at least one athlete and event selection mode from the sidebar.")

def show_event_analysis_page():
    """Event Analysis & Pack Dynamics Page"""
    st.title("📊 Event Analysis & Pack Dynamics")
    st.markdown("Analyze detailed race results, pack dynamics, and individual performance patterns.")
    
    # Data source selection
    data_source = st.radio(
        "Data Source", 
        ["Database (Standard Events)", "Excel Upload (Detailed Results)"],
        help="Choose 'Database' for standard events with basic timing, or 'Excel Upload' for detailed pack dynamics analysis"
    )
    
    if data_source == "Database (Standard Events)":
        show_database_event_analysis()
    else:
        show_excel_event_analysis()

def show_database_event_analysis():
    """Database-based event analysis with standard race data"""
    st.subheader("📋 Standard Event Analysis")
    
    try:
        with st.spinner("Loading events..."):
            ev_df = load_event_names(engine)
    except SQLAlchemyError as e:
        st.error(f"Database error: {e}")
        return
    
    events = ev_df['event_name'].tolist()
    
    # Event selection
    selected_event = st.selectbox("Select Event", events)
    selected_program = st.selectbox("Select Program", ["Elite Men", "Elite Women", "Mixed Relay"])
    
    if selected_event:
        st.info("Database analysis features coming in Phase 2!")
        st.markdown("""
        **Planned Features:**
        - Basic race results and rankings
        - Simple position tracking
        - Athlete performance summaries
        - Time gap analysis
        """)

def show_excel_event_analysis():
    """Excel-based event analysis with detailed timing data"""
    st.subheader("📁 Detailed Event Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Detailed Race Results", 
        type=["xlsx"],
        help="Upload Excel file with detailed timing splits (similar to Hamburg 2025 format)"
    )
    
    if uploaded_file is not None:
        # Process Excel file
        excel_file = process_uploaded_excel(uploaded_file, uploaded_file.name)
        
        if excel_file is not None:
            # Event name input
            event_name = st.text_input(
                "Event Name", 
                value="Hamburg 2025",  # Default value
                help="Enter the name of this event"
            )
            
            # Sheet selection
            sheet_names = excel_file.sheet_names
            selected_sheet = st.selectbox("Select Gender/Category", sheet_names)
            
            # Process selected sheet
            race_data = process_excel_sheet(excel_file, selected_sheet)
            
            if race_data is not None:
                st.success(f"Successfully loaded {len(race_data)} athletes from {selected_sheet}")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(race_data.head())
                    st.write(f"**Columns:** {', '.join(race_data.columns)}")
                
                # Race Overview Dashboard
                st.subheader("🏁 Race Overview")
                
                metrics = create_race_overview_metrics(race_data)
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Racers", metrics['total_racers'])
                    with col2:
                        st.metric("DNF Rate", f"{metrics['dnf_rate']:.1f}%")
                    with col3:
                        st.metric("Winning Time", metrics['winning_time'])
                    with col4:
                        st.metric("Time Spread (Top 10)", metrics['time_spread'])
                
                # Race Results Table
                st.subheader("🏃‍♂️ Race Results")
                
                # Filtering options
                col1, col2 = st.columns(2)
                with col1:
                    athlete_filter = st.multiselect(
                        "Focus on Specific Athletes", 
                        race_data['Name'].tolist() if 'Name' in race_data.columns else []
                    )
                with col2:
                    top_n_filter = st.slider("Show Top N Athletes", 10, len(race_data), min(30, len(race_data)))
                
                # Apply filters
                display_data = race_data.copy()
                if athlete_filter:
                    display_data = display_data[display_data['Name'].isin(athlete_filter)]
                else:
                    display_data = display_data.head(top_n_filter)
                
                st.dataframe(display_data, use_container_width=True)
                
                # Pack Dynamics Analysis (if detailed timing available)
                st.subheader("🏃‍♂️ Pack Dynamics Analysis")
                
                # Pack analysis controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_gap_to_leader = st.slider(
                        "Max Gap to Leader Threshold (seconds)",
                        min_value=1,
                        max_value=10,
                        value=2,
                        help="Maximum time gap from the leader of the pack to be considered in the same pack"
                    )
                    max_gap_within_pack = st.slider(
                        "Max Gap Within Pack Threshold (seconds)",
                        min_value=1,
                        max_value=10,
                        value=1,
                        help="Maximum time gap within the pack to be considered in the same pack"
                    )

                with col2:
                    # Expanded checkpoint options: all major checkpoints
                    checkpoint_options = [
                        "After Swim",
                        "After T1",
                        "After Bike Lap 1",
                        "After Bike Lap 2",
                        "After Bike Lap 3",
                        "After Bike Lap 4",
                        "After Bike Lap 5",
                        "After Bike Lap 6",
                        "After T2",
                        "After Run Seg 1",
                        "After Run Seg 2",
                        "After Run Seg 3"
                    ]
                    checkpoint_analysis = st.selectbox(
                        "Select Checkpoint for Analysis",
                        options=checkpoint_options,
                        index=0
                    )
                
                # Calculate elapsed times and analyze packs
                try:
                    df_with_elapsed, checkpoints = calculate_elapsed_times(race_data)
                    
                    if checkpoints:
                        # Map user selection to checkpoint column
                        checkpoint_mapping = {
                            "After Swim": "Elapsed_After_Swim",
                            "After T1": "Elapsed_After_T1",
                            "After Bike Lap 1": "Elapsed_After_Bike_Lap_1",
                            "After Bike Lap 2": "Elapsed_After_Bike_Lap_2",
                            "After Bike Lap 3": "Elapsed_After_Bike_Lap_3",
                            "After Bike Lap 4": "Elapsed_After_Bike_Lap_4",
                            "After Bike Lap 5": "Elapsed_After_Bike_Lap_5",
                            "After Bike Lap 6": "Elapsed_After_Bike_Lap_6",
                            "After T2": "Elapsed_After_T2",
                            "After Run Seg 1": "Elapsed_After_Run_Seg_1",
                            "After Run Seg 2": "Elapsed_After_Run_Seg_2",
                            "After Run Seg 3": "Elapsed_After_Run_Seg_3"
                        }
                        selected_checkpoint = checkpoint_mapping.get(checkpoint_analysis)
                        
                        if selected_checkpoint and selected_checkpoint in df_with_elapsed.columns:
                            # Analyze packs at selected checkpoint
                            pack_data, pack_stats = analyze_packs_at_checkpoint(
                                df_with_elapsed, selected_checkpoint, max_gap_to_leader, max_gap_within_pack
                            )
                            
                            if pack_stats:
                                # Display pack composition
                                st.write(f"**Pack Composition at {checkpoint_analysis}**")
                                pack_table = create_pack_composition_table(pack_data, pack_stats)
                                st.table(pack_table)
                                
                                # Gap vs Position Scatter Plot
                                st.write(f"**📊 Position vs Gap to Leader - {checkpoint_analysis}**")
                                gap_scatter = create_gap_position_scatter(
                                    df_with_elapsed, selected_checkpoint, max_gap_to_leader, max_gap_within_pack
                                )
                                if gap_scatter:
                                    st.plotly_chart(gap_scatter, use_container_width=True)
                                
                                # Advanced Gap Analysis (Phase 3)
                                st.write("**📊 Gap Analysis & Race Dynamics**")
                                gap_analysis = create_advanced_gap_analysis(df_with_elapsed, checkpoints)
                                if gap_analysis:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.plotly_chart(gap_analysis['gap_evolution'], use_container_width=True)
                                    with col2:
                                        st.plotly_chart(gap_analysis['gap_distribution'], use_container_width=True)

                                # Position tracking for selected athletes
                                if 'Name' in df_with_elapsed.columns:
                                    st.write("**📍 Position Tracking**")
                                    # Select athletes for tracking
                                    top_10_athletes = df_with_elapsed.head(10)['Name'].tolist()
                                    selected_athletes = st.multiselect(
                                        "Select athletes to track positions:",
                                        options=df_with_elapsed['Name'].tolist(),
                                        default=top_10_athletes[:5],
                                        max_selections=10
                                    )
                                    if selected_athletes:
                                        position_chart = create_position_tracking_chart(
                                            df_with_elapsed, checkpoints, selected_athletes
                                        )
                                        if position_chart:
                                            st.plotly_chart(position_chart, use_container_width=True)
                                        
                                        # Individual Athlete Deep Dive (Phase 3)
                                        st.write("**🔍 Individual Athlete Analysis**")
                                        selected_athlete_detailed = st.selectbox(
                                            "Select athlete for detailed analysis:",
                                            options=selected_athletes,
                                            key="athlete_detailed_analysis"
                                        )
                                        if selected_athlete_detailed:
                                            athlete_analysis = create_individual_athlete_analysis(
                                                df_with_elapsed, checkpoints, selected_athlete_detailed, max_gap_to_leader, max_gap_within_pack
                                            )
                                            if athlete_analysis:
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.plotly_chart(athlete_analysis['pack_membership'], use_container_width=True)
                                                with col2:
                                                    st.plotly_chart(athlete_analysis['gap_to_leader'], use_container_width=True)
                                                # Athlete performance summary
                                                st.write("**Performance Summary:**")
                                                st.dataframe(athlete_analysis['summary'], use_container_width=True)
                                
                                # Detailed pack information
                                with st.expander("📋 Detailed Pack Information", expanded=False):
                                    for pack_name, stats in pack_stats.items():
                                        st.write(f"**{pack_name}**")
                                        st.write(f"- Size: {stats['Size']} athletes")
                                        st.write(f"- Time spread: {stats['Time_Spread']:.1f} seconds")
                                        st.write(f"- Athletes: {', '.join(stats['Athletes'])}")
                                        st.write("")
                                
                                # Placeholder for future features
                                with st.expander("🔮 Preview of Upcoming Features"):
                                    st.markdown("""
                                    **✅ Phase 2 (COMPLETED):**
                                    - Pack assignment algorithms with configurable gap thresholds
                                    - Position tracking charts for selected athletes  
                                    - Basic pack composition tables and statistics
                                    - Pack size distribution visualization
                                    
                                    **✅ Phase 3 (COMPLETED):**
                                    - Interactive pack evolution timeline showing pack membership changes
                                    - Individual athlete deep dive analysis with performance summaries
                                    - Advanced gap analysis and race dynamics visualization
                                    - Pack breakaway and formation detection throughout race
                                    
                                    **🚀 Phase 4 (Coming Next):**
                                    - Advanced tactical positioning insights and strategic recommendations
                                    - Export capabilities for analysis results (PDF reports, data downloads)
                                    - Performance optimization and caching for large datasets
                                    - Multi-event pack dynamics comparison and athlete benchmarking
                                    """)
                            
                            else:
                                st.warning(f"No valid data found for {checkpoint_analysis}. Please try a different checkpoint.")
                        
                        else:
                            st.warning(f"Checkpoint data not available for {checkpoint_analysis}")
                    
                    else:
                        st.warning("Could not calculate elapsed times from the uploaded data. Pack analysis requires detailed timing splits.")
                
                except Exception as e:
                    st.error(f"Error in pack dynamics analysis: {e}")
                    st.info("Pack dynamics analysis requires detailed timing data with segment splits.")
            else:
                st.error("Failed to process the selected sheet. Please check the data format.")
    else:
        st.info("👆 Upload an Excel file to begin detailed event analysis")
        
        # Help section
        with st.expander("ℹ️ Excel File Format Requirements"):
            st.markdown("""
            **Required Columns:**
            - `Name`: Athlete name
            - `Rank`: Final position/rank
            - `Total`: Total race time
            
            **Optional Columns (for pack dynamics):**
            - Detailed timing splits (S1, T1, B1T1, etc.)
            - Intermediate positions
            - Segment times
            
            **Example Format:**
            Similar to the Hamburg 2025 detailed results with timing splits for swim, bike segments, and run.
            """)

# ===== MAIN APP EXECUTION =====
if __name__ == "__main__":
    main()
