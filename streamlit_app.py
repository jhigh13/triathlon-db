import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

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
@st.cache_data(show_spinner=False)
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
    import seaborn as sns
    import matplotlib.pyplot as plt

    z = mat.map(color_code).values
    n = len(mat)
    # Dynamically scale the heatmap size so cells (and text) have enough room
    size = max(8, n * 1.5)
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
        annot_kws={"size":10},
        ax=ax
    )
    ax.set_xlabel('Athlete B')
    ax.set_ylabel('Athlete A')
    ax.set_title(title)
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)

st.title("Triathlon H2H Dashboard")


# Get database URI from environment or Streamlit secrets
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    st.error("Database URI not set. Please configure DB_URI in your environment or Streamlit secrets.")
    st.stop()
engine = create_engine(DB_URI, echo=False)
 
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


# Sidebar selectors with USA Men/Women buttons

with st.sidebar:
    st.markdown("### H2H Selection")
    st.markdown("#### Select Default U.S. Mens or Womens Athletes/Events or Choose Your Own")
    if 'selected_athletes' not in st.session_state:
        st.session_state.selected_athletes = []
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = []
    if st.button("USA Men"):
        st.session_state.selected_athletes = [a for a in USA_MEN if a in athletes]
        st.session_state.selected_events = [e for e in DEFAULT_EVENTS if e in events]
        st.rerun()
    if st.button("USA Women"):
        st.session_state.selected_athletes = [a for a in USA_WOMEN if a in athletes]
        st.session_state.selected_events = [e for e in DEFAULT_EVENTS if e in events]
        st.rerun()
    selected_athletes = st.multiselect("Select Athletes to Compare", athletes, default=st.session_state.selected_athletes, key="athlete_multiselect")
    selected_events = st.multiselect("Select Events to Include", events, default=st.session_state.selected_events, key="event_multiselect")
    # Synchronize session state with multiselects
    st.session_state.selected_athletes = selected_athletes
    st.session_state.selected_events = selected_events
    if st.session_state.selected_events:
        st.markdown("---")
        st.markdown("**Selected Events:**")
        for ev in st.session_state.selected_events:
            st.write(ev)


# --- Tabs for charts ---
if st.session_state.selected_athletes and st.session_state.selected_events:
    try:
        with st.spinner("Computing H2H summary..."):
            df = h2h_summary(st.session_state.selected_athletes, st.session_state.selected_events, engine)
    except SQLAlchemyError as e:
        st.error(f"Error computing H2H summary: {e}")
        st.stop()
    tab_names = ["Overall H2H", "Swim Segment", "T1 Segment", "Bike Segment", "T2 Segment", "Run Segment"]
    tabs = st.tabs(tab_names)
    # Overall
    with tabs[0]:
        mat, annot = build_overall_matrix(df)
        plot_heatmap(mat, annot, "Overall H2H Record & Time Gaps")
    # Segments
    segments = ["swim", "t1", "bike", "t2", "run"]
    for i, seg in enumerate(segments, start=1):
        with tabs[i]:
            mat, annot = build_segment_matrix(df, seg)
            plot_heatmap(mat, annot, f"{seg.capitalize()} Segment H2H Record & Time Gaps")
else:
    st.write("Please select at least one athlete and event from the sidebar.")

# --- Show selected event names at the bottom ---
