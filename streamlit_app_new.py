import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ===== H2H ANALYSIS FUNCTIONS =====

@st.cache_data(ttl=600)
def h2h_summary(_engine, events_list, athletes_list):
    """Cache H2H summary to avoid repeated DB calls for same selections"""
    if not events_list or not athletes_list:
        return pd.DataFrame()
    
    events_tuple = tuple(events_list)
    athletes_tuple = tuple(athletes_list)
    
    query = """
    WITH eligible_results AS (
      SELECT rr.*
      FROM race_results rr
      JOIN events e ON rr.event_id = e.event_id
      JOIN athlete a ON rr.athlete_id = a.athlete_id
      WHERE e.event_name = ANY(%(events)s)
        AND a.full_name = ANY(%(athletes)s)
        AND rr.position IS NOT NULL
    ),
    h2h_pairs AS (
      SELECT 
        r1.athlete_id as athlete_id_a,
        r2.athlete_id as athlete_id_b,
        r1.prog_id,
        r1.position < r2.position as wins_a,
        r1.swimrank < r2.swimrank as swim_wins_a,
        r1.t1rank < r2.t1rank as t1_wins_a,
        r1.bikerank < r2.bikerank as bike_wins_a,
        r1.t2rank < r2.t2rank as t2_wins_a,
        r1.runrank < r2.runrank as run_wins_a,
        r1.elapsedrun - r2.elapsedrun as time_diff_sec
      FROM eligible_results r1
      JOIN eligible_results r2 ON r1.prog_id = r2.prog_id
      WHERE r1.athlete_id < r2.athlete_id
    )
    SELECT 
      a1.full_name as athlete_name_a,
      a2.full_name as athlete_name_b,
      COUNT(*) as matches,
      SUM(CASE WHEN wins_a THEN 1 ELSE 0 END) as wins_a,
      SUM(CASE WHEN swim_wins_a THEN 1 ELSE 0 END) as swim_wins_a,
      SUM(CASE WHEN t1_wins_a THEN 1 ELSE 0 END) as t1_wins_a,
      SUM(CASE WHEN bike_wins_a THEN 1 ELSE 0 END) as bike_wins_a,
      SUM(CASE WHEN t2_wins_a THEN 1 ELSE 0 END) as t2_wins_a,
      SUM(CASE WHEN run_wins_a THEN 1 ELSE 0 END) as run_wins_a,
      AVG(time_diff_sec) as avg_time_diff_sec
    FROM h2h_pairs h
    JOIN athlete a1 ON h.athlete_id_a = a1.athlete_id
    JOIN athlete a2 ON h.athlete_id_b = a2.athlete_id
    GROUP BY a1.full_name, a2.full_name
    HAVING COUNT(*) >= 2
    ORDER BY a1.full_name, a2.full_name
    """
    
    return pd.read_sql(query, _engine, params={'events': list(events_tuple), 'athletes': list(athletes_tuple)})

def build_overall_matrix(df):
    """Build win percentage matrix for overall results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    athletes = sorted(set(df['athlete_name_a'].unique()).union(set(df['athlete_name_b'].unique())))
    matrix = pd.DataFrame(index=athletes, columns=athletes, dtype=float)
    annotations = pd.DataFrame(index=athletes, columns=athletes, dtype=object)
    
    for _, row in df.iterrows():
        a, b = row['athlete_name_a'], row['athlete_name_b']
        matches = row['matches']
        wins_a = row['wins_a']
        win_pct_a = wins_a / matches if matches > 0 else 0
        win_pct_b = 1 - win_pct_a
        avg_gap = row['avg_time_diff_sec']
        
        matrix.loc[a, b] = win_pct_a
        matrix.loc[b, a] = win_pct_b
        
        annotations.loc[a, b] = f"{win_pct_a:.1%}\\n({wins_a}-{matches-wins_a})\\n{seconds_to_hms(avg_gap)}"
        annotations.loc[b, a] = f"{win_pct_b:.1%}\\n({matches-wins_a}-{wins_a})\\n{seconds_to_hms(-avg_gap)}"
    
    # Fill diagonal with 50% (self vs self)
    for athlete in athletes:
        matrix.loc[athlete, athlete] = 0.5
        annotations.loc[athlete, athlete] = "50%\\n(0-0)\\n00:00"
    
    return matrix.fillna(0), annotations.fillna("")

def build_segment_matrix(df, segment):
    """Build win percentage matrix for a specific segment"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    wins_col = f"{segment}_wins_a"
    if wins_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    athletes = sorted(set(df['athlete_name_a'].unique()).union(set(df['athlete_name_b'].unique())))
    matrix = pd.DataFrame(index=athletes, columns=athletes, dtype=float)
    annotations = pd.DataFrame(index=athletes, columns=athletes, dtype=object)
    
    for _, row in df.iterrows():
        a, b = row['athlete_name_a'], row['athlete_name_b']
        matches = row['matches']
        wins_a = row[wins_col]
        win_pct_a = wins_a / matches if matches > 0 else 0
        win_pct_b = 1 - win_pct_a
        
        matrix.loc[a, b] = win_pct_a
        matrix.loc[b, a] = win_pct_b
        
        annotations.loc[a, b] = f"{win_pct_a:.1%}\\n({wins_a}-{matches-wins_a})"
        annotations.loc[b, a] = f"{win_pct_b:.1%}\\n({matches-wins_a}-{wins_a})"
    
    # Fill diagonal with 50% (self vs self)
    for athlete in athletes:
        matrix.loc[athlete, athlete] = 0.5
        annotations.loc[athlete, athlete] = "50%\\n(0-0)"
    
    return matrix.fillna(0), annotations.fillna("")

def plot_heatmap(matrix, annotations, title):
    """Plot heatmap using matplotlib/seaborn"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix.astype(float),
        annot=annotations,
        fmt='',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Win Percentage'},
        ax=ax
    )
    ax.set_xlabel('Athlete B')
    ax.set_ylabel('Athlete A')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)

# ===== EVENT ANALYSIS FUNCTIONS =====

@st.cache_data
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
    
    total_finishers = len(df)
    dnf_count = df['Total'].isna().sum() if 'Total' in df.columns else 0
    dnf_rate = (dnf_count / len(df)) * 100 if len(df) > 0 else 0
    
    winning_time = df.loc[df['Rank'] == 1, 'Total'].iloc[0] if 'Total' in df.columns and not df[df['Rank'] == 1].empty else "N/A"
    
    top_10 = df.head(10)
    if 'Total' in df.columns and len(top_10) >= 10:
        # Calculate time spread for top 10 (placeholder logic)
        time_spread = "Calculate based on time format"
    else:
        time_spread = "N/A"
    
    return {
        'total_finishers': total_finishers,
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
                df = h2h_summary(engine, selected_events_for_h2h, st.session_state.selected_athletes)
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
                        st.metric("Total Finishers", metrics['total_finishers'])
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
                st.info("Pack dynamics analysis features coming in Phase 2!")
                st.markdown("""
                **Planned Features:**
                - Interactive pack formation visualization
                - Gap threshold configuration
                - Pack evolution timeline
                - Position tracking charts
                - Tactical positioning insights
                """)
                
                # Placeholder for future features
                with st.expander("🔮 Preview of Upcoming Features"):
                    st.markdown("""
                    **Phase 2 (Next Week):**
                    - Pack assignment algorithms
                    - Position tracking charts
                    - Basic pack composition tables
                    
                    **Phase 3:**
                    - Interactive pack evolution timeline
                    - Individual athlete deep dive
                    - Gap analysis and race dynamics
                    
                    **Phase 4:**
                    - Advanced visualizations
                    - Export capabilities
                    - Performance optimization
                    """)
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

if __name__ == "__main__":
    main()
