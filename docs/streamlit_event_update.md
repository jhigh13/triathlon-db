# Streamlit Event Analysis Page - Detailed Specification

## Overview
Add a new page to the existing Streamlit triathlon analysis app for detailed event-specific race analysis, incorporating pack dynamics, position tracking, and tactical insights from detailed race timing data.

## Current App Context
- **Existing App**: H2H (Head-to-Head) analysis with athlete comparison matrices
- **Data Source**: PostgreSQL database with race results, athletes, events tables
- **Architecture**: Multi-page app with sidebar navigation and cached data loading
- **Features**: Athlete selection, event filtering, win/loss matrices, time gap analysis

## New Page: "Event Analysis" 

### 1. Core Functionality

#### 1.1 Data Source Integration
```pseudocode
# Primary data source options (user selectable)
data_source = st.radio("Data Source", ["Database (Standard Events)", "Excel Upload (Detailed Results)"])

if data_source == "Database":
    # Use existing database connection and standard event data
    # Limited to position and basic split times
    events = load_events_from_db()
    
elif data_source == "Excel Upload":
    # Upload detailed timing data (like Hamburg 2025 format)
    # Enables full pack dynamics analysis
    uploaded_file = st.file_uploader("Upload Detailed Results", type=["xlsx"])
    if uploaded_file:
        event_data = process_detailed_excel(uploaded_file)
```

#### 1.2 Event Selection & Filtering
```pseudocode
# Event selection (different based on data source)
if data_source == "Database":
    selected_event = st.selectbox("Select Event", events_list)
    selected_program = st.selectbox("Select Program", ["Elite Men", "Elite Women"])
    race_data = load_race_data(selected_event, selected_program)
    
elif data_source == "Excel Upload":
    # Auto-detect event name from file or user input
    event_name = st.text_input("Event Name", value=auto_detect_event_name())
    sheet_names = get_excel_sheets(uploaded_file)
    selected_sheet = st.selectbox("Select Gender/Category", sheet_names)
    race_data = process_excel_sheet(uploaded_file, selected_sheet)

# Additional filtering options
athlete_filter = st.multiselect("Focus on Specific Athletes", get_athlete_list(race_data))
top_n_filter = st.slider("Show Top N Athletes", 10, len(race_data), 30)
```

### 2. Analysis Modules

#### 2.1 Race Overview Dashboard
```pseudocode
# Key race statistics in metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Finishers", count_finishers(race_data))
with col2:
    st.metric("DNF Rate", calculate_dnf_rate(race_data))
with col3:
    st.metric("Winning Time", get_winning_time(race_data))
with col4:
    st.metric("Time Spread (Top 10)", calculate_time_spread(race_data, 10))

# Race results table with interactive sorting
st.subheader("Race Results")
display_race_results_table(race_data, sortable=True, athlete_filter)
```

#### 2.2 Pack Dynamics Analysis (Excel Data Only)
```pseudocode
if data_source == "Excel Upload" and has_detailed_timing(race_data):
    st.subheader("🏃‍♂️ Pack Dynamics Analysis")
    
    # Pack formation parameters
    gap_threshold = st.slider("Pack Gap Threshold (seconds)", 1, 10, 2)
    
    # Checkpoint selection for pack analysis
    checkpoints = get_available_checkpoints(race_data)
    selected_checkpoint = st.selectbox("Analyze Packs At:", checkpoints)
    
    # Generate pack assignments
    pack_data = assign_packs(race_data, selected_checkpoint, gap_threshold)
    
    # Pack composition table
    st.subheader(f"Pack Composition at {selected_checkpoint}")
    display_pack_composition_table(pack_data)
    
    # Pack evolution timeline
    st.subheader("Pack Evolution Throughout Race")
    pack_evolution_chart = create_pack_evolution_chart(race_data, gap_threshold)
    st.plotly_chart(pack_evolution_chart, use_container_width=True)
```

#### 2.3 Position Tracking & Movement Analysis
```pseudocode
st.subheader("📈 Position Tracking Analysis")

# Individual athlete position tracking
if athlete_filter:
    position_tracking_chart = create_position_tracking_chart(race_data, athlete_filter)
    st.plotly_chart(position_tracking_chart, use_container_width=True)

# Movement analysis (biggest gainers/losers)
movement_analysis = calculate_position_changes(race_data)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Biggest Gainers")
    st.dataframe(movement_analysis["gainers"].head(10))
with col2:
    st.subheader("Biggest Losers") 
    st.dataframe(movement_analysis["losers"].head(10))
```

#### 2.4 Segment Performance Analysis
```pseudocode
st.subheader("🏊‍♂️🚴‍♂️🏃‍♂️ Segment Performance Breakdown")

# Segment selection
segments = get_available_segments(race_data)
selected_segments = st.multiselect("Analyze Segments", segments, default=segments)

for segment in selected_segments:
    with st.expander(f"{segment.title()} Analysis"):
        # Segment leaderboard
        segment_results = get_segment_results(race_data, segment)
        st.dataframe(segment_results.head(10))
        
        # Segment time distribution
        segment_dist_chart = create_segment_distribution_chart(race_data, segment)
        st.plotly_chart(segment_dist_chart, use_container_width=True)
        
        # Segment impact on overall position
        segment_impact = analyze_segment_impact(race_data, segment)
        st.write(f"Average position change from {segment}: {segment_impact:.1f}")
```

#### 2.5 Gap Analysis & Race Dynamics
```pseudocode
st.subheader("⏱️ Gap Analysis")

# Time gap evolution chart
gap_evolution_chart = create_gap_evolution_chart(race_data, reference_athlete="leader")
st.plotly_chart(gap_evolution_chart, use_container_width=True)

# Critical moments identification
critical_moments = identify_critical_moments(race_data)
st.subheader("Critical Race Moments")
for moment in critical_moments:
    st.write(f"**{moment['checkpoint']}**: {moment['description']}")
```

#### 2.6 Interactive Athlete Deep Dive
```pseudocode
st.subheader("🔍 Individual Athlete Analysis")

selected_athlete = st.selectbox("Select Athlete for Deep Dive", get_athlete_list(race_data))

if selected_athlete:
    athlete_data = get_athlete_race_data(race_data, selected_athlete)
    
    # Athlete race summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Position", athlete_data["final_position"])
    with col2:
        st.metric("Total Time", athlete_data["total_time"])
    with col3:
        st.metric("Best Segment Rank", athlete_data["best_segment_rank"])
    
    # Athlete position progression
    athlete_position_chart = create_athlete_position_chart(race_data, selected_athlete)
    st.plotly_chart(athlete_position_chart, use_container_width=True)
    
    # Nearby competitors analysis
    nearby_competitors = find_nearby_competitors(race_data, selected_athlete)
    st.subheader("Key Competitors Throughout Race")
    st.dataframe(nearby_competitors)
```

### 3. Technical Implementation

#### 3.1 Data Processing Functions
```python
# Core data processing functions needed

def process_detailed_excel(file_path, sheet_name):
    """Process uploaded Excel file with detailed timing splits"""
    # Handle different Excel formats (Hamburg 2025 style)
    # Clean and standardize column names
    # Convert time strings to seconds
    # Handle DNF/incomplete data
    return processed_dataframe

def assign_packs(df, checkpoint, gap_threshold=2):
    """Assign pack IDs based on time gaps at specific checkpoint"""
    # Sort by checkpoint time
    # Identify gaps > threshold
    # Assign pack numbers
    # Handle DNF athletes (pack_id = -1)
    return df_with_packs

def calculate_position_changes(df):
    """Calculate position changes between checkpoints"""
    # Track position at each available checkpoint
    # Calculate net position changes
    # Identify biggest gainers/losers
    return position_changes_dict

def create_pack_evolution_chart(df, gap_threshold):
    """Create interactive pack evolution visualization"""
    # Plotly timeline chart showing pack formation
    # Color-coded packs
    # Interactive checkpoint selection
    return plotly_figure
```

#### 3.2 Caching Strategy
```python
@st.cache_data
def load_event_data_from_db(event_id, program_id):
    """Cached database loading for standard events"""
    return event_dataframe

@st.cache_data
def process_uploaded_file(file_contents, file_name):
    """Cache processed Excel data to avoid re-processing"""
    return processed_dataframe

@st.cache_data
def calculate_pack_dynamics(df, gap_threshold):
    """Cache pack calculations for different threshold values"""
    return pack_dynamics_results
```

#### 3.3 Page Navigation Integration
```python
# Add to main streamlit app navigation
def main():
    st.set_page_config(page_title="Triathlon Analysis", layout="wide")
    
    # Navigation
    pages = {
        "H2H Analysis": show_h2h_page,
        "Event Analysis": show_event_analysis_page  # NEW PAGE
    }
    
    page = st.sidebar.selectbox("Choose Analysis Type", list(pages.keys()))
    pages[page]()

def show_event_analysis_page():
    """Main function for the new event analysis page"""
    st.title("📊 Event Analysis & Pack Dynamics")
    # Implementation of all features above
```

### 4. UI/UX Design Considerations

#### 4.1 Page Layout
- **Header**: Page title with brief explanation
- **Sidebar**: Data source selection, event/athlete filtering
- **Main Content**: Tabbed interface for different analysis types
- **Footer**: Data source information and refresh options

#### 4.2 Responsive Design
- Use `st.columns()` for side-by-side metrics and charts
- Implement collapsible sections with `st.expander()`
- Mobile-friendly chart sizing with `use_container_width=True`

#### 4.3 Interactive Elements
- Real-time filtering and updates
- Downloadable analysis results
- Shareable chart configurations
- Export functionality for key insights

### 5. Error Handling & Data Validation

#### 5.1 Data Quality Checks
```python
def validate_excel_format(df):
    """Validate uploaded Excel has required columns"""
    required_columns = ["Name", "Rank", "Total"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def handle_missing_data(df):
    """Handle missing timing data gracefully"""
    # Identify athletes with incomplete data
    # Mark as DNF where appropriate
    # Provide warnings about data limitations
    return cleaned_df
```

#### 5.2 User Feedback
- Progress bars for data processing
- Clear error messages for invalid uploads
- Warnings for data limitations
- Help tooltips for complex features

### 6. Future Enhancements

#### 6.1 Advanced Analytics
- Predictive race outcome modeling
- Weather impact analysis
- Course-specific performance patterns
- Historical event comparisons

#### 6.2 Export & Sharing
- PDF report generation
- Export charts as images
- Shareable analysis URLs
- Integration with Power BI datasets

#### 6.3 Real-Time Features
- Live race tracking (if timing APIs available)
- Real-time pack updates during events
- Push notifications for critical race moments

### 7. Implementation Timeline

**Phase 1** (Week 1): Core infrastructure
- Page navigation setup
- Basic Excel upload functionality
- Database integration for standard events

**Phase 2** (Week 2): Pack dynamics analysis
- Pack assignment algorithms
- Basic pack composition tables
- Position tracking charts

**Phase 3** (Week 3): Advanced visualizations
- Pack evolution timeline
- Interactive athlete deep dive
- Gap analysis charts

**Phase 4** (Week 4): Polish & optimization
- Performance optimization
- Error handling refinement
- User experience improvements

### 8. Technical Dependencies

#### 8.1 Additional Python Packages
```
plotly>=5.0.0          # Interactive charts
openpyxl>=3.0.0        # Excel file processing  
streamlit-aggrid       # Enhanced data tables (optional)
```

#### 8.2 Database Schema Extensions
No database changes required - page supports both:
- Existing database structure for basic analysis
- Excel upload for detailed pack dynamics analysis

### 9. Success Metrics

- **User Engagement**: Time spent on page, feature usage rates
- **Data Quality**: Successful file uploads, error rates
- **Performance**: Page load times, chart rendering speed
- **User Feedback**: Feature requests, usability ratings

This specification provides a comprehensive roadmap for implementing event-specific analysis capabilities while maintaining integration with the existing H2H analysis features.
