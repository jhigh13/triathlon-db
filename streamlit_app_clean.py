import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our H2H analysis functions
from tri_analysis.h2h_analysis import (
    h2h_summary, 
    calculate_time_gaps, 
    create_h2h_heatmap,
    get_available_athletes,
    get_athlete_matchups_count,
    quick_h2h_analysis
)

# Set page config
st.set_page_config(
    page_title="Triathlon H2H Analysis",
    page_icon="🏊‍♂️",
    layout="wide"
)

# Title and description
st.title("🏊‍♂️ Triathlon Head-to-Head Analysis")
st.markdown("""
This dashboard provides head-to-head analysis between triathlon athletes, showing:
- **Win-loss records** between athlete pairs
- **Time gap statistics** (average, minimum, maximum)
- **Segment-level analysis** (swim, T1, bike, T2, run)
""")

# Sidebar controls
st.sidebar.header("Analysis Settings")

# Get available athletes
try:
    available_athletes = get_available_athletes(min_races=5, limit=100)
    
    # Athlete selection
    selected_athletes = st.sidebar.multiselect(
        "Select Athletes to Compare",
        options=available_athletes,
        default=available_athletes[:5] if len(available_athletes) >= 5 else available_athletes,
        help="Select 2 or more athletes to compare"
    )
    
    # Segment selection
    segment = st.sidebar.selectbox(
        "Analysis Segment",
        options=['overall', 'swim', 't1', 'bike', 't2', 'run'],
        index=0,
        help="Choose which segment to analyze"
    )
    
    # Time gap type
    gap_type = st.sidebar.selectbox(
        "Time Gap Display",
        options=['avg', 'min', 'max'],
        index=0,
        help="Choose which time gap statistic to display"
    )
    
    # Event filter (optional)
    event_filter = st.sidebar.text_input(
        "Event Filter (Optional)",
        placeholder="e.g., EventID = 123",
        help="Optional SQL WHERE clause to filter events"
    )
    
    # Show time gaps toggle
    show_time_gaps = st.sidebar.checkbox(
        "Show Time Gaps",
        value=True,
        help="Include time gap annotations in the heatmap"
    )
    
    # Main content
    if len(selected_athletes) >= 2:
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Athletes Selected", 
                len(selected_athletes)
            )
        
        with col2:
            st.metric(
                "Analysis Segment", 
                segment.title()
            )
        
        # Generate the analysis
        with st.spinner("Generating head-to-head analysis..."):
            try:
                # Get the data
                win_matrix = h2h_summary(
                    selected_athletes, 
                    event_filter if event_filter else None, 
                    segment
                )
                
                # Get matchup counts for context
                matchup_counts = get_athlete_matchups_count(
                    selected_athletes,
                    event_filter if event_filter else None
                )
                
                # Get time gaps if requested
                time_gaps = None
                if show_time_gaps:
                    time_gaps = calculate_time_gaps(
                        selected_athletes,
                        event_filter if event_filter else None,
                        segment
                    )
                
                # Create the visualization
                segment_title = segment.title() if segment != 'overall' else 'Overall'
                title = f"Head-to-Head Analysis - {segment_title}"
                if event_filter:
                    title += " (Filtered)"
                
                fig = create_h2h_heatmap(
                    win_matrix, 
                    time_gaps, 
                    title, 
                    gap_type,
                    figsize=(12, 10)
                )
                
                # Display the heatmap
                st.pyplot(fig)
                
                # Show summary statistics
                st.subheader("📊 Analysis Summary")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Win Matrix", "Time Gaps", "Matchup Counts"])
                
                with tab1:
                    st.markdown("**Win counts between athletes (row beats column):**")
                    st.dataframe(win_matrix, use_container_width=True)
                
                with tab2:
                    if time_gaps:
                        st.markdown(f"**{gap_type.title()} time gaps (seconds):**")
                        st.dataframe(time_gaps[gap_type], use_container_width=True)
                    else:
                        st.info("Enable 'Show Time Gaps' to see time gap statistics")
                
                with tab3:
                    st.markdown("**Number of head-to-head matchups:**")
                    st.dataframe(matchup_counts, use_container_width=True)
                
                # Additional insights
                st.subheader("💡 Insights")
                
                # Find the athlete with most wins
                total_wins = win_matrix.sum(axis=1)
                if not total_wins.empty:
                    best_athlete = total_wins.idxmax()
                    best_wins = total_wins.max()
                    
                    st.success(f"**{best_athlete}** has the most wins ({best_wins}) against this group")
                
                # Find most competitive matchup
                if not matchup_counts.empty:
                    # Get the maximum matchup count (excluding diagonal)
                    matchup_counts_no_diag = matchup_counts.copy()
                    np.fill_diagonal(matchup_counts_no_diag.values, 0)
                    
                    max_matchups = matchup_counts_no_diag.max().max()
                    if max_matchups > 0:
                        max_location = matchup_counts_no_diag.stack().idxmax()
                        athlete1, athlete2 = max_location
                        
                        st.info(f"**{athlete1}** and **{athlete2}** have raced head-to-head the most ({max_matchups} times)")
                        
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
                st.info("Please check your athlete selection and event filter")
    
    else:
        st.warning("Please select at least 2 athletes to compare")
        
        # Show some example athletes
        if available_athletes:
            st.subheader("Top Athletes by Race Count")
            example_df = pd.DataFrame({
                'Athlete': available_athletes[:10],
                'Rank': range(1, 11)
            })
            st.dataframe(example_df, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your database connection and configuration")
    
    # Show debugging info
    with st.expander("Debug Information"):
        st.text(f"Error details: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Triathlon H2H Analysis Dashboard - Built with Streamlit*")
