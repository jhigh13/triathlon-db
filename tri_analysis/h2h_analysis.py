# tri_analysis/h2h_analysis.py
"""
Head-to-head analysis functions for triathlon athlete comparisons.

This module provides functions to:
1. Calculate win-loss records between athlete pairs
2. Generate time gap statistics (average, min, max)
3. Create annotated heatmaps for visualization
4. Support both overall and segment-level analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from .config import DB_URI


def get_engine():
    """Create and return SQLAlchemy engine for database connection."""
    return create_engine(DB_URI)


def h2h_summary(athlete_list: List[str], 
                event_filter: Optional[str] = None,
                segment: str = 'overall') -> pd.DataFrame:
    """
    Generate head-to-head summary matrix for a list of athletes.
    
    Parameters:
    -----------
    athlete_list : List[str]
        List of athlete names to analyze
    event_filter : Optional[str]
        SQL WHERE clause to filter events (e.g., "EventID = 123")
    segment : str
        Analysis segment: 'overall', 'swim', 't1', 'bike', 't2', 'run'
        
    Returns:
    --------
    pd.DataFrame
        Matrix with win counts for each athlete pair
    """
    engine = get_engine()
    
    # Format athlete list for SQL IN clause
    athlete_names_str = "', '".join(athlete_list)
    
    # Build the base query
    base_conditions = f"r1.athlete_name IN ('{athlete_names_str}') AND r2.athlete_name IN ('{athlete_names_str}')"
    
    if event_filter:
        base_conditions += f" AND {event_filter}"
    
    # Define segment-specific comparisons
    segment_comparisons = {
        'overall': "r1.TotalTime < r2.TotalTime",
        'swim': "r1.swim_time < r2.swim_time",
        't1': "r1.t1_time < r2.t1_time", 
        'bike': "r1.bike_time < r2.bike_time",
        't2': "r1.t2_time < r2.t2_time",
        'run': "r1.run_time < r2.run_time"
    }
    
    comparison = segment_comparisons.get(segment, segment_comparisons['overall'])
    
    query = f"""
    SELECT 
        r1.athlete_name as athlete1,
        r2.athlete_name as athlete2,
        COUNT(*) as wins
    FROM race_results r1
    JOIN race_results r2 ON r1.EventID = r2.EventID 
                        AND r1.ProgramID = r2.ProgramID
                        AND r1.athlete_name != r2.athlete_name
    WHERE {base_conditions}
      AND {comparison}
      AND r1.TotalTime IS NOT NULL 
      AND r2.TotalTime IS NOT NULL
    GROUP BY r1.athlete_name, r2.athlete_name
    """
    
    # Execute query and create matrix
    df = pd.read_sql(query, engine)
    
    # Create a complete matrix with all athletes
    matrix = pd.DataFrame(0, index=athlete_list, columns=athlete_list)
    
    # Fill in the wins
    for _, row in df.iterrows():
        if row['athlete1'] in matrix.index and row['athlete2'] in matrix.columns:
            matrix.loc[row['athlete1'], row['athlete2']] = row['wins']
    
    return matrix


def calculate_time_gaps(athlete_list: List[str], 
                       event_filter: Optional[str] = None,
                       segment: str = 'overall') -> Dict[str, pd.DataFrame]:
    """
    Calculate time gap statistics (average, min, max) between athlete pairs.
    
    Parameters:
    -----------
    athlete_list : List[str]
        List of athlete names to analyze
    event_filter : Optional[str]
        SQL WHERE clause to filter events
    segment : str
        Analysis segment: 'overall', 'swim', 't1', 'bike', 't2', 'run'
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with 'avg', 'min', 'max' time gap matrices
    """
    engine = get_engine()
    
    # Format athlete list for SQL IN clause
    athlete_names_str = "', '".join(athlete_list)
    
    # Build the base query
    base_conditions = f"r1.athlete_name IN ('{athlete_names_str}') AND r2.athlete_name IN ('{athlete_names_str}')"
    
    if event_filter:
        base_conditions += f" AND {event_filter}"
    
    # Define segment-specific time columns
    segment_columns = {
        'overall': ('r1.TotalTime', 'r2.TotalTime'),
        'swim': ('r1.swim_time', 'r2.swim_time'),
        't1': ('r1.t1_time', 'r2.t1_time'),
        'bike': ('r1.bike_time', 'r2.bike_time'),
        't2': ('r1.t2_time', 'r2.t2_time'),
        'run': ('r1.run_time', 'r2.run_time')
    }
    
    time_col1, time_col2 = segment_columns.get(segment, segment_columns['overall'])
    
    query = f"""
    SELECT 
        r1.athlete_name as athlete1,
        r2.athlete_name as athlete2,
        AVG({time_col1} - {time_col2}) as avg_gap,
        MIN({time_col1} - {time_col2}) as min_gap,
        MAX({time_col1} - {time_col2}) as max_gap
    FROM race_results r1
    JOIN race_results r2 ON r1.EventID = r2.EventID 
                        AND r1.ProgramID = r2.ProgramID
                        AND r1.athlete_name != r2.athlete_name
    WHERE {base_conditions}
      AND {time_col1} IS NOT NULL 
      AND {time_col2} IS NOT NULL
    GROUP BY r1.athlete_name, r2.athlete_name
    HAVING COUNT(*) >= 1
    """
    
    # Execute query
    df = pd.read_sql(query, engine)
    
    # Create matrices for each statistic
    matrices = {}
    for stat in ['avg_gap', 'min_gap', 'max_gap']:
        matrix = pd.DataFrame(np.nan, index=athlete_list, columns=athlete_list)
        
        # Fill in the values
        for _, row in df.iterrows():
            if row['athlete1'] in matrix.index and row['athlete2'] in matrix.columns:
                matrix.loc[row['athlete1'], row['athlete2']] = row[stat]
        
        matrices[stat.replace('_gap', '')] = matrix
    
    return matrices


def format_time_annotation(seconds: float) -> str:
    """
    Format time difference in seconds to MM:SS format for annotation.
    
    Parameters:
    -----------
    seconds : float
        Time difference in seconds
        
    Returns:
    --------
    str
        Formatted time string (e.g., "1:23" for 83 seconds)
    """
    if pd.isna(seconds):
        return ""
    
    abs_seconds = abs(seconds)
    minutes = int(abs_seconds // 60)
    secs = int(abs_seconds % 60)
    sign = "-" if seconds < 0 else "+"
    
    return f"{sign}{minutes}:{secs:02d}"


def create_h2h_heatmap(win_matrix: pd.DataFrame, 
                      time_gaps: Optional[Dict[str, pd.DataFrame]] = None,
                      title: str = "Head-to-Head Win Matrix",
                      gap_type: str = 'avg',
                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create an annotated heatmap for head-to-head analysis.
    
    Parameters:
    -----------
    win_matrix : pd.DataFrame
        Matrix of win counts between athletes
    time_gaps : Optional[Dict[str, pd.DataFrame]]
        Dictionary of time gap matrices ('avg', 'min', 'max')
    title : str
        Title for the heatmap
    gap_type : str
        Type of time gap to show ('avg', 'min', 'max')
    figsize : Tuple[int, int]
        Figure size for the plot
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create annotations combining wins and time gaps
    annotations = win_matrix.astype(str)
    
    if time_gaps and gap_type in time_gaps:
        gap_matrix = time_gaps[gap_type]
        for i in range(len(win_matrix)):
            for j in range(len(win_matrix.columns)):
                wins = win_matrix.iloc[i, j]
                gap = gap_matrix.iloc[i, j]
                
                if wins > 0 and not pd.isna(gap):
                    time_str = format_time_annotation(gap)
                    annotations.iloc[i, j] = f"{wins}\n{time_str}"
                elif wins > 0:
                    annotations.iloc[i, j] = str(wins)
                else:
                    annotations.iloc[i, j] = ""
    
    # Create the heatmap
    sns.heatmap(win_matrix, 
                annot=annotations, 
                fmt='', 
                cmap='RdYlBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Wins'},
                ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Opponent', fontsize=12, fontweight='bold')
    ax.set_ylabel('Athlete', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig


def get_athlete_matchups_count(athlete_list: List[str], 
                              event_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Get the number of times each pair of athletes has raced against each other.
    
    Parameters:
    -----------
    athlete_list : List[str]
        List of athlete names to analyze
    event_filter : Optional[str]
        SQL WHERE clause to filter events
        
    Returns:
    --------
    pd.DataFrame
        Matrix showing number of head-to-head matchups
    """
    engine = get_engine()
    
    # Format athlete list for SQL IN clause
    athlete_names_str = "', '".join(athlete_list)
    
    # Build the base query
    base_conditions = f"r1.athlete_name IN ('{athlete_names_str}') AND r2.athlete_name IN ('{athlete_names_str}')"
    
    if event_filter:
        base_conditions += f" AND {event_filter}"
    
    query = f"""
    SELECT 
        r1.athlete_name as athlete1,
        r2.athlete_name as athlete2,
        COUNT(*) as matchups
    FROM race_results r1
    JOIN race_results r2 ON r1.EventID = r2.EventID 
                        AND r1.ProgramID = r2.ProgramID
                        AND r1.athlete_name != r2.athlete_name
    WHERE {base_conditions}
      AND r1.TotalTime IS NOT NULL 
      AND r2.TotalTime IS NOT NULL
    GROUP BY r1.athlete_name, r2.athlete_name
    """
    
    # Execute query and create matrix
    df = pd.read_sql(query, engine)
    
    # Create a complete matrix with all athletes
    matrix = pd.DataFrame(0, index=athlete_list, columns=athlete_list)
    
    # Fill in the matchup counts
    for _, row in df.iterrows():
        if row['athlete1'] in matrix.index and row['athlete2'] in matrix.columns:
            matrix.loc[row['athlete1'], row['athlete2']] = row['matchups']
    
    return matrix


def get_available_athletes(min_races: int = 10, 
                          event_filter: Optional[str] = None,
                          limit: int = 50) -> List[str]:
    """
    Get a list of athletes with sufficient race data for analysis.
    
    Parameters:
    -----------
    min_races : int
        Minimum number of races an athlete must have
    event_filter : Optional[str]
        SQL WHERE clause to filter events
    limit : int
        Maximum number of athletes to return
        
    Returns:
    --------
    List[str]
        List of athlete names suitable for analysis
    """
    engine = get_engine()
    
    where_clause = "WHERE TotalTime IS NOT NULL"
    if event_filter:
        where_clause += f" AND {event_filter}"
    
    query = f"""
    SELECT 
        athlete_name,
        COUNT(*) as race_count
    FROM race_results
    {where_clause}
    GROUP BY athlete_name
    HAVING COUNT(*) >= {min_races}
    ORDER BY race_count DESC
    LIMIT {limit}
    """
    
    df = pd.read_sql(query, engine)
    return df['athlete_name'].tolist()


# Convenience function for quick analysis
def quick_h2h_analysis(athlete_list: List[str], 
                      event_filter: Optional[str] = None,
                      segment: str = 'overall',
                      show_time_gaps: bool = True,
                      gap_type: str = 'avg') -> plt.Figure:
    """
    Perform a complete head-to-head analysis and return the visualization.
    
    Parameters:
    -----------
    athlete_list : List[str]
        List of athlete names to analyze
    event_filter : Optional[str]
        SQL WHERE clause to filter events
    segment : str
        Analysis segment: 'overall', 'swim', 't1', 'bike', 't2', 'run'
    show_time_gaps : bool
        Whether to include time gap annotations
    gap_type : str
        Type of time gap to show ('avg', 'min', 'max')
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with the heatmap
    """
    # Get win matrix
    win_matrix = h2h_summary(athlete_list, event_filter, segment)
    
    # Get time gaps if requested
    time_gaps = None
    if show_time_gaps:
        time_gaps = calculate_time_gaps(athlete_list, event_filter, segment)
    
    # Create title
    segment_title = segment.title() if segment != 'overall' else 'Overall'
    title = f"Head-to-Head Analysis - {segment_title}"
    if event_filter:
        title += " (Filtered)"
    
    # Create heatmap
    return create_h2h_heatmap(win_matrix, time_gaps, title, gap_type)
