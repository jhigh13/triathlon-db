import pytest
import pandas as pd
from ml.data_extraction import DataExtractor
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with test data for testing."""
    # Create an in-memory SQLite engine for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create tables and insert test data all in one script
    with engine.connect() as conn:
        conn.connection.executescript(
            """
            CREATE TABLE athlete (
                athlete_id INTEGER PRIMARY KEY,
                full_name TEXT,
                gender TEXT,
                country TEXT,
                age INTEGER
            );
            
            CREATE TABLE events (
                EventID INTEGER PRIMARY KEY,
                EventName TEXT,
                EventDate TEXT,
                Venue TEXT,
                Country TEXT,
                CategoryName TEXT,
                EventSpecifications TEXT
            );
            
            CREATE TABLE race_results (
                athlete_id INTEGER,
                EventID INTEGER,
                TotalTime TEXT,
                Position INTEGER,
                CategoryName TEXT,
                EventSpecifications TEXT
            );
            
            INSERT INTO athlete VALUES (1, 'Alice', 'F', 'USA', 30);
            INSERT INTO athlete VALUES (2, 'Bob', 'M', 'GBR', 28);
            
            INSERT INTO events VALUES (10, 'Test Event', '2023-01-01', 'London', 'GBR', 'Elite Men', 'Triathlon, Sprint');
            INSERT INTO events VALUES (11, 'Test Event 2', '2023-02-01', 'Paris', 'FRA', 'Elite Women', 'Triathlon, Olympic');
            
            INSERT INTO race_results VALUES (1, 10, '01:00:00', 5, 'Elite Men', 'Triathlon, Sprint');
            INSERT INTO race_results VALUES (2, 10, '01:05:00', 8, 'Elite Men', 'Triathlon, Sprint');
            INSERT INTO race_results VALUES (1, 11, '02:10:00', 3, 'Elite Women', 'Triathlon, Olympic');
            """
        )
    return engine


def test_data_extractor_loads_tables(in_memory_engine):
    """Test that DataExtractor can load and join tables correctly."""
    extractor = DataExtractor(in_memory_engine)
    
    # Test individual table loading first
    athletes_df, events_df, results_df = extractor.load_raw_tables()
    
    # Verify individual tables have data
    assert not athletes_df.empty, "Athletes table is empty"
    assert not events_df.empty, "Events table is empty"
    assert not results_df.empty, "Results table is empty"
    
    # Check expected columns exist
    assert 'athlete_id' in athletes_df.columns
    assert 'full_name' in athletes_df.columns
    assert 'EventID' in events_df.columns
    assert 'EventName' in events_df.columns
    assert 'athlete_id' in results_df.columns
    assert 'TotalTime' in results_df.columns
    
    # Test the join
    joined_df = extractor.join_tables(athletes_df, events_df, results_df)
    assert not joined_df.empty, "Joined dataset is empty"
    assert len(joined_df) >= 2, "Should have at least 2 race results"
    
    # Verify join worked correctly - should have columns from all tables
    assert 'athlete_id' in joined_df.columns
    assert 'EventID' in joined_df.columns
    assert 'TotalTime' in joined_df.columns
    assert 'full_name' in joined_df.columns
    assert 'EventName' in joined_df.columns
    
    # Extract full dataset (end-to-end test)
    df = extractor.extract_base_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) >= 2


def test_data_extractor_join_logic(in_memory_engine):
    """Test the specific join logic between tables."""
    extractor = DataExtractor(in_memory_engine)
    athletes_df, events_df, results_df = extractor.load_raw_tables()
    
    # Test the join produces expected results
    joined_df = extractor.join_tables(athletes_df, events_df, results_df)
    
    # Should have 3 race results total
    assert len(joined_df) == 3
    
    # Alice should appear twice (events 10 and 11)
    alice_results = joined_df[joined_df['full_name'] == 'Alice']
    assert len(alice_results) == 2
    
    # Bob should appear once (event 10)
    bob_results = joined_df[joined_df['full_name'] == 'Bob']
    assert len(bob_results) == 1
    
    # All results should have proper event names
    assert 'Test Event' in joined_df['EventName'].values
    assert 'Test Event 2' in joined_df['EventName'].values


def test_data_extractor_handles_empty_tables(in_memory_engine):
    """Test DataExtractor behavior with empty tables."""
    # Create engine with empty tables
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.connection.executescript(
            """
            CREATE TABLE athlete (
                athlete_id INTEGER PRIMARY KEY,
                full_name TEXT,
                gender TEXT,
                country TEXT,
                age INTEGER
            );
            CREATE TABLE events (
                EventID INTEGER PRIMARY KEY,
                EventName TEXT,
                EventDate TEXT,
                Venue TEXT,
                Country TEXT,
                CategoryName TEXT,
                EventSpecifications TEXT
            );
            CREATE TABLE race_results (
                athlete_id INTEGER,
                EventID INTEGER,
                TotalTime TEXT,
                Position INTEGER,
                CategoryName TEXT,
                EventSpecifications TEXT
            );
            """
        )
    
    extractor = DataExtractor(engine)
    athletes_df, events_df, results_df = extractor.load_raw_tables()
    
    # All should be empty but with correct structure
    assert athletes_df.empty
    assert events_df.empty
    assert results_df.empty
    
    # Join should also be empty
    joined_df = extractor.join_tables(athletes_df, events_df, results_df)
    assert joined_df.empty
