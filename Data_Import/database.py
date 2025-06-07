def create_test_tables():
    """
    Create test_athlete and test_athlete_rankings tables in the real database for integration testing.
    """
    engine = get_engine()
    from sqlalchemy import text
    with engine.begin() as conn:
        # Drop if exists for clean test runs
        conn.execute(text("DROP TABLE IF EXISTS test_athlete_rankings"))
        conn.execute(text("DROP TABLE IF EXISTS test_athlete"))
        # Create test_athlete
        conn.execute(text('''
            CREATE TABLE test_athlete (
                athlete_id INTEGER PRIMARY KEY,
                full_name VARCHAR(255) NOT NULL
            )
        '''))
        # Create test_athlete_rankings
        conn.execute(text('''
            CREATE TABLE test_athlete_rankings (
                athlete_id INTEGER,
                athlete_name VARCHAR(255) NOT NULL,
                ranking_cat_name VARCHAR(255) NOT NULL,
                ranking_cat_id INTEGER NOT NULL,
                rank_position INTEGER NOT NULL,
                total_points FLOAT NOT NULL,
                retrieved_at DATE NOT NULL,
                PRIMARY KEY (athlete_name, ranking_cat_name, retrieved_at)
            )
        '''))
    print("Test tables test_athlete and test_athlete_rankings created.")
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Date, Boolean, PrimaryKeyConstraint, Float, BigInteger
)
import os
from config.config import (
    ATHLETE_TABLE_NAME,
    EVENTS_TABLE_NAME,
    RACE_RESULTS_TABLE_NAME,
)
# DB_URI default imported for reference; actual URI chosen at runtime
from config.config import DB_URI as DEFAULT_DB_URI

def get_engine(echo: bool = False):
    """
    Create a SQLAlchemy engine using DB_URI from environment or default.
    """
    uri = os.environ.get('DB_URI', DEFAULT_DB_URI)
    return create_engine(uri, echo=echo)

def initialize_database():
    engine = get_engine()
    metadata = MetaData()

    # Event dimension table
    Table(
        EVENTS_TABLE_NAME, metadata,
        Column('EventID',            Integer, primary_key=True),
        Column('EventName',          String,  nullable=False),
        Column('EventDate',          Date),
        Column('Venue',              String),
        Column('Country',            String),
        Column('CategoryName',       String),
        Column('EventSpecifications',String),
        PrimaryKeyConstraint('EventID', name='pk_evnets')
    )

    # Athlete dimension table
    Table(
        ATHLETE_TABLE_NAME, metadata,
        Column('athlete_id',         Integer, primary_key=True),
        Column('full_name',          String),
        Column('gender',             String),
        Column('country',            String),
        Column('age',                Integer),
        Column('category_to',        Boolean),
        Column('category_coach',     Boolean),
        Column('category_athlete',   Boolean),
        Column('category_medical',   Boolean),
        Column('category_paratriathlete', Boolean),
    )

    # Race‐results fact table with composite PK (athlete_id, EventID, TotalTime)
    Table(
        RACE_RESULTS_TABLE_NAME, metadata,
        Column('athlete_id',     Integer, nullable=False),
        Column('EventID',        Integer, nullable=False),
        Column('ProgID',         Integer),
        Column('Program',        String),
        Column('CategoryName',   String),
        Column('EventSpecifications', String),
        Column('Position',       String),
        Column('TotalTime',      String),
        Column('SwimTime',       String),
        Column('T1',             String),
        Column('BikeTime',       String),
        Column('T2',             String),
        Column('RunTime',        String),
        # checkpoint elapsed times (seconds)
        Column('ElapsedSwim',    BigInteger),
        Column('ElapsedT1',      BigInteger),
        Column('ElapsedBike',    BigInteger),
        Column('ElapsedT2',      BigInteger),
        Column('ElapsedRun',     BigInteger),        # seconds behind leader by program
        Column('BehindSwim',     BigInteger),
        Column('BehindT1',       BigInteger),
        Column('BehindBike',     BigInteger),
        Column('BehindT2',       BigInteger),
        Column('BehindRun',      BigInteger),
        # position rankings at each checkpoint
        Column('Position_at_Swim',     Integer),
        Column('Position_at_T1',       Integer),
        Column('Position_at_Bike',     Integer),
        Column('Position_at_T2',       Integer),
        Column('Position_at_Run',      Integer),
        # position changes between checkpoints (negative = gained positions)
        Column('Swim_to_T1_pos_change',   Integer),
        Column('T1_to_Bike_pos_change',   Integer),
        Column('Bike_to_T2_pos_change',   Integer),
        Column('T2_to_Run_pos_change',    Integer),
        # individual split rankings (rank for each split time within event/program)
        Column('SwimRank',     Integer),
        Column('T1Rank',       Integer),
        Column('BikeRank',     Integer),
        Column('T2Rank',       Integer),
        Column('RunRank',      Integer),
        # Primary key constraint for upsert conflict target (NOT deferrable)
        PrimaryKeyConstraint('athlete_id', 'EventID', 'TotalTime', name='pk_race_results')
    )

    Table(
        'athlete_rankings', metadata,
        Column('athlete_id',      Integer, nullable=True),  # allow null until API matching
        Column('athlete_name',    String,  nullable=False),
        Column('ranking_cat_name', String, nullable=False),
        Column('ranking_cat_id',  Integer, nullable=False),
        Column('rank_position',   Integer, nullable=False),
        Column('total_points',    Float,   nullable=False),
        Column('retrieved_at',    Date,    nullable=False),
        PrimaryKeyConstraint('athlete_name','ranking_cat_name','retrieved_at', name='pk_athlete_rankings')
    )
    # Staging table for historical rankings (no constraints)
    Table(
        'staging_rankings', metadata,
        Column('athlete_id',      Integer, nullable=True),
        Column('athlete_name',    String,  nullable=False),
        Column('ranking_cat_name', String, nullable=False),
        Column('ranking_cat_id',  Integer, nullable=False),
        Column('rank_position',   Integer, nullable=False),
        Column('total_points',    Float,   nullable=False),
        Column('retrieved_at',    Date,    nullable=False)
    )

    metadata.create_all(engine)
    # Ensure unique index for upsert ON CONFLICT target on race_results
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{RACE_RESULTS_TABLE_NAME}_conflict '
                f'ON "{RACE_RESULTS_TABLE_NAME}" (athlete_id, "EventID", "TotalTime")'
            )
        )
    print("Database tables and conflict index ensured.")

if __name__ == "__main__":
    initialize_database()