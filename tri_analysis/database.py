import os
import sys

# Add project root to path for package imports (only if not already present)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import (
    ATHLETE_TABLE_NAME,
    EVENTS_TABLE_NAME,
    RACE_RESULTS_TABLE_NAME,
    DB_URI as DEFAULT_DB_URI 
)

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Date, Boolean, PrimaryKeyConstraint, Float, BigInteger, text
)

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
                year INTEGER NOT NULL,
                retrieved_at DATE NOT NULL,
                PRIMARY KEY (athlete_name, ranking_cat_name, retrieved_at)
            )
        '''))
    print("Test tables test_athlete and test_athlete_rankings created.")

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
        Column('prog_id',            Integer, primary_key=True),
        Column('event_id',           Integer, primary_key=True),
        Column('prog_name',         String),
        Column('prog_distance_category',String),
        Column('swim_laps',         Integer),
        Column('swim_distance',     Float),
        Column('bike_laps',         Integer),
        Column('bike_distance',     Float),
        Column('run_laps',          Integer),
        Column('run_distance',      Float),
        Column('event_name',        String),
        Column('event_venue',       String),
        Column('event_date',        Date),
        Column('event_country',     String),
        Column('event_latitude',    Float),
        Column('event_longitude',   Float),
        Column('cat_name',          String),
        Column('temperature_water', Float),
        Column('temperature_air',   Float),
        Column('humidity',          Float),
        Column('wbgt',              Float),
        Column('wind',              Float),
        Column('weather',           String),
        Column('wetsuit',           String),
        PrimaryKeyConstraint('event_id', 'prog_id', name='pk_events')
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

    # Race‐results fact table with composite PK (athlete_id, prog_id, total_time)
    Table(
        RACE_RESULTS_TABLE_NAME, metadata,
        Column('event_id',        Integer),
        Column('prog_id',         Integer, primary_key=True),  
        Column('athlete_id',      Integer, primary_key=True),
        Column('athlete_full_name',   String),
        Column('swimtime',        String),
        Column('t1time',          String),
        Column('biketime',        String),
        Column('t2time',          String),
        Column('runtime',         String),
        Column('position',        String),
        Column('total_time',      String, primary_key=True),
        Column('start_num',       String),
        # Primary key constraint for upsert conflict target (NOT deferrable)
        PrimaryKeyConstraint('athlete_id', 'prog_id', 'total_time', name='pk_race_results')
    )

    Table(
        'athlete_rankings', metadata,
        Column('athlete_id',      Integer, nullable=True),  # allow null until API matching
        Column('athlete_name',    String),
        Column('ranking_cat_name', String),
        Column('ranking_cat_id',  Integer),
        Column('rank_position',   Integer),
        Column('total_points',    Float),
        Column('year',            Integer),  # year of the ranking
        Column('retrieved_at',    Date),
        PrimaryKeyConstraint('athlete_name','ranking_cat_name', 'year', 'retrieved_at', name='pk_athlete_rankings')
    )

    Table(
        'position_metrics', metadata,
        # keys
        Column('athlete_id',      Integer),
        Column('event_id',        Integer),
        Column('prog_id',         Integer),
        # total elapsed times for each segment
        Column('elapsedswim',    BigInteger),
        Column('elapsedt1',      BigInteger),
        Column('elapsedbike',    BigInteger),
        Column('elapsedt2',      BigInteger),
        Column('elapsedrun',     BigInteger),        
        Column('behindswim',     BigInteger),
        Column('behindt1',       BigInteger),
        Column('behindbike',     BigInteger),
        Column('behindt2',       BigInteger),
        Column('behindrun',      BigInteger),
        # position rankings at each checkpoint
        Column('position_at_swim',     Integer),
        Column('position_at_t1',       Integer),
        Column('position_at_bike',     Integer),
        Column('position_at_t2',       Integer),
        Column('position_at_run',      Integer),
        # position changes between checkpoints (negative = gained positions)
        Column('swim_to_t1_pos_change',   Integer),
        Column('t1_to_bike_pos_change',   Integer),
        Column('bike_to_t2_pos_change',   Integer),
        Column('t2_to_run_pos_change',    Integer),
        # individual split rankings (rank for each split time within event/program)
        Column('swimrank',     Integer),
        Column('t1rank',       Integer),
        Column('bikerank',     Integer),
        Column('t2rank',       Integer),
        Column('runrank',      Integer),
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
        Column('year',            Integer, nullable=False),  # year of the ranking
        Column('retrieved_at',    Date,    nullable=False)
    )

    metadata.create_all(engine)
    # Ensure unique index for upsert ON CONFLICT target on race_results
    with engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{RACE_RESULTS_TABLE_NAME}_conflict '
                f'ON "{RACE_RESULTS_TABLE_NAME}" (athlete_id, "prog_id", "total_time")'
            )
        )
    print("Database tables and conflict index ensured.")

if __name__ == "__main__":
    initialize_database()