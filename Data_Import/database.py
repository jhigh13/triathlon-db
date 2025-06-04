

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Date, Boolean, PrimaryKeyConstraint, Float, BigInteger
)
from config.config import (
    ATHLETE_TABLE_NAME,
    EVENTS_TABLE_NAME,
    RACE_RESULTS_TABLE_NAME,
)
from config.config import DB_URI

def get_engine(echo: bool = False):
    return create_engine(DB_URI, echo=echo)

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
        # Primary key constraint for upsert conflict target (NOT deferrable)
        PrimaryKeyConstraint('athlete_id', 'EventID', 'TotalTime', name='pk_race_results')
    )

    Table(
        'athlete_rankings', metadata,
        Column('athlete_id',      Integer, nullable=False),
        Column('ranking_cat_id',  Integer, nullable=False),
        Column('rank_position',   Integer, nullable=False),
        Column('total_points',    Float,   nullable=False),
        Column('retrieved_at',    Date,    nullable=False),
        PrimaryKeyConstraint('athlete_id','ranking_cat_id','retrieved_at', name='pk_athlete_rankings')
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