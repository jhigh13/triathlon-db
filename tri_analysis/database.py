import os
import sys

# Add project root to path for package imports (only if not already present)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Preferred: package-qualified import
    from tri_analysis.config import (
        ATHLETE_TABLE_NAME,
        EVENTS_TABLE_NAME,
        RACE_RESULTS_TABLE_NAME,
        DB_URI as DEFAULT_DB_URI,
    )
except ImportError:  # pragma: no cover
    # Back-compat for older entrypoints executed with altered sys.path
    from config import (  # type: ignore
        ATHLETE_TABLE_NAME,
        EVENTS_TABLE_NAME,
        RACE_RESULTS_TABLE_NAME,
        DB_URI as DEFAULT_DB_URI,
    )

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Date, Boolean, PrimaryKeyConstraint, Float, BigInteger, DateTime, text
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
        Column('is_para',           Boolean),
    Column('swim_laps',         Float),
        Column('swim_distance',     Float),
    Column('bike_laps',         Float),
        Column('bike_distance',     Float),
    Column('run_laps',          Float),
        Column('run_distance',      Float),
        Column('event_name',        String),
        Column('event_venue',       String),
        Column('event_date',        Date),
        Column('event_country',     String),
        Column('event_latitude',    Float),
        Column('event_longitude',   Float),
        Column('cat_name',          String),
        Column('temperature_water', String),
        Column('temperature_air',   String),
        Column('humidity',          String),
        Column('wbgt',              String),
        Column('wind',              String),
        Column('weather',           String),
        Column('wetsuit',           String),
        # Open-Meteo enriched weather columns
        Column('wind_speed_kmh',    Float),
        Column('wind_gust_kmh',     Float),
        Column('apparent_temp',     Float),
        Column('precipitation_mm',  Float),
        Column('cloud_cover_pct',   Float),
        Column('wet_bulb_temp',     Float),
        Column('weather_source',    String),
        Column('weather_fetched_at', String),
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
        # New structured fields for finish vs non-finish handling and sorting
        Column('finish_status',   String),   # e.g., FINISH, DNF, DNS, DSQ, LAP
        Column('finish_position', Integer),  # numeric position when finished
        Column('position_sort',   Integer),  # numeric sort key (finishers first by place, then DNFs/DNSs)
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
        Column('events_current_period',  Integer, nullable=True),
        Column('events_previous_period', Integer, nullable=True),
        PrimaryKeyConstraint('athlete_name','ranking_cat_name', 'year', 'retrieved_at', name='pk_athlete_rankings')
    )

    Table(
        'athlete_ranking_breakdown', metadata,
        Column('athlete_id',        Integer),
        Column('ranking_cat_id',    Integer),
        Column('event_id',          Integer),
        Column('event_title',       String),
        Column('event_finish_date', Date),
        Column('points',            Float,   nullable=True),
        Column('period',            Integer, nullable=True),  # 1=current (0-1yr), 2=previous (1-2yr)
        Column('position',          Integer, nullable=True),
        Column('retrieved_at',      Date),
        PrimaryKeyConstraint('athlete_id', 'ranking_cat_id', 'event_id', name='pk_athlete_ranking_breakdown')
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
        # race-level aggregates for fast feature building
        Column('n_finishers',     Integer),
        Column('median_total_sec', Float),
    )

    # WTCS pack membership (precomputed, deterministic; full field per event_id+prog_id)
    Table(
        'wtcs_pack_membership', metadata,
        Column('event_id', Integer, nullable=False),
        Column('prog_id', Integer, nullable=False),
        Column('athlete_id', Integer, nullable=False),
        Column('checkpoint', String, nullable=False),  # 'swim'|'bike'|'run'
        Column('elapsed_sec', BigInteger, nullable=False),
        Column('pos_at_checkpoint', Integer, nullable=False),
        Column('gap_to_prev_sec', Integer, nullable=True),
        Column('gap_to_leader_sec', Integer, nullable=False),
        Column('pack_id', Integer, nullable=False),
        Column('pack_size', Integer, nullable=False),
        Column('max_gap_to_prev_sec', Integer, nullable=False),
        Column('algo_version', String, nullable=False),
        Column('computed_at', DateTime, nullable=False),
        PrimaryKeyConstraint('event_id', 'prog_id', 'athlete_id', 'checkpoint', name='pk_wtcs_pack_membership')
    )

    # Program entries (start lists). One row per entry_id per (event_id, prog_id).
    # Note: we intentionally keep key athlete fields denormalized for quick display,
    # but athlete_id still joins to the athlete dimension when present.
    Table(
        'program_entries', metadata,
        Column('event_id', Integer, nullable=False),
        Column('prog_id', Integer, nullable=False),
        Column('entry_id', Integer, nullable=False),
        Column('entry_type', String, nullable=False),
        Column('approved', Boolean, nullable=True),
        Column('start_num', String, nullable=True),
        Column('api_updated_at', DateTime(timezone=True), nullable=True),
        Column('last_fetched_at', DateTime(timezone=True), nullable=False),
        Column('is_active', Boolean, nullable=False),
        Column('removed_at', DateTime(timezone=True), nullable=True),
        Column('athlete_id', Integer, nullable=True),
        Column('athlete_full_name', String, nullable=True),
        Column('athlete_first', String, nullable=True),
        Column('athlete_last', String, nullable=True),
        Column('athlete_gender', String, nullable=True),
        Column('athlete_country_id', Integer, nullable=True),
        Column('athlete_country_name', String, nullable=True),
        Column('athlete_noc', String, nullable=True),
        Column('athlete_yob', Integer, nullable=True),
        Column('athlete_slug', String, nullable=True),
        Column('validated', Boolean, nullable=True),
        PrimaryKeyConstraint('event_id', 'prog_id', 'entry_id', name='pk_program_entries')
    )
    
    # Elo ratings for athletes (computed by elo_ratings.py)
    Table(
        'athlete_elo_ratings', metadata,
        Column('athlete_id', Integer, nullable=False),
        Column('gender', String, nullable=False),       # 'male' or 'female'
        Column('elo_rating', Float, nullable=False),     # current Elo rating
        Column('elo_peak', Float, nullable=False),       # all-time peak rating
        Column('elo_races', Integer, nullable=False),    # total races processed
        Column('last_race_date', Date, nullable=True),   # date of most recent race
        Column('last_updated', DateTime, nullable=False),
        PrimaryKeyConstraint('athlete_id', 'gender', name='pk_athlete_elo_ratings')
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
    
    # Head-to-Head summary table for Power BI integration
    Table(
        'h2h_summary', metadata,
        Column('athlete_a', String, nullable=False),
        Column('athlete_b', String, nullable=False),
        Column('total_matches', Integer, nullable=False),
        Column('wins_a', Integer, nullable=False),
        Column('wins_b', Integer, nullable=False),
        Column('win_pct_a', Float, nullable=False),
        Column('win_pct_b', Float, nullable=False),
        # Segment statistics
        Column('swim_wins_a', Integer),
        Column('swim_wins_b', Integer),
        Column('swim_win_pct_a', Float),
        Column('swim_total', Integer),
        Column('t1_wins_a', Integer),
        Column('t1_wins_b', Integer),
        Column('t1_win_pct_a', Float),
        Column('t1_total', Integer),
        Column('bike_wins_a', Integer),
        Column('bike_wins_b', Integer),
        Column('bike_win_pct_a', Float),
        Column('bike_total', Integer),
        Column('t2_wins_a', Integer),
        Column('t2_wins_b', Integer),
        Column('t2_win_pct_a', Float),
        Column('t2_total', Integer),
        Column('run_wins_a', Integer),
        Column('run_wins_b', Integer),
        Column('run_win_pct_a', Float),
        Column('run_total', Integer),
        # Metadata
        Column('date_range_start', Date),
        Column('date_range_end', Date),
        Column('last_updated', Date, nullable=False),
        Column('min_matches_filter', Integer, nullable=False)
    )
    
    # Apply lightweight migrations to align existing DB schema with updated types
    try:
        with engine.begin() as conn:
            # Convert weather metrics from numeric to text if needed
            # NOTE: DO $$ blocks cannot use bind parameters; inline table name safely.
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'temperature_water' AND data_type <> 'text'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN temperature_water TYPE VARCHAR USING temperature_water::text';
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'temperature_air' AND data_type <> 'text'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN temperature_air TYPE VARCHAR USING temperature_air::text';
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'humidity' AND data_type <> 'text'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN humidity TYPE VARCHAR USING humidity::text';
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'wbgt' AND data_type <> 'text'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN wbgt TYPE VARCHAR USING wbgt::text';
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'wind' AND data_type <> 'text'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN wind TYPE VARCHAR USING wind::text';
                    END IF;
                END$$;
            """))
            # Widen lap columns to double precision (FLOAT) to allow NaN and avoid integer range issues
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'swim_laps' AND data_type <> 'double precision'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN swim_laps TYPE DOUBLE PRECISION';
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'bike_laps' AND data_type <> 'double precision'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN bike_laps TYPE DOUBLE PRECISION';
                    END IF;
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{EVENTS_TABLE_NAME}' AND column_name = 'run_laps' AND data_type <> 'double precision'
                    ) THEN
                        EXECUTE 'ALTER TABLE "{EVENTS_TABLE_NAME}" ALTER COLUMN run_laps TYPE DOUBLE PRECISION';
                    END IF;
                END$$;
            """))
    except Exception:
        # Non-fatal; schema may already match
        pass

    metadata.create_all(engine)
    # Ensure unique index for upsert ON CONFLICT target on race_results
    with engine.begin() as conn:
        # Add is_para to events if missing (safe for existing databases)
        conn.execute(
            text(
                f'ALTER TABLE "{EVENTS_TABLE_NAME}" ADD COLUMN IF NOT EXISTS is_para BOOLEAN'
            )
        )
        # Backfill heuristic: mark events as para when program name or category indicates para
        conn.execute(
            text(
                f"""
                UPDATE "{EVENTS_TABLE_NAME}"
                SET is_para = TRUE
                WHERE (is_para IS DISTINCT FROM TRUE)
                  AND (
                        "prog_name" ILIKE 'PTWC%%'
                     OR "prog_name" ILIKE 'PTS2%%'
                     OR "prog_name" ILIKE 'PTS3%%'
                     OR "prog_name" ILIKE 'PTS4%%'
                     OR "prog_name" ILIKE 'PTS5%%'
                     OR "prog_name" ILIKE 'PTVI%%'
                     OR COALESCE("cat_name", '') ILIKE '%para%'
                  )
                """
            )
        )
        # Add Open-Meteo enriched weather columns if missing
        for col, col_type in [
            ("wind_speed_kmh", "DOUBLE PRECISION"),
            ("wind_gust_kmh", "DOUBLE PRECISION"),
            ("apparent_temp", "DOUBLE PRECISION"),
            ("precipitation_mm", "DOUBLE PRECISION"),
            ("cloud_cover_pct", "DOUBLE PRECISION"),
            ("wet_bulb_temp", "DOUBLE PRECISION"),
            ("weather_source", "VARCHAR"),
            ("weather_fetched_at", "VARCHAR"),
        ]:
            conn.execute(text(f'ALTER TABLE "{EVENTS_TABLE_NAME}" ADD COLUMN IF NOT EXISTS {col} {col_type}'))
        # Add finish-related columns to race_results if missing (safe migrations)
        conn.execute(text(f'ALTER TABLE "{RACE_RESULTS_TABLE_NAME}" ADD COLUMN IF NOT EXISTS finish_status VARCHAR'))
        conn.execute(text(f'ALTER TABLE "{RACE_RESULTS_TABLE_NAME}" ADD COLUMN IF NOT EXISTS finish_position INTEGER'))
        conn.execute(text(f'ALTER TABLE "{RACE_RESULTS_TABLE_NAME}" ADD COLUMN IF NOT EXISTS position_sort INTEGER'))
        # Add new race-level aggregates to position_metrics if missing
        conn.execute(text('ALTER TABLE "position_metrics" ADD COLUMN IF NOT EXISTS n_finishers INTEGER'))
        conn.execute(text('ALTER TABLE "position_metrics" ADD COLUMN IF NOT EXISTS median_total_sec DOUBLE PRECISION'))
        # Backfill finish_status and finish_position from existing position values (cast numeric positions)
        conn.execute(
            text(
                f"""
                UPDATE "{RACE_RESULTS_TABLE_NAME}"
                SET finish_position = CASE
                        WHEN position ~ '^[0-9]+$' THEN position::integer
                        ELSE NULL
                    END,
                    finish_status = CASE
                        WHEN position ~ '^[0-9]+$' THEN 'FINISH'
                        ELSE UPPER(TRIM(position))
                    END
                WHERE finish_status IS NULL OR finish_position IS NULL
                """
            )
        )
        # Backfill position_sort: finishers use finish_position; non-finishers use integer sentinels (higher = worse)
        conn.execute(
            text(
                f"""
                UPDATE "{RACE_RESULTS_TABLE_NAME}"
                SET position_sort = COALESCE(
                    finish_position,
                    CASE finish_status
                        WHEN 'DNF' THEN 1000001
                        WHEN 'DNS' THEN 1000002
                        WHEN 'DSQ' THEN 1000003
                        WHEN 'LAP' THEN 1000004
                        ELSE 1000009
                    END
                )
                WHERE position_sort IS NULL
                """
            )
        )
        # Ensure finish_position and position_sort are stored as INTEGER in case older schemas used VARCHAR
        # NOTE: DO $$ blocks cannot use bind parameters; inline table name safely.
        conn.execute(text(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = '{RACE_RESULTS_TABLE_NAME}' AND column_name = 'finish_position' AND data_type <> 'integer'
                ) THEN
                    EXECUTE 'ALTER TABLE "{RACE_RESULTS_TABLE_NAME}" ALTER COLUMN finish_position TYPE INTEGER USING NULLIF(TRIM(finish_position), '''')::integer';
                END IF;
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = '{RACE_RESULTS_TABLE_NAME}' AND column_name = 'position_sort' AND data_type <> 'integer'
                ) THEN
                    EXECUTE 'ALTER TABLE "{RACE_RESULTS_TABLE_NAME}" ALTER COLUMN position_sort TYPE INTEGER USING NULLIF(TRIM(position_sort), '''')::integer';
                END IF;
            END$$;
            """
        ))
        conn.execute(
            text(
                f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{RACE_RESULTS_TABLE_NAME}_conflict '
                f'ON "{RACE_RESULTS_TABLE_NAME}" (athlete_id, "prog_id", "total_time")'
            )
        )

        # Helpful indexes for WTCS pack membership lookups
        conn.execute(text(
            'CREATE INDEX IF NOT EXISTS idx_wtcs_pack_membership_event_prog_checkpoint '
            'ON wtcs_pack_membership (event_id, prog_id, checkpoint)'
        ))
        conn.execute(text(
            'CREATE INDEX IF NOT EXISTS idx_wtcs_pack_membership_athlete_checkpoint '
            'ON wtcs_pack_membership (athlete_id, checkpoint)'
        ))

        # Helpful indexes for program entries lookups
        conn.execute(text(
            'CREATE INDEX IF NOT EXISTS idx_program_entries_event_prog_active '
            'ON program_entries (event_id, prog_id, entry_type, is_active)'
        ))
        conn.execute(text(
            'CREATE INDEX IF NOT EXISTS idx_program_entries_athlete '
            'ON program_entries (athlete_id)'
        ))
        # Elo ratings indexes
        conn.execute(text(
            'CREATE INDEX IF NOT EXISTS idx_athlete_elo_ratings_rating '
            'ON athlete_elo_ratings (gender, elo_rating DESC)'
        ))

        # Ensure critical integer columns are wide enough (avoid smallint overflows)
        # Note: ALTER TYPE to same type is a no-op; this is safe to run repeatedly.
        #conn.execute(text(f'ALTER TABLE "{RACE_RESULTS_TABLE_NAME}" ALTER COLUMN finish_position TYPE INTEGER'))
        #conn.execute(text(f'ALTER TABLE "{RACE_RESULTS_TABLE_NAME}" ALTER COLUMN position_sort TYPE INTEGER'))
    print("Database tables and conflict index ensured.")

if __name__ == "__main__":
    initialize_database()