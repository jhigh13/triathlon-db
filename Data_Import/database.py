# Data_Import/database.py

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Date, Boolean, PrimaryKeyConstraint, Float
)
from config.config import DB_URI

def get_engine(echo: bool = False):
    return create_engine(DB_URI, echo=echo)

def initialize_database():
    engine = get_engine()
    metadata = MetaData()

    # Event dimension table
    Table(
        'events', metadata,
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
        'athlete', metadata,
        Column('athlete_id',         Integer, primary_key=True),
        Column('full_name',          String),
        Column('gender',             String),
        Column('country_name',       String),
        Column('age',                Integer),
        Column('category_to',        Boolean),
        Column('category_coach',     Boolean),
        Column('category_athlete',   Boolean),
        Column('category_medical',   Boolean),
        Column('category_paratriathlete', Boolean),
    )

    # Race‐results fact table with composite PK (athlete_id, EventID, TotalTime)
    Table(
        'race_results', metadata,
        Column('athlete_id',   Integer, nullable=False),
        Column('EventID',      Integer, nullable=False),
        Column('ProgID',    Integer),
        Column('Program',      String),
        Column('CategoryName', String),
        Column('EventSpecifications', String),
        Column('Position',     String),
        Column('TotalTime',    String),
        Column('SwimTime',     String),
        Column('T1',           String),
        Column('BikeTime',     String),
        Column('T2',           String),
        Column('RunTime',      String),
        #PrimaryKeyConstraint('athlete_id', 'EventID', 'TotalTime', name='pk_race_results')
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
    print("Database tables created (if not existing).")

if __name__ == "__main__":
    initialize_database()
