# database/models/predictions.py
"""
PredictionTable schema for storing model predictions in the database.
"""
from sqlalchemy import Column, Integer, Float, Interval, String, DateTime, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PredictionTable(Base):
    __tablename__ = 'athlete_predictions'
    id = Column(Integer, primary_key=True)
    athlete_id = Column(Integer, nullable=False)
    event_id = Column(Integer, nullable=False)
    predicted_time = Column(Interval, nullable=False)
    confidence = Column(Float)
    split_swim = Column(Float)
    split_bike = Column(Float)
    split_run = Column(Float)
    predicted_position = Column(Integer)
    ci_lower = Column(Float)
    ci_upper = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime)
    feature_importance = Column(String)
    accuracy = Column(Float)
