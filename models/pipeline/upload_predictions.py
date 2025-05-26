# models/pipeline/upload_predictions.py
"""
Upload model predictions to the athlete_predictions table in the database.
"""
from sqlalchemy.orm import sessionmaker
from database.models.predictions import PredictionTable
from sqlalchemy import create_engine
from config.config import DB_URI
import pandas as pd

def upload_predictions(predictions_df: pd.DataFrame):
    engine = create_engine(DB_URI)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        for _, row in predictions_df.iterrows():
            pred = PredictionTable(
                athlete_id = row['athlete_id'],
                event_id = row['event_id'],
                predicted_time = row['predicted_time'],
                confidence = row.get('confidence'),
                split_swim = row.get('split_swim'),
                split_bike = row.get('split_bike'),
                split_run = row.get('split_run'),
                predicted_position = row.get('predicted_position'),
                ci_lower = row.get('ci_lower'),
                ci_upper = row.get('ci_upper'),
                model_version = row.get('model_version'),
                created_at = row.get('created_at'),
                feature_importance = row.get('feature_importance'),
                accuracy = row.get('accuracy'),
            )
            session.add(pred)
        session.commit()
    finally:
        session.close()
