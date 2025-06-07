import logging
from typing import Optional
from sqlalchemy import text
from Data_Import.database import get_engine

logger = logging.getLogger(__name__)

def match_athlete_id(athlete_name: str) -> Optional[int]:
    """
    Match the given athlete_name to an existing athlete_id in the athlete table.
    Returns the athlete_id if found, otherwise None.
    """
    engine = get_engine()
    query = text("SELECT athlete_id FROM athlete WHERE full_name = :name")
    with engine.connect() as conn:
        result = conn.execute(query, {"name": athlete_name}).fetchone()
        if result:
            return result[0]
        else:
            logger.warning(f"No athlete_id found for name '{athlete_name}'")
            return None

