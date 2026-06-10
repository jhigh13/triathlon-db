"""
INSCYD integration: fetch athlete metabolic test data (polynomials + scalars),
evaluate the curves, and cache payloads to disk.

Typical usage::

    from tri_analysis.inscyd import InscydClient, MetabolicTest, save_raw_tests

    client = InscydClient()                       # reads INSCYD_* from .env
    client.verify_access(user_email="me@org.com") # light auth check first
    raw = client.get_all_data(user_email="me@org.com", sport_id=BIKE_SPORT_ID)
    save_raw_tests(raw, sport_id=BIKE_SPORT_ID)   # fetch once, reuse later
    tests = [MetabolicTest.from_api(r) for r in raw]
    vo2 = tests[0].curve("oxygen_uptake").evaluate(300)  # ml/min/kg at 300 W
"""

from .client import InscydAPIError, InscydClient
from .polynomial import (
    METRIC_KEYS,
    SCALAR_KEYS,
    MetabolicPolynomial,
    MetabolicTest,
)
from .storage import DEFAULT_DIR, load_raw_tests, save_raw_tests

__all__ = [
    "InscydClient",
    "InscydAPIError",
    "MetabolicPolynomial",
    "MetabolicTest",
    "METRIC_KEYS",
    "SCALAR_KEYS",
    "save_raw_tests",
    "load_raw_tests",
    "DEFAULT_DIR",
]
