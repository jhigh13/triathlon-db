"""Offline tests for INSCYD polynomial parsing/evaluation (no network)."""

from __future__ import annotations

import math

import numpy as np

from tri_analysis.inscyd.polynomial import (
    METRIC_KEYS,
    MetabolicPolynomial,
    MetabolicTest,
)


def test_ascending_coefficient_order():
    # A + B*x + C*x**2 with [1, 2, 3] at x=2 -> 1 + 4 + 12 = 17
    poly = MetabolicPolynomial(parameters=(1.0, 2.0, 3.0))
    assert poly.evaluate(2.0) == 17.0
    assert poly(0.0) == 1.0  # __call__ alias, constant term


def test_linear_from_api_matches_email_example():
    # oxygen_demand from the API docs (test id 169996740170): "A + B * x"
    payload = {
        "sse": 2.8208424326745623,
        "function": "A + B * x",
        "parameters": [1.0245463830136188, 0.1822562213342966],
    }
    poly = MetabolicPolynomial.from_api(payload)
    assert poly is not None
    assert poly.degree == 1
    assert math.isclose(poly.evaluate(0.0), 1.0245463830136188, rel_tol=1e-12)
    assert math.isclose(poly.evaluate(100.0), 1.0245463830136188 + 0.1822562213342966 * 100.0, rel_tol=1e-12)


def test_evaluate_vectorized():
    poly = MetabolicPolynomial(parameters=(0.0, 1.0))  # f(x) = x
    out = poly.evaluate(np.array([1.0, 2.0, 3.0]))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_from_api_empty_returns_none():
    assert MetabolicPolynomial.from_api(None) is None
    assert MetabolicPolynomial.from_api({"parameters": []}) is None


def test_non_polynomial_function_rejected():
    bad = {"function": "A + B * exp(x)", "parameters": [1.0, 2.0]}
    try:
        MetabolicPolynomial.from_api(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-polynomial function")


def test_metabolic_test_from_api_coerces_string_scalars():
    record = {
        "id": 169996740170,
        "sport_id": 61,
        "athlete_display_id": 154662600022,
        "first_name": "Test",
        "last_name": "Athlet",
        "created_at": "2023-11-14T13:10:08.590286Z",
        "vo2max": "80.00",   # string from API -> must coerce to float
        "vlamax": "0.70",
        "anaerobic_threshold": 0.0,
        "at_percent_vo2max": 78.3,
        "oxygen_demand": {
            "sse": 2.82,
            "function": "A + B * x",
            "parameters": [1.0245463830136188, 0.1822562213342966],
        },
        "carbohydrate": {
            "sse": 296.9,
            "function": "A + B * x",
            "parameters": [10.0, 0.5],
        },
    }
    test = MetabolicTest.from_api(record)
    assert test.test_id == 169996740170
    assert test.athlete_display_id == 154662600022
    assert test.sport_id == 61
    assert test.athlete_name == "Test Athlet"
    assert math.isclose(test.scalars["vo2max"], 80.0)
    assert math.isclose(test.scalars["vlamax"], 0.70)
    # Only the curves present in the record are parsed.
    assert set(test.curves) == {"oxygen_demand", "carbohydrate"}
    assert math.isclose(test.curve("carbohydrate").evaluate(100.0), 60.0)
    # All metric keys are known constants.
    assert "oxygen_uptake" in METRIC_KEYS
