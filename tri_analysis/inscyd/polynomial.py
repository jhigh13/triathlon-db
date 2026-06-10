"""
INSCYD metabolic polynomials.

The INSCYD external API returns each metabolic curve not as sampled points but
as a fitted polynomial:

    {
      "sse": <float>,                         # fit error
      "function": "A + B * x + C * x**2 ...", # human-readable form
      "parameters": [A, B, C, ...]            # coefficients, ASCENDING order
    }

`parameters` are in ascending order (constant term first), which is exactly the
order ``numpy.polynomial.polynomial.polyval`` expects, so evaluation is direct.

For cycling tests the independent variable ``x`` is power in watts. Output units
follow INSCYD's conventions (VO2 in ml/min/kg, lactate metrics in mmol/l/min,
fat/carbohydrate in kcal/h). Validate against a known test before trusting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.polynomial import polynomial as npoly

# The eight metabolic curves returned as polynomials by /athlete/all_data/.
# (last_data / fueling_metrics return a subset of these.)
METRIC_KEYS: tuple[str, ...] = (
    "oxygen_demand",
    "oxygen_uptake",
    "lactate_production",
    "max_aerobic",
    "lackof_pyruvate",
    "lactate_accumulation",
    "fat",
    "carbohydrate",
)

# Scalar fields returned alongside the curves (units per INSCYD).
SCALAR_KEYS: tuple[str, ...] = (
    "vo2max",
    "vlamax",
    "anaerobic_threshold",
    "anaerobic_threshold_absolute",
    "fatmax",
    "fatmax_power",
    "carb_max",
    "at_percent_vo2max",
    "tau_value",
)


def _to_float(value: Any) -> float | None:
    """Coerce API values (sometimes strings like \"80.00\") to float, or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MetabolicPolynomial:
    """A single fitted metabolic curve, evaluable at any power ``x``."""

    parameters: tuple[float, ...]  # ascending: A + B*x + C*x**2 + ...
    function: str = ""
    sse: float = float("nan")

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the polynomial at power ``x`` (watts for cycling).

        Accepts a scalar or array. INSCYD parameters are ascending-order, which
        matches numpy's ``polyval`` coefficient convention.
        """
        coeffs = np.asarray(self.parameters, dtype=float)
        result = npoly.polyval(np.asarray(x, dtype=float), coeffs)
        if np.isscalar(x) or np.ndim(x) == 0:
            return float(result)
        return result

    # Allow profile.curve(x) call syntax.
    __call__ = evaluate

    @property
    def degree(self) -> int:
        return max(len(self.parameters) - 1, 0)

    @classmethod
    def from_api(cls, payload: dict | None) -> "MetabolicPolynomial | None":
        """Build from a single ``{sse, function, parameters}`` API object."""
        if not payload:
            return None
        params = payload.get("parameters") or []
        coeffs = tuple(c for c in (_to_float(p) for p in params) if c is not None)
        if not coeffs:
            return None
        func = str(payload.get("function", ""))
        if func and "x" in func and any(tok in func for tok in ("exp", "log", "sin", "cos", "/")):
            # All observed INSCYD functions are plain polynomials; flag anything else.
            raise ValueError(f"Unsupported INSCYD function form (not a polynomial): {func!r}")
        return cls(parameters=coeffs, function=func, sse=_to_float(payload.get("sse")) or float("nan"))


@dataclass
class MetabolicTest:
    """One INSCYD test: identifying info, scalar metrics, and curve polynomials."""

    test_id: int | None
    athlete_display_id: int | None
    sport_id: int | None
    first_name: str | None
    last_name: str | None
    created_at: str | None
    scalars: dict[str, float | None]
    curves: dict[str, MetabolicPolynomial]
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, record: dict) -> "MetabolicTest":
        """Parse one element of the API ``results`` list."""
        curves: dict[str, MetabolicPolynomial] = {}
        for key in METRIC_KEYS:
            poly = MetabolicPolynomial.from_api(record.get(key))
            if poly is not None:
                curves[key] = poly
        scalars = {key: _to_float(record.get(key)) for key in SCALAR_KEYS}
        return cls(
            test_id=_to_int(record.get("id")),
            athlete_display_id=_to_int(record.get("athlete_display_id")),
            sport_id=_to_int(record.get("sport_id")),
            first_name=record.get("first_name"),
            last_name=record.get("last_name"),
            created_at=record.get("created_at"),
            scalars=scalars,
            curves=curves,
            raw=record,
        )

    def curve(self, name: str) -> MetabolicPolynomial | None:
        return self.curves.get(name)

    @property
    def athlete_name(self) -> str:
        return " ".join(p for p in (self.first_name, self.last_name) if p).strip()
