# Time Trial Bike Speed Optimization Model

## Purpose

Build a simulator that estimates the fastest possible bike split for an athlete on a known time trial course by combining:

1. **Physics**: how fast a given power output moves the rider over each course segment.
2. **Metabolic constraints**: whether the athlete can physiologically tolerate the power plan.
3. **Optimization**: what power distribution minimizes total course time.
4. **Monte Carlo simulation**: how sensitive the result is to wind, CdA, Crr, pacing error, and fueling assumptions.

The goal is not just to predict speed. The goal is to produce a practical race execution file: target power by course section, expected split, risk level, and the key assumptions driving the result.

---

## Available Inputs

### Athlete Inputs

- Athlete mass
- Bike/equipment mass
- Height
- CdA
- Estimated or measured Crr
- Drivetrain efficiency
- Power-duration profile
- VO2max
- Anaerobic threshold / MLSS
- VLamax
- Fatmax
- Carbmax
- Available glycogen
- INSCYD-style metabolic curves:
  - VO2 uptake vs power
  - VO2 demand vs power
  - Lactate production vs power
  - Lactate recovery vs power
  - Fat oxidation vs power
  - Carbohydrate combustion vs power

### Course Inputs

- GPX or course file
- Distance by segment
- Elevation by segment
- Grade by segment
- Bearing/direction by segment
- Technical sections
- Cornering/braking zones
- Speed caps for turns, descents, or unsafe sections

### Environmental Inputs

- Air temperature
- Barometric pressure
- Humidity
- Wind speed
- Wind direction
- Gust risk
- Road condition assumptions

### Fueling Assumption

Primary scenario:

```text
Carbohydrate intake = 150 g/hr
```

Important modeling caution:

```text
150 g/hr intake does not necessarily mean 150 g/hr oxidation.
```

Use at least two scenarios:

```text
Optimistic usable carbohydrate: 150 g/hr
Conservative usable carbohydrate: 110-120 g/hr
```

---

## Approximate Athlete Anchors From Current Test Screenshot

Use these only as initial placeholders. Replace with the exact exported test data when available.

| Metric | Approximate Value | Modeling Role |
|---|---:|---|
| VO2max absolute | 5215 ml/min | Aerobic ceiling |
| VO2max relative | 74.5 ml/kg/min | Aerobic quality |
| MLSS / anaerobic threshold | 364 W | Main sustainable high-intensity anchor |
| Threshold relative power | 5.20 W/kg | Sustainable power relative to mass |
| Threshold as %VO2max | 87.1% | Aerobic utilization |
| VLamax | 0.38 mmol/s | Glycolytic capacity / lactate production tendency |
| Fatmax | 249 W | Peak fat oxidation power |
| Carbmax | 265 W | Carbohydrate metabolism reference point |
| Available glycogen | 465 g | Internal carbohydrate reserve |

---

## Modeling Philosophy

Do not start with a complicated black-box model.

Start with a physics simulator, calibrate it against known ride files, then layer in metabolic constraints and optimization.

The most common failure mode is building a beautiful metabolic model on top of a bad speed model. A small error in CdA, Crr, elevation smoothing, or wind direction can overwhelm the value of a sophisticated lactate model.

Practical build order:

1. **Constant-power physics model**
2. **Variable-power segmented course model**
3. **Metabolic feasibility layer**
4. **Power optimization layer**
5. **Monte Carlo uncertainty layer**
6. **Coach-facing pacing report**

---

## Project Setup

Suggested repo name:

```bash
mkdir tt-bike-optimizer
cd tt-bike-optimizer
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install numpy pandas scipy matplotlib pydantic gpxpy pyarrow
```

Optional later additions:

```bash
pip install plotly streamlit scikit-learn fastparquet
```

Suggested structure:

```text
tt-bike-optimizer/
  README.md
  pyproject.toml
  data/
    raw/
      course.gpx
      athlete_profile.csv
      metabolic_curves.csv
      weather_forecast.csv
    processed/
      course_segments.parquet
  outputs/
    pacing_plan.csv
    simulation_summary.csv
    monte_carlo_results.parquet
  src/
    config.py
    course.py
    physics.py
    metabolism.py
    simulator.py
    optimizer.py
    monte_carlo.py
    reporting.py
  notebooks/
    01_course_inspection.ipynb
    02_physics_calibration.ipynb
    03_optimizer_testing.ipynb
```

---

## Core Conceptual Model

For each course segment, solve the speed that matches the rider's power output.

```text
Power at pedals -> drivetrain loss -> power at wheel -> speed over segment
```

The power demand is approximately:

```text
P = P_aero + P_rolling + P_gravity + P_acceleration
```

Where:

```text
P_aero      = 0.5 * rho * CdA * apparent_wind_speed^2 * ground_speed
P_rolling   = Crr * mass * g * ground_speed
P_gravity   = mass * g * grade * ground_speed
P_accel     = mass * acceleration * ground_speed
```

For version 1, ignore acceleration except at obvious braking/cornering zones.

---

## Important Performance Principle

A watt is worth more when the athlete is moving slower.

That usually means:

- More power on climbs
- More power into headwinds
- Less power on fast descents
- Less power with strong tailwinds
- No pointless surges into already-high aero drag
- Avoid early spikes that create metabolic debt without meaningful time gain

---

## Data Models

Use simple dataclasses first.

```python
from dataclasses import dataclass

@dataclass
class Athlete:
    rider_mass_kg: float
    bike_mass_kg: float
    cda: float
    crr: float
    drivetrain_efficiency: float
    mlss_w: float
    vo2max_ml_min: float
    glycogen_available_g: float
    carb_intake_g_hr: float

    @property
    def total_mass_kg(self) -> float:
        return self.rider_mass_kg + self.bike_mass_kg


@dataclass
class CourseSegment:
    segment_id: int
    distance_m: float
    grade: float
    bearing_deg: float
    elevation_gain_m: float
    speed_cap_mps: float | None = None
    technical_factor: float = 1.0


@dataclass
class Weather:
    air_temp_c: float
    pressure_hpa: float
    humidity_pct: float
    wind_speed_mps: float
    wind_direction_deg: float
    air_density_kg_m3: float
```

---

## Physics Module

File:

```text
src/physics.py
```

Initial implementation:

```python
import math
from scipy.optimize import brentq

G = 9.80665


def relative_wind_speed(ground_speed_mps, bearing_deg, wind_speed_mps, wind_direction_deg):
    """
    Estimate apparent wind component along the rider's direction of travel.

    Convention note:
    Weather wind direction usually means direction wind comes FROM.
    You may need to adjust this depending on your weather source.
    """
    angle_rad = math.radians(wind_direction_deg - bearing_deg)
    headwind_component = wind_speed_mps * math.cos(angle_rad)
    return ground_speed_mps + headwind_component


def power_required(
    speed_mps,
    grade,
    bearing_deg,
    athlete,
    weather,
):
    mass = athlete.total_mass_kg
    apparent_wind = relative_wind_speed(
        ground_speed_mps=speed_mps,
        bearing_deg=bearing_deg,
        wind_speed_mps=weather.wind_speed_mps,
        wind_direction_deg=weather.wind_direction_deg,
    )

    # Prevent unrealistic negative apparent wind values from tailwind assumptions.
    apparent_wind = max(0.0, apparent_wind)

    p_aero = 0.5 * weather.air_density_kg_m3 * athlete.cda * apparent_wind**2 * speed_mps
    p_roll = athlete.crr * mass * G * speed_mps
    p_grav = mass * G * grade * speed_mps

    power_at_wheel = p_aero + p_roll + p_grav
    power_at_pedals = power_at_wheel / athlete.drivetrain_efficiency

    return power_at_pedals


def solve_speed_from_power(power_w, segment, athlete, weather):
    """
    Solve for speed where required power equals target power.
    """
    def f(v):
        return power_required(
            speed_mps=v,
            grade=segment.grade,
            bearing_deg=segment.bearing_deg,
            athlete=athlete,
            weather=weather,
        ) - power_w

    # Reasonable cycling speed bounds: 1 m/s to 30 m/s.
    speed = brentq(f, 1.0, 30.0)

    if segment.speed_cap_mps is not None:
        speed = min(speed, segment.speed_cap_mps)

    return speed
```

---

## Metabolic Module

File:

```text
src/metabolism.py
```

The metabolic model should use interpolation from exported curves rather than hand-written equations when possible.

Expected metabolic curve file:

```text
power_w,vo2_uptake_ml_min_kg,vo2_demand_ml_min_kg,lactate_production_mmol_min,lactate_recovery_mmol_min,fat_kcal_hr,carb_kcal_hr,carb_g_hr
200,...
220,...
240,...
```

Initial implementation:

```python
import numpy as np
import pandas as pd


class MetabolicProfile:
    def __init__(self, curve_df: pd.DataFrame):
        self.df = curve_df.sort_values("power_w").copy()
        self.power = self.df["power_w"].to_numpy()

    def interp(self, column: str, power_w: float) -> float:
        return float(np.interp(power_w, self.power, self.df[column].to_numpy()))

    def carb_g_hr(self, power_w: float) -> float:
        return self.interp("carb_g_hr", power_w)

    def vo2_demand(self, power_w: float) -> float:
        return self.interp("vo2_demand_ml_min_kg", power_w)

    def lactate_balance_mmol_min(self, power_w: float) -> float:
        production = self.interp("lactate_production_mmol_min", power_w)
        recovery = self.interp("lactate_recovery_mmol_min", power_w)
        return production - recovery


def update_lactate_state(current_state, power_w, dt_seconds, metabolic_profile):
    """
    Simple proxy model.

    Positive balance increases metabolic debt.
    Negative balance lets the athlete recover toward zero.
    """
    balance_per_min = metabolic_profile.lactate_balance_mmol_min(power_w)
    delta = balance_per_min * (dt_seconds / 60.0)
    return max(0.0, current_state + delta)
```

### Important Modeling Note

This lactate state is not literal blood lactate. Treat it as a **race-specific metabolic debt proxy** until validated against actual testing or race data.

---

## Simulator Module

File:

```text
src/simulator.py
```

```python
import numpy as np
from .physics import solve_speed_from_power
from .metabolism import update_lactate_state


def simulate_course(power_plan, course_segments, athlete, weather, metabolic_profile):
    total_time_s = 0.0
    total_distance_m = 0.0
    glycogen_drawdown_g = 0.0
    total_carb_demand_g = 0.0
    lactate_state = 0.0
    max_lactate_state = 0.0
    time_above_mlss_s = 0.0

    rows = []

    for segment, power_w in zip(course_segments, power_plan):
        speed_mps = solve_speed_from_power(
            power_w=power_w,
            segment=segment,
            athlete=athlete,
            weather=weather,
        )

        dt_s = segment.distance_m / speed_mps
        dt_hr = dt_s / 3600.0

        carb_demand_g = metabolic_profile.carb_g_hr(power_w) * dt_hr
        carb_supplied_g = athlete.carb_intake_g_hr * dt_hr
        glycogen_drawdown_g += max(0.0, carb_demand_g - carb_supplied_g)
        total_carb_demand_g += carb_demand_g

        lactate_state = update_lactate_state(
            current_state=lactate_state,
            power_w=power_w,
            dt_seconds=dt_s,
            metabolic_profile=metabolic_profile,
        )
        max_lactate_state = max(max_lactate_state, lactate_state)

        if power_w > athlete.mlss_w:
            time_above_mlss_s += dt_s

        total_time_s += dt_s
        total_distance_m += segment.distance_m

        rows.append({
            "segment_id": segment.segment_id,
            "distance_m": segment.distance_m,
            "grade": segment.grade,
            "target_power_w": power_w,
            "speed_mps": speed_mps,
            "speed_kmh": speed_mps * 3.6,
            "segment_time_s": dt_s,
            "carb_demand_g": carb_demand_g,
            "glycogen_drawdown_g": glycogen_drawdown_g,
            "lactate_state": lactate_state,
        })

    avg_speed_mps = total_distance_m / total_time_s

    summary = {
        "total_time_s": total_time_s,
        "distance_m": total_distance_m,
        "avg_speed_kmh": avg_speed_mps * 3.6,
        "avg_power_w": float(np.mean(power_plan)),
        "max_lactate_state": max_lactate_state,
        "glycogen_drawdown_g": glycogen_drawdown_g,
        "total_carb_demand_g": total_carb_demand_g,
        "time_above_mlss_s": time_above_mlss_s,
    }

    return summary, rows
```

---

## Optimization Module

File:

```text
src/optimizer.py
```

Use `scipy.optimize.minimize` with bounded segment powers.

### Objective

```text
Minimize total course time
```

### Constraints

Start simple:

```text
Average power <= planned target average
Max power <= surge cap
Glycogen drawdown <= available glycogen - reserve
Max lactate state <= allowed metabolic debt
Power change between adjacent segments <= smoothness limit
```

Initial implementation:

```python
import numpy as np
from scipy.optimize import minimize
from .simulator import simulate_course


def optimize_power_plan(
    course_segments,
    athlete,
    weather,
    metabolic_profile,
    initial_power_w,
    min_power_w=100,
    max_power_w=450,
    glycogen_reserve_g=75,
    max_lactate_state=10.0,
):
    n = len(course_segments)
    x0 = np.full(n, initial_power_w, dtype=float)
    bounds = [(min_power_w, max_power_w) for _ in range(n)]

    def objective(power_plan):
        summary, _ = simulate_course(
            power_plan=power_plan,
            course_segments=course_segments,
            athlete=athlete,
            weather=weather,
            metabolic_profile=metabolic_profile,
        )
        return summary["total_time_s"]

    def glycogen_constraint(power_plan):
        summary, _ = simulate_course(power_plan, course_segments, athlete, weather, metabolic_profile)
        max_allowed = athlete.glycogen_available_g - glycogen_reserve_g
        return max_allowed - summary["glycogen_drawdown_g"]

    def lactate_constraint(power_plan):
        summary, _ = simulate_course(power_plan, course_segments, athlete, weather, metabolic_profile)
        return max_lactate_state - summary["max_lactate_state"]

    constraints = [
        {"type": "ineq", "fun": glycogen_constraint},
        {"type": "ineq", "fun": lactate_constraint},
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    final_summary, final_rows = simulate_course(
        result.x,
        course_segments,
        athlete,
        weather,
        metabolic_profile,
    )

    return result, final_summary, final_rows
```

---

## Pacing Plan Output

The model should produce a course-specific execution plan like this:

| Section | Target Power | Reason |
|---|---:|---|
| Opening flat | 330-345 W | Settle without early metabolic spike |
| Short climb | 380-410 W | High ROI watts because speed is lower |
| Fast descent | 220-280 W | Low ROI watts; recover metabolic debt |
| Exposed headwind | 350-370 W | Extra watts more valuable than in tailwind |
| Tailwind section | 315-335 W | Avoid wasting energy against high aero drag |
| Final 5 min | 360-390 W | Spend remaining reserve if lactate/fuel budget allows |

Final output files:

```text
outputs/pacing_plan.csv
outputs/simulation_summary.csv
outputs/monte_carlo_results.parquet
```

Suggested `pacing_plan.csv` columns:

```text
segment_id
section_name
distance_m
grade
bearing_deg
target_power_w
predicted_speed_kmh
segment_time_s
carb_demand_g
glycogen_drawdown_g
lactate_state
coach_note
```

---

## Monte Carlo Simulation

Once the deterministic model works, add uncertainty.

Randomize:

```text
CdA: +/- 2-5%
Crr: +/- 5-10%
Wind speed: forecast error distribution
Wind direction: +/- 10-25 degrees
Air density: small variation
Power execution: segment target +/- 2-4%
Carbohydrate oxidation: optimistic vs conservative
Cornering speed caps: technical execution uncertainty
```

File:

```text
src/monte_carlo.py
```

Pseudocode:

```python
import copy
import numpy as np
import pandas as pd
from .simulator import simulate_course


def run_monte_carlo(
    base_power_plan,
    course_segments,
    athlete,
    weather,
    metabolic_profile,
    n_sims=5000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    rows = []

    for sim_id in range(n_sims):
        sim_athlete = copy.deepcopy(athlete)
        sim_weather = copy.deepcopy(weather)

        sim_athlete.cda *= rng.normal(1.0, 0.03)
        sim_athlete.crr *= rng.normal(1.0, 0.07)
        sim_weather.wind_speed_mps *= max(0.0, rng.normal(1.0, 0.15))
        sim_weather.wind_direction_deg += rng.normal(0.0, 15.0)

        sim_power_plan = np.array(base_power_plan) * rng.normal(1.0, 0.025, size=len(base_power_plan))

        summary, _ = simulate_course(
            power_plan=sim_power_plan,
            course_segments=course_segments,
            athlete=sim_athlete,
            weather=sim_weather,
            metabolic_profile=metabolic_profile,
        )

        rows.append({
            "sim_id": sim_id,
            **summary,
            "cda": sim_athlete.cda,
            "crr": sim_athlete.crr,
            "wind_speed_mps": sim_weather.wind_speed_mps,
            "wind_direction_deg": sim_weather.wind_direction_deg,
        })

    return pd.DataFrame(rows)
```

Useful outputs:

```text
Median bike split
10th-90th percentile bike split
Probability of sub-target split
Sensitivity to CdA
Sensitivity to wind
Sensitivity to power execution
Risk of excessive glycogen drawdown
Risk of excessive lactate proxy
```

---

## Validation Plan

Before trusting optimization outputs, validate the physics model.

Use known ride files where you have:

- Power
- Speed
- Elevation
- Weather approximation
- Equipment setup
- Rider position / CdA assumption

Validation targets:

```text
Good early target: predicted time within 3-5%
Stronger target: predicted time within 1-3%
Excellent target: segment-level errors explainable by wind, braking, surface, or position changes
```

If the model is off, check in this order:

1. Course elevation smoothing
2. CdA assumption
3. Wind direction convention
4. Crr assumption
5. Drivetrain loss
6. Speed caps / technical braking
7. Athlete position changes

---

## First Implementation Milestones

### Milestone 1: Course Parser

Input:

```text
GPX file
```

Output:

```text
course_segments.parquet
```

Segment fields:

```text
segment_id, distance_m, grade, bearing_deg, elevation_gain_m
```

### Milestone 2: Constant Power Simulator

Test:

```text
What bike split does 340 W produce over the course?
```

Output:

```text
total_time_s, avg_speed_kmh, segment_times
```

### Milestone 3: Manual Pacing Rules

Compare:

```text
Steady 340 W
Climb-biased plan
Headwind-biased plan
Conservative start plan
Aggressive finish plan
```

### Milestone 4: Metabolic Feasibility

Add:

```text
Carb demand
Glycogen drawdown
Time above MLSS
Lactate state proxy
VO2 demand check
```

### Milestone 5: Optimizer

Let the optimizer search for the best power plan under constraints.

### Milestone 6: Monte Carlo

Run 1,000-10,000 simulations and produce confidence intervals.

---

## Early Sanity Checks

The optimizer should naturally discover these patterns:

- Power rises on climbs.
- Power rises into headwinds.
- Power falls on fast descents.
- Power falls in strong tailwinds.
- Power avoids repeated stochastic spikes unless there is a clear course reason.
- The plan spends more power where speed is lower and aerodynamic penalty is lower.
- The plan does not burn the entire glycogen reserve unless the race is short or the final segment allows it.
- The plan allows above-MLSS work only where the time return is high.

If the optimizer does not discover these patterns, the model is probably wrong.

---

## Suggested CLI Entry Point

Create:

```text
scripts/run_optimization.py
```

Example command:

```bash
python scripts/run_optimization.py \
  --course data/raw/course.gpx \
  --metabolic-curves data/raw/metabolic_curves.csv \
  --weather data/raw/weather_forecast.csv \
  --output-dir outputs \
  --initial-power 340 \
  --carb-intake 150 \
  --n-sims 5000
```

Expected console output:

```text
=== Deterministic Optimized Plan ===
Predicted split: 54:20
Average speed: 44.2 km/h
Average power: 347 W
Time above MLSS: 6:42
Estimated carb demand: 165 g
Estimated glycogen drawdown: 42 g
Max lactate proxy: 7.8

=== Monte Carlo Summary ===
Median split: 54:28
10th-90th percentile: 53:49 - 55:21
Probability sub-54:00: 28%
Most important sensitivity: CdA, wind direction, Crr
```

---

## Coach-Facing Interpretation

The final answer should not be:

```text
Ride 347 W average.
```

The useful answer is:

```text
This course rewards controlled over-threshold work on the two climbs and exposed headwind section, but not on the fast descent or tailwind return. The athlete can likely tolerate short 380-410 W surges if he keeps the first 10 minutes controlled and continues fueling. The model is most sensitive to CdA and wind direction, so equipment/position validation matters more than squeezing another 5 W into the plan.
```

---

## Immediate Next Step

Build the constant-power physics model first.

Recommended first test:

```text
Input: course GPX + athlete mass + bike mass + CdA + Crr + weather
Question: what time does 340 W predict?
```

Then compare that to any real-world ride file or expected benchmark. Do not move to complex metabolic optimization until the baseline speed model is directionally right.
