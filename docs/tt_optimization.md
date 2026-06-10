# Time-Trial Bike Speed Optimization Model

## Context

We want a simulator that produces the **fastest realistic bike split** for an athlete on a known
TT course, plus a coach-facing **race-execution plan** (target power by section, expected split, risk
bands, and the assumptions that drive the result). Inputs span physics (mass, CdA, Crr, drivetrain),
metabolism (INSCYD curves: VO2, lactate production/combustion, fat/carb g·hr; plus MLSS, VLamax,
glycogen), fueling, weather, and a GPX course profile.

A detailed brainstorm already exists at
`triathlon-db/docs/tt_bike_speed_model_implementation.md`. This plan **adopts its build philosophy**
(physics first, validate, then layer metabolism → optimization → Monte Carlo) but **corrects several
robustness/accuracy flaws** found during review (below) and **places the work to reuse existing
infrastructure** instead of a standalone repo.

### Decisions locked with the user
- **Architecture**: compute engine as a new `tri_analysis.tt_optimizer` package in **triathlon-db**
  (reuses the Monte-Carlo infra and is already editable-installed into the dashboard); INSCYD client,
  athlete/equipment storage, and the coach UI in **PodiumDashboard**.
- **INSCYD**: connect to the API for **full metabolic curves** (a short Phase-0 spike confirms the
  exact response schema before we commit the curve interchange format).
- **Validation**: user has **real bike/TT ride files** → estimate CdA via virtual-elevation and gate
  the project on predicted-vs-actual split accuracy before trusting the optimizer.
- **Metabolic model**: implement **both** a W′balance (critical-power) model **and** a corrected
  lactate-accumulation proxy behind one interface, and compare them against real efforts.

---

## Corrections baked into the design (from review)

These are deliberate departures from the brainstorm code; each is a real accuracy/robustness fix.

| # | Flaw in brainstorm | Correction in this plan |
|---|---|---|
| P1 | `apparent_wind = max(0, …)` + `apparent_wind**2` discards strong tailwinds and loses sign | Signed aero force `F = 0.5·ρ·CdA·v_air·|v_air|`; tailwinds faster than the rider *reduce* power demand |
| P2 | Only head/tail wind component modeled (`cos`), crosswind dropped | Apparent speed from both components `√((v_g+head)²+cross²)`; optional yaw→CdA table (v2) |
| P3 | `grade ≈ sinθ`, Crr uses full normal force | `θ = arctan(grade)`, gravity `m·g·sinθ·v`, rolling `Crr·m·g·cosθ·v` |
| P4 | Per-segment steady-state speed ignores momentum | Coarse macro-segments for v1; optional forward kinetic-energy integration if fine segmentation needed |
| M1/M2 | Unitless "lactate state", arbitrary `=10.0` cap, `production − recovery` (wrong curves) | Proxy reformulated as `max(0, production − max_aerobic_combustion)`; **W′bal is the primary, validated model**; both compared |
| M4 | `carb_supplied = intake_g_hr` credits 100% of intake | Cap **exogenous oxidation** ~90–110 g·hr (optimistic/conservative scenarios) independent of intake |
| M5 | VO2max / aerobic ceiling never enforced | Sustainable-power ceiling tied to CP; above-CP work draws W′ |
| O1 | SLSQP over per-segment power → O(n²) sims/iter, poor convergence | Optimize **5–15 macro-section power levels**, not per-segment; matches coach output |
| O2 | `max(0,…)` kinks break gradient methods | Smooth constraints or add a derivative-free cross-check (differential evolution) |
| O3 | "avg power ≤ target" pseudo-constraint can block the true optimum | Drop it; physiology (W′/glycogen/VO2) bounds the effort |
| D1 | `np.interp` silently flatlines outside sampled power range | Validate curve power-range covers optimizer bounds; explicit guarded extrapolation |
| D2 | CdA (the #1 sensitivity) has no acquisition path | Virtual-elevation (Chung) from a real ride file = both CdA source and physics validation |
| D-air | `Weather.air_density_kg_m3` taken as manual input | Compute air density from temp/pressure/humidity |

---

## Architecture & dependency direction

```
triathlon-db  (lower layer, pure compute — never imports PodiumDashboard)
  tri_analysis/tt_optimizer/        ← NEW package (engine)
PodiumDashboard  (depends on tri_analysis via editable install)
  app/services/inscyd_api.py        ← NEW INSCYD client
  app/services/tt_optimizer_service.py ← NEW bridge (mirrors prediction.py)
  app/models/ + app/webapp/         ← profile storage + coach UI
```

**Key constraint:** triathlon-db must not import PodiumDashboard. FIT parsing lives in the dashboard
(`fit_analysis.py`); the engine's CdA-calibration consumes a neutral **records interchange file**
(`records.parquet`: `t, power_w, speed_mps, altitude_m, distance_m, lat, lon, temp_c`) that the
dashboard exports via the existing FIT parser. This keeps the dependency arrow one-directional.

---

## Engine module layout — `triathlon-db/tri_analysis/tt_optimizer/`

Follow repo conventions: `from __future__ import annotations`, **dataclasses** (not Pydantic),
module-level config constants, type hints. Mirror `tri_analysis/prediction/` structure.

- `__init__.py` — export public API (`Athlete`, `Equipment`, `CourseSegment`, `Weather`,
  `MetabolicProfile`, `optimize_power_plan`, `simulate_course`, `run_monte_carlo`, `build_report`).
- `config.py` — `G`, default power bounds, glycogen reserve, exo-oxidation caps, MC uncertainty σ's.
- `models.py` — dataclasses: `Athlete` (mass, physiology, CP/W′), `Equipment` (bike mass, CdA, Crr,
  drivetrain η), `CourseSegment`, `MacroSection`, `Weather`, `PacingPlan`, `SimSummary`.
- `course.py` — GPX → segments: haversine distance, **elevation smoothing/resampling** (validation
  flags this as failure-mode #1), grade via `arctan`, per-segment bearing; group fine segments into
  **macro-sections** (climb/descent/flat × headwind/tailwind/cross) for the optimizer.
- `physics.py` — `air_density(temp_c, pressure_hpa, humidity_pct)`; `apparent_wind(...)` returning
  signed along-track + crosswind; `power_required(...)` (corrected P1–P3, Martin et al. 1998 ref);
  `solve_speed_from_power(...)` via `brentq`; **`estimate_cda_virtual_elevation(records_df, ...)`**
  (Chung method) for calibration/validation.
- `metabolism.py` — `MetabolicProfile` (INSCYD curve interp with **guarded range**, D1); a
  `DebtModel` protocol with two implementations: `WPrimeBalanceModel` (Skiba W′bal; CP/W′ fit from
  power-duration curve) and `LactateProxyModel` (corrected `max(0, production − combustion)`);
  `FuelModel` (carb demand from curve vs **capped exogenous oxidation**, glycogen drawdown).
- `simulator.py` — `simulate_course(power_plan, segments, athlete, equipment, weather, metab, debt_model)`
  → path-dependent forward integration returning `SimSummary` + per-segment rows (time, speed, carb,
  glycogen, debt/W′bal, time-above-CP).
- `optimizer.py` — `optimize_power_plan(...)` over **macro-section powers** (reduced dims) with
  physiological constraints from the selected `DebtModel`; SLSQP with a differential-evolution
  cross-check (O2).
- `monte_carlo.py` — `run_monte_carlo(base_plan, …, n_sims=5000, random_state=42)` **mirroring**
  `prediction/simulate.py:1136` (`np.random.default_rng`, array accumulation, `np.percentile`);
  randomize CdA/Crr/wind/air-density/power-execution/carb-oxidation; return DataFrame + percentile
  summary + sensitivities.
- `reporting.py` — `build_report(...)` → `pacing_plan.csv` (section_name, target_power_w,
  predicted_speed_kmh, segment_time_s, carb/glycogen/debt, coach_note) + narrative summary.

CLI: `triathlon-db/scripts/run_tt_optimization.py` (pattern from `scripts/predict_program.py`:
`sys.path` inject, argparse, logging). `pyproject.toml`: add `scipy`, `gpxpy` (and `pyarrow` if not
present).

---

## PodiumDashboard additions

- `app/utils/settings.py` — add `inscyd_api_key`, `inscyd_api_base` (Pydantic `BaseSettings`, .env).
- `app/services/inscyd_api.py` — class-based client following `tp_api.py` house style (`requests`,
  `_headers()`, retry/`raise_for_status`, RuntimeError with context). Fetch full metabolic curves +
  scalars; map to the engine's curve schema.
- `app/services/tt_optimizer_service.py` — bridge mirroring `prediction.py`: assemble
  `Athlete`/`Equipment`/`Weather`/course, call `tri_analysis.tt_optimizer`, cache deterministic
  result so re-simulation only reruns Monte Carlo (same caching idea as `_pred_cache`).
- `app/models/tables.py` — new `AthletePhysicsProfile` (athlete_id FK; body/bike mass, CdA, Crr,
  drivetrain η, CP, W′, MLSS, VLamax, VO2max, Fatmax, Carbmax, glycogen; cached INSCYD curve JSON +
  fetched_at). Optional `TTPlan` to persist generated plans.
- A records exporter (reuse `fit_analysis.analyze(include_records=True)`) → `records.parquet` for the
  engine's CdA calibration (convert FIT semicircle lat/long ×180/2³¹).
- Weather: reuse triathlon-db's Open-Meteo pattern (`tri_analysis/weather.py`) against the **forecast**
  endpoint using event lat/lon; feed temp/pressure/humidity/wind into the engine (air density computed).
- UI: `app/webapp/routes_tt.py` + `templates/tt.html` (full page, extends `base.html`) +
  `partials/tt_pacing_report.html`, `tt_risk.html` (HTMX fragments) following the
  `routes_compare.py` / `compare.html` / `partials/compare_h2h.html` pattern (HTMX `hx-get` partials,
  cache headers). Parameter controls (CdA, carb intake, debt model, weather) re-trigger the partial.

---

## Build sequence (each phase gated)

0. **INSCYD spike** — call the API for one athlete; document response; lock the curve interchange
   schema (columns + units). Small, de-risks the metabolic data path.
1. **Course + physics + constant-power sim** — `course.py`, `physics.py` (corrected), constant-power
   `simulate_course`. Milestone: "what split does 340 W predict?"
2. **CdA calibration + VALIDATION GATE** — virtual-elevation CdA from the user's ride file; compare
   predicted vs actual split. **Do not proceed until within 3–5% (target 1–3%).** Diagnose in the
   doc's order (elevation smoothing → CdA → wind convention → Crr → drivetrain → braking).
3. **Metabolism** — INSCYD curve interp (guarded), fuel model (exo-oxidation cap), **both** debt
   models (W′bal + lactate proxy); add feasibility outputs to the simulator.
4. **Optimizer** — macro-section optimization with physiological constraints (selectable debt model).
   Sanity-check: power rises on climbs/headwinds, falls on descents/tailwinds, W′/glycogen not
   overdrawn. If not, the model is wrong.
5. **Monte Carlo** — mirror `simulate.py`; percentile split bands, sub-target probability,
   CdA/wind/Crr/execution sensitivities.
6. **Reporting + dashboard** — coach narrative + `pacing_plan.csv`; INSCYD client; bridge service;
   `AthletePhysicsProfile` persistence; coach UI page + partials.

---

## Critical files

**Create (triathlon-db):** `tri_analysis/tt_optimizer/{__init__,config,models,course,physics,metabolism,simulator,optimizer,monte_carlo,reporting}.py`, `scripts/run_tt_optimization.py`, tests under `tests/tt_optimizer/`. **Modify:** `pyproject.toml` (deps).

**Create (PodiumDashboard):** `app/services/inscyd_api.py`, `app/services/tt_optimizer_service.py`, `app/webapp/routes_tt.py`, `app/webapp/templates/tt.html` + `partials/tt_*.html`. **Modify:** `app/utils/settings.py`, `app/models/tables.py`, `app/webapp/app.py` (register routes).

## Reused utilities (do not rebuild)
- `app/services/fit_analysis.py` — `RecordPoint`, `analyze(include_records=True)`, `extract_power_curve()` → records export, CdA virtual-elevation, CP/W′ fitting.
- `triathlon-db/tri_analysis/prediction/simulate.py:1136` `run_monte_carlo` — MC patterns (`default_rng`, array accumulation, `np.percentile`).
- `app/services/prediction.py` — bridge/caching pattern for `tt_optimizer_service.py`.
- `app/services/tp_api.py` + `app/utils/settings.py` — external-API house style for `inscyd_api.py`.
- `triathlon-db/tri_analysis/weather.py` + `events` lat/lon — weather forecast plumbing.
- `app/webapp/routes_compare.py` + `compare.html` + `partials/compare_h2h.html` — UI pattern.

---

## Verification
- **Unit tests** (triathlon-db `tests/`): physics analytic cases (flat/no-wind power↔speed; tailwind
  reduces power; climb increases; air-density formula; arctan grade); metabolism (W′bal
  depletion/reconstitution monotonic; curve extrapolation guarded); simulator energy/time sanity;
  optimizer rediscovers the sanity-check pacing patterns on a synthetic course.
- **Validation gate**: `predicted vs actual` split on the user's ride file within 3–5% (then 1–3%).
- **Model comparison**: notebook running W′bal vs lactate proxy on the same real efforts (the
  "both, compared" deliverable) — report which tracks reality better.
- **CLI**: `python scripts/run_tt_optimization.py --course … --metabolic-curves … --weather … --initial-power 340 --carb-intake 150 --n-sims 5000` prints deterministic + Monte-Carlo summary.
- **Dashboard end-to-end**: load athlete (INSCYD curves fetched + cached), pick course + forecast
  weather, render the pacing-report partial; verify parameter changes (CdA, carb, debt model)
  re-run and update.

## Open items / risks
- **INSCYD API schema** unconfirmed until Phase-0 spike (endpoint/field names/units) — engine uses an
  adapter boundary so the mapping is isolated.
- **CP/W′ calibration quality** depends on the athlete having true maximal efforts in the power-duration
  curve; provide a manual override seeded from the MLSS anchor + a W′ estimate.
- **Yaw→CdA** table needs data; v1 uses apparent-speed-with-crosswind, yaw-dependent CdA is a v2 refinement.
- **Forecast weather uncertainty** is itself a Monte-Carlo input (wind speed/direction error).
