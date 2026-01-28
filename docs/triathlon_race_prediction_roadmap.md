# Triathlon Race Prediction (WTCS Draft‑Legal) — Roadmap, Research Takeaways, and Scaffolding

## 1) What we’re building

### Goal
Pre‑race predictions for **draft‑legal WTCS / World Cup** programs using your existing database of results + checkpoint gaps/positions + WTCS pack membership + upcoming start lists.

### Outputs (per upcoming program)
1. **Expected finishing order** (ranked list)
2. **Probabilities**: win, podium, top‑5, top‑10, top‑20 (and “most likely rank range”)
3. **Predicted splits and total time** per athlete:
   - swim, T1, bike, T2, run, total
4. **Explanation layer** for coaches:
   - “Why this athlete is projected top‑5” (top drivers: swim pack likelihood, run strength, etc.)
   - scenario summaries: “If misses front swim pack, podium probability drops from X% → Y%”

### Non‑goals (for now)
- Live in‑race updates
- Non‑draft long‑course modeling
- Athlete physiology / training‑log ingestion (optional future)

---

## 2) Research takeaways that shape the roadmap (practical)
- **Predict time (and splits) first, then derive rank by sorting.** Rank labels are noisier than continuous outcomes.
- **Bike + run matter most for total time**, but in **draft‑legal** the swim matters hugely because it determines pack access and bike dynamics.
- **Monte Carlo simulation is the right way to turn point forecasts into probabilities**, plus it gives robust scenario analysis (“hot day”, “miss pack”, etc.).
- **Course + conditions matter** (weather, currents, wind) and should be added as features ASAP once the baseline works.

---

## 3) What you already have (tables we’ll leverage)

Your schema includes (names as in your project):
- `events` with program + event metadata (distance, lat/long, weather fields, wetsuit, etc.)
- `race_results` fact table with splits + finish status/position sort keys
- `position_metrics` with elapsed/behind times and positions at checkpoints (swim exit, T1, bike exit, T2, run finish)
- `wtcs_pack_membership` with pack id/size and gaps at checkpoints (draft‑legal gold)
- `program_entries` providing upcoming start lists (pre‑race roster)

You also have scripts that build schema and ingest from World Triathlon API (triathlon.org) and can fetch start lists.

---

## 4) Concrete build order (what to implement next)

### Phase 0 — Define “prediction unit” + evaluation protocol (1–2 short sessions)
**Prediction unit:** `(event_id, prog_id)` for a WTCS men/women program.

**Training/test splitting (critical):**
- Use **time‑based splits** to avoid leakage.
- Recommended: for each evaluation run, pick a cutoff date T. Train on events < T, test on events >= T.
- Add a second evaluation: **leave‑one‑event‑out** on a recent season for robustness.

**Metrics you’ll compute every run:**
- **Ranking metrics**
  - `Precision@K`: % of actual top‑K included in predicted top‑K (K ∈ {3,5,10,20})
  - `Mean Reciprocal Rank (MRR)` (winner & podium emphasis)
  - `Spearman` / `Kendall` rank correlation across full field
- **Time metrics**
  - MAE for total time and per split
  - Calibration checks: do simulated 80% intervals contain ~80% of outcomes?

Deliverable: a single `evaluate.py` that takes a model + feature set and prints a consistent report.

---

### Phase 1 — Build a clean modeling dataset (feature table) (core dependency)
Create a reproducible pipeline that outputs a **feature snapshot** for each athlete entry in each historical program.

#### 1A) Create a training view (SQL or pandas)
Create a “wide” dataset keyed by `(event_id, prog_id, athlete_id)` with:
- labels: `total_time_sec`, `swim_sec`, `t1_sec`, `bike_sec`, `t2_sec`, `run_sec`
- rank labels: `finish_position` (int), plus derived:
  - `is_podium`, `is_top5`, `is_top10`, `is_top20`
- program metadata: distances, wetsuit flag, country, date, etc.

Implementation notes:
- Convert your time strings to seconds consistently.
- Filter to **finishers** for time regression labels; keep DNFs for classification tasks if desired.
- Keep a `field_size` feature (# active results in program).

#### 1B) Engineer athlete form features (pre‑race only)
For each athlete and each target program date, compute features from prior history **only**:
- **Recent performance**:
  - last N races splits/total (N=1,3,5)
  - recency‑weighted averages (EWMA) for run/bike/total
  - trend features (slope over last K races)
- **Stability / reliability**
  - variance of splits/total
  - DNF rate in last 12 months
- **Relative strength**
  - “run percentile vs field” in last X races
  - net positions gained on run (`position_at_t2` → `finish_position`) averages
- **Pack / swim access features (draft‑legal specific)**
  - probability of being in **front swim pack**: fraction of races where athlete’s swim gap to leader ≤ threshold (e.g., ≤ 20s or ≤ 0.8% of swim time)
  - typical swim rank percentile
  - historical pack metrics from `wtcs_pack_membership`:
    - average pack_size at swim/bike checkpoints
    - average gap_to_leader_sec at swim checkpoint
    - average gap_to_prev_sec and max_gap_to_prev_sec

#### 1C) Field‑context features (who else is racing)
For an upcoming start list, your model improves if it “knows” the field:
- compute a **seed score** for every entrant (e.g., predicted total time baseline from recent EWMA)
- add race‑level aggregations as features:
  - `field_mean_seed`, `field_top5_mean_seed`, `field_depth_index` (std dev of seeds)
  - `athlete_seed_rank` within start list

This is one of the biggest practical boosts for podium/top‑K probability modeling.

Deliverable: `build_features.py` that produces a parquet file:
- `features_train.parquet` (historical)
- `features_upcoming.parquet` (for inference)

---

### Phase 2 — Baseline models (fast wins) (get to “usable”)
Start with **tabular models**. They’re strong on structured sports data and easy to train locally.

#### 2A) Split + total time regression
Models:
- LightGBM / XGBoost / CatBoost (pick one to start)
Targets:
- `swim_sec`, `bike_sec`, `run_sec`, `t1_sec`, `t2_sec`
- `total_time_sec`
Two options:
1) Predict splits then sum → total (plus an independent total model as a check)
2) Only predict total and use split models for explainability

Output of inference:
- predicted splits + predicted total
- derived rank = sort by predicted total

#### 2B) Top‑K probability heads (optional but recommended)
Train binary classifiers for:
- `is_podium`, `is_top5`, `is_top10`, `is_top20`
This produces usable coach outputs even when exact order is uncertain.

Deliverables:
- `train_models.py` (saves model artifacts)
- `predict_upcoming.py` (runs inference on a given `(event_id, prog_id)` start list)

---

### Phase 3 — Add Monte Carlo for probabilities + scenario analysis (coach‑ready)
Once you have mean predictions, add uncertainty and simulate races.

#### 3A) Estimate uncertainty per athlete and per split
Start simple:
- For each athlete, compute residual SD from training CV (or from historical variability).
- Use global fallback SD for athletes with sparse history.

Advanced later:
- Train a second model that predicts absolute error (heteroscedasticity).

#### 3B) Simulate 10k races per program
For each simulation:
- sample swim/bike/run/transition times for each athlete (with correlation rules below)
- compute total
- rank by total
Aggregate:
- win/podium/top‑K probabilities
- expected rank and 90% rank interval
- time intervals

**Draft‑legal correlation rules (simple, effective):**
- Introduce a latent “race day form” variable per athlete that shifts all splits slightly.
- Add a **pack‑effect** on bike:
  - athletes likely in lead swim pack get tighter bike distributions (less variance) and slightly faster mean
  - athletes missing pack get slower mean bike (and/or larger variance)
This makes the simulation feel like real WTCS racing without heavy physics.

Deliverable: `simulate.py` that takes predicted means + SDs + pack likelihood features.

---

### Phase 4 — Draft‑legal dynamics upgrade (your differentiator)
This is where your database outclasses generic predictors.

#### 4A) Model “front pack after swim” probability explicitly
Train a classifier to predict:
- `in_front_pack_swim` (label from `wtcs_pack_membership.pack_id` or gap threshold)
Features:
- athlete swim history, field swim depth, wetsuit flag, venue type proxy, etc.

#### 4B) Conditional bike/run models
Use **mixture modeling**:
- If in front pack: bike time ~ Model A
- If not: bike time ~ Model B
Similarly for run, conditional on bike pack / gap at T2:
- large chase efforts tend to worsen run

Implementation:
- either two separate regressors
- or single regressor with interaction features including predicted pack membership

Deliverable: updated model family (v2).

---

### Phase 5 — Weather + course enrichment (after baseline is stable)
You store weather text fields in `events` but upcoming races may have only location.

Plan:
1) Add a `weather_history` table (event_date+lat/long keyed, plus event_id/prog_id link).
2) Backfill historical weather with a weather API.
3) For upcoming races, pull a forecast and store it.

Then add weather features:
- air temp, humidity, wind speed, WBGT proxy
- water temp + wetsuit rule impacts

This improves split forecasts and makes scenario analysis meaningful (heat‑stress scenarios).

---

## 5) Suggested repository scaffolding (copy/paste into your project)

```
tri_prediction/
  README.md
  pyproject.toml (or requirements.txt)
  src/
    tri_prediction/
      __init__.py
      config.py
      db.py
      sql/
        base_training_view.sql
      features/
        build_features.py
        time_parsing.py
        feature_defs.py
      models/
        train_regression.py
        train_classification.py
        model_io.py
      simulation/
        monte_carlo.py
        pack_effects.py
      inference/
        predict_program.py
        format_outputs.py
      evaluation/
        metrics.py
        backtest.py
  notebooks/
    01_data_audit.ipynb
    02_feature_build.ipynb
    03_baseline_model.ipynb
    04_simulation.ipynb
  outputs/
    models/
    predictions/
    reports/
```

**Design principle:** keep everything callable from CLI scripts *and* notebook‑friendly.

---

## 6) Implementation details (practical)

### 6A) Canonical time parsing
You need a single function that converts time strings like:
- `HH:MM:SS`, `MM:SS`, sometimes empty or status strings
into seconds or null.

Make it used everywhere (features + labels + inference).

### 6B) Avoid leakage
When building athlete history features, always filter prior races by:
- `event_date < target_event_date`
Never compute stats using the same race you’re trying to predict.

### 6C) Handle sparse history
For athletes with < 2 relevant prior races:
- fall back to:
  - global priors by gender + distance category
  - plus an uncertainty penalty (bigger SD)

### 6D) Outputs format (coach‑usable)
For each program:
- table sorted by expected finish
- columns:
  - athlete_name, country
  - expected_rank, rank_90pct_interval
  - win/podium/top10/top20 probabilities
  - split means + intervals
  - key drivers text (top 3 SHAP features or heuristics)

---

## 7) “Definition of done” for v1
You have v1 when you can:
1) Select an upcoming `(event_id, prog_id)` with a start list from `program_entries`
2) Generate features for all entrants
3) Produce:
   - ranked list (expected finish order)
   - podium/top‑10 probabilities
   - splits + total time estimates with intervals
4) Run an automated backtest on last season and show:
   - precision@3, @10, @20
   - MAE total time and run split
   - calibration plot for top‑10 probabilities

---

## 8) Next questions you’ll answer while implementing (not blockers)
- What threshold defines “front swim pack” best for your goals? (fixed seconds vs % of swim)
- Are you predicting men/women separately? (recommended: separate models)
- How do you want to treat DNFs for probability outputs? (ignore, or model DNF risk as separate head)

---

## 9) Quick start checklist (day‑by‑day)
**Day 1**
- Build the training dataset (labels in seconds, join events + results + position_metrics)
- Write `time_parsing.py`
- Create backtest splitter

**Day 2**
- Build athlete history features (last N, EWMA, stability, swim pack likelihood)
- Add field context features (seed ranks)

**Day 3**
- Train baseline regression (total + splits)
- Evaluate (precision@K, MAE)

**Day 4**
- Add binary heads for podium/top‑K
- Add SHAP explainability outputs

**Day 5**
- Implement Monte Carlo simulation
- Produce probabilities + intervals + scenario summaries
