# Triathlon‑DB ML Pipeline Outline

*Version 0.1 · June 2025*
> **Note:** All ML testing and exploration is currently performed in `notebooks/model_pipeline.ipynb`. As the pipeline stabilizes, reusable code will be moved into versioned Python modules under `ml/`.

---

## 🎯 Goal

Predict an athlete’s **next‑race finish time** (seconds) within **⩽ 600 sec MAE** for each race distance:

| Distance     | Abbrev | Typical Range (sec) |
| ------------ | ------ | ------------------- |
| Sprint       | `SPR`  |  2 800 – 5 000      |
| Standard     | `STD`  |  5 500 – 9 000      |
| Super‑Sprint | `SSP`  |   900 – 2 500       |

Each distance gets its **own model** to avoid multi‑modal targets.

---

## 1  Data Requirements

| Table                 | Key Fields                                                        | Purpose               |
| --------------------- | ----------------------------------------------------------------- | --------------------- |
| `race_results` (fact) | `athlete_id`, `EventID`, `Position`, `TotalTime_sec`, `EventDate` | raw outcomes & labels |
| `events` (dim)        | `EventID`, `distance_category`, `country`, `Venue`                | metadata / dummies    |
| `athlete` (dim)       | `athlete_id`, `gender`, `birth_year`                              | demographics          |

*ETL already refreshes these tables weekly via Docker → Postgres.*

---

## 2  Label & Baseline Features

```text
next_time_sec      = finish‑time of athlete’s next race (shift -1)
prev_time_sec      = finish‑time of previous race (shift +1)
rolling3_time_dist = mean of last 3 times (distance‑specific)
rolling5_time_dist = mean of last 5 times (distance‑specific)
rolling3_pos       = mean of last 3 positions
rolling5_pos       = mean of last 5 positions
days_since_last    = EventDate − previous EventDate
age                = EventDate.year − birth_year
```

*Drop athletes with < 3 historic races within that distance.*

---

## 3  Pre‑processing Graph

```text
┌─────────────┐    num_cols    ┌─────────────┐   PCA (0.95 var)
│  raw df     │──────────────▶│ impute‑median│──────────────▶ PCs
└─────────────┘   cat_cols    └─────────────┘
        │                              ▲
        │ one‑hot + impute‑mode        │
        ▼                              │
  OHE matrix ──────────────────────────┘
          ▼ concat
      final design matrix
  
> **PCA Integration:** The numerical pipeline applies PCA (`n_components=0.95`) after scaling to reduce dimensionality while retaining 95% of variance.
```

### Code Skeleton

```python
num_cols = [...see above...]
cat_cols = ['gender'] + distance_dummies + mode_dummies

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler()),
    ('pca', PCA(n_components=0.95, svd_solver='full'))
])
cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])
preproc = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])
model = GradientBoostingRegressor(random_state=42)
pipe  = Pipeline([('prep', preproc), ('mod', model)])
```

---

## 4  Hyper‑parameter Search (nested CV)

```python
param_grid = {
    'prep__num__pca__n_components': [0.90, 0.95, 0.99],
    'mod__n_estimators': [200, 400],
    'mod__learning_rate': [0.03, 0.05],
    'mod__max_depth': [3, 4]
}
outer = GroupKFold(5); inner = GroupKFold(3)
search = GridSearchCV(pipe, param_grid, cv=inner,
                      scoring='neg_mean_absolute_error', n_jobs=-1)
mae = cross_val_score(search, X, y, cv=outer, groups=groups,
                      scoring='neg_mean_absolute_error')
print('distance', -mae.mean())
```

---

## 5  Acceptance Criteria

* **MAE ≤ 600 sec** on outer CV for each distance subset.
* Nested CV ensures no learner sees the same athlete in train & test.
* `mlflow.log_metric('MAE', value)` per distance (e.g., using a local file or SQLite backend).

---

## 6  Implementation Milestones

| #  | Milestone            | Description                                                            |
| -- | -------------------- | ---------------------------------------------------------------------- |
|  1 | **Data Service**     | Verified distance splits, outlier filter committed                     |
|  2 | **Feature Gen v1**   | SQL + Pandas scripts produce baseline cols incl. `prev_time_sec`       |
|  3 | **Pipeline code**    | `pipeline.py` exposes `build_pipe()` + `train_search()`                |
|  4 | **CI job**           | GitHub Actions runs `pytest` + `python train.py --distance SPR` on PRs |
|  5 | **Benchmark report** | MD file logs baseline vs. model MAE; target met.                       |
|  6 | **Testing**          | Unit & integration tests for feature-engineering and training modules |

---

## 7  Local Run Example

```bash
# Sprint model
python train.py --distance SPR --n_jobs -1
# Evaluate & persist best → models/sprint_best.pkl
```

---

## 8  Future (Out‑of‑Scope)

* Weather data integration (temperature, wind, precipitation) via external APIs
* Detailed course metadata (lap counts, swim/bike conditions, surface type)
* Elevation profile (total climb and descent) via GPS/course APIs
* Equipment data (bike type, wetsuit usage)
* LightGBM & XGBoost hyperparameter sweeps
* Azure ML hyper-drive deployment
