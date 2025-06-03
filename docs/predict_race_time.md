# Feature Spec: Predict Triathlete Race Time & Ranking

**Version 0.1 · June 2025**

## 🎯 Goal
Build a reusable pipeline to generate a labeled, feature-engineered dataset from our Postgres tables (`race_results`, `events`, `athlete`) and configure a scikit-learn workflow (with PCA) that predicts each athlete’s next-race finish time and ranking.

**Directory Layout**: All new pipeline modules, training scripts, and related utilities will live under a top-level `ml/` directory. The existing notebook `notebooks/model_pipeline.ipynb` remains as an exploration sandbox and can be referenced for implementation details.

---

## 1  Functional Requirements

1. **Data Extraction**
   - Pull raw tables `race_results`, `events`, and `athlete` from Postgres via SQLAlchemy.
   - Acceptance: DataFrame shapes and key columns (`athlete_id`, `EventID`, `TotalTime`, `Position`, `EventDate`, `gender`, `birth_year`) match expected ranges.

2. **Label Generation**
   - Compute `next_time_sec` and `next_position` by shifting per `athlete_id` sorted by `EventDate`.
   - Drop rows lacking a valid next race.
   - Acceptance: No nulls in label columns; at least one record per athlete has labels.

3. **Baseline Feature Engineering**
   - Create features: `prev_time_sec`, `rolling3_time_dist`, `rolling5_time_dist`, `rolling3_pos`, `rolling5_pos`, `days_since_last`, `age`.
   - Filter athletes with < 3 past races per distance.
   - Acceptance: Rolling features computed without errors; DataFrame has no missing numeric fields.

4. **Advanced Feature Engineering**
   - Flag DNF/DNS, compute recent DNF rate.
   - Parse `EventSpecifications` into `race_type`, `distance`, and `event_mode` (individual, mixed_relay, paratriathlon).  
   - One-hot encode categorical fields (`gender`, `race_type`, `distance`, `event_mode`).
   - Acceptance: New columns exist and sum of one-hot groups equals number of rows.

5. **Preprocessing & PCA**
   - Build a scikit-learn `ColumnTransformer` pipeline: median imputation → scaling → PCA(95% var) for numeric; most-frequent imputation → one-hot for categorical.
   - Acceptance: Transformed feature matrix has reduced dimensionality and no missing values.

6. **Training Orchestration**
   - Implement `train.py --distance {SPR,STD,SSP}` that runs nested CV (GroupKFold) with `GridSearchCV` over PCA and model hyperparameters (e.g., `n_estimators`, `learning_rate`, `max_depth`).
   - Acceptable models: GradientBoostingRegressor to start; later RandomForest.
   - Acceptance: Script completes without errors and outputs a pickled best-pipe under `models/`.

7. **Experiment Tracking**
   - Integrate MLflow with a local file or SQLite backend (`mlruns/` or `.db`).  
   - Log metrics (outer CV MAE) and parameters for each distance.
   - Acceptance: Runs directory shows artifacts; `mlflow ui` can display metrics.

8. **Testing & CI**
   - Write unit tests for each feature-engineering function using small, synthetic DataFrames.  
   - Add an integration smoke test that runs `train.py --distance SPR --n_jobs 1` on a sample subset.
   - Configure GitHub Actions to run `pytest` and smoke tests on each PR.
   - Acceptance: All tests pass; CI status is green.

---

## Additional Considerations

- **Future Features**: weather data, course metadata (laps, conditions), elevation profile, equipment usage.
- **Alternative Models**: LightGBM and XGBoost in subsequent sweeps.
- **Deployment**: containerize training or schedule with GitHub Actions / Azure ML hyperdrive.

---

*Please review and let me know if you’d like any adjustments or additions.*
