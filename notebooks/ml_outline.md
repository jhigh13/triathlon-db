# Machine Learning Prediction Outline for Triathlon

## 1. Define the Prediction Task

**Target variable(s):**
- Finish time (continuous regression)
- Finish position (ordinal classification)

**Scope:**  
Predict a given athlete's result in their next event.

## 2. Assemble & Label Training Data

- **Historic Results:**  
  Join `race_results` fact table with `events` (event_date, distance category) and `athlete` (age at race, gender, country).  
  Optionally filter to a specific subset (e.g. Elite standard-distance races).

- **Next-Race Labeling:**  
  For each athlete, sort races by date and shift finish time/position forward by one.  
  Drop the final race for each athlete (no "next" label).

## 3. Database & ETL Pipeline

- PostgreSQL schema: athlete (dimension), events (dimension), race_results (fact)
- `master_data_import.py` for full loads
- `update_race_results.py` for incremental upserts
- SQLAlchemy for schema management, concurrency for API calls, batched inserts via `to_sql(method='multi')`

## 4. Data Cleaning & Transformation

- Separated DNF/DNS into valid vs. all datasets
- Parsed race_type, distance, event_mode from event specifications
- Converted times to seconds, computed days_since_last, calculated athlete age

## 5. Feature Engineering

**Athlete Features:**
- Rolling averages (last 3–5 finishes times (distance specific and global), positions)
- Total races in past 6–12 months
- Age at race date

**Event Features:**
- Distance category (Sprint, Standard, Super-Sprint, Relay)
- Time of year (month, ISO week)
- Location metadata (continent, climate zone)
- Venue Characteristics (if available):
  - Elevation gain/loss
  - Swim type (open-water vs. pool)

**Temporal/Contextual:**
- Days since last race (fatigue metric)
- Days until a marquee competition (peaking indicator)
- Interactions:
  - Age × average_speed
  - Variability in finish times

## 6. Label Creation

- Group by `athlete_id`, use `.shift(-1)` to attach next-race labels
- Drop final record per athlete to create `df_model`

## 7. Modeling Roadmap

- Train/test split with GroupKFold (temporal per athlete)
- Baseline models: LinearRegression, RandomForest, GradientBoosting
- Evaluation metrics: MAE/RMSE for time, accuracy/top-N accuracy for position

## 8. Modular Preprocessing Pipeline

- Use `ColumnTransformer` for numeric imputation & scaling
- One-hot or target encoding for categoricals
- Wrap all steps in an `sklearn.Pipeline` for reproducibility and hyperparameter search

## 9. ETL Robustness & Metadata Tracking

- Add metadata table to track last processed event_date or event_id per category
- Centralize DB connection and introduce structured logging
- Ensure idempotent, auditable incremental updates

## 10. Outlier & Missing-Value Handling

- Impute missing split times (median or KNN) and flag imputed rows
- Clip or remove extreme finish times beyond 3σ to stabilize training

## 11. Cross-Validation & Grouping Strategy

- Implement GroupKFold by athlete_id with temporal splits
- Nest hyperparameter tuning inside the group-aware CV loop to avoid leakage

## 12. Additional Data Sources

- Weather: temperature, humidity, wind speed
- Course profile: elevation gain/loss
- Athlete training load (e.g., Strava API)
- Injury history or recent DNFs
- Equipment details (bike, wetsuit)
- Field strength (average competitor ranking)

## 13. Two-Phase Workflow (Notebook → Production)

- Exploratory prototyping in Jupyter (EDA, quick feature tests)
- Refactor code into modules (`features.py`, `models.py`)
- Parameterize via config files or CLI arguments
- Automate end-to-end via scripts scheduled with cron/CI/CD

## 14. Learning Resources

- DataCamp modules on ML fundamentals and scikit-learn pipelines
- [scikit-learn official tutorials](https://scikit-learn.org/stable/tutorial)
- Coursera: Andrew Ng’s Machine Learning; University of Michigan’s Applied ML in Python
- Kaggle Learn micro-courses on Pandas, ML, and pipelines
- Book: Géron’s *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*

## 15. Next Immediate Steps

1. Enroll in a foundational ML course and complete scikit-learn pipeline tutorials
2. Prototype a baseline regression pipeline in Jupyter using ColumnTransformer
3. Implement GroupKFold temporal splits and evaluate baseline performance
4. Extract code into modular scripts and create a standalone CLI entry-point
5. Set up a scheduled job (cron or CI) for ETL + model inference writing to PostgreSQL
6. Integrate the predictions table into Power BI for live dashboards

---

This outline will guide iterative development and production deployment of next-race performance prediction models.

