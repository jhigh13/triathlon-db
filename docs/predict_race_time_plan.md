# Implementation Plan for Predict Triathlete Race Time & Ranking

## Current Status Summary (June 3, 2025)

### ✅ COMPLETED - Working Components:
- **ML Pipeline Foundation**: Complete modular architecture with 5 working modules
- **Data Processing**: Successfully processes 62,063 race results → 50,738 labeled samples  
- **Feature Engineering**: 46 features including rolling averages, DNF flags, event parsing, one-hot encoding
- **Training Script**: Command-line interface with distance-specific training
- **Code Quality**: Black formatter applied, consistent structure, proper logging

### 🔧 IN PROGRESS - Current Issue:
- **Feature Name Data Types**: Fixed pandas Index object compatibility issue
- **Need**: Test the fix by running complete training pipeline

### 📋 NEXT STEPS:
1. Test the feature name fix (run `python train.py --distance Sprint`)
2. Complete end-to-end training and model evaluation
3. Model performance validation (target: MAE ≤ 600 sec)
4. Unit testing and CI/CD integration

---

- [x] Step 1: Scaffold `ml/` Directory
  - **Task**: Create a new top-level folder `ml/` to house pipeline modules and training script. Ensure an `__init__.py` for package recognition.
  - **Files**:
    - `ml/__init__.py`: empty file to mark as Python package
  - **Dependencies**: None (initial directory structure)

- [x] Step 2: Data Extraction Module
  - **Status**: ✅ COMPLETE - `ml/data_extraction.py` implemented
  - **Features**: DataExtractor class with database connection, table loading, and joining
  - **Tested**: Successfully loads 1,909 athletes, 3,349 events, 62,063 race results
  - **Task**: Implement functions to load Postgres tables into pandas DataFrames using SQLAlchemy.
  - **Files**:
    - `ml/data_extraction.py`: 
      ```python
      def load_race_results(engine) -> pd.DataFrame:  # SELECT * FROM race_results
      def load_events(engine) -> pd.DataFrame:
      def load_athletes(engine) -> pd.DataFrame:
      ```
  - **Dependencies**: `pandas`, `sqlalchemy`
  - **User Intervention**: Confirm database URI in `config/config.py` is accessible.

- [x] Step 3: Label Generation Module
  - **Status**: ✅ COMPLETE - `ml/label_generation.py` implemented
  - **Features**: LabelGenerator class creates next_time_sec and next_position targets
  - **Tested**: Successfully generates labels for 50,738 samples (96.4% retention rate)
  - **Task**: Write a function to compute `next_time_sec` and `next_position` per athlete and drop rows without labels.
  - **Files**:
    - `ml/label_generation.py`:
      ```python
      def add_next_race_labels(df: pd.DataFrame) -> pd.DataFrame:
          # groupby athlete_id, shift TotalTime_sec and Position
      ```
  - **Dependencies**: `pandas`

- [x] Step 4: Feature Engineering Module
  - **Status**: ✅ COMPLETE - `ml/features.py` implemented with fixes applied
  - **Features**: FeatureEngineer class with event parsing, rolling averages, DNF handling, categorical encoding
  - **Tested**: Processes 52,938 valid times → 52,640 final features (43 columns) 
  - **Fix Applied**: Enhanced column name string conversion to handle pandas Index objects
  - **Task**: Build baseline and advanced feature functions: rolling means, days since last, age, DNF flags, event_specs parsing, one-hot encoding.
  - **Files**:
    - `ml/features.py`:
      ```python
      def add_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
          # prev_time_sec, rolling windows, days_since_last, age

      def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
          # DNF_flag, recent DNF rate, parse_event_specs(), one-hot
      ```
  - **Dependencies**: `pandas`, `datetime`
  - **User Intervention**: Provide mapping sets for `race_types` and `distances` if updated.

- [x] Step 5: Preprocessing & PCA Module
  - **Status**: ✅ COMPLETE - `ml/preprocessing.py` implemented with fixes applied  
  - **Features**: TriathlonPreprocessor with ColumnTransformer, StandardScaler, optional PCA
  - **Tested**: Identifies 9 numeric + 8 categorical features for 32,903 Sprint samples
  - **Fix Applied**: Robust column name string conversion to prevent sklearn TypeError
  - **Task**: Define a scikit-learn `ColumnTransformer` pipeline for numeric and categorical processing, including PCA(0.95).
  - **Files**:
    - `ml/preprocessing.py`:
      ```python
      def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
          # Pipeline: imputer->scaler->PCA, imputer->ohe
      ```
  - **Dependencies**: `sklearn.pipeline`, `sklearn.compose`, `sklearn.impute`, `sklearn.preprocessing`, `sklearn.decomposition`

- [x] Step 6: Training Orchestration Script
  - **Status**: ✅ COMPLETE - `train.py` implemented with fixes applied
  - **Features**: Full CLI with argument parsing, ML pipeline orchestration, model training
  - **Models**: LinearRegression and RandomForestRegressor with GroupKFold CV
  - **Fix Applied**: Argument parser formatting and print statement escaping corrected
  - **Testing**: Pipeline runs successfully through feature engineering, debugging data type compatibility
  - **Task**: Create `train.py` CLI that parses `--distance`, loads data, applies modules, runs nested CV with `GridSearchCV`, and saves best model.
  - **Files**:
    - `train.py` (in project root):
      ```python
      if __name__ == '__main__':
          args = parse_args()  # --distance, --n_jobs
          engine = get_engine()
          df_raw = extract all tables
          df_labels = add_next_race_labels(df_raw)
          df_feats = add_baseline_features(df_labels)
          df_feats = add_advanced_features(df_feats)
          X, y, groups = filter_by_distance(df_feats, args.distance)
          preproc = build_preprocessor(num_cols, cat_cols)
          pipe = Pipeline([('prep', preproc), ('mod', GradientBoostingRegressor())])
          param_grid = {...}
          search = GridSearchCV(pipe, param_grid, ...)
          search.fit(X, y)
          joblib.dump(search.best_estimator_, f"models/{args.distance}_best.pkl")
      ```
  - **Dependencies**: `argparse`, modules from `ml/`, `sklearn`, `joblib`, `mlflow`
  - **User Intervention**: Ensure `models/` directory exists.

- [ ] Step 7: Experiment Tracking Integration
  - **Task**: Embed MLflow logging in `train.py` for parameters and outer CV MAE.
  - **Files**:
    - Edit `train.py`, around fit and CV:
      ```python
      mlflow.set_tracking_uri('file:///c:/Users/jhigh/mlruns')
      with mlflow.start_run():
          mlflow.log_params(args)
          mlflow.log_metric('MAE', -best_score)
      ```
  - **Dependencies**: `mlflow`

- [ ] Step 8: Unit Tests for Modules
  - **Task**: Write pytest functions to validate each module on toy data.
  - **Files**:
    - `tests/test_data_extraction.py`: mock engine and sample DataFrame
    - `tests/test_label_generation.py`
    - `tests/test_features.py`
    - `tests/test_preprocessing.py`
  - **Dependencies**: `pytest`, `pandas`, `sqlalchemy` (for in-memory SQLite engine)

- [ ] Step 9: Integration Smoke Test & CI Configuration
  - **Task**: Add a smoke test running `train.py --distance SPR --n_jobs 1` on a small sample and configure GitHub Actions.
  - **Files**:
    - `tests/test_smoke_training.py`: trigger subprocess
    - `.github/workflows/ci.yml`: add job steps for `pytest` and smoke test
  - **Dependencies**: GitHub Actions runner, `python`

- [ ] Step 10: Build & Test Final Model
  - **Task**: Execute full training for all distances, evaluate, and verify MAE ≤ 600 sec.
  - **Files**: no code changes; run via terminal:
    ```powershell
    python train.py --distance SPR --n_jobs 4; python train.py --distance STD --n_jobs 4; python train.py --distance SSP --n_jobs 4
    ```
  - **Dependencies**: database availability, compute resources

- [ ] Step 11: Validation of Model Accuracy Tests
  - **Task**: Write tests that confirm the saved model predicts within target MAE on a hold-out dataset.
  - **Files**:
    - `tests/test_prediction_accuracy.py`: load `models/SPR_best.pkl`, run predictions on small known set, assert MAE < 600
  - **Dependencies**: `joblib`, `sklearn.metrics`

- [ ] Step 12: Model Iteration & Robustness Review
  - **Task**: If MAE > 600 sec for any distance, revisit pipeline:
    - Expand feature set (weather, course, splits, rankings)
    - Explore alternative regression algorithms (LightGBM, XGBoost, ensemble)
    - Refine hyperparameter tuning or model stacking
  - **Files**: Update documentation (`docs/predict_race_time.md`) with new ideas and track in `docs/memory.md`
  - **Dependencies**: Domain expertise, additional data sources

---

Accessibility: Each code function should include docstrings and type hints to improve readability and IDE accessibility.

*Please review and let me know if any step or detail needs adjustment before implementation.*
