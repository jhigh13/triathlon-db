# Implementation Plan for Race Prediction Model

## 1. Data Pipeline Enhancement
- [x] **Task**: Extend model_pipeline.ipynb to improve data preprocessing and feature engineering
- **Files**: 
  - `model_pipeline.ipynb`: Add cells for:
    ```python
    # Add field strength calculation
    def calculate_field_strength(event_df):
        return event_df.groupby('EventID').agg({
            'Position': ['count', 'mean', 'std'],
            'athlete_rank': ['min', 'max', 'mean']
        })
    
    # Add split-time processing
    def process_split_times(df):
        for split in ['SwimTime', 'BikeTime', 'RunTime']:
            df[f'{split}_sec'] = pd.to_timedelta(df[split]).dt.total_seconds()
            df[f'{split}_rolling3'] = group_rolling_mean(df, split, 3)
    ```
- **Dependencies**: pandas, numpy

## 2. Model Architecture Setup
- [ ] **Task**: Create modular model pipeline with multi-stage predictions
- **Files**:
  - `ML/models/dnf_classifier.py`:
    ```python
    class DNFRiskClassifier:
        def __init__(self):
            self.model = XGBClassifier()
            self.features = ['recent_5_DNF_rate', 'days_since_last', ...]
    ```
  - `ML/models/split_predictor.py`:
    ```python
    class SplitTimePredictor:
        def __init__(self, split_name):
            self.model = LightGBM()
            self.split = split_name
    ```
  - `ML/models/position_predictor.py`
- **Dependencies**: xgboost, lightgbm, scikit-learn

## 3. Validation Framework
- [ ] **Task**: Implement time-based cross-validation and model evaluation
- **Files**:
  - `ML/validation/cross_validator.py`:
    ```python
    class TemporalCrossValidator:
        def split_by_time(self, df, n_splits=5):
            # Time-based splitting while preserving athlete groups
    ```
  - `ML/validation/metrics.py`
- **Dependencies**: scikit-learn

## 4. Database Integration
- [ ] **Task**: Create predictions table and upload pipeline
- **Files**:
  - `database/models/predictions.py`:
    ```python
    class PredictionTable:
        def __init__(self):
            self.table_name = 'athlete_predictions'
            self.schema = {
                'athlete_id': 'INTEGER',
                'event_id': 'INTEGER',
                'predicted_time': 'INTERVAL',
                'confidence': 'FLOAT'
            }
    ```
  - `ML/pipeline/upload_predictions.py`
- **Dependencies**: SQLAlchemy

## 5. Model Training Pipeline
- [ ] **Task**: Create end-to-end training pipeline
- **Files**:
  - `ML/pipeline/trainer.py`:
    ```python
    class ModelTrainer:
        def train_all_models(self):
            self.train_dnf_model()
            self.train_split_models()
            self.train_position_model()
    ```
  - `ML/config.py`
- **Dependencies**: All previous modules

## 6. Model Testing and Evaluation
- [ ] **Task**: Implement comprehensive testing suite
- **Files**:
  - `tests/test_prediction_accuracy.py`:
    ```python
    def test_model_accuracy():
        # Test MAE < 5% of mean finish time
        # Test position predictions within ±3 places
    ```
  - `tests/test_model_drift.py`
- **Dependencies**: pytest, numpy

## 7. PowerBI Integration
- [ ] **Task**: Create visualization dashboard for predictions
- **Files**:
  - `visualization/prediction_dashboard.pbix`
- **Dependencies**: PowerBI Desktop

User Intervention Required:
1. Review feature importance plots to validate feature selection
2. Set thresholds for model retraining triggers
3. Validate prediction confidence intervals
4. Approve final model selection for production
