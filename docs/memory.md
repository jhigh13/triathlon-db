# Triathlon Database - File Memory

## Recent File Changes (June 2025)

### Core ETL Pipeline Files

#### `Data_Upload/update_race_results.py`
- **Purpose**: Handles incremental updates of race results from recent events
- **Key Features**: 
  - Fetches new events since last database update
  - Processes Elite Men/Women programs specifically
  - UPSERT operations with duplicate handling
- **Recent Changes**: Added NULL value filtering and data preview functionality to prevent constraint violations

#### `Data_Import/master_data_import.py`
- **Purpose**: Full database initialization with complete historical data import
- **Key Features**:
  - Concurrent athlete data fetching
  - Event dimension table creation
  - Race results fact table population
- **Recent Changes**: Added duplicate removal logic based on unique constraint columns before database insertion

#### `main.py`
- **Purpose**: Command-line interface for all database operations
- **Features**: Menu-driven options for full import, incremental updates, or single athlete import
- **CLI Support**: Accepts command-line arguments for automated/scripted execution

### Configuration & Database

#### `config/config.py`
- **Purpose**: Centralized configuration management
- **Contains**: API endpoints, database URIs, authentication headers
- **Security**: Environment variable integration for sensitive credentials

#### `Data_Import/database.py`
- **Purpose**: Database schema definition and connection management
- **Features**: Table creation, constraint definition, connection pooling
- **Schema**: Optimized for triathlon data with proper indexing

### Documentation & Analysis

#### `docs/Summary.md`
- **Purpose**: High-level project overview and current state documentation
- **Updated**: June 2025 with recent bug fixes and architectural improvements
- **Content**: Features, architecture, tech stack, and usage instructions

#### `model_pipeline.ipynb`
- **Purpose**: Machine learning pipeline for athlete performance predictions
- **Status**: In development for race outcome and time predictions
- **ML Stack**: LightGBM, XGBoost, scikit-learn integration

### Machine Learning Pipeline (Production Modules)

#### `ml/` Directory
- **Purpose**: Production machine learning modules for race time prediction
- **Architecture**: Modular design with separate components for each pipeline stage
- **Target**: Predict next-race finish time within ≤600 sec MAE per distance

#### `ml/data_extraction.py`
- **Purpose**: Extracts and joins data from PostgreSQL (race_results, events, athlete tables)
- **Features**: Database connection management, table joining, initial data loading
- **Class**: `DataExtractor` with methods for loading and joining raw tables

#### `ml/features.py`
- **Purpose**: Feature engineering pipeline for triathlon data
- **Features**: Event parsing, DNF handling, rolling averages, categorical encoding
- **Class**: `FeatureEngineer` handles complete feature transformation pipeline

#### `ml/label_generation.py`
- **Purpose**: Creates target labels for next-race prediction
- **Features**: Athlete-grouped time shifting, position labels, model-ready filtering
- **Class**: `LabelGenerator` creates next_time_sec and next_position targets

#### `ml/preprocessing.py`
- **Purpose**: Preprocessing with scaling, imputation, and PCA dimensionality reduction
- **Features**: Feature selection, StandardScaler, optional PCA with configurable components
- **Class**: `TriathlonPreprocessor` manages complete preprocessing pipeline

#### `ml/config.py`
- **Purpose**: Configuration management for ML pipeline
- **Features**: Model parameters, target MAE thresholds, feature engineering settings
- **Classes**: `ModelConfig`, `FeatureConfig`, `DataConfig` for organized settings

#### `train.py`
- **Purpose**: Training orchestration script coordinating complete ML pipeline
- **Features**: Command-line interface, distance-specific training, model persistence
- **Usage**: `python train.py [--distance DISTANCE] [--no-pca] [--pca-components N]`
- **Models**: LinearRegression and RandomForestRegressor with GroupKFold cross-validation

### Testing & Quality Assurance

#### `tests/` Directory
- **test_master_import.py**: Tests for full data import functionality
- **test_smoke.py**: Basic system health checks
- **test_hello_world.py**: Basic environment validation
- **Coverage**: Integrated with GitHub Actions CI/CD pipeline

### Data & Infrastructure

#### `requirements.txt`
- **Purpose**: Python dependency management
- **Key Dependencies**: SQLAlchemy 2.0, pandas, psycopg2-binary, scikit-learn, lightgbm, xgboost
- **Updated**: Maintains compatibility with Python 3.13

#### `docker-compose.yml` & `Dockerfile`
- **Purpose**: Containerized deployment configuration
- **Features**: PostgreSQL service, Python environment setup
- **Benefits**: Consistent development and production environments

## Database Schema Summary

### Tables
- **athlete**: Athlete profiles and biographical information
- **events**: Event details, venues, dates, categories
- **race_results**: Individual race performances with splits and positions
- **athlete_rankings**: Current and historical rankings

### Key Constraints
- **race_results_unique**: `(athlete_id, EventID, TotalTime)` prevents duplicate results
- **Primary Keys**: Proper indexing on all dimension tables
- **Foreign Keys**: Referential integrity between athletes, events, and results

## Recent Bug Fixes & Improvements
1. **Constraint Violation Fix**: Resolved duplicate key errors in race_results table
2. **Data Quality**: Added NULL value filtering before database operations
3. **Validation**: Implemented data preview for troubleshooting and monitoring
4. **Performance**: Optimized concurrent processing with configurable thread pools
5. **Error Handling**: Enhanced logging and graceful failure recovery

# Triathlon Database - File Memory

## ML Pipeline Implementation Status (June 3, 2025)

### 🎯 ACHIEVEMENT SUMMARY
Successfully implemented a complete, modular machine learning pipeline for predicting triathlete race times. The pipeline processes 62,063 race results through sophisticated feature engineering to create a 50,738-sample dataset with 46 features, targeting next-race time prediction within ≤600 sec MAE.

### ✅ COMPLETED COMPONENTS

#### Core ML Pipeline Modules (all in `ml/` directory):

1. **`ml/data_extraction.py`** - DataExtractor class
   - Database connection and table loading (race_results, events, athlete tables)
   - Intelligent joining with proper foreign key relationships
   - **Performance**: Loads 1,909 athletes, 3,349 events, 62,063 race results

2. **`ml/features.py`** - FeatureEngineer class  
   - Event specification parsing (race_type, distance, event_flags)
   - DNF/DNS handling with recent DNF rate calculation
   - Rolling time and position averages (windows: 3, 5 races)
   - Time-based features (days_since_last, age_at_race)
   - One-hot encoding for categorical features (race_type, distance, event_mode)
   - **Output**: 43 engineered features from 25 raw columns

3. **`ml/label_generation.py`** - LabelGenerator class
   - Creates next_time_sec and next_position targets per athlete
   - Athlete-grouped time shifting with proper sorting
   - Filters for modeling-ready dataset
   - **Retention**: 96.4% sample retention (52,640 → 50,738 rows)

4. **`ml/preprocessing.py`** - TriathlonPreprocessor class
   - Feature type identification (numeric vs categorical)
   - ColumnTransformer with StandardScaler and imputation
   - Optional PCA with configurable components (default: 95% variance)
   - Robust column name handling for sklearn compatibility

5. **`ml/config.py`** - Configuration classes
   - ModelConfig, FeatureConfig, DataConfig for organized settings
   - Hyperparameter grids and performance thresholds

#### Training Infrastructure:

6. **`train.py`** - Complete training orchestration script
   - Command-line interface with distance-specific training
   - GroupKFold cross-validation respecting athlete boundaries
   - Multiple model support (LinearRegression, RandomForestRegressor)
   - Model persistence with pickle serialization
   - **CLI**: `python train.py [--distance DISTANCE] [--no-pca] [--pca-components N]`

### 🔧 RECENT FIXES APPLIED

#### Data Type Compatibility (June 3, 2025):
- **Issue**: Feature names had mixed types ('quoted_name', 'str') preventing sklearn compatibility
- **Root Cause**: pandas get_dummies() creating pandas Index objects instead of plain strings
- **Solution**: Enhanced column name conversion in both `features.py` and `preprocessing.py`:
  ```python
  # Before: df.columns = df.columns.astype(str)  
  # After: df.columns = [str(col) for col in df.columns]
  ```

#### Code Quality Improvements:
- Applied Black formatter across all ML modules for consistent formatting
- Fixed argument parser formatting issues in `train.py`
- Enhanced logging throughout pipeline for better debugging

### 📊 PIPELINE PERFORMANCE METRICS

#### Data Processing Flow:
```
Raw Database: 62,063 race results
↓ Initial Cleaning: 59,374 rows (filter triathlon events only)
↓ DNF Handling: 52,938 rows (valid times only)  
↓ Feature Engineering: 52,640 rows (43 features)
↓ Label Generation: 50,738 rows (46 features with targets)
↓ Sprint Distance: 32,903 samples (17 features after filtering)
```

#### Feature Distribution:
- **Numeric Features**: 9 (times, positions, rolling averages, days_since_last)
- **Categorical Features**: 8 (one-hot encoded race_type, distance, event_mode)
- **Target Variables**: next_time_sec, next_position

### 🚧 IMMEDIATE NEXT STEPS

1. **Test Feature Name Fix**: Verify the data type compatibility fix resolves sklearn errors
2. **End-to-End Training**: Complete full model training and evaluation 
3. **Performance Validation**: Confirm MAE ≤ 600 sec target for each distance
4. **Model Persistence**: Validate saved models and prediction capability

### 📁 FILE ARCHITECTURE
```
ml/
├── __init__.py              # Package initialization
├── config.py               # Configuration classes  
├── data_extraction.py      # Database loading and joining
├── features.py            # Feature engineering pipeline
├── label_generation.py    # Target variable creation
└── preprocessing.py       # sklearn pipeline and PCA
train.py                   # CLI training orchestration
```

### 🎓 TECHNICAL LEARNING NOTES
- **GroupKFold**: Prevents data leakage by keeping athlete data within single folds
- **Rolling Windows**: Capture athlete performance trends without future information
- **PCA**: Dimensionality reduction while preserving 95% variance by default
- **Modular Design**: Each pipeline stage isolated for testing and maintenance