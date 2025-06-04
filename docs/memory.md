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

#### `docs/WTO_Report.pbix`
- **Purpose**: Power BI dashboard for triathlon analytics
- **Features**: Podium analysis, split times, performance trends
- **Integration**: Direct PostgreSQL connection with refresh capabilities

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

- June 2025: Updated .gitignore to include `.venv/` for ignoring Python virtual environment directory.