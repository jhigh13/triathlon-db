# Triathlon Database - File Memory

## Recent File Changes (June 2025)

### Core ETL Pipeline Files

#### `Data_Upload/update_race_results.py`
- **Purpose**: Handles incremental updates of race results from recent events
- **Key Features**: 
  - Fetches new events since last database update
  - Processes Elite Men/Women programs specifically
  - UPSERT operations with duplicate handling
  - Position tracking and race dynamics analysis
  - **Individual split rankings**: SwimRank, T1Rank, BikeRank, T2Rank, RunRank
- **Recent Changes**: 
  - Added NULL value filtering and data preview functionality to prevent constraint violations
  - **June 2025**: Recreated file to fix corruption and add position tracking features
  - **June 2025**: Added same position rankings and position change calculations as master import
  - **June 2025**: Added individual split rankings for each segment (swim, T1, bike, T2, run)

#### `Data_Import/master_data_import.py`
- **Purpose**: Full database initialization with complete historical data import
- **Key Features**:
  - Concurrent athlete data fetching
  - Event dimension table creation
  - Race results fact table population
  - Position tracking and race dynamics analysis
  - **Individual split rankings**: SwimRank, T1Rank, BikeRank, T2Rank, RunRank
- **Recent Changes**: 
  - Added duplicate removal logic based on unique constraint columns before database insertion
  - **June 2025**: Added position rankings at each checkpoint (Position_at_Swim, Position_at_T1, etc.)
  - **June 2025**: Added position change tracking between checkpoints (negative values = gained positions)
  - **June 2025**: Added individual split rankings for each segment (swim, T1, bike, T2, run)

#### `streamlit_app.py`
- **Purpose**: Multi-page Streamlit application for triathlon analysis with complete pack dynamics system
- **Features**: 
  - **Page Navigation**: H2H Analysis and Event Analysis pages
  - **H2H Analysis**: Head-to-head athlete comparison functionality with heatmap matrices
  - **Event Analysis**: Advanced event-specific race analysis with complete pack dynamics system
- **Key Features**: 
  - **Dual Data Sources**: Database integration for standard events, Excel upload for detailed analysis
  - **Race Overview**: Metrics dashboard with finisher count, DNF rate, winning time
  - **Data Processing**: Excel file validation and processing for detailed timing data
  - **Pack Dynamics**: COMPLETE Phase 4 - advanced tactical analysis with export capabilities
  - **H2H Matrices**: Interactive heatmaps for overall and segment-by-segment athlete comparisons
  - **Tactical Insights**: Breakaway detection, draft zone analysis, strategic recommendations
  - **Export System**: JSON and CSV download capabilities for analysis reports
  - **Advanced Visualizations**: Pack stability analysis, competitive pressure tracking
- **Recent Changes**: 
  - **July 2025**: Restructured app to support multiple pages with sidebar navigation
  - **July 2025**: Added Event Analysis page with Excel upload functionality
  - **July 2025**: Implemented Phase 1 infrastructure for pack dynamics analysis
  - **July 2025**: COMPLETED Phase 2 pack dynamics with gap thresholds, position tracking, pack composition
  - **July 2025**: COMPLETED Phase 3 pack dynamics with evolution timeline, individual athlete analysis, advanced gap analysis
  - **July 2025**: COMPLETED Phase 4 pack dynamics with tactical insights, export capabilities, performance benchmarks, advanced visualizations
  - **July 2025**: FIXED H2H analysis bug - reverted to working pandas-based approach from broken SQL approach
  - **July 2025**: Added comprehensive data validation and error handling
  - **July 2025**: COMPLETED Phase 2 pack dynamics with gap thresholds, position tracking, pack composition
  - **July 2025**: FIXED H2H analysis bug - reverted to working pandas-based approach from broken SQL approach

### Configuration & Database

#### `config/config.py`
- **Purpose**: Centralized configuration management
- **Contains**: API endpoints, database URIs, authentication headers
- **Security**: Environment variable integration for sensitive credentials

#### `Data_Import/database.py`
- **Purpose**: Database schema definition and connection management
- **Features**: Table creation, constraint definition, connection pooling
- **Schema**: Optimized for triathlon data with proper indexing
- **Recent Changes**: 
  - **June 2025**: Added position tracking columns to race_results table
  - **June 2025**: Enhanced schema with checkpoint position rankings and position change metrics
  - **June 2025**: Added individual split ranking columns (SwimRank, T1Rank, BikeRank, T2Rank, RunRank)

### Documentation & Analysis

#### `docs/historical-rankings-scraping.md`
- **Purpose**: Comprehensive functional specification for historical triathlon rankings data scraping feature
- **Key Features**:
  - Web scraping infrastructure for old.triathlon.org historical rankings
  - Data validation and quality assurance processes
  - Integration with existing athlete_rankings database table
  - Athlete name matching algorithms
  - Incremental processing and monitoring capabilities
- **Scope**: Historical World Triathlon Championship Series rankings (2009-2024)
- **Target Data**: Rank position, athlete name, country, points by year and gender
- **ML Integration**: Enhanced feature engineering for career trajectory and ranking trends analysis
- **Created**: June 2025 as detailed implementation roadmap

#### `docs/Summary.md`
- **Purpose**: High-level project overview and current state documentation
- **Updated**: June 2025 with recent bug fixes and architectural improvements
- **Content**: Features, architecture, tech stack, and usage instructions

#### `model_pipeline.ipynb`
- **Purpose**: Machine learning pipeline for athlete performance predictions
- **Status**: In development for race outcome and time predictions
- **ML Stack**: LightGBM, XGBoost, scikit-learn integration

#### `proj_pod/pp_podiums.ipynb`
- **Purpose**: Notebook to tally and visualize Project Podium athlete podiums vs Canada, Mexico, and Other USA across selected events.
- **Outputs**: Saves raw and cleaned results plus charts to `proj_pod/` (`raw_results.csv`, `cleaned_results.csv`, `podium_counts_overall.csv`, `podium_counts_by_event.csv`, `podium_overall.png`, `podium_by_event.png`).
- **Config**: Uses `proj_pod/.env` (example provided) with `DATABASE_URL`/`DB_URI` and optional table/column overrides.
- **Seed Files**: `proj_pod/project_podium_athletes.csv`, `proj_pod/events.csv`, `proj_pod/queries.sql`.
 - **New (Aug 2025)**:
   - Added all-events query and tallies for wins and podiums across Elite Men program (includes non-continental-cup events) — saves `pp_all_results.csv` and `pp_wins_podiums_totals.csv`.
   - Added Plotly visualization `pp_wins_podiums.html` (+ optional PNG) for wins/podiums per athlete.
   - Added PowerPoint export `Project_Podium_Report.pptx` with:
     - Per-event slides where a Project Podium athlete podiumed (Athlete, Pos, Program)
     - Totals slide (wins/podiums per athlete) and chart image if available
     - Continental Cup comparison slides using existing charts (`podium_overall.png`, `podium_by_event.png`)

#### `docs/WTO_Report.pbix`
- **Purpose**: Power BI dashboard for triathlon analytics
- **Features**: Podium analysis, split times, performance trends
- **Integration**: Direct PostgreSQL connection with refresh capabilities

### Analysis & ML Pipeline

#### `tri_analysis/h2h.ipynb`
- **Purpose**: Head-to-head (H2H) athlete comparison analysis system
- **Key Features**:
  - Dynamic athlete pair comparison with win/loss statistics
  - Segment-by-segment performance analysis (swim, T1, bike, T2, run)
  - Scalable parameterized queries for Power BI integration
  - Flexible filtering by athlete, event, country, and date range
  - Win percentage matrices and statistical aggregations
- **Technical Implementation**:
  - Joins race_results with position_metrics tables using program ID
  - Generates all possible athlete pairs per event/program
  - Calculates win flags for overall and segment performance
  - Aggregates H2H metrics with minimum match thresholds
  - Provides Power BI-ready Python script for dynamic queries
- **Power BI Integration**: Complete setup with parameterized queries to avoid loading millions of rows
- **Created**: June 2025 as robust solution for athlete comparison analytics

#### `tri_analysis/pack_dynamics_analysis.ipynb`
- **Purpose**: Triathlon pack dynamics analysis for race strategy and tactical insights
- **Key Features**:
  - Automated pack detection using time gap thresholds (default: 2 seconds)
  - Multi-segment pack evolution tracking (swim, bike, run transitions)
  - Strategic positioning analysis for drafting opportunities
  - Interactive visualizations for coaches and athletes
  - Gap analysis between consecutive athletes and pack formation patterns
- **Analysis Concepts**:
  - Pack timeline visualization showing group formation/dissolution
  - Pack size distribution and time spread analysis
  - Athlete movement patterns between different packs
  - Strategic insights for race tactics and positioning
- **Target Data**: Hamburg 2025 detailed results with potential for expansion to other races
- **Applications**: Race strategy planning, training focus identification, performance analysis, competitive intelligence
- **Created**: July 2025 for advanced tactical race analysis

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