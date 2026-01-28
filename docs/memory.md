# Triathlon Database - File Memory

## Recent File Changes (January 2026)

### Prediction Anchoring Fix (2026-01-26)

#### Problem Diagnosed
- Model had inverted learning: athletes with unusually fast EMAs (like Reese Vannerson) were predicted SLOWER
- Root cause: tree-based model learned U-shaped relationship - very fast EMA correlates with racing at higher tiers
- When fast athlete races at lower tier event, model predicts "regression to mean" (unrealistically slow)
- Example: Vannerson (EMA=52.7m, best=49.9m) predicted at 63.6m (10.8 min slower than EMA!)

#### `tri_analysis/prediction/predict.py` (Fixed 2026-01-26)
- **Added**: Prediction anchoring to prevent absurd slowdowns
- **Logic**: Cap predictions at 110% of athlete's EMA total time
- **Effect**: Vannerson rank improved from 22nd to 6th (now appropriately top tier)

#### `tri_analysis/prediction/sql.py` (Fixed 2026-01-26)
- **Added**: `distance_category` parameter to `fetch_pack_history()`
- **Fixed**: Pack metrics (ema_bike_gap_sec_7, etc.) now filter by distance
- **Impact**: Prevented Olympic distance gaps (~4500s) mixing with Sprint gaps (~1000s)

### Event Tier Classification & Sample Weighting

#### `tri_analysis/prediction/features.py` (Updated 2026-01-26)
- **Added**: `EVENT_TIER_PATTERNS` dict - regex patterns for classifying event tiers
  - Tier 1: WTCS/Championship Series, World Championship Finals
  - Tier 2: World Cup events
  - Tier 3: Regional (Continental Cup, Americas Cup, etc.)
  - Tier 4: Other events
- **Added**: `TIER_SAMPLE_WEIGHTS` = {1: 4.0, 2: 2.0, 3: 1.5, 4: 1.0}
- **Added**: `classify_event_tier(event_name)` function
- **Added**: `event_tier` as feature column (now 18 total features)
- **Updated**: `build_features_for_program()` now passes distance_category to pack fetch

#### `tri_analysis/prediction/train.py` (Updated 2026-01-26)
- **Added**: `use_sample_weights` parameter to `train_baseline_models()`
- **Added**: Sample weighting based on event tier during training
- **Effect**: WTCS events get 4x weight, World Cups 2x, Regional 1.5x

#### `tri_analysis/prediction/sql.py` (Fixed 2026-01-26)
- **Fixed**: `fetch_event_metadata()` was missing `event_name` column in SELECT
- **Impact**: Without this fix, ALL events were classified as tier 4

### Model Performance (2026-01-26)
- **Previous**: Precision@10 ~22%, Spearman ~0.40
- **After pack features**: Precision@10 44%, Spearman 0.11  
- **After tier weighting fix**: Precision@10 47.9%, Spearman 0.105
- **Training data**: 22,811 samples across tiers (2,187 T1, 3,869 T2, 7,782 T3, 8,973 T4)
- **Model v5**: 38,328 samples (2018-2025), includes distance-filtered pack features

### Prediction Pipeline (`tri_analysis/prediction/`)

#### `tri_analysis/prediction/__init__.py`
- **Purpose**: Package init exposing key utilities (parse_time_to_seconds, seconds_to_hms, ProgramKey)

#### `tri_analysis/prediction/utils_time.py`
- **Purpose**: Time string parsing ("mm:ss", "hh:mm:ss") to/from integer seconds
- **Key Functions**: `parse_time_to_seconds()`, `seconds_to_hms()`, `parse_time_columns()`
- **Handles**: None, empty, DNF/DNS/DSQ strings gracefully

#### `tri_analysis/prediction/sql.py`
- **Purpose**: Parameterized SQL queries for prediction data extraction
- **Key Functions**: 
  - `fetch_program_results()` - labeled rows for a program
  - `fetch_athlete_history()` - prior finisher results (no leakage)
  - `fetch_pack_history()` - wtcs_pack_membership history (now with distance filtering)
  - `fetch_start_list()` - program_entries for upcoming races
- **Uses**: ProgramKey(event_id, prog_id) dataclass

#### `tri_analysis/prediction/features.py`
- **Purpose**: Feature engineering for athlete form, pack metrics, and field context
- **Key Functions**:
  - `compute_athlete_form_features()` - EWMA splits, std, days since race
  - `compute_pack_features()` - front_pack_rate, avg_swim_gap_leader
  - `compute_field_context_features()` - seed_total_rank, n_entrants
  - `build_features_for_program()` - complete feature matrix for a program
- **MVP Features**: ema_{swim,bike,run,total}_sec_5, std_total_sec_24m, days_since_last_race, front_pack_rate, avg_swim_gap_leader, seed_total_rank, n_entrants

#### `tri_analysis/prediction/train.py`
- **Purpose**: Model training and persistence
- **Key Components**:
  - `ModelBundle` dataclass: stores model_swim, model_bike, model_run, model_total, feature_columns, metadata
  - `train_baseline_models()` - trains HistGradientBoostingRegressor (or LightGBM if available)
  - `save_model_bundle()` / `load_model_bundle()` - joblib persistence
  - `build_training_dataset()` - creates training DataFrame from historical programs

#### `tri_analysis/prediction/predict.py`
- **Purpose**: Deterministic predictions from trained models
- **Key Functions**:
  - `predict_splits_and_total()` - adds pred_{swim,bike,run,total}_sec, predicted_rank
  - `format_prediction_output()` - clean display DataFrame

#### `tri_analysis/prediction/simulate.py`
- **Purpose**: Monte Carlo simulation for probability estimates
- **Key Functions**:
  - `estimate_uncertainty()` - adds sigma columns from std_total_sec_24m
  - `run_monte_carlo()` - 10k simulations with pack effects
  - Returns: prob_win, prob_podium, prob_top5, prob_top10, prob_top20, expected_rank, rank intervals, time intervals
- **Draft-Legal Features**: Pack bonus/penalty effects on bike segment

#### `tri_analysis/prediction/evaluate.py`
- **Purpose**: Evaluation metrics and backtesting
- **Key Functions**:
  - `precision_at_k()` - fraction of true top-K in predicted top-K
  - `spearman_rank_corr()` - rank correlation
  - `compute_mae()` - mean absolute error for times
  - `backtest_events()` - rolling backtest harness

### CLI Scripts

#### `scripts/train_models.py`
- **Purpose**: Train prediction models from historical data
- **Usage**: `python scripts/train_models.py --start_date 2022-01-01 --end_date 2025-12-31 --output models/bundle.joblib`

#### `scripts/predict_program.py`
- **Purpose**: Predict outcomes for an upcoming program
- **Usage**: `python scripts/predict_program.py --event_id 123 --prog_id 456 --model_path models/bundle.joblib`
- **Outputs**: Prints top 20 table, saves CSV to outputs/

### Test Files

#### `tests/test_prediction.py`
- **Purpose**: Unit tests for prediction pipeline
- **Coverage**: Time parsing, feature computation, simulation, evaluation metrics

---

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

#### `notebooks/option_b_non_wtcs_analysis.ipynb`
- **Purpose**: Option B exploratory view of 2023–2025 race results excluding WTCS categories (keeps DNFs) to spotlight USA team averages and Sullivan Middaugh bike-pack overlaps.
- **Key Points**:
  - Pulls race_results + events + athlete + position_metrics via SQLAlchemy with adjustable date window and WTCS keyword override.
  - Produces USA seasonal averages, event-level scatter (finish vs DNF rate), and pack case-study tables/plots driven by pack gap logic (2s leader / 1s intra-pack).
  - Serves as quick-launch notebook for storytelling/visuals prior to promoting insights into Streamlit or Power BI.

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

### New (Sept 2025)

#### `tri_analysis/wtcs_performance.py`
  - `fetch_wtcs_us_dataset` – joins race_results, position_metrics, events, athlete; filters by event_name pattern ("World Triathlon Championship Series") and USA country code.
  - (Sept 2025 update) Broadened filtering to accept both "World Triathlon Championship Series" and legacy "World Triathlon Series" plus a generic conjunction (World Triathlon + Series) to guard against naming variance.
  - (Sept 2025 update) Added `para_filter` (None=all, True=para only, False=championship only) applied to events.is_para.
  - (Sept 2025 update) Added chart shaping helpers: `coerce_finish_position`, `melt_checkpoint_positions` for Streamlit visualization layer.
  - (Sept 2025 update) Expanded country filtering logic: multiple U.S. aliases (USA, United States, United States of America) with optional override via UI.
  - (Sept 2025 update) Simplified Streamlit WTCS page: replaced free-text country input with USA Only / All Countries toggle; country filter only applied when USA Only selected.
  - (Sept 2025 update) Added gender normalization (Male/Female and M/F both accepted) plus diagnostics panel (matched events count, rows after filters, distinct genders/countries) when no results.
  - (Sept 2025 update) Hardened gender filter: case-insensitive LOWER(a.gender) matching; added debug checkbox and fallback diagnostics to reveal distinct genders if filtered result empty.
  - `aggregate_checkpoint_metrics` – computes per-athlete average positions, gaps, ranks, and position deltas with race counts & min-event flag.
    - (Sept 2025 update) Added defensive numeric coercion for aggregation targets (finish_position, positions, gaps, ranks, deltas) converting non-numeric tokens like 'DNF' to NaN to avoid pandas TypeError during mean aggregation.
  - `select_best_worst_races` – placeholder logic selecting best (lowest finish) and worst (highest finish) event ids per athlete.
  - (Sept 2025 update) Streamlit integration refactored to single-athlete view: UI now forces selection of one athlete; aggregation table and charts scoped to that athlete only; athlete_id hidden from all displays for readability.
  - (Sept 2025 update) Added primary chart: chronological Finish Position line (event-order, reversed y-axis) plus secondary checkpoint positions chart; improves quick assessment of result trajectory.
  - (Sept 2025 update) Enforced finish position chart y-axis always starting at 1 (top). Implemented manual reversed range and phantom data point insertion when athlete has no 1st-place finishes to guarantee label visibility.

### New (Dec 2025)

#### `tri_analysis/wtcs_pack_metrics.py`
- **Purpose**: Precompute and persist full-field WTCS pack membership to keep Streamlit fast.
- **Definition**: Field is `(event_id, prog_id)`; checkpoints are swim/bike/run; pack assignment uses chain rule where a new pack starts when `gap_to_prev_sec > 2`.
- **Output Table**: `wtcs_pack_membership` (one row per athlete per checkpoint per event/program, only when elapsed checkpoint time is present).
- **Usage**: Run `python -m tri_analysis.wtcs_pack_metrics` to compute for all WTCS event/program pairs already in `events` / `position_metrics`.

#### `tri_analysis/wtcs_radar.py`
- **Purpose**: WTCS end-of-season “report card” radar scores for Swim/Bike/Run/Transitions.
- **Scoring**: Computes per-race percentiles vs the full WTCS field within `(event_id, prog_id)`; field percentiles are computed from `race_results` + `position_metrics` (does not require `athlete` rows for all non-USA athletes). Aggregates over season.
- **Output**: 1..10 values where 10 is strength and 1 is weakness; requires ≥3 WTCS races per category or leaves that category blank.

#### `tri_analysis/export_wtcs_radar.py`
- **Purpose**: CLI exporter for WTCS radar charts (SVG/PNG) for easy insertion into PowerPoint.
- **Usage**: Run `python -m tri_analysis.export_wtcs_radar --year YYYY --athlete-name "Full Name" --format svg` (or `--athlete-id`).
  - (Sept 2025 update) Replaced Plotly finish position chart with Seaborn/Matplotlib line: inverted y-axis, dynamic tick thinning (every 2 or 5), podium shading (1–3), improved label readability.


### New (Dec 2025)

#### `docs/para_standards_plan.md`
- **Purpose**: Implementation plan for para-category performance standards using Paris 2024 medalists as the benchmark set.
- **Key Points**: Auto-derives top-3 medalists from the Paris 2024 benchmark race for the selected para `prog_name`; filters to major para events; outputs trend charts by discipline.

#### `tri_analysis/time_utils.py`
- **Purpose**: Shared time parsing and pace conversion helpers (string times → seconds, seconds → formatted, sec/100m and sec/km pace helpers).
- **Key Points**: Returns `None` for invalid/missing times (avoids biasing metrics by treating missing as zero).

#### `tri_analysis/para_standards.py`
- **Purpose**: Standalone CLI report generator to compare a selected USA athlete vs Paris medalists for a para category since 2021.
- **Key Points**: Finds Paris benchmark event via `events` (major games + `is_para`), extracts medalists from `race_results`, filters history to major para events, outputs HTML+PNG charts and a CSV dataset.

#### `tests/test_para_standards.py`
- **Purpose**: Unit tests for time parsing and pace conversion helpers used by para standards reporting.
#### `tri_analysis/future_tri_events.py`
- Purpose: Fetch upcoming continental cups, world cups, WTCS, and para series events and compute nomination dates (Tuesday 30 days prior).
- How it works:
  - Calls World Triathlon event listing API from today to +1 year, filtered by CATEGORY_IDS (env override supported) using HEADERS from config.
  - Computes nomination_date for each event as Tuesday on/after (event_date - 30 days).
  - Prints a table to stdout and writes CSV to `tri_analysis/outputs/future_tri_events.csv`.
- Default categories covered: 340, 341, 342, 623, 352, 624, 348, 349, 449, 350.
- **Notes**: Initial skeleton powering new Streamlit page. Future expansion: pack classification persistence, lead/chase participation rates, PDF export staging.

#### `tri_analysis/update_program_entries.py`
- Purpose: Pull upcoming event start lists (Program Entries) and persist them to the DB for downstream prediction/simulation workflows.
- How it works:
  - Uses `fetch_events_ids` (date window) and `fetch_program_ids` (broad program selection) to build (event_id, prog_id) pairs.
  - Calls the Program Entries endpoint (`/events/{event_id}/programs/{prog_id}/entries?type=start`) and normalizes rows via `normalize_program_entries`.
  - Upserts into `program_entries` and deactivates entries that disappeared since the last successful pull.
- Notes:
  - Does not store waitlists or raw JSON snapshots (by design for now).
  - Tunable concurrency via env vars (`PROGRAM_WORKERS`, `ENTRY_WORKERS`) for faster extraction.

#### `streamlit_app.py` (Sept 2025 addition)
- Added new navigation option "WTCS US Performance" with skeleton page rendering summary table and best/worst race IDs using new helper module functions.

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
 
## December 2025 Update
- Metrics exclusion rule: Added global exclusion in `tri_analysis/metrics.py` so rows with any zero split (`swimsecs`, `t1secs`, `bikesecs`, `t2secs`, or `runsecs`) are excluded from minima and ranking calculations. Prevents 00:00:00 splits from creating artificial minimum thresholds across segments.

### New Utility (Dec 2025)
- `tri_analysis/stats_queries.py`: CLI script to print USA medal details/counts and country coverage stats directly from the database using `DB_URI` (env override). Provides medal detail table, medal tallies by category/host country, per-athlete country counts, and distinct country totals. Run with `python -m tri_analysis.stats_queries`.

- `tri_analysis/metrics.py`: Computes `position_metrics` and now supports `--latest-events` to only recompute metrics for event IDs in `latest_events.txt`; writes using an event-scoped refresh (delete those events’ metrics, then append) so prior events remain.