# Triathlon Database: A Comprehensive Data Pipeline and Analysis Tool

## Project Overview
The **Triathlon Database** project is a production-ready data engineering solution that automates the process of fetching, storing, and analyzing triathlon data. It integrates with the World Triathlon API to gather athlete profiles, race results, and rankings, storing over **65,000+ race result rows** across **2,000+ athletes** and **3,400+ events** in a PostgreSQL database for analysis and reporting.

## Key Features

**✅ Completed** 
### 1. Configuration Management, [folder](../config/)
- Centralized configuration system with environment variables and API endpoints
- Secure credential management with `.env` file support
- Modular configuration for different pipeline components
- Easy deployment configuration with Docker support

**✅ Completed**
### 2. Data Import Pipeline, [folder](../Data_Import/)
- **Master Data Import**: Full database initialization with athlete profiles, events, and historical race results
- **Concurrent Processing**: High-performance data fetching using `concurrent.futures` with configurable thread pools
- **Database Schema**: Robust PostgreSQL schema with proper indexing and constraints
- **Data Validation**: Automatic duplicate detection and removal to prevent constraint violations
- **Error Handling**: Comprehensive error handling with detailed logging for debugging

**✅ Completed**
### 3. Data Upload Pipeline, [folder](../Data_Upload/)
- **Incremental Updates**: Smart detection of new events since last update
- **Real-time Processing**: Fetches and processes new race results as events conclude
- **Elite Program Filtering**: Automatically identifies and processes Elite Men/Women programs
- **Upsert Operations**: Intelligent conflict resolution using PostgreSQL UPSERT functionality
- **Data Quality**: Pre-processing validation to ensure data integrity before database insertion

**✅ Completed**
### 4. Database Management & Analytics
- **PostgreSQL 15**: Production-grade database with optimized schema and indexing
- **Power BI Integration**: Direct database connection with automated refresh capabilities
- **Jupyter Notebooks**: Interactive analysis tools (`get_top_triathletes.ipynb`, `model_pipeline.ipynb`)
- **Reporting Dashboard**: Comprehensive Power BI report (`WTO_Report.pbix`) with podium analysis, fastest splits, and lifetime trends
- **Data Quality Monitoring**: Built-in validation and constraint checking

**🔄 In Progress**
### 5. Machine Learning & Predictions
- **Predictive Modeling**: ML pipeline for athlete performance and race outcome predictions
- **Feature Engineering**: Advanced feature extraction from historical performance data
- **Model Selection**: Comparative analysis of different ML algorithms (LightGBM, XGBoost, Scikit-learn)
- **Performance Metrics**: MSE optimization for accurate race time and position predictions
- **Integration**: Seamless integration with existing database and Power BI infrastructure

## Architecture & Implementation

### Database Schema
- **PostgreSQL 15** with optimized indexing and unique constraints
- **Tables**: `athlete`, `events`, `race_results`, `athlete_rankings`
- **Constraints**: Unique constraint on `(athlete_id, EventID, TotalTime)` prevents duplicate race results
- **Data Types**: Proper typing for timestamps, numeric values, and text fields

### ETL Pipeline
- **Extract**: World Triathlon API integration with rate limiting and error handling
- **Transform**: Data normalization, validation, and duplicate removal
- **Load**: UPSERT operations with conflict resolution and data integrity checks

### Recent Improvements (2025)
- **🐛 Bug Fix**: Resolved PostgreSQL unique constraint violations in ETL processes
- **🔧 Data Quality**: Added duplicate removal logic before database insertion
- **📊 Validation**: Implemented data preview functionality for race result upserts
- **⚡ Performance**: Optimized concurrent processing with configurable thread pools
- **🧪 Testing**: Added comprehensive test suite with CI/CD integration

## Technologies Used
- **Programming Language**: Python 3.13
- **Database**: PostgreSQL 15
- **Core Libraries**: 
  - `pandas` - Data manipulation and analysis
  - `SQLAlchemy 2.0` - Database ORM and connection management
  - `concurrent.futures` - Parallel processing
  - `requests` - API communication
- **ML Stack**: `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`
- **Analytics**: Power BI, Jupyter Notebooks
- **Infrastructure**: Docker, GitHub Actions CI/CD
- **Data Format**: Parquet for ML datasets

## Getting Started

### Prerequisites
```bash
# Install Python 3.13
# Install PostgreSQL 15
# Clone the repository
git clone https://github.com/jhigh13/triathlon-db.git
cd triathlon-db
```

### Quick Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Initialize database and import data
python main.py 1

# Or use Docker
docker-compose up
```

### Usage Options
1. **Full Import**: Complete database rebuild with all historical data
2. **Incremental Update**: Add recent events and results
3. **Single Athlete**: Import specific athlete data by name

## Data Quality & Reliability
- **Duplicate Prevention**: Automatic detection and removal of duplicate records
- **Data Validation**: Pre-insertion validation of required fields
- **Error Handling**: Comprehensive logging and graceful error recovery
- **Constraint Management**: Database-level integrity enforcement
- **Testing**: Automated test suite with coverage reporting

## Impact & Use Cases
This project serves as a comprehensive solution for:
- **Sports Analytics**: Performance tracking and trend analysis
- **Predictive Modeling**: Race outcome and athlete performance predictions
- **Business Intelligence**: Strategic insights for triathlon organizations
- **Research**: Academic analysis of triathlon performance data
- **Fan Engagement**: Interactive dashboards and visualizations

The system demonstrates modern data engineering best practices including robust ETL pipelines, data quality management, and scalable architecture design.