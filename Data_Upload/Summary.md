# Triathlon Database: A Comprehensive Data Pipeline and Analysis Tool

## Project Overview
The **Triathlon Database** project automates the process of fetching, storing, and analyzing data related to triathletes, events, and race results. It integrates with the World Triathlon API to gather athlete profiles, race results, and rankings, and stores the data in a PostgreSQL database for further analysis and reporting.

## Key Features

### 1. Data Import Pipeline
- Fetches athlete profiles and race results concurrently using Python's `concurrent.futures`.
- Processes and organizes data into structured tables: `athletes`, `events`, and `race_results`.
- Stores the data in a PostgreSQL database for long-term storage and querying.

### 2. API Integration
- Utilizes the World Triathlon API to retrieve up-to-date information on athletes, events, and rankings.
- Configurable endpoints for searching athletes, fetching results, and accessing event details.

### 3. Database Management
- A robust database schema to store and manage triathlon data.
- Includes scripts for initializing and updating the database.

### 4. Data Analysis and Reporting
- Jupyter notebooks for exploratory data analysis (e.g., `get_top_triathletes.ipynb`).
- Integration with Power BI for creating detailed reports and visualizations (e.g., `WTO_Report.pbix`).

### 5. Modular Design
- Organized into modules for data import (`Data_Import`), data upload (`Data_Upload`), and configuration (`config`).
- Easily extensible for additional features or data sources.

## Technologies Used
- **Programming Language**: Python
- **Database**: PostgreSQL
- **Libraries**: Pandas, SQLAlchemy, Concurrent Futures
- **Visualization**: Power BI, Jupyter Notebooks
- **API**: World Triathlon API

## Impact
This project is a powerful tool for triathlon enthusiasts, analysts, and organizations to gain insights into athlete performance, event trends, and rankings. It showcases the use of modern data engineering practices to streamline data collection and analysis.