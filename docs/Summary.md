# Triathlon Database: A Comprehensive Data Pipeline and Analysis Tool

## Project Overview
The **Triathlon Database** project automates the process of fetching, storing, and analyzing data related to triathletes, events, and race results. It integrates with the World Triathlon API to gather athlete profiles, race results, and rankings, and stores the data in a PostgreSQL database for further analysis and reporting.

## Key Features

**Completed** 
### 1. Configuration, [folder](../config/)
- Holds global variables and key links for API calls. 
- Organized into modules for data import (`Data_Import`), data upload (`Data_Upload`), and configuration (`config`).
- Easily extensible for additional features or data sources.

**Completed**
### 2. Data Import Pipeline, [folder](../Data_Import/)
- A robust database schema to store and manage triathlon data.
- Includes scripts for initializing and updating the database.
- Calls the WTO API to query for raw JSON data attributed to World Triathlon athletes. 
- Fetches athlete profiles and race results concurrently using Python's `concurrent.futures`.
- Processes and organizes data into structured tables: `athletes`, `events`, and `race_results`, `athlete_rankings`.
- Stores the data in a PostgreSQL database for long-term storage and querying.

**Completed**
### 3. Data Upload Pipeline, [folder](../Data_Upload/)
- Utilizes the World Triathlon API to retrieve up-to-date information on athletes, events, and rankings.
- Configurable endpoints for fetching recent race results, and accessing event details.

**Completed**
### 4. Database Management

- Jupyter notebooks for exploratory data analysis (e.g., `get_top_triathletes.ipynb`).
- Integration with Power BI for creating detailed reports and visualizations (e.g., `WTO_Report.pbix`).
- PowerBI uses direct import with the PostgreSQL database and can refresh after data uploads following events. 

**In Progress**
### 5. Race and Athlete Prediction 

- Utilize past athlete performances to predict upcoming events for the World Triathlon Series.
- Jupyter notebooks for exploratory data analysis, feature engineer, labelling, and model selection (`model_pipeline.ipynb`)
- Analyze data to determine necessary data cleaning, and feature variables to improve MSE and enable accurate athlete race time and position predictions. 
- Store as additional table in the database and pull into PowerBI for analysis in the report. 

## Technologies Used
- **Programming Language**: Python
- **Database**: PostgreSQL
- **Libraries**: Pandas, SQLAlchemy, Concurrent Futures
- **Visualization**: Power BI, Jupyter Notebooks
- **API**: World Triathlon API

## Impact
This project is a powerful tool for triathlon enthusiasts, analysts, and organizations to gain insights into athlete performance, event trends, and rankings. It showcases the use of modern data engineering practices to streamline data collection and analysis.