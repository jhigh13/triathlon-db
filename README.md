# Triathlon‑DB  ** OUTDATED ** 

[![CI](https://github.com/jhigh13/triathlon-db/actions/workflows/ci.yml/badge.svg)](https://github.com/jhigh13/triathlon-db/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/coveralls/github/jhigh13/triathlon-db?style=flat)](https://coveralls.io/github/jhigh13/triathlon-db)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/jhigh13/triathlon-db)](LICENSE)

> **Analyze current & historical triathlete performance across 65 K race‑result rows with a real‑time Python → Postgres → Power BI pipeline.**
---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Quick‑Start](#quick-start)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Example Analysis](#example-analysis)
7. [Road‑Map](#road-map)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## Features

* **Real‑time data ingest** from the public [World Triathlon API](https://developers.triathlon.org/reference/athletes-api-overview) 💧
* **65 K+ race‑result rows** across 2 K athletes / 3.4 K events stored in **Postgres 15** 📈
* **Python ETL** scripts with menu‑driven CLI – full import, incremental event update, or single‑athlete pull 🐍
* **Power BI analytics** – podiums, fastest splits, lifetime trends (public embed coming soon) 📊
* Ready for **weekly scheduled refresh** via GitHub Actions ⏲️
* Future: ML model to **predict athlete finish time & rank** 🧠

---

## Architecture

```
┌────────────┐      REST      ┌──────────────┐       SQL        ┌──────────────┐
│ World      │ ───────────▶  │  ETL Scripts │ ───────────────▶ │ PostgreSQL   │
│ Triathlon  │               │  (Python)    │                 │   15         │
└────────────┘               └──────────────┘                 └──────┬───────┘
                                                                      │
                                                                      │ (Direct query)
                                                                      ▼
                                                              Power BI Dashboard
```

---

## Quick Start

```bash
# 1. Clone
$ git clone https://github.com/jhigh13/triathlon-db.git
$ cd triathlon-db

# 2. Create and activate a virtual environment (Python 3.13.3)
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install deps
$ pip install -r requirements.txt

# 4. Copy env template and add your World Triathlon key
$ cp .env.example .env
$ echo "TRI_API_KEY=<your_key>" >> .env

# 5. Run full import (Warning: overwrites tables)
$ python main.py   # choose option 1
```

### Docker (optional)

A `docker-compose.yml` will be added in v2 to spin‑up Postgres + ETL in one command.

---

## Configuration

```dotenv
# .env.example
TRI_API_KEY=
DB_URI=postgresql+psycopg2://postgres:<password>@localhost:5432/triathlon_results
```

All config variables are loaded via `config/config.py`.

---

## Usage

```bash
# Incrementally add new events (option 2)
$ python main.py
# Add a single athlete by name (option 3)
$ python main.py
```

---

## Example Analysis

```sql
-- Top 10 fastest swim splits in 2024 (seconds)
SELECT athlete_name, event_name, swim_time
FROM race_results
WHERE EXTRACT(YEAR FROM event_date) = 2024
ORDER BY swim_time ASC
LIMIT 10;
```

> *Expected result*: 10 rows, columns: `athlete_name | event_name | swim_time`.

## Power BI Report: WTO_Report

I have included both the raw `.pbix` and a static PDF export below.

- **Download the PBIX file** (requires Power BI Desktop):  
  [WTO_Report_Rankings.pbix](docs/WTO_Report_Rankings.pbix)

- **View the PDF in‐browser** (no download needed):  
  [▶ View Static Report (PDF)](https://jhigh13.github.io/triathlon-db/WTO_Report.pdf)

---

## Road‑Map

* Integrate into Docker container for easy replication 
* Conduct PCA for exploratory data analysis 
* Perform feature engineering to prepare data for ML pipeline testing 
* Implement model to predict athlete's future race time and project event placements 
* Integrate conversational UI using Azure Bot Service & OpenAI API to allow users to query race stats and receive insights 

---

## Race Prediction (v1)

Pre-race predictions for WTCS draft-legal events using historical results, pack dynamics, and Monte Carlo simulation.

### Quick Start

```bash
# 1. Train models on historical data (2022-2025)
python scripts/train_models.py \
    --start_date 2022-01-01 \
    --end_date 2025-12-31 \
    --output models/bundle.joblib

# 2. Predict an upcoming program
python scripts/predict_program.py \
    --event_id 178882 \
    --prog_id 553345 \
    --model_path models/bundle.joblib
```

### Features Used (MVP)
- **Athlete Form**: EWMA swim/bike/run/total (last 5), std_total_24m, days_since_last_race
- **Draft-Legal Pack Metrics**: front_pack_rate, avg_swim_gap_leader (from `wtcs_pack_membership`)
- **Field Context**: seed_total_rank, n_entrants

### Outputs
- Predicted finishing order (deterministic, sorted by total time)
- Predicted splits (swim, bike, run) and total time
- Win/podium/top-5/top-10/top-20 probabilities (Monte Carlo)
- Rank intervals (10th-90th percentile)
- CSV export to `outputs/`

### Backtesting
```python
from tri_analysis.prediction.evaluate import backtest_events
from tri_analysis.database import get_engine

engine = get_engine()
results = backtest_events(
    engine,
    event_prog_keys=[(178882, 553345), (178883, 553346)],
    bundle_path="models/bundle.joblib"
)
print(results[["event_id", "precision_at_10", "spearman_corr", "mae_total_sec"]])
```

### Module Structure
```
tri_analysis/prediction/
├── __init__.py      # Package exports
├── utils_time.py    # Time parsing (mm:ss → seconds)
├── sql.py           # Database queries (no leakage)
├── features.py      # Feature engineering
├── train.py         # Model training + persistence
├── predict.py       # Deterministic predictions
├── simulate.py      # Monte Carlo probabilities
└── evaluate.py      # Metrics + backtesting
```

---


## License

MIT – see [LICENSE](LICENSE) for details.

---

## Contact

📧 [johnkhigh@outlook.com](mailto:johnkhigh@outlook.com)
🔗 [www.linkedin.com/in/john-high](http://www.linkedin.com/in/john-high)