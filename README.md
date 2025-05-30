# Triathlon‑DB

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

A Power BI report with interactive tabs (World Triathlon Series, Athlete Overview, etc.) will be publicly embedded **(local only for now – open `reports/triathlon_dashboard.pbix` in Power BI Desktop)**

---

## Road‑Map

* Integrate into Docker container for easy replication 
* Conduct PCA for exploratory data analysis 
* Perform feature engineering to prepare data for ML pipeline testing 
* Implement model to predict athlete's future race time and project event placements 
* Integrate conversational UI using Azure Bot Service & OpenAI API to allow users to query race stats and receive insights 

---

## Contributing

Pull requests welcome. Please open an issue first to discuss major changes.

```bash
# run tests (will exist after Sprint‑2)
pytest -q
```

---

## License

MIT – see [LICENSE](LICENSE) for details.

---

## Contact

📧 [johnkhigh@outlook.com](mailto:johnkhigh@outlook.com)
🔗 [www.linkedin.com/in/john-high](http://www.linkedin.com/in/john-high)