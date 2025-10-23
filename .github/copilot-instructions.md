## GitHub Copilot / AI Agent Project Instructions

These rules make an AI assistant immediately productive in this repository. Keep responses concise, actionable, and tailored to THIS codebase (not generic Python advice).

### Core Principles
1. User is strong in general coding, still learning production + ML: add brief rationale when introducing patterns (why, not just what).
2. Use Windows PowerShell formatting: chain multiple commands with ';'. Example: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt`.
3. Before non‑trivial edits: search for symbol usage (avoid breaking ETL invariants, ranking logic, pack metrics).
4. When you create a NEW file: append a short entry to `docs/memory.md` (purpose + key points). For edits, no memory update unless behavior/role meaningfully changes.
5. After completing a cohesive feature/module, ask: "Commit changes now? (y/n)".

### Architecture Snapshot
Data pipeline: World Triathlon API -> ETL scripts (`Data_Import/master_data_import.py` full load; `Data_Upload/update_race_results.py` incremental) -> PostgreSQL (tables: `athlete`, `events`, `race_results`, `athlete_rankings`, `position_metrics`) -> Analytics (Streamlit `streamlit_app.py`, Power BI `.pbix` files, notebooks, ML pipeline).
Key augmentation columns: per‑checkpoint positions (`Position_at_Swim` etc.), position deltas (negative = gained places), segment ranks (`SwimRank` ... `RunRank`). Maintain these when transforming results.

### Frequent Workflows
Full historical import (DESTRUCTIVE): `python main.py` (option 1). Incremental new events: same menu (option 2). Single athlete pull: option 3.
Run Streamlit app: `streamlit run streamlit_app.py` (ensure DB + `.env` with `TRI_API_KEY`, `DB_URI`).
Run tests: `pytest -q` (add focused tests for new ETL/ranking behaviors). Add new test files under `tests/` mirroring feature area.
ML experimentation lives in notebooks (`model_pipeline.ipynb`)—avoid embedding heavyweight model code inside ETL scripts; isolate feature engineering utilities if reused.

### Conventions & Patterns
Time parsing helpers (`time_to_seconds`, `convert_time_to_seconds`, `seconds_to_hms`) appear in multiple places—reuse, do NOT re‑invent. Centralize improvements instead of duplicating.
Cache in Streamlit via `@st.cache_data(ttl=600)` for stable lookups; if modifying queries, keep column names stable for downstream matrix builders.
Head‑to‑head logic builds pairwise combinations with `athlete_id_a < athlete_id_b` ordering; preserve this to avoid duplicates and asymmetry bugs.
Pack dynamics: thresholds (`max_gap_to_leader`, `max_gap_within_pack`) drive grouping—surface changes as parameters rather than hard‑coding new constants.
Database interactions: Use SQLAlchemy engine + pandas `read_sql`; bulk logic handled in import scripts—avoid ad‑hoc DDL changes inside analytical code.
Unique constraint on `race_results` requires pre‑deduplication (already implemented)—when adding columns, prefer `ALTER TABLE` migration script over silent schema drift.

### Quality & Safety
Before modifying ETL core logic: scan for same function name in both full + incremental scripts to keep parity (e.g., ranking + position change calculations live in both).
Always handle DNF / missing split times gracefully (keep `NaN` rather than zero; do not bias averages/gaps).
If adding new analytical export (CSV/JSON) follow existing naming style: snake_case, descriptive (`pp_wins_podiums_totals.csv`).

### When Lacking Context
1. Perform a code search for domain terms first (e.g., "Position_at_Swim", "SwimRank").
2. If external library uncertainty: resolve library id then fetch docs (Context7 tools) before guessing.

### Pull Request / Commit Guidance
Group related file + test + doc updates. Include: reason, data impact, backward compatibility note. Ask user before committing (see principle 5).

### Minimal Example Answers
Add column to ranking metrics? -> Update schema (if needed), adjust both import + update scripts, update Streamlit H2H or packs if they display it, add test asserting non‑null proportion.
Add new pack threshold control? -> Parameterize in function + expose Streamlit sidebar control with sane defaults.

### Do Not
Hard‑code secrets or API keys. Remove broad try/except swallowing SQL errors. Introduce silent schema changes. Duplicate time conversion utilities.

### Legacy Rules (Preserved)
Original learning focus and context acquisition rules still apply; integrate them with above actionable guidance.