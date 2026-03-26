## GitHub Copilot / AI Agent Project Instructions

These rules should keep an AI assistant productive in this repository as it exists today. Keep responses concise, actionable, and specific to this codebase rather than giving generic Python or ML advice.

### Core Principles
1. The user is strong in general coding and wants brief rationale when you introduce production or ML patterns. Explain why a change matters, not just what to type.
2. Use Windows PowerShell formatting for shell examples and chain commands with ';'. Prefer the project virtual environment at `.venv`.
3. Before non-trivial edits, search for symbol usage and mirrored logic across `tri_analysis/`, `scripts/`, `streamlit_app.py`, and `tests/`.
4. When you create a NEW file, append a short entry to `docs/memory.md` with its purpose and the key decisions. For edits, only update `docs/memory.md` when behavior or project structure meaningfully changes.
5. After completing a cohesive feature or module, ask: `Commit changes now? (y/n)`.

### Current Architecture Snapshot
- Core package: `tri_analysis/`
- Prediction system: `tri_analysis/prediction/{features,train,predict,simulate,evaluate,sql,utils_time}.py`
- ETL / data workflows: `tri_analysis/build_database.py`, `tri_analysis/api_handling.py`, `tri_analysis/metrics.py`, `tri_analysis/wtcs_pack_metrics.py`, `tri_analysis/weather.py`
- Entry points: `scripts/train_models.py`, `scripts/run_backtest.py`, `scripts/predict_program.py`, `scripts/debug_diagnostics.py`, plus analysis and sweep scripts under `scripts/`
- Analytics surfaces: `streamlit_app.py`, notebooks, Power BI assets under `docs/power_bi_files/`, and CSV/JSON outputs under `outputs/`
- Active reference docs: `CLAUDE.md`, `docs/prediction_status.md`, `docs/experiment-log.md`, `docs/model_improvement_brainstorm.md`, and `docs/memory.md`

### Prediction Baseline And Workflow Rules
- Current documented champion: `models/bundle_elite_v45.joblib`
- Current documented deterministic baseline: P@10 `74.2%`, P@3 `57.0%`, Spearman `0.796`
- Training cutoff is `2025-06-30`. Do not train or evaluate with later data unless the user explicitly changes the backtest protocol.
- Backtest period is H2 2025. Deterministic backtest with `--no_sim` is the primary ranking evaluation.
- Monte Carlo simulation is valuable for probability outputs, not for primary ranking accuracy. Do not present MC as the preferred ranking path.
- Never overwrite an existing model bundle. Always increment the `bundle_elite_v{N}.joblib` version.

### Frequent Workflows
- Train a new model: `python scripts/train_models.py --output models/bundle_elite_v{N}.joblib [extra args]`
- Run primary backtest: `python scripts/run_backtest.py --model models/bundle_elite_v{N}.joblib --no_sim`
- Run prediction for an event: `python scripts/predict_program.py --event_id {ID} --prog_id {ID} --model_path models/bundle_elite_v{N}.joblib`
- Run diagnostics: `python scripts/debug_diagnostics.py --event_id {ID} --prog_id {ID} --model_path {model} --section overview`
- Find events: `python scripts/find_events.py`
- Run the app: `streamlit run streamlit_app.py`
- Run tests: `pytest -q`

### Codebase Conventions
- Reuse time helpers from `tri_analysis/prediction/utils_time.py` or existing shared utilities. Do not add another copy of time parsing and formatting logic.
- Keep database access on the existing SQLAlchemy path via `tri_analysis.database.get_engine()` and pandas `read_sql` where the codebase already follows that pattern.
- Preserve head-to-head pair ordering with `athlete_id_a < athlete_id_b` to avoid duplicate and asymmetric records.
- Treat pack thresholds and related tactical assumptions as parameters when possible. Avoid hard-coded experimental constants buried in analytical code.
- Handle DNF, DNS, DSQ, and missing split times as missing values where appropriate. Do not coerce them to zero.
- If you modify prediction features or data access, check downstream impact across `features.py`, `train.py`, `predict.py`, `simulate.py`, `evaluate.py`, `sql.py`, and the corresponding scripts.
- If you change output files, keep names descriptive and snake_case, matching the existing `outputs/` style.

### Quality And Safety
- When changing ranking, pack, or metric logic, search for the same domain concept across ETL utilities, Streamlit views, scripts, and tests before editing.
- Avoid silent schema drift. If a database shape must change, make the change explicit and describe the migration path.
- Do not hard-code secrets or credentials. Use `.env` and existing configuration paths.
- Do not swallow SQL or data quality errors with broad `try/except` blocks.

### When Lacking Context
1. Search the repo for the domain term first, especially in `tri_analysis/`, `scripts/`, and `docs/`.
2. Read `CLAUDE.md` and `docs/prediction_status.md` before making modeling assumptions.
3. If library behavior is uncertain, fetch documentation before guessing.

### Example Guidance
- Adding a new prediction feature: update the feature builder, training path, prediction path, and backtest coverage together; then verify the feature does not leak post-event information.
- Adjusting pack logic: update the core pack utility, any simulation dependency, any Streamlit or export consumer, and add or revise a targeted test.
- Investigating a model regression: run diagnostics first, compare against the champion baseline, and summarize the tier-level tradeoffs instead of only reporting overall metrics.

### Do Not
Do not reference the old `main.py` menu workflow or legacy `Data_Import/` and `Data_Upload/` paths as the primary architecture. Do not overwrite model files. Do not train past `2025-06-30`. Do not duplicate time utilities or introduce silent schema changes.