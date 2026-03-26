---
mode: 'agent'
description: 'Implement a repo-aware plan step by step'
---
Your task is to implement the approved plan step by step in this repository.

Before making non-trivial changes:
- Review `CLAUDE.md`, `docs/prediction_status.md`, and the plan document.
- Search for symbol usage and mirrored logic across `tri_analysis/`, `scripts/`, `streamlit_app.py`, and `tests/`.

Implementation rules:
- Treat the plan as guidance, not a straitjacket. If you need to adjust the sequence, explain why.
- Keep changes focused and minimal. Do not refactor unrelated code.
- If the work touches the prediction pipeline, maintain parity across training, prediction, simulation, evaluation, and CLI entry points as needed.
- If the work touches models, preserve the current protocol: no training cutoff past `2025-06-30`, no bundle overwrite, deterministic backtests are primary.
- Add or update targeted tests when behavior changes.
- Update relevant docs when workflow or behavior materially changes.

Execution requirements:
- Mark each completed step in the plan Markdown file before moving to the next one.
- Report blockers or ambiguous requirements as soon as they are clear.
- Finish by summarizing what changed, what was validated, and any follow-up risk.