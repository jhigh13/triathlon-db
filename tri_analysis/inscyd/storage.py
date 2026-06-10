"""
On-disk cache for INSCYD test payloads.

Pulling polynomials from the API is comparatively heavy, so we fetch once and
store the raw API ``results`` as JSON. The engine reads these files later to
build metabolic profiles, so we never re-hit the API at optimization time.

Default location: ``triathlon-db/data/inscyd/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# repo_root/data/inscyd  (storage.py is at tri_analysis/inscyd/storage.py)
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "inscyd"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_filename(athlete_display_id: int | str | None, sport_id: int | str | None) -> str:
    parts = [str(athlete_display_id or "all"), str(sport_id or "sport")]
    return "_".join(parts) + ".json"


def save_raw_tests(
    results: list[dict],
    *,
    athlete_display_id: int | str | None = None,
    sport_id: int | str | None = None,
    source: str = "all_data",
    filename: str | None = None,
    out_dir: Path | str = DEFAULT_DIR,
) -> Path:
    """Persist raw API result dicts as a JSON payload. Returns the file path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = filename or default_filename(athlete_display_id, sport_id)
    path = out_dir / name
    payload: dict[str, Any] = {
        "fetched_at": _timestamp(),
        "source": source,
        "athlete_display_id": athlete_display_id,
        "sport_id": sport_id,
        "count": len(results),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_raw_tests(path: Path | str) -> dict:
    """Load a previously saved payload (``{fetched_at, source, ..., results}``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
