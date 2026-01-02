from pathlib import Path

import pytest

from tri_analysis.metrics import load_latest_event_ids


def test_load_latest_event_ids_ignores_invalid_lines(tmp_path: Path):
    p = tmp_path / "latest_events.txt"
    p.write_text("\n".join(["194291", "", "abc", "194292", "194291  ", "  194313"]) + "\n", encoding="utf-8")
    assert load_latest_event_ids(p) == [194291, 194292, 194313]


def test_load_latest_event_ids_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_latest_event_ids(tmp_path / "does_not_exist.txt")
