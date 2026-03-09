# Para Standards (Paris Medalist Benchmark) — Implementation Plan

## Goal
Build a standalone analysis script that, for a selected paratriathlon program (e.g., `PTWC Women`) and a selected USA athlete, automatically:

1. Identifies the Paris 2024 Paralympic race for that program.
2. Extracts the top-3 finishers (Paris medalists) from that race as the benchmark set.
3. Pulls “major event” history since `2021-01-01` for the selected USA athlete and those medalists.
4. Produces trend charts for Swim/Bike/Run/Overall plus T1/T2, split into Sprint vs Standard panels.
5. Outputs interactive HTML and static PNG charts, plus a CSV of the underlying filtered dataset.

This is designed to scale to any para program where Paris 2024 data exists.

---

## Data Sources (current DB)

Tables used:
- `events`:
  - keys: `event_id`, `prog_id`
  - fields used: `prog_name`, `prog_distance_category`, `is_para`, `event_name`, `event_date`, `cat_name`, distances (`swim_distance`, `bike_distance`, `run_distance`)
- `race_results`:
  - keys: `athlete_id`, `prog_id`, `total_time`
  - fields used: `swimtime`, `t1time`, `biketime`, `t2time`, `runtime`, `total_time`, `finish_status`, `finish_position`
- `athlete`:
  - fields used: `athlete_id`, `full_name`, `country`

Key assumptions:
- `events.*_distance` are meters (as confirmed).
- `events.is_para = TRUE` can be used to exclude Olympic (non-para) “Major Games”.
- “Major events only” can be reliably filtered using `events.cat_name` labels, plus `is_para`.

---

## Definitions

### Major events (para)
Filter to events where `events.is_para = TRUE` and `events.cat_name` indicates one of:
- Major Games (category id 343)
- Para Cups (category id 449)
- Para Series (category id 448)

Implementation detail: since category IDs are not stored directly, filter uses `cat_name ILIKE` patterns:
- `%Major Games%`
- `%Para Cup%`
- `%Para Series%`

(We keep these patterns configurable in the script.)

### Benchmark race (Paris 2024)
For the selected program:
- `events.prog_name = :category`
- `events.is_para = TRUE`
- `events.event_date` in year `2024`
- `events.cat_name ILIKE '%Major Games%'`
- `events.event_name ILIKE '%Paris%'` (fallback patterns can be added if needed)

If no benchmark race is found: fail with a clear message.

### Medalist standard set
From `race_results` for that benchmark race:
- Select rows with `finish_position IN (1,2,3)` and `finish_status = 'FINISH'`.
- Extract `athlete_id` values (and resolve names from `athlete` dimension for display).

If fewer than 3 finishers are found: fail clearly.

---

## Metrics

### Time parsing
Split times are stored as strings (e.g., `HH:MM:SS` or `MM:SS`).
- Parse to seconds, returning `None` for blank/invalid.

### Normalized pace (distance-adjusted)
Using distances from `events` (meters):
- Swim pace: seconds per 100m
- Bike pace: seconds per km
- Run pace: seconds per km

### Transitions
T1/T2 have no distance normalization:
- Chart raw seconds.

### Overall
Overall is shown as total time (seconds / formatted), split into Sprint vs Standard panels.

### Non-finish handling
- DNFs/DNS/DSQ/LAP appear as markers/annotations.
- They are excluded from averages/trendlines.

---

## Outputs

For the selected program + USA athlete:
- Interactive HTML report containing:
  - Swim pace trends (Sprint/Standard panels)
  - Bike pace trends (Sprint/Standard panels)
  - Run pace trends (Sprint/Standard panels)
  - T1 seconds trends (Sprint/Standard panels)
  - T2 seconds trends (Sprint/Standard panels)
  - Overall total time trends (Sprint/Standard panels)
  - Benchmark medalist names and benchmark event metadata
- PNG exports for each figure (via `kaleido`)
- CSV export of the filtered dataset used to generate charts

---

## CLI Interface (first version)
Proposed invocation:

```bash
python -m tri_analysis.para_standards \
  --category "PTWC Women" \
  --usa-athlete-name "Kendall Gretsch" \
  --since 2021-01-01 \
  --benchmark-year 2024 \
  --benchmark-city "Paris" \
  --outdir tri_analysis/outputs/para_standards
```

Notes:
- `--usa-athlete-name` resolves to `athlete_id` (interactive disambiguation if multiple matches).
- Later, a separate Streamlit app can wrap the same core functions for dropdown selectors.

---

## Testing Strategy
- Unit tests (no DB dependency):
  - time parsing (valid formats, invalid inputs)
  - pace calculations (meters → sec/100m, sec/km)
  - non-finish marker classification

DB integration is left to manual runs against the real DB.

---

## Implementation Steps
1. Add shared time parsing helpers in `tri_analysis/time_utils.py`.
2. Implement `tri_analysis/para_standards.py`:
   - DB queries (benchmark event, medalists, history)
   - metric calculation
   - plotting + exports
3. Add unit tests in `tests/test_para_standards.py`.
4. Ensure dependencies include `plotly` and `kaleido`.

---

## Edge Cases / Failure Modes
- Paris benchmark race not present in DB → fail with clear message.
- Multiple candidate Paris events → pick the latest date and print a warning.
- Missing distances for some events → pace becomes null for those rows (still chart raw split seconds if desired later).
- Name collisions for athlete selection → interactive prompt to select the correct athlete_id.
