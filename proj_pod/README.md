# Project Podium — Archive (2025 Season)

One-off analysis tracking 9 USA Triathlon "Project Podium" athletes at 2025 Americas/World Triathlon Cup events. Compared their podium/win counts against Canada, Mexico, Other USA, and Other Countries.

## Athletes
| Name | Wins | Podiums | Races |
|---|---|---|---|
| Reese Vannerson | 3 | 4 | 6 |
| Braxton Legg | 1 | 3 | 7 |
| Mathis Beaulieu | 1 | 2 | 4 |
| Keller Norland | 1 | 1 | 7 |
| Porter Middaugh | 1 | 1 | 6 |
| Luke Anthony | 1 | 1 | 4 |
| Sullivan Middaugh | 0 | 2 | 5 |
| Blake Bullard | 0 | 1 | 4 |
| Blake Harris | 0 | 0 | 7 |

## Target Events (Continental Cups)
- 2025 Americas Triathlon Cup La Habana
- 2025 Americas Triathlon Cup Magog
- 2025 Americas Triathlon Cup Miami
- 2025 Americas Triathlon Cup Montreal
- 2025 Americas Triathlon Cup Salinas
- 2025 Americas Triathlon Cup Kelowna
- 2025 Americas Triathlon Cup La Paz

## Key Findings
- **Podiums (Top 3)**: Project Podium 12, Other USA 2, Canada 2, Mexico 2, Other Countries 3
- **Wins**: Project Podium 7, all others 0 at the continental cup level
- Only non-PP USA podiums: Darr Smith (3rd, Miami) and Morgan Pearson (2nd, Montreal)

## Podiums by Event
| Event | PP | Other USA | Canada | Mexico | Other |
|---|---|---|---|---|---|
| Kelowna | 2 | 0 | 1 | 0 | 0 |
| La Habana | 2 | 0 | 0 | 0 | 1 |
| La Paz | 2 | 0 | 0 | 1 | 0 |
| Magog | 2 | 0 | 1 | 0 | 0 |
| Miami | 1 | 1 | 0 | 0 | 1 |
| Montreal | 2 | 1 | 0 | 0 | 0 |
| Salinas | 1 | 0 | 0 | 1 | 1 |

## SQL Query Pattern
```sql
-- Fetch results for target events, Elite Men program
SELECT e.event_name, e.event_date,
       a.full_name AS athlete_name,
       COALESCE(a.country, rr.country) AS country,
       rr.position, e.prog_name AS program
FROM race_results rr
JOIN events e   ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
JOIN athlete a  ON rr.athlete_id = a.athlete_id
WHERE e.event_name = :event_name
  AND e.prog_name = 'Elite Men'
  AND e.event_date >= '2025-01-01' AND e.event_date < '2026-01-01'
  AND rr.position IS NOT NULL;

-- All results for specific athletes
SELECT e.event_name, e.event_date, e.prog_name AS program,
       a.full_name AS athlete_name, a.country, rr.position
FROM race_results rr
JOIN athlete a ON rr.athlete_id = a.athlete_id
JOIN events  e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
WHERE a.full_name = ANY(:pp_names)
  AND e.prog_name = 'Elite Men'
  AND e.event_date >= '2025-01-01' AND e.event_date < '2026-01-01';
```

## Workflow Summary
1. Define athlete list and target event list
2. Query DB (or load from `raw_results.csv` offline) for all event results
3. Normalize country names → USA/CAN/MEX/etc.
4. Filter to podium positions (≤ 3)
5. Categorize each result: Project Podium / Other USA / Canada / Mexico / Other Countries
6. Aggregate counts overall and per event
7. Visualize with bar charts; export PowerPoint report

## Reusable Script
See `project_podium_analysis.py` — a clean, single-file version of the notebook workflow.
Requires: `pandas`, `sqlalchemy`, `python-dotenv`, `plotly` (optional for charts).

## Environment
```env
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
```
