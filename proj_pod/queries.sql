-- Template query for event results joined to athlete and events
-- Adapt table/column names if needed via .env overrides
SELECT
  e.event_name,
  e.event_date,
  a.full_name AS athlete_name,
  COALESCE(a.country, rr.country) AS country,
  rr.position,
  e.prog_name AS program
FROM race_results rr
JOIN events e   ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
JOIN athlete a  ON rr.athlete_id = a.athlete_id
WHERE e.event_name = :event_name
  AND e.prog_name = 'Elite Men'
  AND rr.position IS NOT NULL;
