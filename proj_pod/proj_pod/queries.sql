-- Template query for event results joined to athlete and events
SELECT
  e.event_name,
  e.event_date,
  a.full_name AS athlete_name,
  a.country,
  rr.position
FROM race_results rr
JOIN events e   ON rr.event_id = e.event_id
JOIN athlete a  ON rr.athlete_id = a.athlete_id
WHERE e.event_name = :event_name
  AND rr.position IS NOT NULL;