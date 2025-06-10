-- Update staging_rankings table with manual name mappings and athlete IDs
-- Based on names_mapping.csv file

-- Update Summer Cook -> Summer Rappaport with athlete_id 83096
UPDATE staging_rankings 
SET athlete_name = 'Summer Rappaport', 
    athlete_id = 83096
WHERE athlete_name = 'Summer Cook' 
  AND athlete_id IS NULL;

-- Update Chris Deegan -> Christopher Deegan with athlete_id 165609
UPDATE staging_rankings 
SET athlete_name = 'Christopher Deegan', 
    athlete_id = 165609
WHERE athlete_name = 'Chris Deegan' 
  AND athlete_id IS NULL;

-- Update Steffen Justus with athlete_id 5747 (name stays same)
UPDATE staging_rankings 
SET athlete_id = 5747
WHERE athlete_name = 'Steffen Justus' 
  AND athlete_id IS NULL;

-- Update Ai Ueda with athlete_id 5576 (name stays same)
UPDATE staging_rankings 
SET athlete_id = 5576
WHERE athlete_name = 'Ai Ueda' 
  AND athlete_id IS NULL;

-- Check results
SELECT athlete_name, athlete_id, COUNT(*) as records_updated
FROM staging_rankings 
WHERE athlete_name IN ('Summer Rappaport', 'Christopher Deegan', 'Steffen Justus', 'Ai Ueda')
GROUP BY athlete_name, athlete_id;
