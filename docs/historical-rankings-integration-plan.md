# Historical Rankings Integration and Persistence Plan

## Overview
We need to adapt the current scraper to:
1. **Temporarily store scraped ranking data** in a staging table (no upsert) so we can inspect and debug raw imports.
2. Enhance the import pipeline:
   - For each scraped athlete record:
     - **Lookup** `athlete_id` in the existing `athlete` table by full name.
     - If found, assign it; otherwise **fetch** via the API:
       1. Use `search_id.get_athlete_id(full_name)` to obtain a new `athlete_id`.
       2. Retrieve athlete details with `athlete_data.get_athlete_info_df(athlete_id)`.
       3. **Upsert** this new athlete record into the `athlete` table.
     - **Collect** the athlete’s historical results with `athlete_data.get_athlete_results_df(athlete_id)` if needed.
3. **Upsert** ranking records into `athlete_rankings` with correct `athlete_id` linkage.
4. **Recalculate** position metrics if additional results have been ingested:
   - Either reload full `race_results` data and run `calculate_position_metrics`, or
   - Implement a targeted `update_position_metrics` function to adjust metrics for newly inserted records.

## Proposed Implementation Steps

### 1. Temporary Staging Table (Completed - saved into staging_rankings)
- Create a new table `staging_rankings` with the same columns as `athlete_rankings`, but without constraints or conflicts.
- Modify `upsert_rankings` (temporarily) to insert all scraped records into `staging_rankings`.
- This allows inspection of raw data before upserting.

### 2. Athlete ID Resolution
- Extract full athlete name (`"First Last"`) for each record.
- Query `athlete` table:
  ```sql
  SELECT athlete_id FROM athlete WHERE full_name = :name
  ```
- If missing:
  - Call `search_id.get_athlete_id(name)` to get an ID.
  - Fetch athlete details via `athlete_data.get_athlete_info_df(id)`.
  - Upsert into `athlete` table.

### 3. Ranking Upsert
- After ensuring `athlete_id` exists, upsert the ranking row into `athlete_rankings`:
  - Use existing ON CONFLICT logic.

### 4. Position Metrics Update
-  **Full refresh**
  - Reload all rows from `race_results` into a DataFrame.
  - Call `calculate_position_metrics` and upsert full results.

## Potential Challenges
- **API rate limits** when fetching missing athlete IDs and details.
- **Name variations** (e.g., diacritics, middle initials) may require fuzzy matching.
- **Data consistency**: ensuring race results used for metrics match ranking data timeline.
- **Performance**: full recalculation can be heavy for large datasets—incremental updates may be preferred.

---

_Next steps:_
- Confirm staging table schema and naming.
- Review API usage and error handling for athlete lookups.
- Design incremental metrics update interface.

_No code changes applied yet—awaiting confirmation._
