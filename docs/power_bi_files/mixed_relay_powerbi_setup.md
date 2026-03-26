# Mixed Relay Power BI Setup

## Recommendation

Yes, Power BI is a good fit for this analysis.

For implementation, start with the simplest working model first.

Do not start by drilling from a relay team row into an individual-race page. A relay team contains four athletes and usually maps to two different individual programs:

- Elite Women
- Elite Men

So a team row does not have one unique `individual_prog_key`.

The correct drillthrough path is:

1. country
2. relay team
3. relay athlete slot
4. comparable individual race

The current export structure already supports a strong drillthrough workflow:

- high-level country and event-class summary
- country-specific team analysis for Great Britain, Germany, and United States
- drillthrough from country or event class into specific relay races
- drillthrough again into athlete slots within a race

The best current semantic model is:

1. one combined individual-race lens
2. one broader country-strength lens

In the current exports, those are:

- `individual_*`: same-event first, then same-weekend fallback if needed
- `prior_365_*`: previous 365-day country form

The slot export now also carries the exact comparable individual race keys:

- `individual_event_id`
- `individual_prog_id`
- `individual_prog_name`

These are the fields you should use when connecting the relay drillthrough to a broader `race_results` table in Power BI.

## Important Data Constraint

If you want the second lens to be `nearest athlete_rankings snapshot to the event date`, the current database is only partly ready for that.

Current ranking coverage in `athlete_rankings`:

- `World Rankings - Male` and `World Rankings - Female` exist only as a current 2026 snapshot
- `World Triathlon Series - Male` and `World Triathlon Series - Female` have historical coverage from 2016 to 2025

That means:

- a true historical `nearest world ranking snapshot` lens is not available yet
- a historical `nearest WTCS ranking snapshot` lens is available
- but WTCS ranking is not ideal for the exact question you raised, because some relay athletes are not ranked highly enough to appear in WTCS rankings

So for the current state of the repo, I recommend:

1. Use `individual_*` as the primary selection lens
2. Use `prior_365_*` as the current country-strength lens
3. If you want the ranking-snapshot lens later, first import historical world ranking snapshots into `athlete_rankings`

## Files To Load

Load these CSVs from [outputs](outputs):

- [outputs/mixed_relay_selection_slot_report.csv](outputs/mixed_relay_selection_slot_report.csv)
- [outputs/mixed_relay_selection_team_report.csv](outputs/mixed_relay_selection_team_report.csv)
- [outputs/mixed_relay_selection_event_class_summary.csv](outputs/mixed_relay_selection_event_class_summary.csv)
- [outputs/mixed_relay_selection_country_summary.csv](outputs/mixed_relay_selection_country_summary.csv)

You can skip `event_class_summary` and `country_summary` if you want a cleaner star schema and prefer all summaries to be computed by DAX from the slot and team tables.

If your Power BI model already has direct database or imported-table access to the full race results, also load:

- `race_results`
- `events`
- `athlete`

## Recommended Data Model

### Simple model that works first

Use this first. It is the easiest version to get working.

- `RelayTeams`
  - one row per country per relay event

- `RelaySlots`
  - one row per relay athlete slot

- `IndividualRaceResults`
  - one row per athlete result in one Elite Men or Elite Women program

- `DimCountry`
  - one row per country

Relationships:

- `DimCountry[country]` -> `RelayTeams[country]`
- `DimCountry[country]` -> `RelaySlots[country]`
- `DimCountry[country]` -> `IndividualRaceResults[country]`
- `RelayTeams[team_key]` -> `RelaySlots[team_key]`
- `RelaySlots[individual_prog_key]` <-> `IndividualRaceResults[prog_key]`

For the last relationship, use:

- cardinality: many-to-many
- cross-filter direction: both

This is not the prettiest semantic model, but it is the most pragmatic way to get the drillthrough working now.

Only after that works should you consider replacing it with a bridge table.

### More formal model

Use this model in Power BI:

- `RelayTeams`
  - source: `mixed_relay_selection_team_report.csv`
  - grain: one row per relay team in one event

- `RelaySlots`
  - source: `mixed_relay_selection_slot_report.csv`
  - grain: one row per relay athlete slot

- `DimEvent`
  - reference table created from `RelayTeams`
  - one row per event/program

- `DimCountry`
  - reference table created from `RelayTeams`
  - one row per country

- `DimEventClass`
  - reference table created from `RelayTeams`
  - one row per event class

- `IndividualRaceResults`
  - source: full `race_results` joined to `events` and `athlete`
  - grain: one athlete result in one event program

- `DimIndividualProgram`
  - reference table created from `IndividualRaceResults`
  - one row per individual elite event/program

Recommended relationships:

- `DimCountry[country]` -> `RelayTeams[country]` one-to-many
- `DimCountry[country]` -> `RelaySlots[country]` one-to-many
- `DimEvent[event_key]` -> `RelayTeams[event_key]` one-to-many
- `DimEvent[event_key]` -> `RelaySlots[event_key]` one-to-many
- `DimEventClass[event_class]` -> `RelayTeams[event_class]` one-to-many
- `DimEventClass[event_class]` -> `RelaySlots[event_class]` one-to-many

For the full individual-race drillthrough, add a program-level key in both `RelaySlots` and `IndividualRaceResults`:

- `individual_prog_key = Text.From([individual_event_id]) & "-" & Text.From([individual_prog_id])`
- `prog_key = Text.From([event_id]) & "-" & Text.From([prog_id])`

Do not start here unless the simple model is already working.

If you later want a cleaner semantic model, replace the direct many-to-many relationship with a bridge table.

Do not relate `RelaySlots` directly to `IndividualRaceResults` in the formal version if both tables contain repeated program keys. That becomes a fact-to-fact many-to-many pattern and is the source of the inactive relationship problem.

Instead, create a bridge table named `DimIndividualProgram` with one row per `prog_key`.

Then create these active relationships:

- `DimIndividualProgram[prog_key]` -> `RelaySlots[individual_prog_key]`
- `DimIndividualProgram[prog_key]` -> `IndividualRaceResults[prog_key]`

Both should be one-to-many, single direction from `DimIndividualProgram` to the fact tables.

Use single-direction filtering from dimensions to facts.

## Power Query Steps

### 1. Load the CSVs

In Power BI Desktop:

1. Home -> Get Data -> Text/CSV
2. Select `mixed_relay_selection_team_report.csv`
3. Select `mixed_relay_selection_slot_report.csv`
4. Click Transform Data

### 2. Create event keys

In both `RelayTeams` and `RelaySlots`, add a custom column named `event_key`:

```powerquery
Text.From([event_id]) & "-" & Text.From([prog_id])
```

In `RelaySlots`, also add:

```powerquery
Text.From([individual_event_id]) & "-" & Text.From([individual_prog_id])
```

Name that column `individual_prog_key`.

In the full individual-race results table, add:

```powerquery
Text.From([event_id]) & "-" & Text.From([prog_id])
```

Name that column `prog_key`.

Also create team keys:

In `RelayTeams`:

```powerquery
[event_key] & "|" & [country]
```

In `RelaySlots`:

```powerquery
[event_key] & "|" & [country]
```

Name that column `team_key` in both tables.

### 3. Set data types

Set these explicitly:

- `event_date` or `individual_event_date`: Date
- `individual_event_id`: Whole Number
- `individual_prog_id`: Whole Number
- `team_finish`: Whole Number
- `individual_top2_slots`: Whole Number
- `prior_365_top2_slots`: Whole Number
- `avg_individual_country_rank`: Decimal Number
- `avg_prior_365_country_rank`: Decimal Number
- boolean flags like `medal_team`, `all_individual_top2`, `all_prior_365_top2`: True/False

### 4. Create dimensions

Reference `RelayTeams` three times:

- `DimCountry`
  - keep only `country`
  - remove duplicates

- `DimEventClass`
  - keep only `event_class`
  - remove duplicates

- `DimEvent`
  - keep `event_key`, `event_id`, `prog_id`, `event_name`, `event_class`, `event_date`
  - remove duplicates

- `DimIndividualProgram`
  - reference `IndividualRaceResults`
  - keep `prog_key`, `event_id`, `prog_id`, `prog_name`, `event_name`, `event_date`
  - remove duplicates

### 5. Optional: country focus flag

In `DimCountry`, add a custom column:

```powerquery
if [country] = "Great Britain" or [country] = "Germany" or [country] = "United States" then "Priority" else "Other"
```

This gives you a clean slicer for the three countries you care about.

## Full Individual Event Drillthrough

This is the extra step you asked for: once a relay team is selected, you want to see not just the relay athlete slot row, but the full comparable individual field and the subset of athletes from that same country.

That is now supported by the new slot columns.

Important limitations:

Not every relay athlete row has a comparable individual program.

If `raced_individual_comparison = FALSE`, then the row has no usable `individual_prog_id` and cannot drill through into an individual field. In those cases, Power BI should show blank or you should filter those rows out of the drillthrough entry table.

Also, do not drill through from the relay team table into the individual-race page. That will often fail or show nothing because the team row does not correspond to one individual elite program.

### Build the full individual results table

If you already have a full race-results connection in Power BI, create a table or query with these columns:

- `event_id`
- `prog_id`
- `prog_name`
- `event_name`
- `event_date`
- `athlete_id`
- `full_name`
- `country`
- `gender`
- `finish_position`

Name it `IndividualRaceResults`.

Important: keep only individual elite programs if you want the drillthrough page to stay clean:

- `Elite Men`
- `Elite Women`

### Best filtering pattern

On the athlete/team drillthrough page, use the selected row from `RelaySlots` to carry:

- `individual_event_id`
- `individual_prog_id`
- `country`

Then show two visuals:

1. full comparable individual field
2. same-country athletes in that field

Best practice: only allow drillthrough from rows where:

- `raced_individual_comparison = TRUE`
- `individual_prog_id` is not blank

### Page design

Create a page named `Comparable Individual Race`.

Add these drillthrough fields:

- `RelaySlots[individual_prog_key]`
- `RelaySlots[country]`
- optionally `RelaySlots[relay_athlete]`

Turn on `Keep all filters`.

Add a Back button.

Add page filters:

- `RelaySlots[raced_individual_comparison] = TRUE`
- `RelaySlots[individual_prog_key]` is not blank

### Visual 1: Full field table

Use `IndividualRaceResults` and add:

- `finish_position`
- `full_name`
- `country`
- `gender`
- `prog_name`

Filter this visual to the selected `individual_event_id` and `individual_prog_id`.

If the direct many-to-many relationship is active, the selected `individual_prog_key` will filter `IndividualRaceResults` automatically.

### Visual 2: Same-country athletes table

Duplicate the same table.

Then add a visual-level filter:

- `country` equals the selected `RelaySlots[country]`

This gives you exactly what you described:

- the relay athlete's comparable individual race
- the other athletes from that country who raced in that same individual field

### Optional helper measure

In `IndividualRaceResults`, add:

```DAX
Selected Country Athlete =
IF(
  SELECTEDVALUE(RelaySlots[country]) = SELECTEDVALUE(IndividualRaceResults[country]),
  1,
  0
)
```

This is most useful when the page is already filtered to one country. If the relationship approach is awkward, use a direct visual filter on `country` instead.

### Even cleaner approach

If you want the least fragile version, create a Power Query table from the database that already joins the comparable individual program key to the result rows. Then Power BI does not need to infer anything.

But with the bridge table plus the new exported columns, you do not need that immediately.

## Working Build Order

If you are rebuilding from scratch, do it in this order:

1. Load `RelayTeams`, `RelaySlots`, and `IndividualRaceResults`
2. Create `event_key`, `team_key`, `individual_prog_key`, and `prog_key`
3. Create `DimCountry`
4. Build these relationships first:
  - `DimCountry` to all three tables on `country`
  - `RelayTeams[team_key]` to `RelaySlots[team_key]`
  - `RelaySlots[individual_prog_key]` to `IndividualRaceResults[prog_key]` as many-to-many, both
5. Build the main page
6. Build the relay athlete slot table
7. Only then build the `Comparable Individual Race` drillthrough page

If the drillthrough menu shows nothing, check these in order:

1. Are you clicking a row in `RelaySlots` rather than `RelayTeams`?
2. Does that row have `raced_individual_comparison = TRUE`?
3. Does that row have a non-blank `individual_prog_key`?
4. Is the relationship to `IndividualRaceResults[prog_key]` active?
5. Is the drillthrough page using `RelaySlots[individual_prog_key]` rather than `DimIndividualProgram[prog_key]`?

## Recommended DAX Measures

Create these in `RelayTeams` unless noted.

### Core team counts

```DAX
Teams = COUNTROWS(RelayTeams)
```

```DAX
Medal Teams = CALCULATE(COUNTROWS(RelayTeams), RelayTeams[medal_team] = TRUE())
```

```DAX
Medal Rate = DIVIDE([Medal Teams], [Teams])
```

```DAX
Average Finish = AVERAGE(RelayTeams[team_finish])
```

### Individual-race lens

```DAX
Avg Individual Top2 Slots = AVERAGE(RelayTeams[individual_top2_slots])
```

```DAX
Avg Individual Rank = AVERAGE(RelayTeams[avg_individual_country_rank])
```

```DAX
Teams All Individual Top2 = CALCULATE(COUNTROWS(RelayTeams), RelayTeams[all_individual_top2] = TRUE())
```

```DAX
All Individual Top2 Rate = DIVIDE([Teams All Individual Top2], [Teams])
```

```DAX
Matched Individual Slots Avg = AVERAGE(RelayTeams[matched_individual_slots])
```

### Country-strength lens

```DAX
Avg Prior365 Top2 Slots = AVERAGE(RelayTeams[prior_365_top2_slots])
```

```DAX
Avg Prior365 Rank = AVERAGE(RelayTeams[avg_prior_365_country_rank])
```

```DAX
Teams All Prior365 Top2 = CALCULATE(COUNTROWS(RelayTeams), RelayTeams[all_prior_365_top2] = TRUE())
```

```DAX
All Prior365 Top2 Rate = DIVIDE([Teams All Prior365 Top2], [Teams])
```

### Athlete-slot lens

Create these in `RelaySlots` or as measures over that table.

```DAX
Relay Slots = COUNTROWS(RelaySlots)
```

```DAX
Individual Top2 Slot Rate =
DIVIDE(
    CALCULATE(COUNTROWS(RelaySlots), RelaySlots[individual_top2] = TRUE()),
    [Relay Slots]
)
```

```DAX
Prior365 Top2 Slot Rate =
DIVIDE(
    CALCULATE(COUNTROWS(RelaySlots), RelaySlots[prior_365_top2] = TRUE()),
    [Relay Slots]
)
```

### Diagnostic comparison measure

```DAX
Selection Gap = [Avg Individual Top2 Slots] - [Avg Prior365 Top2 Slots]
```

This is useful because it shows whether a country tends to select athletes who are stronger in the event-weekend field than they are in broader country form.

## Recommended Calculated Columns

In `RelayTeams`:

```DAX
Finish Band =
SWITCH(
    TRUE(),
    RelayTeams[team_finish] = 1, "Gold",
    RelayTeams[team_finish] = 2, "Silver",
    RelayTeams[team_finish] = 3, "Bronze",
    RelayTeams[team_finish] <= 8, "Top 8",
    "Other"
)
```

```DAX
Individual Top2 Band =
SWITCH(
    TRUE(),
    RelayTeams[individual_top2_slots] = 4, "4 of 4",
    RelayTeams[individual_top2_slots] = 3, "3 of 4",
    RelayTeams[individual_top2_slots] = 2, "2 of 4",
    RelayTeams[individual_top2_slots] = 1, "1 of 4",
    "0 of 4"
)
```

## Recommended Report Pages

### 1. Overview page

Visuals:

- Cards
  - Teams
  - Medal Rate
  - Avg Individual Top2 Slots
  - Avg Prior365 Top2 Slots

- Clustered bar chart
  - Axis: `event_class`
  - Values: `Avg Individual Top2 Slots`, `Avg Prior365 Top2 Slots`

- Matrix
  - Rows: `country`
  - Values: `Teams`, `Medal Teams`, `Average Finish`, `Avg Individual Top2 Slots`, `Avg Prior365 Top2 Slots`

- Scatter plot
  - X: `Avg Individual Rank`
  - Y: `Average Finish`
  - Size: `Teams`
  - Legend: `country`

Use slicers for:

- country
- event class
- event date
- priority country flag

### 2. Country focus page

Set a page filter or slicer default to:

- Great Britain
- Germany
- United States

Visuals:

- Line and clustered column chart
  - Axis: `event_date`
  - Column values: `individual_top2_slots`
  - Line values: `team_finish`

- Matrix
  - Rows: `event_name`
  - Values: `team_finish`, `individual_top2_slots`, `prior_365_top2_slots`, `avg_individual_country_rank`, `avg_prior_365_country_rank`

- Decomposition tree
  - Analyze: `Average Finish`
  - Explain by: `country`, `event_class`, `Finish Band`, `Individual Top2 Band`

### 3. Race drillthrough page

Add a drillthrough page using `event_key`.

Include:

- cards for event name, date, event class, country, team finish
- table of athlete slots with:
  - `relay_athlete`
  - `gender`
  - `individual_event_name`
  - `individual_country_rank`
  - `prior_365_country_rank`
  - `prior_365_starts`
  - `prior_365_form_score`

This page answers: who was selected, what was their comparable individual rank, and how did that differ from broader country form?

### 4. Athlete detail page

Optional second drillthrough from `RelaySlots` using `relay_athlete` plus `country`.

Include:

- table of all relay appearances
- table of event-level slot metrics
- trend line of `prior_365_form_score` over time

## Drillthrough Setup

For the race page:

1. Create a new page named `Race Drillthrough`
2. Add `event_key` to the Drillthrough filters pane
3. Turn on `Keep all filters`
4. Add a Back button

Recommended source visuals for drillthrough:

- country matrix
- event-class bar chart
- country focus race table

## Best Visual Story For Your Question

If your core question is whether relay teams are mostly just the best overall athletes, the cleanest visual story is:

1. Overview card: average number of top-2 comparable individual-race athletes per relay team
2. Comparison card: average number of top-2 prior-365 country-form athletes per relay team
3. Country matrix for Great Britain, Germany, United States
4. Drillthrough table showing exactly which athletes were and were not top-2 under each lens

That keeps the logic clear:

- event-weekend selection reality
- broader country strength
- concrete athlete examples

## Suggested Next Enhancement

If you want the second lens to be rankings-based rather than form-based, the right next step is not a Power BI change first. The right next step is data:

1. import historical world ranking snapshots into `athlete_rankings`
2. export a new event-athlete ranking snapshot table keyed to the latest snapshot on or before each event date
3. add that table to Power BI as a second country-strength fact

Until then, `prior_365_*` is the most complete and defensible second lens currently available in this repo.