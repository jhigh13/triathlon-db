
# Triathlon Power BI Report: Visualization Brainstorm Overview

Below is a menu of ideas for visuals, grouped by the questions a High-Performance Director, coach, or sport scientist might ask during a post-race debrief. Items marked 🟢 can be built immediately with existing columns; 🔵 require a small additional measure or ETL tweak; 🟠 assume bringing in more data (e.g., rankings, weather).

---

## 1. “How did the *race itself* unfold?”  (Event-level pacing & pack dynamics)

| Visual | Why it helps | Build notes |
|---|---|---|
| **🟢 Animated bump chart** (play axis on StageIndex) | Lets management *watch* positions shuffle every few seconds; highlights decisive moves | Turn the existing bump chart into an “auto-play” using the built-in Power BI play axis custom visual or Charticulator |
| **🟢 Cumulative time-gap ribbon** | Area chart with shaded band = time back from leader by stage for **top-N** athletes | Use `BehindSwim … BehindRun`; add a Top N slicer on final rank so the ribbon isn’t cluttered |
| **🟢 “Traffic-light” grid (heatmap)** | Table where rows = Athletes, columns = Stages, cell colour = Rank percentile; coaches scan for red (poor split) | Create a disconnected “Percentile thresholds” table (0-25-50-75-100) and conditional formatting |
| **🟢 Density plot / ridgeline of gaps** | Shows compression or fragmentation of the field at each split | Use violin or histogram custom visual; facet by Stage |
| **🔵 Chord diagram of place changes** | Quickly spot big movers/losers between two checkpoints | Need a 2-column table *FromRank* / *ToRank*; you can unpivot your existing `Position_at_*` columns in Power Query |

---

## 2. “Where were OUR athletes strong or weak?”  (Athlete- vs Field comparison)

| Visual | Why it helps | Build notes |
|---|---|---|
| **🟢 Split-time percentile radar** | One web-like chart per athlete: each axis = stage percentile | Measure: `Percentile = DIVIDE(SplitRank, COUNTROWS(EventProgram))` |
| **🟢 Box-and-whisker of each split** with selected athlete overlaid | Instantly flags whether 4th-fastest swim is actually *good* or *bad* depending on distribution | Use `SwimSecs` etc.; highlight with diverging colour if athlete < quartile 1 |
| **🟢 KPI card strip (“hero metrics”)** | • Final Pos  
• Total Time vs Winner  
• Fastest split rank  
• Net places gained | All columns exist; put at top of athlete page |
| **🔵 “If I took 30 sec off the bike…” What-If slider** | Engages coaches: move a slider, watch hypothetical new rank propagate | Parameter table with seconds delta; adjust `HypoElapsedBike = ElapsedBike – Slider[sec]`; recalc downstream measures |

---

## 3. “How are we trending across the season?”  (Longitudinal)

| Visual | Why it helps | Build notes |
|---|---|---|
| **🟢 Small-multiple line charts** of finish rank by event date (one per athlete) | Surfaces consistency vs sporadic spikes | Use `EventDate` from *events* dim; filter on athlete list |
| **🔵 Slopegraph: World Triathlon ranking → race finish** | Are higher-ranked athletes *meeting* expectations? | Requires the `athlete_rankings` table you already ingest; relate on `athlete_id` & choose latest `retrieved_at` snapshot |
| **🟠 Micro-heatmap of training stimulus vs race rank** | If you eventually land daily load data, a simple calendar map shows taper effectiveness | Would need external TSS or HR-based load feed |

---

## 4. “How did **packs** form on the bike?”  (Group tactics)

| Visual | Why it helps | Build notes |
|---|---|---|
| **🟢 Scatter: BikeElapsed vs SwimElapsed coloured by BikeSplitRank** | Quickly shows who bridged up, who stayed away | Axis already in fact table |
| **🔵 Cluster-detect table** (“Pack ID”) | Label each athlete with the *same* behind-time (± 5 sec) at each 2-km loop; then summarise pack size | Easy in Python: groupby loop checkpoint, round `BehindBike` to nearest 5 |
| **🔵 Sankey of pack membership over laps** | Visual storytelling of breakaway splits/merges | Needs the Pack ID column above plus Lap dimension |

---

## 5. “Which *country/program* is outperforming?”  (Management scoreboard)

| Visual | Why it helps | Build notes |
|---|---|---|
| **🟢 Medal table matrix** | Rows = Country, Columns = Gold/Silver/Bronze counts, with sort on total | Count rows where `Position = 1`, etc. |
| **🟢 Stacked bar: Points scored per event** | If you assign World Triathlon points to top 30, management sees ROI of sending athletes | Build static Points lookup table (place → points) and SUMX |
| **🟠 Bubble chart: Federation spend vs medals** | Strategic budget conversation | Needs external spend data |

---

## 6. Data / ETL tweaks you *might not be considering yet*

| Idea | Benefit | Effort |
|---|---|---|
| **Lap-level split ingestion** | Unlocks in-depth bike/run pacing charts, pack analysis | API has `/laps` endpoint; add into `race_laps` fact |
| **Weather & course profile** | Contextualise bad days; filter windy vs calm races | Dark Sky / Open-Meteo API + course elevation CSV |
| **Live GPS (once available)** | Real-time dashboard for on-site coaches | Stream to Azure Event Hubs, push to Power BI streaming dataset |
| **Physiological profiling link** | Merge lab VO2max and lactate curves to predict split potential | Secure athlete consent + OneDrive Excel link |

---

### Anything missing?

- Do you want **interactive storytelling** (buttons/bookmarks that walk management through “Key Moments of the Race”)?  
- Is **mobile layout** a priority for on-deck tablets?  
- Should we attach **alert rules** (e.g., email if an athlete’s gap exceeds 45 sec by mid-bike)?

---

*End of overview.*
