"""
World Triathlon ranking points computation engine.

Implements the official WT ranking criteria (2025 version) to compute
points earned per athlete per event, then simulate weekly rankings.

Key rules:
  - Points decay 7.5% per position: points = base * 0.925^(pos-1)
  - Sprint events score at 25% of base points
  - Athletes must finish within 108% of winner's time (cut-off)
  - Continental events get a Quality of Field (QoF) boost
  - Top-5 at Continental Championships get a bonus (25/20/15/10/5%)
  - Rolling 2-year window: best 6 events current period (full) +
    best 6 previous period (1/3 value), max 12 total
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Event classification: (substring, base_points, uses_qof, event_type_key)
# Evaluated in priority order — first match wins.
# ---------------------------------------------------------------------------
EVENT_POINTS_MAP: list[tuple[str, float, bool, str]] = [
    ("World Championship Finals",    1250.0, False, "champ_finals"),
    ("Championship Finals",          1250.0, False, "champ_finals"),
    ("Olympic",                      1000.0, False, "olympics"),
    ("World Championship Series",    1000.0, False, "wtcs"),
    ("T100",                          500.0, False, "t100"),
    ("World Triathlon Cup",           500.0, False, "world_cup"),
    ("World Cup",                     500.0, False, "world_cup"),
    ("Continental Championships",     400.0, True,  "cont_champs"),
    ("Continental Cup",               250.0, True,  "cont_cup"),
    ("Regional Championships",        150.0, False, "regional"),
    ("Development Regional Cup",      125.0, False, "dev_regional"),
    ("National Championships",         50.0, False, "national"),
]

# Positions that get a bonus at Continental Championships
CONT_CHAMPS_BONUS: dict[int, float] = {
    1: 0.25,
    2: 0.20,
    3: 0.15,
    4: 0.10,
    5: 0.05,
}

# ---------------------------------------------------------------------------
# Country → continent mapping (IOC country names used in athlete table)
# ---------------------------------------------------------------------------
COUNTRY_CONTINENT: dict[str, str] = {
    # Europe
    "Albania": "Europe", "Andorra": "Europe", "Armenia": "Europe",
    "Austria": "Europe", "Azerbaijan": "Europe", "Belarus": "Europe",
    "Belgium": "Europe", "Bosnia and Herzegovina": "Europe",
    "Bulgaria": "Europe", "Croatia": "Europe", "Cyprus": "Europe",
    "Czech Republic": "Europe", "Czechia": "Europe", "Denmark": "Europe",
    "Estonia": "Europe", "Finland": "Europe", "France": "Europe",
    "Georgia": "Europe", "Germany": "Europe", "Great Britain": "Europe",
    "Greece": "Europe", "Hungary": "Europe", "Iceland": "Europe",
    "Ireland": "Europe", "Israel": "Europe", "Italy": "Europe",
    "Kazakhstan": "Europe", "Kosovo": "Europe", "Latvia": "Europe",
    "Liechtenstein": "Europe", "Lithuania": "Europe", "Luxembourg": "Europe",
    "Malta": "Europe", "Moldova": "Europe", "Monaco": "Europe",
    "Montenegro": "Europe", "Netherlands": "Europe", "North Macedonia": "Europe",
    "Norway": "Europe", "Poland": "Europe", "Portugal": "Europe",
    "Romania": "Europe", "Russia": "Europe", "San Marino": "Europe",
    "Serbia": "Europe", "Slovakia": "Europe", "Slovenia": "Europe",
    "Spain": "Europe", "Sweden": "Europe", "Switzerland": "Europe",
    "Turkey": "Europe", "Ukraine": "Europe",
    # Americas
    "Antigua and Barbuda": "Americas", "Argentina": "Americas",
    "Aruba": "Americas", "Bahamas": "Americas", "Barbados": "Americas",
    "Belize": "Americas", "Bermuda": "Americas", "Bolivia": "Americas",
    "Brazil": "Americas", "Canada": "Americas", "Cayman Islands": "Americas",
    "Chile": "Americas", "Colombia": "Americas", "Costa Rica": "Americas",
    "Cuba": "Americas", "Dominican Republic": "Americas", "Ecuador": "Americas",
    "El Salvador": "Americas", "Guatemala": "Americas", "Guyana": "Americas",
    "Haiti": "Americas", "Honduras": "Americas", "Jamaica": "Americas",
    "Mexico": "Americas", "Nicaragua": "Americas", "Panama": "Americas",
    "Paraguay": "Americas", "Peru": "Americas", "Puerto Rico": "Americas",
    "Trinidad and Tobago": "Americas", "United States": "Americas",
    "Uruguay": "Americas", "Venezuela": "Americas", "Virgin Islands": "Americas",
    "Virgin Islands, US": "Americas",
    # Asia
    "Afghanistan": "Asia", "Bahrain": "Asia", "Bangladesh": "Asia",
    "Cambodia": "Asia", "China": "Asia", "Chinese Taipei": "Asia",
    "Hong Kong": "Asia", "Hong Kong, China": "Asia", "India": "Asia",
    "Indonesia": "Asia", "Iran": "Asia", "Iraq": "Asia", "Japan": "Asia",
    "Jordan": "Asia", "Kuwait": "Asia", "Kyrgyzstan": "Asia",
    "Lebanon": "Asia", "Malaysia": "Asia", "Mongolia": "Asia",
    "Myanmar": "Asia", "Nepal": "Asia", "North Korea": "Asia",
    "Oman": "Asia", "Pakistan": "Asia", "Philippines": "Asia",
    "Qatar": "Asia", "Saudi Arabia": "Asia", "Singapore": "Asia",
    "South Korea": "Asia", "Sri Lanka": "Asia", "Syria": "Asia",
    "Tajikistan": "Asia", "Thailand": "Asia", "Timor-Leste": "Asia",
    "Turkmenistan": "Asia", "United Arab Emirates": "Asia",
    "Uzbekistan": "Asia", "Vietnam": "Asia", "Yemen": "Asia",
    # Africa
    "Algeria": "Africa", "Angola": "Africa", "Benin": "Africa",
    "Botswana": "Africa", "Burkina Faso": "Africa", "Burundi": "Africa",
    "Cameroon": "Africa", "Cape Verde": "Africa",
    "Central African Republic": "Africa", "Chad": "Africa",
    "Comoros": "Africa", "Congo": "Africa", "DR Congo": "Africa",
    "Djibouti": "Africa", "Egypt": "Africa", "Eritrea": "Africa",
    "Eswatini": "Africa", "Ethiopia": "Africa", "Gabon": "Africa",
    "Gambia": "Africa", "Ghana": "Africa", "Guinea": "Africa",
    "Kenya": "Africa", "Lesotho": "Africa", "Liberia": "Africa",
    "Libya": "Africa", "Madagascar": "Africa", "Malawi": "Africa",
    "Mali": "Africa", "Mauritania": "Africa", "Mauritius": "Africa",
    "Morocco": "Africa", "Mozambique": "Africa", "Namibia": "Africa",
    "Niger": "Africa", "Nigeria": "Africa", "Rwanda": "Africa",
    "Senegal": "Africa", "Sierra Leone": "Africa", "Somalia": "Africa",
    "South Africa": "Africa", "South Sudan": "Africa", "Sudan": "Africa",
    "Tanzania": "Africa", "Togo": "Africa", "Tunisia": "Africa",
    "Uganda": "Africa", "Zambia": "Africa", "Zimbabwe": "Africa",
    # Oceania
    "American Samoa": "Oceania", "Australia": "Oceania", "Cook Islands": "Oceania",
    "Fiji": "Oceania", "Guam": "Oceania", "Kiribati": "Oceania",
    "Marshall Islands": "Oceania", "Micronesia": "Oceania", "Nauru": "Oceania",
    "New Caledonia": "Oceania", "New Zealand": "Oceania", "Palau": "Oceania",
    "Papua New Guinea": "Oceania", "Samoa": "Oceania",
    "Solomon Islands": "Oceania", "Tonga": "Oceania", "Tuvalu": "Oceania",
    "Vanuatu": "Oceania",
}

CONTINENTS = ["Europe", "Americas", "Asia", "Africa", "Oceania"]

# ---------------------------------------------------------------------------
# QoF factors derived from real athlete_ranking_breakdown data (2025)
# Continental Cup: max 20% (Europe), others proportional
# Continental Championships: max 30% (Europe), others proportional
# Keyed by continent name.
# ---------------------------------------------------------------------------
QOF_CUP_DEFAULT: dict[str, float] = {
    "Europe":   1.20,
    "Americas": 1.12,
    "Asia":     1.06,
    "Oceania":  1.05,
    "Africa":   1.02,
}
QOF_CHAMPS_DEFAULT: dict[str, float] = {
    "Europe":   1.30,
    "Americas": 1.17,
    "Asia":     1.08,
    "Oceania":  1.08,
    "Africa":   1.04,
}


def classify_event(cat_name: str) -> tuple[float, bool, str] | None:
    """
    Map a cat_name string to (base_points, uses_qof, event_type).
    Returns None if the event should not score (e.g. para, junior, age-group).
    Uses priority-ordered substring matching.
    """
    if not cat_name:
        return None
    # Skip non-scoring categories
    skip = ("para", "junior", "age-group", "age group", "youth",
            "u23", "paratriathlon", "mixed relay", "indoor")
    lower = cat_name.lower()
    if any(s in lower for s in skip):
        return None

    for substring, base_pts, uses_qof, event_type in EVENT_POINTS_MAP:
        if substring.lower() in lower:
            return (base_pts, uses_qof, event_type)
    return None


def country_to_continent(country: str) -> str:
    """Return the continent for a country name. Defaults to 'Unknown'."""
    return COUNTRY_CONTINENT.get(country, "Unknown")


def parse_time_seconds(time_str: str) -> Optional[float]:
    """
    Parse HH:MM:SS or MM:SS into total seconds.
    Returns None if unparseable.
    """
    if not time_str or not time_str.strip():
        return None
    parts = time_str.strip().split(":")
    try:
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        return None


def is_sprint(prog_distance_category: str, swim_distance_m: Optional[float]) -> bool:
    """
    Return True if this program is a sprint-distance event.
    Primary: check prog_distance_category field.
    Fallback: swim distance < 1000m implies sprint.
    """
    if prog_distance_category:
        cat = prog_distance_category.lower().replace("-", "_")
        if cat == "sprint":
            return True
        if cat in ("standard", "olympic", "long_distance", "middle_distance",
                   "long", "middle", "super_sprint"):
            return False
    # Fallback: swim < 1000m → sprint
    if swim_distance_m is not None and swim_distance_m > 0:
        return swim_distance_m < 1000.0
    return False


def points_for_position(
    base_points: float,
    position: int,
    sprint: bool = False,
    qof_factor: float = 1.0,
    cont_champs_bonus: bool = False,
) -> float:
    """
    Compute ranking points for a given finish position.

    Args:
        base_points: Winner's base points for this event type (e.g. 1000 for WTCS)
        position: Finish position (1-indexed)
        sprint: If True, apply 25% sprint reduction
        qof_factor: Quality of Field multiplier (1.0 = no boost)
        cont_champs_bonus: If True, apply position-based bonus for Continental Champs
    """
    pts = base_points * (0.925 ** (position - 1))
    if sprint:
        pts *= 0.25
    pts *= qof_factor
    if cont_champs_bonus and position in CONT_CHAMPS_BONUS:
        pts *= (1.0 + CONT_CHAMPS_BONUS[position])
    return round(pts, 4)


def compute_qof_factors(
    engine,
    year: int,
    gender: str,  # "Male" or "Female"
    ranking_cat_id: int,
) -> dict[str, dict[str, float]]:
    """
    Compute annual QoF factors from Dec 31 rankings of the prior year.
    Returns {"cup": {continent: factor}, "champs": {continent: factor}}.

    Falls back to QOF_CUP_DEFAULT / QOF_CHAMPS_DEFAULT if data is unavailable.
    """
    from sqlalchemy import text

    prior_year = year - 1

    with engine.connect() as conn:
        # Get the latest ranking snapshot from prior year
        snap = conn.execute(text("""
            SELECT MAX(retrieved_at) FROM athlete_rankings
            WHERE ranking_cat_id = :cat_id AND year = :yr
        """), {"cat_id": ranking_cat_id, "yr": prior_year}).scalar()

        if snap is None:
            # Try earliest snapshot of current year as fallback
            snap = conn.execute(text("""
                SELECT MIN(retrieved_at) FROM athlete_rankings
                WHERE ranking_cat_id = :cat_id AND year = :yr
            """), {"cat_id": ranking_cat_id, "yr": year}).scalar()

        if snap is None:
            return {"cup": QOF_CUP_DEFAULT, "champs": QOF_CHAMPS_DEFAULT}

        # Fetch top 400 athletes from that snapshot
        rows = conn.execute(text("""
            SELECT ar.athlete_id, ar.rank_position, a.country
            FROM athlete_rankings ar
            LEFT JOIN athlete a ON a.athlete_id = ar.athlete_id
            WHERE ar.ranking_cat_id = :cat_id
              AND ar.retrieved_at = :snap
            ORDER BY ar.rank_position
            LIMIT 400
        """), {"cat_id": ranking_cat_id, "snap": snap}).fetchall()

    if not rows:
        return {"cup": QOF_CUP_DEFAULT, "champs": QOF_CHAMPS_DEFAULT}

    # Assign descending values: rank 1 = 400, rank 400 = 1
    # Sum by continent
    n = len(rows)
    continent_totals: dict[str, float] = {c: 0.0 for c in CONTINENTS}
    for i, row in enumerate(rows):
        country = row[2] or ""
        continent = country_to_continent(country)
        value = n - i  # top athlete gets n points
        if continent in continent_totals:
            continent_totals[continent] += value

    max_total = max(continent_totals.values()) if continent_totals else 1.0

    # Cup: max continent gets 20%, others proportional
    cup_factors: dict[str, float] = {}
    for cont in CONTINENTS:
        if max_total > 0:
            pct = math.ceil((continent_totals[cont] / max_total) * 20)
        else:
            pct = 0
        cup_factors[cont] = 1.0 + pct / 100.0

    # Champs: max continent gets 30%, others proportional
    champs_factors: dict[str, float] = {}
    for cont in CONTINENTS:
        if max_total > 0:
            pct = math.ceil((continent_totals[cont] / max_total) * 30)
        else:
            pct = 0
        champs_factors[cont] = 1.0 + pct / 100.0

    return {"cup": cup_factors, "champs": champs_factors}


def event_country_to_continent(event_country: str) -> str:
    """Map event host country to continent for QoF lookup."""
    return country_to_continent(event_country)


def compute_event_points_batch(engine) -> int:
    """
    Score all elite race results from 2020-01-01 onward.
    Inserts/replaces rows in computed_event_points.
    Returns number of rows written.
    """
    from sqlalchemy import text

    # Cache QoF factors per year per gender
    qof_cache: dict[tuple[int, str], dict] = {}

    def get_qof(year: int, gender: str) -> dict:
        key = (year, gender)
        if key not in qof_cache:
            cat_id = 13 if gender == "Male" else 14
            qof_cache[key] = compute_qof_factors(engine, year, gender, cat_id)
        return qof_cache[key]

    with engine.connect() as conn:
        # Load all elite events from 2020 onward with their results
        print("Loading elite events and results from 2020+...")
        events = conn.execute(text("""
            SELECT
                e.event_id,
                e.prog_id,
                e.cat_name,
                e.prog_name,
                e.prog_distance_category,
                e.swim_distance,
                e.event_date,
                e.event_country,
                rr.athlete_id,
                rr.finish_position,
                rr.total_time,
                rr.finish_status
            FROM events e
            JOIN race_results rr ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
            WHERE e.prog_name IN ('Elite Men', 'Elite Women')
              AND e.event_date >= '2020-01-01'
              AND rr.finish_position IS NOT NULL
            ORDER BY e.event_id, e.prog_id, rr.finish_position
        """)).fetchall()

    print(f"Loaded {len(events):,} result rows. Classifying and scoring...")

    # Group by (event_id, prog_id)
    from collections import defaultdict
    programs: dict[tuple, list] = defaultdict(list)
    for row in events:
        key = (row[0], row[1])  # event_id, prog_id
        programs[key].append(row)

    rows_to_insert = []

    for (event_id, prog_id), results in programs.items():
        # Grab event metadata from first row
        first = results[0]
        cat_name        = first[2] or ""
        prog_name       = first[3] or ""
        dist_cat        = first[4] or ""
        swim_dist       = first[5]
        event_date      = first[6]
        event_country   = first[7] or ""

        gender = "Male" if prog_name == "Elite Men" else "Female"

        # Classify the event
        classification = classify_event(cat_name)
        if classification is None:
            continue

        base_points, uses_qof, event_type = classification

        # Sprint detection
        sprint = is_sprint(dist_cat, swim_dist)

        # QoF factor
        qof_factor = 1.0
        if uses_qof and event_date:
            year = event_date.year
            qof = get_qof(year, gender)
            continent = event_country_to_continent(event_country)
            if event_type == "cont_cup":
                qof_factor = qof["cup"].get(continent, QOF_CUP_DEFAULT.get(continent, 1.0))
            elif event_type == "cont_champs":
                qof_factor = qof["champs"].get(continent, QOF_CHAMPS_DEFAULT.get(continent, 1.0))

        is_cont_champs = (event_type == "cont_champs")

        # Find winner's time for cut-off
        winner_time = None
        for r in results:
            if r[9] == 1 and r[10]:  # finish_position == 1
                winner_time = parse_time_seconds(r[10])
                break

        cutoff_time = (winner_time * 1.08) if winner_time else None

        for r in results:
            athlete_id     = r[8]
            position       = r[9]
            total_time_str = r[10]
            finish_status  = r[11]

            if finish_status != "FINISH" or position is None:
                continue

            # Cut-off check
            scored = True
            if cutoff_time is not None and total_time_str:
                athlete_time = parse_time_seconds(total_time_str)
                if athlete_time is not None and athlete_time > cutoff_time:
                    scored = False

            pts = 0.0
            if scored:
                pts = points_for_position(
                    base_points, position,
                    sprint=sprint,
                    qof_factor=qof_factor,
                    cont_champs_bonus=is_cont_champs,
                )

            rows_to_insert.append({
                "athlete_id":      athlete_id,
                "event_id":        event_id,
                "prog_id":         prog_id,
                "event_date":      event_date,
                "finish_position": position,
                "base_points":     base_points,
                "qof_factor":      qof_factor,
                "sprint_factor":   0.25 if sprint else 1.0,
                "points":          pts,
                "scored":          scored,
                "event_type":      event_type,
                "gender":          gender,
            })

    print(f"Inserting {len(rows_to_insert):,} scored rows...")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS computed_event_points (
                athlete_id      INTEGER NOT NULL,
                event_id        INTEGER NOT NULL,
                prog_id         INTEGER NOT NULL,
                event_date      DATE    NOT NULL,
                gender          VARCHAR NOT NULL,
                finish_position INTEGER,
                base_points     FLOAT,
                qof_factor      FLOAT DEFAULT 1.0,
                sprint_factor   FLOAT DEFAULT 1.0,
                points          FLOAT,
                scored          BOOLEAN,
                event_type      VARCHAR,
                PRIMARY KEY (athlete_id, event_id, prog_id)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cep_date
            ON computed_event_points(event_date, gender)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cep_athlete
            ON computed_event_points(athlete_id, gender)
        """))

        # Batch upsert
        BATCH = 500
        for i in range(0, len(rows_to_insert), BATCH):
            batch = rows_to_insert[i:i+BATCH]
            conn.execute(text("""
                INSERT INTO computed_event_points
                    (athlete_id, event_id, prog_id, event_date, gender,
                     finish_position, base_points, qof_factor, sprint_factor,
                     points, scored, event_type)
                VALUES
                    (:athlete_id, :event_id, :prog_id, :event_date, :gender,
                     :finish_position, :base_points, :qof_factor, :sprint_factor,
                     :points, :scored, :event_type)
                ON CONFLICT (athlete_id, event_id, prog_id) DO UPDATE SET
                    points       = EXCLUDED.points,
                    scored       = EXCLUDED.scored,
                    qof_factor   = EXCLUDED.qof_factor,
                    sprint_factor = EXCLUDED.sprint_factor,
                    event_type   = EXCLUDED.event_type
            """), batch)

    return len(rows_to_insert)


def simulate_weekly_rankings(engine, start_date: date, end_date: date) -> int:
    """
    Generate weekly ranking snapshots for every Sunday between start_date and end_date.
    Inserts into computed_weekly_rankings.
    Returns total rows written.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS computed_weekly_rankings (
                ranking_date    DATE    NOT NULL,
                ranking_cat_id  INTEGER NOT NULL,
                athlete_id      INTEGER NOT NULL,
                rank_position   INTEGER NOT NULL,
                total_points    FLOAT   NOT NULL,
                events_current  INTEGER,
                events_previous INTEGER,
                PRIMARY KEY (ranking_date, ranking_cat_id, athlete_id)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cwr_date_cat
            ON computed_weekly_rankings(ranking_date, ranking_cat_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cwr_athlete
            ON computed_weekly_rankings(athlete_id, ranking_cat_id)
        """))

    # Walk every Sunday
    sundays = []
    d = start_date
    while d.weekday() != 6:  # advance to first Sunday
        d += timedelta(days=1)
    while d <= end_date:
        sundays.append(d)
        d += timedelta(weeks=1)

    total_rows = 0

    for cat_id, gender in [(13, "Male"), (14, "Female")]:
        print(f"\nSimulating {len(sundays)} weeks for cat {cat_id} ({gender})...")

        for ranking_date in sundays:
            curr_start = ranking_date - timedelta(weeks=52)
            prev_start = ranking_date - timedelta(weeks=104)
            prev_end   = curr_start - timedelta(days=1)

            rows = _compute_ranking_for_date(
                engine, ranking_date, cat_id, gender,
                curr_start, ranking_date, prev_start, prev_end,
            )

            if rows:
                with engine.begin() as conn:
                    conn.execute(text("""
                        DELETE FROM computed_weekly_rankings
                        WHERE ranking_date = :rd AND ranking_cat_id = :cat
                    """), {"rd": ranking_date, "cat": cat_id})

                    conn.execute(text("""
                        INSERT INTO computed_weekly_rankings
                            (ranking_date, ranking_cat_id, athlete_id,
                             rank_position, total_points,
                             events_current, events_previous)
                        VALUES
                            (:ranking_date, :ranking_cat_id, :athlete_id,
                             :rank_position, :total_points,
                             :events_current, :events_previous)
                    """), rows)
                total_rows += len(rows)

        print(f"  Done cat {cat_id}. Running total rows: {total_rows:,}")

    return total_rows


def _compute_ranking_for_date(
    engine,
    ranking_date: date,
    cat_id: int,
    gender: str,
    curr_start: date,
    curr_end: date,
    prev_start: date,
    prev_end: date,
) -> list[dict]:
    """
    Compute one weekly ranking snapshot using computed_event_points.
    Returns a list of row dicts ready for insertion.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT athlete_id, event_date, points, scored
            FROM computed_event_points
            WHERE gender = :gender
              AND event_date BETWEEN :prev_start AND :curr_end
              AND scored = TRUE
            ORDER BY athlete_id, event_date
        """), {
            "gender":     gender,
            "prev_start": prev_start,
            "curr_end":   curr_end,
        }).fetchall()

    if not rows:
        return []

    # Group by athlete
    from collections import defaultdict
    athlete_events: dict[int, list[tuple[date, float, str]]] = defaultdict(list)
    for r in rows:
        athlete_id, event_date, points, scored = r
        period = "curr" if curr_start <= event_date <= curr_end else "prev"
        athlete_events[athlete_id].append((event_date, points, period))

    results = []
    for athlete_id, events in athlete_events.items():
        curr_pts = sorted(
            [pts for _, pts, p in events if p == "curr"],
            reverse=True
        )[:6]
        prev_pts = sorted(
            [pts for _, pts, p in events if p == "prev"],
            reverse=True
        )[:6]

        total = sum(curr_pts) + sum(p / 3.0 for p in prev_pts)
        if total <= 0:
            continue

        results.append({
            "athlete_id":      athlete_id,
            "total_points":    round(total, 4),
            "events_current":  len(curr_pts),
            "events_previous": len(prev_pts),
        })

    # Sort by points descending → assign ranks
    results.sort(key=lambda x: x["total_points"], reverse=True)
    output = []
    for rank, r in enumerate(results, start=1):
        output.append({
            "ranking_date":    ranking_date,
            "ranking_cat_id":  cat_id,
            "athlete_id":      r["athlete_id"],
            "rank_position":   rank,
            "total_points":    r["total_points"],
            "events_current":  r["events_current"],
            "events_previous": r["events_previous"],
        })
    return output


def validate_against_official(engine, cat_id: int) -> dict:
    """
    Compare computed weekly rankings to official API snapshots.
    Returns a dict of validation metrics per snapshot date.
    """
    from sqlalchemy import text
    import statistics

    with engine.connect() as conn:
        # Get all official snapshot dates for this category
        snap_dates = conn.execute(text("""
            SELECT DISTINCT retrieved_at FROM athlete_rankings
            WHERE ranking_cat_id = :cat_id
            ORDER BY retrieved_at
        """), {"cat_id": cat_id}).fetchall()

    results = {}
    for (snap_date,) in snap_dates:
        with engine.connect() as conn:
            official = conn.execute(text("""
                SELECT athlete_id, rank_position, total_points
                FROM athlete_rankings
                WHERE ranking_cat_id = :cat_id AND retrieved_at = :snap
                ORDER BY rank_position
            """), {"cat_id": cat_id, "snap": snap_date}).fetchall()

        # Find the closest computed Sunday to this official snapshot date
        with engine.connect() as conn:
            candidate_dates = conn.execute(text("""
                SELECT DISTINCT ranking_date
                FROM computed_weekly_rankings
                WHERE ranking_cat_id = :cat_id
                  AND ranking_date BETWEEN :d1 AND :d2
            """), {
                "cat_id": cat_id,
                "d1": snap_date - timedelta(days=7),
                "d2": snap_date + timedelta(days=7),
            }).fetchall()

        if not candidate_dates:
            continue

        closest_date = min(
            (r[0] for r in candidate_dates),
            key=lambda d: abs((d - snap_date).days)
        )

        with engine.connect() as conn:
            computed = conn.execute(text("""
                SELECT athlete_id, rank_position, total_points
                FROM computed_weekly_rankings
                WHERE ranking_cat_id = :cat_id
                  AND ranking_date = :closest
                ORDER BY rank_position
            """), {"cat_id": cat_id, "closest": closest_date}).fetchall()

        if not official or not computed:
            continue

        official_ranks = {r[0]: r[1] for r in official}
        computed_ranks = {r[0]: r[1] for r in computed}

        # Athletes in both
        common = set(official_ranks) & set(computed_ranks)
        if not common:
            continue

        # Top-N accuracy
        top10_official = {r[0] for r in official if r[1] <= 10}
        top10_computed = {r[0] for r in computed if r[1] <= 10}
        top10_overlap = len(top10_official & top10_computed)

        # Rank deltas for top 50 (using absolute rank positions)
        top50 = [a for a, r in official_ranks.items() if r <= 50 and a in computed_ranks]
        deltas = [abs(official_ranks[a] - computed_ranks[a]) for a in top50]
        mean_delta = statistics.mean(deltas) if deltas else None

        # Spearman: re-rank within shared athletes so scores are comparable
        if len(top50) >= 5:
            # Sort by official rank, get position in that ordering
            top50_sorted = sorted(top50, key=lambda a: official_ranks[a])
            off_positions = list(range(1, len(top50_sorted) + 1))
            # Assign computed position by sorting same athletes by computed rank
            computed_order = sorted(top50_sorted, key=lambda a: computed_ranks[a])
            comp_pos_map = {a: i+1 for i, a in enumerate(computed_order)}
            comp_positions = [comp_pos_map[a] for a in top50_sorted]
            spearman = _spearman(off_positions, comp_positions)
        else:
            spearman = None

        results[str(snap_date)] = {
            "common_athletes": len(common),
            "top10_overlap": top10_overlap,
            "top10_official": len(top10_official),
            "top50_mean_rank_delta": round(mean_delta, 2) if mean_delta else None,
            "top50_spearman": round(spearman, 4) if spearman else None,
        }

    return results


def _spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    d2 = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
    return 1.0 - (6 * d2) / (n * (n ** 2 - 1))
