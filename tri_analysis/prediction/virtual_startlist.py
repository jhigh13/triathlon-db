"""
virtual_startlist.py -- Build a *virtual* Olympic start list from world rankings.

The Olympic triathlon field is not an open start list: each NOC (country) may
enter a limited number of athletes per gender, and the field is capped. This
module reconstructs "who would be on the LA start line if the race were today"
by walking down the current world ranking and applying the qualification quota:

    * Field capped at `field_size` athletes (default 60).
    * At most `max_per_country` athletes per country (default 3).
    * A country's 3rd athlete is only admitted if that athlete is ranked inside
      the top `top_n_for_third` of the world ranking (default 30). Otherwise the
      slot rolls down to the next eligible athlete from another country.
    * Retired / moved-on athletes can be excluded by name or id.

This mirrors the World Triathlon Olympic quota logic (a NOC needs 3 athletes in
the top 30 of the qualification ranking to send 3).

Returns the selected field plus the athletes who were bumped (for transparency).
"""
from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Ranking category ids in athlete_rankings, by (ranking source, gender).
RANKING_CAT_IDS = {
    "world":   {"men": 13, "women": 14},   # World Rankings
    "olympic": {"men": 11, "women": 12},   # Olympic Qualification Ranking
    "wtcs":    {"men": 15, "women": 16},   # World Triathlon Series Ranking
}

# Default file of athletes to treat as retired / no longer contending.
# One name (or "id:12345") per line; blank lines and #-comments ignored.
_DEFAULT_EXCLUSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "olympic_exclusions.txt",
)


def _norm(name: str) -> str:
    """Casefold + collapse whitespace for tolerant name matching."""
    return " ".join(str(name).split()).casefold()


def load_exclusions(path: str | None = None) -> tuple[set[str], set[int]]:
    """Load excluded athlete names and ids from a text file (if it exists)."""
    path = path or _DEFAULT_EXCLUSION_FILE
    names: set[str] = set()
    ids: set[int] = set()
    if not os.path.exists(path):
        return names, ids
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("id:"):
                try:
                    ids.add(int(line[3:].strip()))
                except ValueError:
                    logger.warning("Bad exclusion id line: %r", line)
            else:
                names.add(_norm(line))
    logger.info("Loaded exclusions: %d names, %d ids from %s", len(names), len(ids), path)
    return names, ids


# athlete.country placeholder values that are not real nations -> fall back to entry NOC.
_PLACEHOLDER_COUNTRIES = {"world triathlon", "unknown", "", "none"}


def fetch_world_ranking(engine: Engine, cat_id: int) -> pd.DataFrame:
    """Latest snapshot of one ranking category, with a robust country resolution.

    Country prefers ``athlete.country``; when that is missing or a placeholder
    (e.g. 'World Triathlon'), it falls back to the athlete's most recent
    ``program_entries.athlete_country_name``.
    """
    q = text("""
        WITH latest AS (
            SELECT MAX(retrieved_at) AS mx
            FROM athlete_rankings WHERE ranking_cat_id = :cid
        )
        SELECT
            ar.rank_position,
            ar.athlete_id,
            ar.athlete_name,
            ar.total_points,
            a.country      AS ath_country,
            a.gender,
            pe.athlete_country_name AS entry_country
        FROM athlete_rankings ar
        LEFT JOIN athlete a ON a.athlete_id = ar.athlete_id
        LEFT JOIN LATERAL (
            SELECT pe2.athlete_country_name
            FROM program_entries pe2
            WHERE pe2.athlete_id = ar.athlete_id
              AND pe2.athlete_country_name IS NOT NULL
            ORDER BY pe2.event_id DESC
            LIMIT 1
        ) pe ON TRUE
        WHERE ar.ranking_cat_id = :cid
          AND ar.retrieved_at = (SELECT mx FROM latest)
          AND ar.athlete_id IS NOT NULL
        ORDER BY ar.rank_position
    """)
    df = pd.read_sql(q, engine, params={"cid": cat_id})

    def _resolve(row) -> str:
        ath = (row["ath_country"] or "").strip()
        if ath.lower() not in _PLACEHOLDER_COUNTRIES:
            return ath
        entry = (row["entry_country"] or "").strip()
        return entry if entry else (ath or "Unknown")

    df["country"] = df.apply(_resolve, axis=1)
    return df.drop(columns=["ath_country", "entry_country"])


def fetch_recent_elite_counts(
    engine: Engine,
    athlete_ids: list[int],
    as_of: str,
    window_months: int = 36,
) -> pd.DataFrame:
    """Per-athlete recent elite race counts used to gate cold-start athletes.

    Returns columns: athlete_id, n_std_sprint (all elite standard/sprint finishes)
    and n_top_races (those at WTCS / World Cup / Olympic level = tier 1-2). The
    model over-rates athletes who only race weaker fields, so n_top_races is the
    reliable "genuine Olympic-level record" signal.
    """
    if not athlete_ids:
        return pd.DataFrame(columns=["athlete_id", "n_std_sprint", "n_top_races"])
    q = text("""
        SELECT rr.athlete_id,
               COUNT(*) FILTER (
                   WHERE e.prog_distance_category IN ('standard', 'sprint')
               ) AS n_std_sprint,
               COUNT(*) FILTER (
                   WHERE e.prog_distance_category IN ('standard', 'sprint')
                     AND e.event_name ~* 'World Triathlon Championship|World Cup|World Triathlon Cup|Olympic'
               ) AS n_top_races
        FROM race_results rr
        JOIN events e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        WHERE rr.athlete_id = ANY(:ids)
          AND rr.finish_status = 'FINISH'
          AND e.prog_name ~* '(Elite Men|Elite Women)'
          AND e.event_date >= (CAST(:as_of AS date) - make_interval(months => :months))
          AND e.event_date <= CAST(:as_of AS date)
        GROUP BY rr.athlete_id
    """)
    return pd.read_sql(q, engine, params={"ids": athlete_ids, "as_of": as_of, "months": window_months})


def build_virtual_olympic_startlist(
    engine: Engine,
    gender: str,
    field_size: int = 60,
    max_per_country: int = 3,
    top_n_for_third: int = 30,
    ranking: str = "world",
    exclude_names: set[str] | None = None,
    exclude_ids: set[int] | None = None,
    exclusion_file: str | None = None,
    min_top_races: int = 0,
    history_window_months: int = 36,
    as_of: str | None = None,
) -> dict:
    """
    Construct the virtual Olympic field.

    Returns a dict:
        selected : DataFrame (ordered) of the qualified field
        bumped   : DataFrame of athletes skipped by the quota rules
        meta     : dict of the parameters used
    """
    gender = gender.lower()
    if gender not in ("men", "women"):
        raise ValueError("gender must be 'men' or 'women'")
    if ranking not in RANKING_CAT_IDS:
        raise ValueError(f"ranking must be one of {list(RANKING_CAT_IDS)}")

    cat_id = RANKING_CAT_IDS[ranking][gender]
    ranking_df = fetch_world_ranking(engine, cat_id)
    if ranking_df.empty:
        raise ValueError(f"No ranking rows for cat_id={cat_id} ({ranking} {gender})")

    # Merge file-based + caller-provided exclusions
    file_names, file_ids = load_exclusions(exclusion_file)
    ex_names = set(file_names) | {_norm(n) for n in (exclude_names or set())}
    ex_ids = set(file_ids) | set(exclude_ids or set())

    excluded_mask = ranking_df["athlete_id"].isin(ex_ids) | ranking_df["athlete_name"].map(_norm).isin(ex_names)
    excluded_df = ranking_df[excluded_mask].copy()
    cleaned = ranking_df[~excluded_mask].reset_index(drop=True)

    # Attach recent elite race counts (for transparency + optional cold-start gate).
    as_of = as_of or date.today().isoformat()
    counts = fetch_recent_elite_counts(
        engine, cleaned["athlete_id"].astype(int).tolist(), as_of, history_window_months
    )
    cleaned = cleaned.merge(counts, on="athlete_id", how="left")
    cleaned[["n_std_sprint", "n_top_races"]] = cleaned[["n_std_sprint", "n_top_races"]].fillna(0).astype(int)

    # Cold-start gate: drop athletes without enough recent top-tier (WTCS/WC/Olympic)
    # starts — the model cannot reliably rate them (weak-field inflation).
    gated_df = pd.DataFrame()
    if min_top_races > 0:
        gate_mask = cleaned["n_top_races"] < min_top_races
        gated_df = cleaned[gate_mask].copy()
        cleaned = cleaned[~gate_mask].reset_index(drop=True)

    # Walk the cleaned ranking; apply per-country quota with the top-30 third rule.
    selected_rows, bumped_rows = [], []
    country_count: dict[str, int] = {}
    for _, row in cleaned.iterrows():
        c = row["country"]
        cnt = country_count.get(c, 0)
        official_rank = int(row["rank_position"])

        if cnt >= max_per_country:
            bumped_rows.append({**row.to_dict(), "reason": f"{c} already has {max_per_country}"})
            continue
        if cnt == (max_per_country - 1) and official_rank > top_n_for_third:
            # Would be the 3rd (or max-th) but is outside the top-N cutoff.
            bumped_rows.append({
                **row.to_dict(),
                "reason": f"{c} #{cnt + 1}: rank {official_rank} outside top {top_n_for_third}",
            })
            continue

        note = ""
        if cnt == max_per_country - 1:
            note = f"{c} #{cnt + 1} (in top {top_n_for_third})"
        selected_rows.append({**row.to_dict(), "note": note})
        country_count[c] = cnt + 1
        if len(selected_rows) >= field_size:
            break

    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected.insert(0, "field_pos", range(1, len(selected) + 1))
    bumped = pd.DataFrame(bumped_rows)

    meta = {
        "gender": gender,
        "ranking": ranking,
        "ranking_cat_id": cat_id,
        "field_size": field_size,
        "max_per_country": max_per_country,
        "top_n_for_third": top_n_for_third,
        "min_top_races": min_top_races,
        "history_window_months": history_window_months,
        "as_of": as_of,
        "n_excluded": int(excluded_mask.sum()),
        "excluded_names": sorted(excluded_df["athlete_name"].tolist()),
        "n_gated": int(len(gated_df)),
        "n_selected": len(selected),
    }
    logger.info(
        "Virtual %s field (%s ranking): %d selected, %d bumped, %d excluded, %d gated (min_top_races=%d)",
        gender, ranking, len(selected), len(bumped), int(excluded_mask.sum()), len(gated_df), min_top_races,
    )
    return {"selected": selected, "bumped": bumped, "gated": gated_df, "meta": meta}


def format_country_summary(selected: pd.DataFrame) -> pd.DataFrame:
    """Athletes-per-country in the selected field, sorted desc."""
    if selected.empty:
        return pd.DataFrame(columns=["country", "athletes"])
    return (selected.groupby("country").size()
            .reset_index(name="athletes")
            .sort_values(["athletes", "country"], ascending=[False, True])
            .reset_index(drop=True))
