from __future__ import annotations

import math
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tri_analysis.database import get_engine
except ImportError:  # pragma: no cover
    from database import get_engine  # type: ignore


OUTPUT_DIR = Path("outputs")
GENERIC_EVENT_TOKENS = {
    "world", "triathlon", "championship", "championships", "series", "mixed", "relay",
    "sprint", "event", "events", "games", "olympic", "qualification", "qualifier",
    "cup", "cups", "europe", "european", "americas", "american", "asian", "africa",
    "african", "oceania", "continental", "club", "clubs", "level", "premium", "water",
    "grand", "final", "finals", "supertri",
}


def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def _safe_write_text(content: str, path: Path) -> Path:
    try:
        path.write_text(content, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        fallback.write_text(content, encoding="utf-8")
        return fallback


def classify_event_class(event_name: str) -> str:
    name = (event_name or "").upper()
    if "OLYMPIC QUALIFICATION" in name:
        return "Olympic Qualifier"
    if "OLYMPIC GAMES" in name or "EUROPEAN GAMES" in name or "ASIAN GAMES" in name or "PAN-AMERICAN GAMES" in name:
        return "Major Games"
    if "MIXED RELAY CHAMPIONSHIPS" in name or "SPRINT AND RELAY CHAMPIONSHIPS" in name or "MIXED RELAY WORLD CHAMPIONSHIPS" in name:
        return "World Championship Level"
    if "CHAMPIONSHIP SERIES" in name or "WTCS" in name:
        return "WTCS"
    if "CLUB CHAMPIONSHIPS" in name:
        return "Club Championship"
    if "CHAMPIONSHIPS" in name:
        return "Continental Championship"
    if "CUP" in name:
        return "Cup"
    return "Other"


def _event_tokens(event_name: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", str(event_name).lower()):
        if token.isdigit() or len(token) <= 3 or token in GENERIC_EVENT_TOKENS:
            continue
        tokens.add(token)
    return tokens


def _query_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    relay_sql = text(
        """
        SELECT rr.event_id,
               rr.prog_id,
               e.event_name,
               e.event_date,
               a.country,
               a.gender,
               a.athlete_id,
               a.full_name,
               rr.finish_position
        FROM race_results rr
        JOIN events e
          ON rr.event_id = e.event_id
         AND rr.prog_id = e.prog_id
        JOIN athlete a
          ON rr.athlete_id = a.athlete_id
        WHERE e.prog_name = 'Mixed Relay'
          AND rr.athlete_id IS NOT NULL
          AND rr.finish_position IS NOT NULL
          AND e.event_name !~* '(Junior|Youth|Age-Group|Para|School|Student|University)'
        """
    )
    individual_sql = text(
        """
        SELECT rr.event_id,
               rr.prog_id,
               e.event_name,
               e.event_date,
               e.prog_name,
               a.country,
               a.gender,
               a.athlete_id,
               a.full_name,
               rr.finish_position,
               COUNT(*) OVER (PARTITION BY rr.event_id, rr.prog_id) AS field_size
        FROM race_results rr
        JOIN events e
          ON rr.event_id = e.event_id
         AND rr.prog_id = e.prog_id
        JOIN athlete a
          ON rr.athlete_id = a.athlete_id
        WHERE e.prog_name IN ('Elite Men', 'Elite Women')
          AND rr.athlete_id IS NOT NULL
          AND rr.finish_position IS NOT NULL
          AND e.event_name !~* '(Junior|Youth|Age-Group|Para|School|Student|University)'
        """
    )
    engine = get_engine()
    with engine.connect() as conn:
        relay = pd.read_sql(relay_sql, conn)
        individual = pd.read_sql(individual_sql, conn)
    return relay, individual


def _prepare_standard_teams(relay: pd.DataFrame) -> pd.DataFrame:
    relay = relay.copy()
    relay["event_date"] = pd.to_datetime(relay["event_date"])
    relay["finish_position_num"] = pd.to_numeric(relay["finish_position"], errors="coerce")
    relay = relay[relay["finish_position_num"].notna()].copy()
    team_counts = (
        relay.groupby(["event_id", "prog_id", "event_name", "event_date", "country", "gender"])["athlete_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    team_counts.columns.name = None
    if "female" not in team_counts.columns:
        team_counts["female"] = 0
    if "male" not in team_counts.columns:
        team_counts["male"] = 0
    standard = team_counts[(team_counts["female"] == 2) & (team_counts["male"] == 2)][["event_id", "prog_id", "country"]]
    relay = relay.merge(standard, on=["event_id", "prog_id", "country"], how="inner")
    relay["event_class"] = relay["event_name"].apply(classify_event_class)
    return relay


def _prepare_individual(individual: pd.DataFrame) -> pd.DataFrame:
    individual = individual.copy()
    individual["event_date"] = pd.to_datetime(individual["event_date"])
    individual["finish_position_num"] = pd.to_numeric(individual["finish_position"], errors="coerce")
    individual = individual[individual["finish_position_num"].notna() & (individual["field_size"] > 1)].copy()
    individual["score"] = 1 - (individual["finish_position_num"] - 1) / (individual["field_size"] - 1)
    return individual


def _build_same_event_ranks(individual: pd.DataFrame) -> pd.DataFrame:
    same_event = (
        individual.groupby(["event_id", "prog_id", "prog_name", "country", "gender", "athlete_id", "full_name"], as_index=False)
        .agg(same_event_individual_finish=("finish_position_num", "min"))
        .sort_values(["event_id", "prog_id", "country", "gender", "same_event_individual_finish", "athlete_id"])
        .reset_index(drop=True)
    )
    same_event["same_event_country_rank"] = same_event.groupby(["event_id", "country", "gender"]).cumcount() + 1
    same_event_pool = same_event.groupby(["event_id", "country", "gender"], as_index=False).agg(
        same_event_pool_size=("athlete_id", "nunique")
    )
    return same_event.merge(same_event_pool, on=["event_id", "country", "gender"], how="left")


def _build_same_weekend_event_map(relay: pd.DataFrame, individual: pd.DataFrame) -> pd.DataFrame:
    relay_events = relay[["event_id", "prog_id", "event_name", "event_date"]].drop_duplicates().copy()
    individual_events = individual[["event_id", "event_name", "event_date"]].drop_duplicates().copy()
    rows: list[dict] = []
    for relay_event in relay_events.itertuples(index=False):
        relay_tokens = _event_tokens(relay_event.event_name)
        if not relay_tokens:
            continue
        relay_date = pd.Timestamp(relay_event.event_date)
        candidates = individual_events[
            (individual_events["event_id"] != relay_event.event_id)
            & (individual_events["event_date"] >= relay_date - pd.Timedelta(days=1))
            & (individual_events["event_date"] <= relay_date + pd.Timedelta(days=1))
        ]
        best = None
        best_score = None
        for candidate in candidates.itertuples(index=False):
            candidate_tokens = _event_tokens(candidate.event_name)
            shared = relay_tokens & candidate_tokens
            if not shared:
                continue
            score = (len(shared), -abs((pd.Timestamp(candidate.event_date) - relay_date).days), -int(candidate.event_id))
            if best_score is None or score > best_score:
                best_score = score
                best = candidate
        if best is None:
            continue
        rows.append(
            {
                "event_id": int(relay_event.event_id),
                "prog_id": int(relay_event.prog_id),
                "weekend_match_event_id": int(best.event_id),
                "weekend_match_event_name": best.event_name,
                "weekend_match_event_date": pd.Timestamp(best.event_date),
                "weekend_match_shared_tokens": " | ".join(sorted(relay_tokens & _event_tokens(best.event_name))),
            }
        )
    return pd.DataFrame(rows)


def _build_same_weekend_ranks(individual: pd.DataFrame, weekend_map: pd.DataFrame) -> pd.DataFrame:
    if weekend_map.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "prog_id",
                "country",
                "gender",
                "athlete_id",
                "same_weekend_individual_finish",
                "same_weekend_country_rank",
                "same_weekend_pool_size",
                "same_weekend_event_name",
                "same_weekend_event_date",
                "same_weekend_shared_tokens",
            ]
        )
    candidate_rows = individual.merge(
        weekend_map,
        left_on="event_id",
        right_on="weekend_match_event_id",
        how="inner",
        suffixes=("_individual", "_relay"),
    )
    same_weekend = (
        candidate_rows.groupby(
            [
                "event_id_relay",
                "prog_id_relay",
                "country",
                "gender",
                "athlete_id",
                "full_name",
                "prog_id_individual",
                "prog_name",
                "weekend_match_event_name",
                "weekend_match_event_date",
                "weekend_match_shared_tokens",
            ],
            as_index=False,
        )
        .agg(same_weekend_individual_finish=("finish_position_num", "min"))
        .sort_values(["event_id_relay", "prog_id_relay", "country", "gender", "same_weekend_individual_finish", "athlete_id"])
        .reset_index(drop=True)
    )
    same_weekend["same_weekend_country_rank"] = same_weekend.groupby(["event_id_relay", "prog_id_relay", "country", "gender"]).cumcount() + 1
    pool = same_weekend.groupby(["event_id_relay", "prog_id_relay", "country", "gender"], as_index=False).agg(
        same_weekend_pool_size=("athlete_id", "nunique")
    )
    same_weekend = same_weekend.merge(pool, on=["event_id_relay", "prog_id_relay", "country", "gender"], how="left")
    same_weekend.rename(columns={"event_id_relay": "event_id", "prog_id_relay": "prog_id"}, inplace=True)
    return same_weekend


def _build_slot_report(
    relay: pd.DataFrame,
    individual: pd.DataFrame,
    same_event: pd.DataFrame,
    same_weekend_map: pd.DataFrame,
    same_weekend: pd.DataFrame,
) -> pd.DataFrame:
    slots = relay[
        [
            "event_id", "prog_id", "event_name", "event_class", "event_date",
            "country", "gender", "athlete_id", "full_name", "finish_position_num",
        ]
    ].drop_duplicates()
    slots = slots.merge(
        same_event[
            [
                "event_id", "prog_id", "prog_name", "country", "gender", "athlete_id",
                "same_event_individual_finish", "same_event_country_rank", "same_event_pool_size",
            ]
        ].rename(columns={"prog_id": "same_event_prog_id", "prog_name": "same_event_prog_name"}),
        on=["event_id", "country", "gender", "athlete_id"],
        how="left",
    )
    slots = slots.merge(same_weekend_map, on=["event_id", "prog_id"], how="left")
    slots = slots.merge(
        same_weekend[
            [
                "event_id", "prog_id", "country", "gender", "athlete_id",
                "same_weekend_individual_finish", "same_weekend_country_rank", "same_weekend_pool_size",
                "prog_id_individual", "prog_name",
                "weekend_match_event_name", "weekend_match_event_date", "weekend_match_shared_tokens",
            ]
        ].rename(
            columns={
                "prog_id_individual": "same_weekend_prog_id",
                "prog_name": "same_weekend_prog_name",
                "weekend_match_event_name": "same_weekend_event_name",
                "weekend_match_event_date": "same_weekend_event_date",
                "weekend_match_shared_tokens": "same_weekend_shared_tokens",
            }
        ),
        on=["event_id", "prog_id", "country", "gender", "athlete_id"],
        how="left",
    )
    slots["raced_same_event_individual"] = slots["same_event_country_rank"].notna()
    slots["raced_same_weekend_individual"] = slots["same_weekend_country_rank"].notna()

    prior_rows: list[dict] = []
    for slot in slots.itertuples(index=False):
        window_start = slot.event_date - pd.Timedelta(days=365)
        athlete_history = individual[
            (individual["athlete_id"] == slot.athlete_id)
            & (individual["event_date"] < slot.event_date)
            & (individual["event_date"] >= window_start)
        ]
        athlete_score = athlete_history["score"].mean() if not athlete_history.empty else math.nan
        athlete_starts = int(athlete_history["event_name"].nunique()) if not athlete_history.empty else 0

        pool = individual[
            (individual["country"] == slot.country)
            & (individual["gender"] == slot.gender)
            & (individual["event_date"] < slot.event_date)
            & (individual["event_date"] >= window_start)
        ]
        if pool.empty:
            prior_rows.append(
                {
                    "event_id": slot.event_id,
                    "prog_id": slot.prog_id,
                    "athlete_id": slot.athlete_id,
                    "prior_365_country_rank": math.nan,
                    "prior_365_pool_size": 0,
                    "prior_365_form_score": athlete_score,
                    "prior_365_starts": athlete_starts,
                }
            )
            continue

        pool_summary = (
            pool.groupby(["athlete_id", "full_name"], as_index=False)
            .agg(prior_365_form_score=("score", "mean"), prior_365_starts=("event_name", "nunique"))
            .sort_values(["prior_365_form_score", "prior_365_starts"], ascending=[False, False])
            .reset_index(drop=True)
        )
        pool_summary["prior_365_country_rank"] = pool_summary.index + 1
        rank_row = pool_summary[pool_summary["athlete_id"] == slot.athlete_id]
        prior_rows.append(
            {
                "event_id": slot.event_id,
                "prog_id": slot.prog_id,
                "athlete_id": slot.athlete_id,
                "prior_365_country_rank": float(rank_row["prior_365_country_rank"].iloc[0]) if not rank_row.empty else math.nan,
                "prior_365_pool_size": int(len(pool_summary)),
                "prior_365_form_score": athlete_score,
                "prior_365_starts": athlete_starts,
            }
        )

    slots = slots.merge(pd.DataFrame(prior_rows), on=["event_id", "prog_id", "athlete_id"], how="left")
    slots.rename(columns={"full_name": "relay_athlete", "finish_position_num": "team_finish"}, inplace=True)

    slots["individual_event_name"] = slots["event_name"]
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_event_name"] = slots["same_weekend_event_name"]
    slots["individual_event_date"] = slots["event_date"]
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_event_date"] = slots["same_weekend_event_date"]
    slots["individual_event_id"] = slots["event_id"]
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_event_id"] = slots["weekend_match_event_id"]
    slots["individual_prog_id"] = slots["same_event_prog_id"]
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_prog_id"] = slots["same_weekend_prog_id"]
    slots["individual_prog_name"] = slots["same_event_prog_name"]
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_prog_name"] = slots["same_weekend_prog_name"]
    slots["individual_country_rank"] = slots["same_event_country_rank"]
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_country_rank"] = slots["same_weekend_country_rank"]
    slots["individual_pool_size"] = slots["same_event_pool_size"]
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_pool_size"] = slots["same_weekend_pool_size"]
    slots["individual_shared_tokens"] = pd.NA
    slots.loc[~slots["raced_same_event_individual"] & slots["raced_same_weekend_individual"], "individual_shared_tokens"] = slots["same_weekend_shared_tokens"]
    slots["raced_individual_comparison"] = slots["individual_country_rank"].notna()

    slots["medal_team"] = slots["team_finish"] <= 3
    slots["individual_top2"] = slots["individual_country_rank"] <= 2
    slots["individual_top3"] = slots["individual_country_rank"] <= 3
    slots["prior_365_top2"] = slots["prior_365_country_rank"] <= 2
    slots["prior_365_top3"] = slots["prior_365_country_rank"] <= 3

    drop_columns = [
        "same_event_individual_finish",
        "same_event_country_rank",
        "same_event_pool_size",
        "same_event_prog_id",
        "same_event_prog_name",
        "same_weekend_individual_finish",
        "same_weekend_country_rank",
        "same_weekend_pool_size",
        "same_weekend_event_name",
        "same_weekend_event_date",
        "same_weekend_shared_tokens",
        "same_weekend_prog_id",
        "same_weekend_prog_name",
        "raced_same_event_individual",
        "raced_same_weekend_individual",
        "individual_shared_tokens",
        "comparison_event_name",
        "comparison_event_date",
        "comparison_country_rank",
        "comparison_pool_size",
        "comparison_shared_tokens",
        "same_event_top2",
        "same_event_top3",
        "same_weekend_top2",
        "same_weekend_top3",
        "comparison_top2",
        "comparison_top3",
    ]
    return slots.drop(columns=drop_columns, errors="ignore")


def _build_team_report(slots: pd.DataFrame) -> pd.DataFrame:
    team_summary = (
        slots.groupby(["event_id", "prog_id", "event_name", "event_class", "event_date", "country"], as_index=False)
        .agg(
            team_finish=("team_finish", "min"),
            relay_athletes=("relay_athlete", lambda s: " | ".join(sorted(set(s)))),
            women=("gender", lambda s: int((pd.Series(s) == "female").sum())),
            men=("gender", lambda s: int((pd.Series(s) == "male").sum())),
            matched_individual_slots=("raced_individual_comparison", "sum"),
            individual_top2_slots=("individual_top2", "sum"),
            individual_top3_slots=("individual_top3", "sum"),
            avg_individual_country_rank=("individual_country_rank", "mean"),
            matched_prior_365_slots=("prior_365_country_rank", lambda s: s.notna().sum()),
            prior_365_top2_slots=("prior_365_top2", "sum"),
            prior_365_top3_slots=("prior_365_top3", "sum"),
            avg_prior_365_country_rank=("prior_365_country_rank", "mean"),
            avg_prior_365_form_score=("prior_365_form_score", "mean"),
        )
    )
    team_summary["medal_team"] = team_summary["team_finish"] <= 3
    team_summary["all_individual_top2"] = team_summary["individual_top2_slots"] == 4
    team_summary["all_individual_top3"] = team_summary["individual_top3_slots"] == 4
    team_summary["all_prior_365_top2"] = team_summary["prior_365_top2_slots"] == 4
    team_summary["all_prior_365_top3"] = team_summary["prior_365_top3_slots"] == 4
    return team_summary.sort_values(["event_date", "team_finish", "country"], ascending=[False, True, True])


def _summarize_event_classes(slots: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    summary = teams.groupby("event_class", as_index=False).agg(
        teams=("country", "count"),
        medal_teams=("medal_team", "sum"),
        avg_finish=("team_finish", "mean"),
        fully_individual_matched_teams=("matched_individual_slots", lambda s: int((s == 4).sum())),
        all_individual_top2_rate=("all_individual_top2", "mean"),
        avg_individual_country_rank=("avg_individual_country_rank", "mean"),
        avg_prior_365_country_rank=("avg_prior_365_country_rank", "mean"),
    )
    slot_summary = slots.groupby("event_class", as_index=False).agg(
        slot_count=("relay_athlete", "count"),
        individual_slot_top2_rate=("individual_top2", "mean"),
        prior_365_slot_top2_rate=("prior_365_top2", "mean"),
    )
    return summary.merge(slot_summary, on="event_class", how="left").sort_values("teams", ascending=False)


def _summarize_countries(teams: pd.DataFrame) -> pd.DataFrame:
    return (
        teams.groupby("country", as_index=False)
        .agg(
            teams=("team_finish", "count"),
            medals=("medal_team", "sum"),
            avg_finish=("team_finish", "mean"),
            avg_individual_country_rank=("avg_individual_country_rank", "mean"),
            avg_prior_365_country_rank=("avg_prior_365_country_rank", "mean"),
            all_individual_top2_rate=("all_individual_top2", "mean"),
            all_prior_365_top2_rate=("all_prior_365_top2", "mean"),
        )
        .sort_values(["medals", "avg_finish", "teams"], ascending=[False, True, False])
    )


def _summarize_nonmatched(slots: pd.DataFrame) -> pd.DataFrame:
    return (
        slots[~slots["raced_individual_comparison"]]
        .groupby(["relay_athlete", "country", "gender"], as_index=False)
        .agg(
            relay_slots=("event_id", "count"),
            events=("event_name", "nunique"),
            first_event=("event_date", "min"),
            last_event=("event_date", "max"),
        )
        .sort_values(["relay_slots", "events", "last_event"], ascending=[False, False, False])
    )


def _format_report_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> pd.DataFrame:
    sample = df.loc[:, columns].head(max_rows).copy()
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].dt.strftime("%Y-%m-%d")
    for col in sample.columns:
        if sample[col].dtype == bool:
            sample[col] = sample[col].map({True: "Yes", False: "No"})
    for col in [col for col in sample.columns if col.endswith("_rate")]:
        sample[col] = sample[col].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    for col in [col for col in sample.columns if "avg_" in col and "rank" in col]:
        sample[col] = sample[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    if "avg_finish" in sample.columns:
        sample["avg_finish"] = sample["avg_finish"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    if "team_finish" in sample.columns:
        sample["team_finish"] = sample["team_finish"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    return sample


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    sample = _format_report_table(df, columns, max_rows=max_rows)
    if sample.empty:
        return "_No rows._"
    header = "| " + " | ".join(sample.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(sample.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in sample.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _write_report(event_class_summary: pd.DataFrame, country_summary: pd.DataFrame, team_report: pd.DataFrame, nonmatched: pd.DataFrame) -> Path:
    headline = (
        "Most adult mixed relay teams are built from strong overall triathletes, using a combined individual-race lens that first checks the same event and then falls back to a related same-weekend race when needed."
    )
    report_path = OUTPUT_DIR / "mixed_relay_selection_draft_report.md"
    medal_examples = team_report[(team_report["medal_team"]) & (~team_report["all_individual_top2"])][
        [
            "event_date", "event_class", "event_name", "country", "team_finish",
            "individual_top2_slots", "prior_365_top2_slots", "avg_individual_country_rank",
            "avg_prior_365_country_rank",
        ]
    ].sort_values(["avg_individual_country_rank", "team_finish"])
    glossary = [
        ("individual_*", "Combined individual-race comparison. It uses the same event first and falls back to a related same-weekend elite race if no exact individual result exists."),
        ("prior_365_*", "Previous-365-day form ranking within country/gender, based on average normalized finish score where 1.0 is best in field and 0.0 is last in field."),
        ("individual_top2_slots", "How many of the 4 relay athletes were top-2 within their country/gender under the combined individual-race comparison."),
        ("prior_365_top2_slots", "How many of the 4 relay athletes were top-2 within their country/gender on prior-365-day form, even if they were not in that event's individual field."),
    ]
    slides = [
        "Slide 1: Headline and method. Combined individual-race check first, prior-365 country form second.",
        "Slide 2: Event-class summary using the combined individual-race columns as the primary measure.",
        "Slide 3: Country scorecard using average individual-race rank and all-individual-top2 rate.",
        "Slide 4: Exceptions and roster strategy, including relay-only or schedule-managed athletes.",
    ]
    report = "\n".join(
        [
            "# Mixed Relay Selection Draft Report",
            "",
            "## Headline",
            headline,
            "",
            "## How To Read The Main Fields",
            "| Field | Plain-English Meaning |",
            "| --- | --- |",
            *[f"| {field} | {meaning} |" for field, meaning in glossary],
            "",
            "## Event-Class Summary",
            _markdown_table(
                event_class_summary,
                [
                    "event_class", "teams", "medal_teams", "fully_individual_matched_teams",
                    "individual_slot_top2_rate", "all_individual_top2_rate",
                    "prior_365_slot_top2_rate", "avg_individual_country_rank", "avg_prior_365_country_rank",
                ],
            ),
            "",
            "## Country Scorecard",
            _markdown_table(
                country_summary,
                [
                    "country", "teams", "medals", "avg_finish",
                    "avg_individual_country_rank", "avg_prior_365_country_rank",
                    "all_individual_top2_rate", "all_prior_365_top2_rate",
                ],
            ),
            "",
            "## Medal Teams That Weren't Full Top-2 Rosters",
            _markdown_table(
                medal_examples,
                [
                    "event_date", "event_class", "event_name", "country", "team_finish",
                    "individual_top2_slots", "prior_365_top2_slots",
                    "avg_individual_country_rank", "avg_prior_365_country_rank",
                ],
            ),
            "",
            "## Recurring Relay-Only Or Schedule-Managed Athletes",
            _markdown_table(
                nonmatched,
                ["relay_athlete", "country", "gender", "relay_slots", "events", "last_event"],
            ),
            "",
            "## Draft Slide Structure",
            *[f"- {line}" for line in slides],
        ]
    )
    return _safe_write_text(report, report_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    relay, individual = _query_frames()
    relay = _prepare_standard_teams(relay)
    individual = _prepare_individual(individual)
    same_event = _build_same_event_ranks(individual)
    same_weekend_map = _build_same_weekend_event_map(relay, individual)
    same_weekend = _build_same_weekend_ranks(individual, same_weekend_map)
    slot_report = _build_slot_report(relay, individual, same_event, same_weekend_map, same_weekend)
    team_report = _build_team_report(slot_report)
    event_class_summary = _summarize_event_classes(slot_report, team_report)
    country_summary = _summarize_countries(team_report)
    nonmatched = _summarize_nonmatched(slot_report)

    written_paths = [
        _safe_write_csv(slot_report, OUTPUT_DIR / "mixed_relay_selection_slot_report.csv"),
        _safe_write_csv(team_report, OUTPUT_DIR / "mixed_relay_selection_team_report.csv"),
        _safe_write_csv(event_class_summary, OUTPUT_DIR / "mixed_relay_selection_event_class_summary.csv"),
        _safe_write_csv(country_summary, OUTPUT_DIR / "mixed_relay_selection_country_summary.csv"),
        _safe_write_csv(nonmatched, OUTPUT_DIR / "mixed_relay_selection_nonmatched_athletes.csv"),
        _write_report(event_class_summary, country_summary, team_report, nonmatched),
    ]
    for path in written_paths:
        print(f"Wrote {path.as_posix()}")


if __name__ == "__main__":
    main()