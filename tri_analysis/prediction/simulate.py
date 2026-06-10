"""
Monte Carlo simulation for probability estimates.

Models the causal chain of draft-legal triathlon:
    Swim → T1 → Pack Formation (swim+T1) → Bike (with pack drafting) → T2 → Run → Total

Key design principles:
- Swim determines pack membership via chain-rule algorithm (>2s consecutive gap = new pack)
- Multiple packs form naturally: front pack, chase pack, third pack, solo riders
- Pack drafting benefit scales with pack_size (larger pack = more aero benefit)
- Front pack gets full draft benefit, later packs get reduced benefit, solo riders penalized
- Pack effects are learned from historical data
- Run is individual effort (no pack dynamics)
- Shared form factor creates realistic correlation across splits
"""

from __future__ import annotations
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from .utils_time import parse_time_to_seconds

logger = logging.getLogger(__name__)

# Default per-split uncertainty (seconds, std dev) — calibrated for standard distance
DEFAULT_SIGMA_SWIM = 15.0
DEFAULT_SIGMA_T1 = 5.0
DEFAULT_SIGMA_BIKE = 45.0
DEFAULT_SIGMA_T2 = 4.0
DEFAULT_SIGMA_RUN = 30.0
DEFAULT_SIGMA_TOTAL = 90.0  # Fallback when splits unavailable

# Distance-specific sigma multipliers relative to standard distance.
# Sprint races are ~half the duration, so absolute variance is smaller.
# Super-sprint even shorter, middle/long distances longer.
DISTANCE_SIGMA_MULTIPLIER = {
    "super-sprint": 0.45,
    "super_sprint": 0.45,
    "sprint": 0.65,
    "standard": 1.0,
    "olympic": 1.0,
    "middle": 1.4,
    "middle_distance": 1.4,
    "long": 1.8,
    "long_distance": 1.8,
}

# Fraction of per-split variance attributable to shared "form factor"
# (good day / bad day correlation across splits)
DEFAULT_FORM_SHARE = 0.1


# ── Pack Effect Model ────────────────────────────────────────────────

# Distance-specific gap thresholds for pack formation.
# Sprint: tighter swim fields, T1 noise can shuffle order → use wider gap.
# Standard: typical WTCS distance, 3s works well.
# Middle/long: non-drafting, pack effects should not apply.
DISTANCE_PACK_GAP_SEC = {
    "super-sprint": 5.0,
    "super_sprint": 5.0,
    "sprint": 5.0,
    "standard": 3.0,
    "olympic": 3.0,
    "middle": 0.0,       # Non-drafting
    "middle_distance": 0.0,
    "long": 0.0,         # Non-drafting
    "long_distance": 0.0,
}

# Distances where drafting is legal and pack effects should apply
DRAFT_LEGAL_DISTANCES = {"super-sprint", "super_sprint", "sprint", "standard", "olympic"}


@dataclass
class PackEffectParams:
    """
    Learned parameters for the multi-pack bike-effect model.

    Pack formation uses the chain-rule algorithm: consecutive gaps > max_gap_sec
    create a new pack. Each pack gets a bike time adjustment based on:
    - Pack order (pack 0 = front, pack 1 = first chase, etc.)
    - Pack size (larger packs provide more drafting benefit)
    - Solo riders (pack_size=1) get a penalty

    Attributes:
        front_pack_bonus_sec: Bike time adjustment for front-pack athletes (negative = faster)
        chase_penalty_sec: Bike time adjustment for solo/late athletes (positive = slower)
        max_gap_sec: Consecutive gap threshold for pack splitting (default 2s)
        min_pack_size_for_draft: Minimum pack size that provides drafting benefit
        draft_size_scale: How much pack size affects drafting (larger pack = more benefit)
        n_observations: Number of athlete-race rows used to learn these parameters
        n_races: Number of distinct races in the training data
        distance_category: Which distance these params were learned from (or None for all)
    """
    front_pack_bonus_sec: float = -25.0
    chase_penalty_sec: float = 15.0
    max_gap_sec: float = 3.0
    min_pack_size_for_draft: int = 3
    draft_size_scale: float = 0.15  # Each additional rider adds ~15% more draft benefit
    n_observations: int = 0
    n_races: int = 0
    distance_category: str | None = None


@dataclass
class MergeParams:
    """
    Learned parameters for dynamic pack merging during the bike leg.

    Models the probability that a small group (1-2 riders) ahead of a larger
    chase pack will be absorbed during the bike. Merge probability is:

        P(merge) = sigmoid(beta_0 + beta_gap * gap + beta_chase_size * chase_size - breakaway_bias)

    Where gap is seconds between chase front and ahead group's tail, and
    chase_size is the number of riders in the chase pack.

    Attributes:
        beta_0: Intercept (base log-odds of merging at zero gap)
        beta_gap: Coefficient on gap_sec (negative: larger gap = less merge)
        beta_chase_size: Coefficient on chase_pack_size (positive: bigger = more merge)
        max_merge_gap_sec: Hard cutoff beyond which merge never happens
        min_chase_size_for_merge: Minimum chase pack size to trigger merge logic
        n_observations: Number of merge-candidate situations in training data
        n_races: Number of distinct races in training data
    """
    beta_0: float = 2.5
    beta_gap: float = -0.35
    beta_chase_size: float = 0.25
    max_merge_gap_sec: float = 15.0
    min_chase_size_for_merge: int = 3
    n_observations: int = 0
    n_races: int = 0


def assign_packs_chain(
    sorted_times: np.ndarray, max_gap_sec: float = 3.0
) -> np.ndarray:
    """
    Assign pack IDs using the chain-rule algorithm.

    A new pack starts when the gap between consecutive athletes (sorted by
    time) exceeds max_gap_sec.

    Args:
        sorted_times: Array of times sorted ascending (swim exit times)
        max_gap_sec: Maximum gap between consecutive athletes in same pack

    Returns:
        Array of pack IDs (0 = front pack, 1 = first chase, etc.)
        aligned with the SORTED order (not original athlete order).
    """
    n = len(sorted_times)
    if n == 0:
        return np.array([], dtype=np.int32)

    pack_ids = np.zeros(n, dtype=np.int32)
    current_pack = 0

    for i in range(1, n):
        if (sorted_times[i] - sorted_times[i - 1]) > max_gap_sec:
            current_pack += 1
        pack_ids[i] = current_pack

    return pack_ids


def compute_pack_sizes(pack_ids: np.ndarray) -> np.ndarray:
    """Compute the size of each athlete's pack from pack ID assignments."""
    if len(pack_ids) == 0:
        return np.array([], dtype=np.int32)

    # Count athletes per pack
    max_pack = pack_ids.max() + 1
    counts = np.bincount(pack_ids, minlength=max_pack)

    # Map each athlete's pack_id to its size
    return counts[pack_ids]


def multi_pack_bike_effect(
    pack_ids: np.ndarray,
    pack_sizes: np.ndarray,
    params: PackEffectParams,
) -> np.ndarray:
    """
    Compute bike time adjustment based on multi-pack membership.

    Pack effects model:
    - Pack 0 (front): Full drafting benefit, scaled by pack size
    - Pack 1+ (chase): Reduced benefit that decays with pack order
    - Solo riders (pack_size < min_pack_size): Penalty (no drafting partner)

    The pack-size scaling models that larger groups create more efficient
    aerodynamic drafting (peloton effect).

    Args:
        pack_ids: Pack ID per athlete (0 = front pack)
        pack_sizes: Pack size per athlete
        params: PackEffectParams with learned values

    Returns:
        Array of bike time adjustments in seconds.
        Negative = faster (drafting), positive = slower (solo).
    """
    n = len(pack_ids)
    effects = np.zeros(n, dtype=np.float64)

    bonus = params.front_pack_bonus_sec   # e.g., -25s
    penalty = params.chase_penalty_sec     # e.g., +15s
    min_size = params.min_pack_size_for_draft
    size_scale = params.draft_size_scale

    for i in range(n):
        pid = pack_ids[i]
        psize = pack_sizes[i]

        if psize < min_size:
            # Solo or tiny group: no drafting benefit, full penalty
            effects[i] = penalty
        elif pid == 0:
            # Front pack: full base bonus, augmented by pack size
            # Larger front pack = slightly more drafting benefit
            size_factor = 1.0 + size_scale * max(0, psize - min_size)
            effects[i] = bonus * min(size_factor, 1.5)  # Cap at 1.5x
        else:
            # Chase pack: interpolate between bonus and penalty based on pack order
            # Pack 1 gets ~70% of bonus, pack 2 gets ~40%, pack 3+ gets penalty
            decay = min(1.0, pid * 0.35)
            base_effect = bonus + decay * (penalty - bonus)
            # Still scale by pack size (large chase pack drafts well)
            size_factor = 1.0 + size_scale * max(0, psize - min_size)
            effects[i] = base_effect * min(size_factor, 1.3)

    return effects


def learn_pack_effects_from_data(
    pack_effect_df: pd.DataFrame,
    front_pack_threshold: float = 5.0,
    chase_threshold: float = 15.0,
) -> PackEffectParams:
    """
    Learn pack effect parameters from historical swim-pack + bike-time data.

    For each race, identifies the front swim pack's median bike time as the
    baseline. Then computes each athlete's bike time deviation from this
    baseline, grouped by swim gap to determine the drafting benefit curve.

    Args:
        pack_effect_df: DataFrame from fetch_pack_effect_data with columns:
            event_id, prog_id, athlete_id, swim_pack_id, swim_gap_to_leader,
            swimtime, biketime, prog_distance_category
        front_pack_threshold: Swim gap (sec) defining front pack membership
        chase_threshold: Swim gap (sec) beyond which athletes are solo/chase

    Returns:
        PackEffectParams with empirically learned values
    """
    df = pack_effect_df.copy()

    # Parse times to seconds
    for col in ["swimtime", "biketime", "runtime", "total_time"]:
        if col in df.columns:
            df[col + "_sec"] = df[col].apply(parse_time_to_seconds)

    # Filter to valid bike times
    df = df.dropna(subset=["biketime_sec"])
    df = df[df["biketime_sec"] > 0]

    if len(df) < 50:
        logger.warning(
            f"Insufficient data for pack effect learning ({len(df)} rows), "
            "using defaults"
        )
        return PackEffectParams()

    # For each race, compute OVERALL median bike time as the neutral baseline.
    # Front-pack athletes should be faster than this (negative delta = bonus).
    # Chase athletes should be slower (positive delta = penalty).
    race_median_bike = (
        df.groupby(["event_id", "prog_id"])["biketime_sec"]
        .median()
        .reset_index()
        .rename(columns={"biketime_sec": "race_median_bike"})
    )

    # Only keep races with enough athletes for meaningful stats
    race_counts = df.groupby(["event_id", "prog_id"]).size()
    valid_races = race_counts[race_counts >= 10].index
    race_median_bike = race_median_bike.set_index(["event_id", "prog_id"])
    race_median_bike = race_median_bike.loc[race_median_bike.index.isin(valid_races)].reset_index()

    if race_median_bike.empty:
        logger.warning("No races with enough athletes, using defaults")
        return PackEffectParams()

    # Merge back and compute bike delta relative to race median
    df = df.merge(race_median_bike, on=["event_id", "prog_id"], how="inner")
    df["bike_delta"] = df["biketime_sec"] - df["race_median_bike"]

    # Compute parameters by swim-gap group
    front_mask = df["swim_gap_to_leader"] <= front_pack_threshold
    chase_mask = df["swim_gap_to_leader"] > chase_threshold

    front_bonus = df.loc[front_mask, "bike_delta"].median()
    chase_penalty = df.loc[chase_mask, "bike_delta"].median()

    n_races = df[["event_id", "prog_id"]].drop_duplicates().shape[0]

    # Log detailed breakdown by gap bin for transparency
    bins = [0, 5, 10, 15, 20, 30, 60, 9999]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-60", "60+"]
    df["gap_bin"] = pd.cut(df["swim_gap_to_leader"], bins=bins, labels=labels, right=True)
    bin_stats = df.groupby("gap_bin", observed=True)["bike_delta"].agg(["median", "count"])
    logger.info(f"Pack effect by swim gap bin:\n{bin_stats.to_string()}")

    logger.info(
        f"Learned pack effects from {len(df)} observations across {n_races} races: "
        f"front_bonus={front_bonus:.1f}s, chase_penalty={chase_penalty:.1f}s"
    )

    return PackEffectParams(
        front_pack_bonus_sec=float(front_bonus),
        chase_penalty_sec=float(chase_penalty),
        n_observations=len(df),
        n_races=n_races,
    )


def pack_params_to_dict(params: PackEffectParams) -> dict:
    """Serialize PackEffectParams to a plain dict for storage in ModelBundle metadata."""
    return {
        "front_pack_bonus_sec": params.front_pack_bonus_sec,
        "chase_penalty_sec": params.chase_penalty_sec,
        "max_gap_sec": params.max_gap_sec,
        "min_pack_size_for_draft": params.min_pack_size_for_draft,
        "draft_size_scale": params.draft_size_scale,
        "n_observations": params.n_observations,
        "n_races": params.n_races,
        "distance_category": params.distance_category,
    }


def pack_params_from_dict(d: dict) -> PackEffectParams:
    """Reconstruct PackEffectParams from a dict (e.g., from bundle metadata)."""
    return PackEffectParams(**{k: v for k, v in d.items() if k in PackEffectParams.__dataclass_fields__})


def merge_params_to_dict(params: MergeParams) -> dict:
    """Serialize MergeParams to a plain dict for storage in ModelBundle metadata."""
    return {
        "beta_0": params.beta_0,
        "beta_gap": params.beta_gap,
        "beta_chase_size": params.beta_chase_size,
        "max_merge_gap_sec": params.max_merge_gap_sec,
        "min_chase_size_for_merge": params.min_chase_size_for_merge,
        "n_observations": params.n_observations,
        "n_races": params.n_races,
    }


def merge_params_from_dict(d: dict) -> MergeParams:
    """Reconstruct MergeParams from a dict (e.g., from bundle metadata)."""
    return MergeParams(**{k: v for k, v in d.items() if k in MergeParams.__dataclass_fields__})


def get_distance_pack_params(
    bundle_metadata: dict,
    distance_category: str | None,
    event_tier: int | None = None,
) -> PackEffectParams | None:
    """
    Look up pack params from bundle metadata.

    Lookup order: tier-specific → distance-specific → overall.
    Returns None if the distance is non-drafting (middle/long).

    Args:
        bundle_metadata: Model bundle metadata dict
        distance_category: Race distance ('sprint', 'standard', etc.)
        event_tier: Event tier (1=WTCS, 2=World Cup, 3=Continental, 4=Other).
                   When provided, tries tier-specific params first.
    """
    if distance_category is None:
        d = bundle_metadata.get("pack_effect_params")
        return pack_params_from_dict(d) if d else None

    dist_key = distance_category.lower().strip()
    if dist_key == "olympic":
        dist_key = "standard"

    if dist_key not in DRAFT_LEGAL_DISTANCES:
        logger.info(f"Distance '{dist_key}' is non-drafting, disabling pack effects")
        return None

    gap_sec = DISTANCE_PACK_GAP_SEC.get(dist_key, 3.0)

    # Try tier-specific params first
    if event_tier is not None:
        tier_key = f"tier{event_tier}"
        by_tier = bundle_metadata.get("pack_effect_params_by_tier", {})
        if tier_key in by_tier:
            params = pack_params_from_dict(by_tier[tier_key])
            params.max_gap_sec = gap_sec
            params.distance_category = dist_key
            logger.info(
                f"Using tier-specific pack params ({tier_key}, {dist_key}): "
                f"bonus={params.front_pack_bonus_sec:.1f}s, penalty={params.chase_penalty_sec:.1f}s"
            )
            return params

    # Try distance-specific params
    by_distance = bundle_metadata.get("pack_effect_params_by_distance", {})
    if dist_key in by_distance:
        params = pack_params_from_dict(by_distance[dist_key])
        params.max_gap_sec = gap_sec
        params.distance_category = dist_key
        logger.info(
            f"Using distance-specific pack params for '{dist_key}': "
            f"bonus={params.front_pack_bonus_sec:.1f}s, penalty={params.chase_penalty_sec:.1f}s, "
            f"gap={params.max_gap_sec:.1f}s"
        )
        return params

    # Fall back to overall params with distance-specific gap
    d = bundle_metadata.get("pack_effect_params")
    if d:
        params = pack_params_from_dict(d)
        params.max_gap_sec = gap_sec
        params.distance_category = dist_key
        logger.info(
            f"No distance/tier-specific pack params for '{dist_key}', "
            f"using overall (gap overridden to {params.max_gap_sec:.1f}s)"
        )
        return params

    return None


def get_distance_merge_params(
    bundle_metadata: dict,
    distance_category: str | None,
    event_tier: int | None = None,
) -> MergeParams | None:
    """
    Look up merge params from bundle metadata.

    Lookup order: tier-specific → distance-specific → overall.
    Returns None for non-drafting distances.
    """
    if distance_category is None:
        d = bundle_metadata.get("merge_params")
        return merge_params_from_dict(d) if d else None

    dist_key = distance_category.lower().strip()
    if dist_key == "olympic":
        dist_key = "standard"

    if dist_key not in DRAFT_LEGAL_DISTANCES:
        return None

    # Try tier-specific first
    if event_tier is not None:
        tier_key = f"tier{event_tier}"
        by_tier = bundle_metadata.get("merge_params_by_tier", {})
        if tier_key in by_tier:
            params = merge_params_from_dict(by_tier[tier_key])
            logger.info(
                f"Using tier-specific merge params ({tier_key}, {dist_key}): "
                f"beta_0={params.beta_0:.3f}, beta_gap={params.beta_gap:.3f}"
            )
            return params

    # Try distance-specific
    by_distance = bundle_metadata.get("merge_params_by_distance", {})
    if dist_key in by_distance:
        params = merge_params_from_dict(by_distance[dist_key])
        logger.info(
            f"Using distance-specific merge params for '{dist_key}': "
            f"beta_0={params.beta_0:.3f}, beta_gap={params.beta_gap:.3f}"
        )
        return params

    d = bundle_metadata.get("merge_params")
    return merge_params_from_dict(d) if d else None


# ── Learn Merge Parameters ───────────────────────────────────────────


def learn_merge_params_from_data(
    transitions_df: pd.DataFrame,
    min_observations: int = 30,
) -> MergeParams:
    """
    Learn merge probability parameters from historical swim-to-bike pack transitions.

    For each race, identifies merge-candidate situations: a small group (1-2
    riders) at swim exit with a larger group (3+) close behind (gap <= 15s).
    Then checks whether those riders ended up in the same bike-exit pack.

    Fits logistic regression: merged ~ gap_sec + chase_pack_size

    Args:
        transitions_df: DataFrame from fetch_swim_to_bike_transitions()
        min_observations: Minimum merge-candidate observations required to fit

    Returns:
        MergeParams with fitted coefficients (or defaults if insufficient data)
    """
    if transitions_df.empty:
        logger.warning("Empty transitions data, returning default MergeParams")
        return MergeParams()

    observations = []  # list of (gap_sec, chase_size, merged)

    race_groups = transitions_df.groupby(["event_id", "prog_id"])
    n_races = len(race_groups)

    for (event_id, prog_id), race_df in race_groups:
        race_df = race_df.sort_values("swim_pos").reset_index(drop=True)

        # Build swim packs from the data (using swim_pack_id)
        pack_ids = race_df["swim_pack_id"].values
        unique_packs = sorted(race_df["swim_pack_id"].unique())

        if len(unique_packs) < 2:
            continue

        # Build pack info: athletes grouped by swim pack
        pack_info = {}
        for pid in unique_packs:
            mask = pack_ids == pid
            pack_athletes = race_df[mask]
            pack_info[pid] = {
                "size": len(pack_athletes),
                "max_swim_gap": pack_athletes["swim_gap_to_leader_sec"].max(),
                "bike_pack_ids": set(pack_athletes["bike_pack_id"].unique()),
                "athlete_ids": set(pack_athletes["athlete_id"].values),
            }

        # Check consecutive pack pairs for merge candidates
        for i in range(len(unique_packs) - 1):
            ahead_pid = unique_packs[i]
            chase_pid = unique_packs[i + 1]
            ahead = pack_info[ahead_pid]
            chase = pack_info[chase_pid]

            ahead_size = ahead["size"]
            chase_size = chase["size"]

            # Only consider: small group ahead, larger group behind
            if ahead_size > 2 or chase_size < 3:
                continue

            # Compute gap between the groups: chase front's swim gap minus
            # ahead group's last swimmer's swim gap (both relative to leader)
            chase_athletes = race_df[race_df["swim_pack_id"] == chase_pid]
            ahead_athletes = race_df[race_df["swim_pack_id"] == ahead_pid]
            gap = chase_athletes["swim_gap_to_leader_sec"].min() - ahead_athletes["swim_gap_to_leader_sec"].max()

            if gap <= 0 or gap > 15.0:
                continue

            # Did they merge? Check if any ahead athletes share a bike pack
            # with any chase athletes
            merged = bool(ahead["bike_pack_ids"] & chase["bike_pack_ids"])

            observations.append((gap, chase_size, int(merged)))

    if len(observations) < min_observations:
        logger.warning(
            f"Only {len(observations)} merge-candidate observations "
            f"(need {min_observations}), using default MergeParams"
        )
        return MergeParams(n_observations=len(observations), n_races=n_races)

    # Fit logistic regression
    from sklearn.linear_model import LogisticRegression

    obs_arr = np.array(observations)
    X = obs_arr[:, :2]  # gap_sec, chase_size
    y = obs_arr[:, 2]   # merged (0 or 1)

    merge_rate = y.mean()
    logger.info(
        f"Merge candidates: {len(observations)} situations from {n_races} races, "
        f"merge rate: {merge_rate:.1%}"
    )

    # Log distribution by gap bins
    gap_bins = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    for lo, hi in gap_bins:
        mask = (obs_arr[:, 0] >= lo) & (obs_arr[:, 0] < hi)
        if mask.sum() > 0:
            bin_rate = obs_arr[mask, 2].mean()
            logger.info(f"  Gap [{lo:2d}-{hi:2d}s]: {mask.sum():4d} obs, merge rate {bin_rate:.1%}")

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(X, y)

    beta_0 = float(lr.intercept_[0])
    beta_gap = float(lr.coef_[0][0])
    beta_chase_size = float(lr.coef_[0][1])

    params = MergeParams(
        beta_0=beta_0,
        beta_gap=beta_gap,
        beta_chase_size=beta_chase_size,
        n_observations=len(observations),
        n_races=n_races,
    )

    logger.info(
        f"Learned merge params: beta_0={beta_0:.3f}, beta_gap={beta_gap:.3f}, "
        f"beta_chase_size={beta_chase_size:.3f}"
    )

    # Log example probabilities
    for gap, cs in [(2, 3), (5, 5), (8, 3), (12, 4)]:
        lo = beta_0 + beta_gap * gap + beta_chase_size * cs
        p = 1.0 / (1.0 + np.exp(-lo))
        logger.info(f"  P(merge | gap={gap}s, chase={cs}): {p:.1%}")

    return params


# ── Dynamic Pack Merging ─────────────────────────────────────────────


def apply_pack_merges(
    swim_times: np.ndarray,
    pack_params: PackEffectParams,
    merge_params: MergeParams,
    rng: np.random.Generator,
    breakaway_bias: float = 0.0,
) -> np.ndarray:
    """
    Apply dynamic pack merging and return bike time effects.

    After assigning initial packs from swim times, scans for merge candidates:
    small groups (1-2 riders) ahead of larger chase packs (3+). For each
    candidate, samples a merge based on learned probability. Merged athletes
    get the chase pack's drafting benefit.

    Scans from back to front so cascading merges are handled: if pack 3
    catches pack 2, the enlarged group may then catch pack 1.

    Args:
        swim_times: Simulated swim times (seconds), original athlete order
        pack_params: PackEffectParams for bike effect calculation
        merge_params: MergeParams with learned merge probabilities
        rng: Random number generator for this simulation iteration
        breakaway_bias: Coach override. 0.0 = learned probabilities.
            Positive = breakaways stick more. Negative = merges more likely.

    Returns:
        Array of bike time adjustments (seconds), same order as swim_times.
    """
    n = len(swim_times)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Sort by swim time
    sorted_idx = np.argsort(swim_times)
    sorted_swim = swim_times[sorted_idx]

    # Assign initial packs from swim exit
    pack_ids = assign_packs_chain(sorted_swim, pack_params.max_gap_sec)

    # Build pack metadata: list of {start, end, size} per pack (sorted order)
    packs = []
    current_pid = pack_ids[0]
    start = 0
    for i in range(1, n):
        if pack_ids[i] != current_pid:
            packs.append({"start": start, "end": i - 1, "size": i - start})
            current_pid = pack_ids[i]
            start = i
    packs.append({"start": start, "end": n - 1, "size": n - start})

    # Scan for merges (repeat until stable to handle cascades)
    max_passes = len(packs)
    for _ in range(max_passes):
        merged_any = False
        new_packs = []
        i = len(packs) - 1

        while i >= 0:
            if i == 0:
                new_packs.insert(0, packs[i])
                break

            chase = packs[i]
            ahead = packs[i - 1]

            ahead_size = ahead["size"]
            chase_size = chase["size"]

            # Merge candidate: small group ahead, larger group behind
            can_merge = (
                ahead_size <= 2
                and chase_size >= merge_params.min_chase_size_for_merge
                and chase["start"] <= n - 1  # safety
            )

            if can_merge:
                # Gap = chase front swim time - ahead tail swim time
                gap = sorted_swim[chase["start"]] - sorted_swim[ahead["end"]]

                if 0 < gap <= merge_params.max_merge_gap_sec:
                    log_odds = (
                        merge_params.beta_0
                        + merge_params.beta_gap * gap
                        + merge_params.beta_chase_size * chase_size
                        - breakaway_bias
                    )
                    p_merge = 1.0 / (1.0 + np.exp(-log_odds))

                    if rng.random() < p_merge:
                        # Merge: absorb ahead into chase
                        merged_pack = {
                            "start": ahead["start"],
                            "end": chase["end"],
                            "size": ahead_size + chase_size,
                        }
                        new_packs.insert(0, merged_pack)
                        merged_any = True
                        i -= 2  # skip both packs
                        continue

            new_packs.insert(0, packs[i])
            i -= 1

        packs = new_packs
        if not merged_any:
            break

    # Rebuild pack_ids from merged packs
    final_pack_ids = np.zeros(n, dtype=np.int32)
    for pack_order, pack in enumerate(packs):
        for j in range(pack["start"], pack["end"] + 1):
            final_pack_ids[j] = pack_order

    final_pack_sizes = compute_pack_sizes(final_pack_ids)

    # Compute bike effects from merged configuration
    effects_sorted = multi_pack_bike_effect(final_pack_ids, final_pack_sizes, pack_params)

    # Un-sort back to original athlete order
    effects = np.empty(n, dtype=np.float64)
    effects[sorted_idx] = effects_sorted

    return effects


# ── Continuous Gap-Based Bike Effect ─────────────────────────────────


def continuous_gap_bike_effect(
    swim_times: np.ndarray,
    params: PackEffectParams,
    density_window_sec: float | None = None,
) -> np.ndarray:
    """
    Compute bike time adjustment as a continuous function of swim gap to leader.

    Instead of discrete pack assignments with a binary threshold, the effect
    transitions smoothly from front-pack bonus to chase penalty based on each
    athlete's gap to the swim leader. A local density factor (how many athletes
    are nearby) scales the drafting benefit — larger groups draft better.

    The effect curve:
    - gap ≤ max_gap_sec (default 2s): full front_pack_bonus
    - max_gap_sec < gap < chase_threshold (15s): linear interpolation
    - gap ≥ chase_threshold: full chase_penalty

    Pack-size scaling uses local density: count of athletes within
    ±density_window_sec of this athlete's swim time.

    Args:
        swim_times: Array of swim times (seconds).
        params: PackEffectParams with learned bonus/penalty values.
        density_window_sec: Window (seconds) for counting nearby athletes.

    Returns:
        Array of bike time adjustments (seconds), same length as swim_times.
        Negative = faster (drafting), positive = slower (solo).
    """
    n = len(swim_times)
    if n == 0:
        return np.array([], dtype=np.float64)

    effects = np.zeros(n, dtype=np.float64)
    leader_time = swim_times.min()

    bonus = params.front_pack_bonus_sec    # e.g., -25s
    penalty = params.chase_penalty_sec      # e.g., +15s
    full_benefit_gap = params.max_gap_sec   # Within this gap = full bonus
    chase_threshold = 15.0                  # Beyond this gap = full penalty
    min_size = params.min_pack_size_for_draft
    size_scale = params.draft_size_scale

    # Use pack gap threshold as density window if not specified
    if density_window_sec is None:
        density_window_sec = params.max_gap_sec

    # Precompute sorted times for efficient density calculation
    sorted_times = np.sort(swim_times)

    for i in range(n):
        gap = swim_times[i] - leader_time

        # Base effect from continuous gap function
        if gap <= full_benefit_gap:
            base_effect = bonus
        elif gap >= chase_threshold:
            base_effect = penalty
        else:
            # Linear interpolation between bonus and penalty
            t = (gap - full_benefit_gap) / (chase_threshold - full_benefit_gap)
            base_effect = bonus + t * (penalty - bonus)

        # Local density: count athletes within ±density_window_sec
        low = np.searchsorted(sorted_times, swim_times[i] - density_window_sec, side="left")
        high = np.searchsorted(sorted_times, swim_times[i] + density_window_sec, side="right")
        local_density = high - low  # includes self

        # Scale effect by local density (more nearby athletes = better drafting)
        if local_density < min_size:
            # Solo or tiny group: penalize regardless of gap
            effects[i] = penalty
        else:
            size_factor = 1.0 + size_scale * max(0, local_density - min_size)
            if base_effect < 0:
                # Drafting benefit: amplify with larger group (cap at 1.5x)
                effects[i] = base_effect * min(size_factor, 1.5)
            else:
                # Penalty zone: density doesn't reduce the penalty
                effects[i] = base_effect

    return effects


# ── Uncertainty Estimation ───────────────────────────────────────────


def estimate_uncertainty(
    pred_df: pd.DataFrame,
    sigma_total_col: str = "std_total_sec_24m",
    default_sigma: float = DEFAULT_SIGMA_TOTAL,
    distance_category: str | None = None,
    bundle_metadata: dict | None = None,
) -> pd.DataFrame:
    """
    Add per-athlete, per-split uncertainty (sigma) columns.

    When learned residual stats are available in bundle_metadata, uses empirical
    per-split sigmas from training data (distance-specific when possible).
    Otherwise falls back to hardcoded defaults with distance multiplier scaling.

    Per-athlete heteroscedastic scaling is preserved: athletes with higher
    std_total_sec_24m get proportionally wider uncertainty bands.

    Args:
        pred_df: DataFrame with predictions
        sigma_total_col: Column name for athlete's historical std dev
        default_sigma: Default sigma if not available
        distance_category: Race distance ('sprint', 'standard', etc.) for
                          scaling uncertainty. None defaults to standard.
        bundle_metadata: Model bundle metadata dict (may contain 'residual_stats')

    Returns:
        DataFrame with sigma_total, sigma_swim, sigma_bike, sigma_run, etc. added
    """
    df = pred_df.copy()

    # ── Strategy: Normalized percentile variance × field spread ──
    # Use std of each athlete's split percentile rank across races (std_swim_pct_24m).
    # This is completely course-agnostic — measures how much an athlete's RELATIVE
    # POSITION varies race-to-race.
    #
    # Convert to seconds using FIELD SPREAD (not predicted time!):
    #   sigma_swim = std_swim_pct × field_swim_spread
    # where field_swim_spread = p90 - p10 of predicted swim times in this race.
    #
    # Why field spread, not pred_time: std_pct=0.08 means position varies by ±8% of
    # the field. If field is spread over 80s, that's ±6.4s. If we used pred_time (600s),
    # we'd get 48s — a ~7× overestimate that swamps position-level gaps.

    has_norm_variance = all(
        col in df.columns and df[col].notna().any()
        for col in ["std_swim_pct_24m", "std_bike_pct_24m", "std_run_pct_24m"]
    )

    if has_norm_variance:
        # Default std_pct for athletes without enough race history
        # 0.12 = moderate variability (~12% of field swing race-to-race)
        DEFAULT_STD_PCT = 0.12

        std_swim_pct = pd.to_numeric(df["std_swim_pct_24m"], errors="coerce").fillna(DEFAULT_STD_PCT)
        std_bike_pct = pd.to_numeric(df["std_bike_pct_24m"], errors="coerce").fillna(DEFAULT_STD_PCT)
        std_run_pct = pd.to_numeric(df["std_run_pct_24m"], errors="coerce").fillna(DEFAULT_STD_PCT)

        # Compute field spread from predicted split times (p90 - p10)
        # This captures how spread out THIS race's field is
        def _field_spread(col, fallback):
            """Compute p90-p10 field spread for a split, with fallback."""
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) >= 5:
                    return vals.quantile(0.9) - vals.quantile(0.1)
            return fallback

        # Field spreads: use predicted splits if available, else typical sprint defaults
        spread_swim = _field_spread("pred_swim_sec", 80.0)
        spread_bike = _field_spread("pred_bike_sec", 160.0)
        spread_run = _field_spread("pred_run_sec", 120.0)

        df["sigma_swim"] = std_swim_pct * spread_swim
        df["sigma_bike"] = std_bike_pct * spread_bike
        df["sigma_run"] = std_run_pct * spread_run
        # T1/T2: proportional to swim/run sigma
        df["sigma_t1"] = df["sigma_swim"] * 0.25
        df["sigma_t2"] = df["sigma_run"] * 0.15

        # Clamp per-split: prevent extreme values
        df["sigma_swim"] = df["sigma_swim"].clip(2, 20)
        df["sigma_bike"] = df["sigma_bike"].clip(4, 40)
        df["sigma_run"] = df["sigma_run"].clip(3, 30)
        df["sigma_t1"] = df["sigma_t1"].clip(0.5, 5)
        df["sigma_t2"] = df["sigma_t2"].clip(0.5, 4)
        df["sigma_total"] = df["sigma_swim"] + df["sigma_t1"] + df["sigma_bike"] + df["sigma_t2"] + df["sigma_run"]

        logger.info(
            f"Normalized sigma (field-spread): swim=[{df['sigma_swim'].min():.1f}-{df['sigma_swim'].max():.1f}] "
            f"(spread={spread_swim:.0f}s), bike=[{df['sigma_bike'].min():.1f}-{df['sigma_bike'].max():.1f}] "
            f"(spread={spread_bike:.0f}s), run=[{df['sigma_run'].min():.1f}-{df['sigma_run'].max():.1f}] "
            f"(spread={spread_run:.0f}s), total=[{df['sigma_total'].min():.1f}-{df['sigma_total'].max():.1f}]"
        )
    else:
        # ── Fallback: ratio-based scaling from learned residuals ──
        learned_sigmas = _get_learned_sigmas(bundle_metadata, distance_category)
        FIELD_SIGMA_SCALE = 0.5

        if learned_sigmas:
            base_swim = learned_sigmas.get("swim", DEFAULT_SIGMA_SWIM) * FIELD_SIGMA_SCALE
            base_t1 = learned_sigmas.get("t1", DEFAULT_SIGMA_T1) * FIELD_SIGMA_SCALE
            base_bike = learned_sigmas.get("bike", DEFAULT_SIGMA_BIKE) * FIELD_SIGMA_SCALE
            base_t2 = learned_sigmas.get("t2", DEFAULT_SIGMA_T2) * FIELD_SIGMA_SCALE
            base_run = learned_sigmas.get("run", DEFAULT_SIGMA_RUN) * FIELD_SIGMA_SCALE
            base_total = base_swim + base_t1 + base_bike + base_t2 + base_run
            logger.info(
                f"Fallback: learned residual sigmas (×{FIELD_SIGMA_SCALE}): swim={base_swim:.1f}, "
                f"bike={base_bike:.1f}, run={base_run:.1f}"
            )
        else:
            dist_mult = 1.0
            if distance_category:
                dist_mult = DISTANCE_SIGMA_MULTIPLIER.get(
                    distance_category.lower().strip(), 1.0
                )
            base_swim = DEFAULT_SIGMA_SWIM * dist_mult
            base_t1 = DEFAULT_SIGMA_T1 * dist_mult
            base_bike = DEFAULT_SIGMA_BIKE * dist_mult
            base_t2 = DEFAULT_SIGMA_T2 * dist_mult
            base_run = DEFAULT_SIGMA_RUN * dist_mult
            base_total = DEFAULT_SIGMA_TOTAL * dist_mult

        if sigma_total_col in df.columns:
            df["sigma_total"] = df[sigma_total_col].fillna(default_sigma)
            df["sigma_total"] = df["sigma_total"].clip(15, 90)
        else:
            df["sigma_total"] = default_sigma

        ratio = (df["sigma_total"] / base_total) ** 1.8
        df["sigma_swim"] = base_swim * ratio
        df["sigma_t1"] = base_t1 * ratio
        df["sigma_bike"] = base_bike * ratio
        df["sigma_t2"] = base_t2 * ratio
        df["sigma_run"] = base_run * ratio

    # Weather-based sigma adjustments
    # Heat increases variance (athletes respond to heat differently)
    if "wbgt" in df.columns:
        wbgt_val = pd.to_numeric(df["wbgt"], errors="coerce").fillna(20.0)
        heat_mult = 1.0 + (wbgt_val - 22.0).clip(lower=0) * 0.02  # +2% per °C above 22
        df["sigma_bike"] *= heat_mult
        df["sigma_run"] *= heat_mult
        df["sigma_total"] *= heat_mult

    # Wind increases bike variance
    if "wind_speed_kmh" in df.columns:
        wind_val = pd.to_numeric(df["wind_speed_kmh"], errors="coerce").fillna(10.0)
        wind_mult = 1.0 + (wind_val - 15.0).clip(lower=0) * 0.01  # +1% per km/h above 15
        df["sigma_bike"] *= wind_mult

    # Rain increases transition and bike variance
    if "precipitation_mm" in df.columns:
        precip_val = pd.to_numeric(df["precipitation_mm"], errors="coerce").fillna(0.0)
        rain_mult = pd.Series(np.where(precip_val > 1.0, 1.1, 1.0), index=df.index)
        df["sigma_t1"] *= rain_mult
        df["sigma_t2"] *= rain_mult
        df["sigma_bike"] *= rain_mult

    return df


def _get_learned_sigmas(
    metadata: dict | None, distance_category: str | None
) -> dict | None:
    """
    Retrieve per-split empirical sigmas from bundle metadata.

    Looks up distance-specific residual stats first, falls back to overall.
    Returns None if no residual stats are available.
    """
    if not metadata:
        return None

    residual_stats = metadata.get("residual_stats")
    if not residual_stats:
        return None

    # Try distance-specific first
    if distance_category:
        dist_key = distance_category.lower().strip()
        if dist_key == "olympic":
            dist_key = "standard"
        dist_stats = residual_stats.get(dist_key)
        if dist_stats and "per_split_sigma" in dist_stats:
            return dist_stats["per_split_sigma"]

    # Fall back to overall
    overall = residual_stats.get("overall")
    if overall and "per_split_sigma" in overall:
        return overall["per_split_sigma"]

    return None


def _get_residual_cov(
    metadata: dict | None, distance_category: str | None
) -> np.ndarray | None:
    """
    Retrieve 5×5 residual covariance matrix from bundle metadata.

    Looks up distance-specific first, falls back to overall.
    Returns None if not available (triggers legacy noise model).
    """
    if not metadata:
        return None

    residual_stats = metadata.get("residual_stats")
    if not residual_stats:
        return None

    # Try distance-specific first
    if distance_category:
        dist_key = distance_category.lower().strip()
        if dist_key == "olympic":
            dist_key = "standard"
        dist_stats = residual_stats.get(dist_key)
        if dist_stats and "cov_matrix" in dist_stats:
            return np.array(dist_stats["cov_matrix"])

    # Fall back to overall
    overall = residual_stats.get("overall")
    if overall and "cov_matrix" in overall:
        return np.array(overall["cov_matrix"])

    return None


def _get_swim_gap_distribution(bundle_metadata: dict | None, distance_category: str | None) -> dict | None:
    """
    Retrieve swim gap distribution from bundle metadata.

    Looks up distance-specific first, falls back to overall.
    Returns None if not available.
    """
    if not bundle_metadata:
        return None

    gap_dists = bundle_metadata.get("swim_gap_distributions")
    if not gap_dists:
        return None

    # Try distance-specific first
    if distance_category:
        dist_key = distance_category.lower().strip()
        if dist_key == "olympic":
            dist_key = "standard"
        dist_dist = gap_dists.get(dist_key)
        if dist_dist:
            return dist_dist

    # Fall back to overall
    return gap_dists.get("overall")


def _percentile_to_gap(percentiles: np.ndarray, gap_dist: dict) -> np.ndarray:
    """
    Convert swim exit percentiles to gap-to-leader using learned distribution.

    Uses linear interpolation between quantile points.

    Args:
        percentiles: Array of swim exit percentiles (0=first, 1=last)
        gap_dist: Dict with 'quantiles' and 'gap_at_quantile' arrays

    Returns:
        Array of gap-to-leader values in seconds
    """
    q_points = np.array(gap_dist["quantiles"])
    g_points = np.array(gap_dist["gap_at_quantile"])

    # Clamp percentiles to valid range
    pct_clamped = np.clip(percentiles, q_points[0], q_points[-1])

    # Linear interpolation
    gaps = np.interp(pct_clamped, q_points, g_points)

    # Ensure non-negative
    return np.maximum(gaps, 0.0)


# ── Monte Carlo Simulation ──────────────────────────────────────────


def run_monte_carlo(
    pred_df: pd.DataFrame,
    n_sims: int = 10000,
    random_state: int = 42,
    use_pack_effects: bool = True,
    pack_params: PackEffectParams | None = None,
    merge_params: MergeParams | None = None,
    breakaway_bias: float = 0.0,
    form_share: float = DEFAULT_FORM_SHARE,
    distance_category: str | None = None,
    bundle_metadata: dict | None = None,
    swim_mode: str = "legacy",
) -> pd.DataFrame:
    """
    Run causal-chain Monte Carlo simulation.

    Models the causal structure of draft-legal triathlon racing:
    1. Simulate swim times (form + noise) → swim exit order
    2. Simulate T1 times (form + noise) → bike entry order
    3. Compute swim+T1 gaps → pack formation → dynamic merging → bike effect
    4. Simulate bike times (form + noise + pack effect from swim+T1)
    5. Simulate T2 times (form + noise)
    6. Simulate run times (form + noise, individual effort only)
    7. Total = swim + T1 + bike + T2 + run → rank

    The pack effect is centered around baseline predictions so that the
    simulation's mean matches the deterministic prediction. Variance comes
    from the causal chain: good swim → front pack → faster bike.

    When merge_params are provided, the simulation dynamically merges packs
    during the bike leg: chase packs probabilistically absorb small groups
    (1-2 riders) ahead of them, and merged athletes get drafting benefit.

    Falls back to simple total-time simulation if split predictions are
    not available.

    Args:
        pred_df: DataFrame with predictions (needs pred_swim_sec, pred_bike_sec,
                 pred_run_sec for causal chain; or pred_total_sec for fallback)
        n_sims: Number of simulations (default 10000)
        random_state: Random seed for reproducibility
        use_pack_effects: Whether to apply pack effects on bike
        pack_params: Learned pack effect parameters (uses defaults if None)
        merge_params: Learned merge parameters for dynamic pack merging.
                     If None, uses static swim-exit packs (original behavior).
        breakaway_bias: Coach override for merge probability. 0.0 = learned
                       probabilities. Positive = breakaways stick. Negative =
                       packs merge more. Range: roughly -3 to +3.
        form_share: Fraction of per-split variance from shared form factor (0-1).
                    Higher = more correlation across splits. Ignored when MVN
                    covariance matrix is available (correlations are learned).
        distance_category: Race distance ('sprint', 'standard', etc.) for
                          scaling uncertainty. None defaults to standard.
        bundle_metadata: Model bundle metadata dict. When it contains
                        'residual_stats' with a covariance matrix, the simulation
                        uses multivariate normal sampling instead of independent
                        noise + form factor. This naturally captures split
                        correlations from the training data.
        swim_mode: How to simulate swim times. Options:
                  - "legacy": Add noise to predicted swim times (original approach)
                  - "percentile": Sample swim exit percentiles, convert to gaps
                    using learned distributions. Requires pred_swim_exit_pct
                    column and swim_gap_distributions in bundle_metadata.

    Returns:
        DataFrame with original columns plus:
        - prob_win, prob_podium, prob_top5, prob_top10, prob_top20
        - expected_rank, rank_p10, rank_p50, rank_p90
        - total_p10, total_p50, total_p90 (seconds)
    """
    if pred_df.empty:
        logger.warning("Empty pred_df, returning empty DataFrame")
        return pred_df.copy()

    df = estimate_uncertainty(
        pred_df, distance_category=distance_category,
        bundle_metadata=bundle_metadata,
    )
    n_athletes = len(df)
    rng = np.random.default_rng(random_state)

    # Check for learned covariance matrix (MVN noise model)
    cov_matrix = _get_residual_cov(bundle_metadata, distance_category)
    use_mvn = cov_matrix is not None

    # Check if split predictions are available for causal chain
    has_splits = all(
        c in df.columns and df[c].notna().all()
        for c in ["pred_swim_sec", "pred_bike_sec", "pred_run_sec"]
    )

    if not has_splits:
        logger.info(
            "Split predictions not fully available, using total-time simulation"
        )
        return _run_monte_carlo_total(df, n_sims, rng, use_pack_effects)

    # ── Percentile swim mode setup ──
    use_pct_swim = False
    pred_swim_pct = None
    swim_gap_dist = None

    if swim_mode == "percentile":
        has_pct = "pred_swim_exit_pct" in df.columns and df["pred_swim_exit_pct"].notna().all()
        swim_gap_dist = _get_swim_gap_distribution(bundle_metadata, distance_category)

        if has_pct and swim_gap_dist is not None:
            use_pct_swim = True
            pred_swim_pct = df["pred_swim_exit_pct"].clip(0.01, 0.99).values.astype(np.float64)
            logger.info(
                f"Using percentile-based swim simulation "
                f"(gap dist: {swim_gap_dist.get('n_races', '?')} races, "
                f"median gap={swim_gap_dist.get('gap_p50', '?'):.1f}s)"
            )
        else:
            reasons = []
            if not has_pct:
                reasons.append("no pred_swim_exit_pct")
            if swim_gap_dist is None:
                reasons.append("no swim_gap_distributions in metadata")
            logger.warning(
                f"swim_mode='percentile' requested but falling back to legacy "
                f"({', '.join(reasons)})"
            )

    # ── Causal chain simulation with multi-pack formation ──
    # Swim → T1 → Pack Formation (swim+T1) → Bike (with drafting) → T2 → Run → Total

    # Defensive .fillna() to prevent None arithmetic errors from partial missing data
    pred_swim = df["pred_swim_sec"].fillna(df["pred_swim_sec"].median()).values.astype(np.float64)
    pred_bike = df["pred_bike_sec"].fillna(df["pred_bike_sec"].median()).values.astype(np.float64)
    pred_run = df["pred_run_sec"].fillna(df["pred_run_sec"].median()).values.astype(np.float64)

    # T1/T2 predictions (critical for accurate pack formation and total accounting)
    # Use .any() to check if we have at least some T1/T2 data, and .fillna()
    # to handle partial missing values defensively (prevents None arithmetic errors)
    has_t1 = "pred_t1_sec" in df.columns and df["pred_t1_sec"].notna().any()
    has_t2 = "pred_t2_sec" in df.columns and df["pred_t2_sec"].notna().any()

    if has_t1:
        pred_t1 = df["pred_t1_sec"].fillna(30.0).values.astype(np.float64)
    else:
        pred_t1 = np.full(n_athletes, 30.0)
        logger.warning("No T1 predictions available, using 30s default")

    if has_t2:
        pred_t2 = df["pred_t2_sec"].fillna(25.0).values.astype(np.float64)
    else:
        pred_t2 = np.full(n_athletes, 25.0)
        logger.warning("No T2 predictions available, using 25s default")

    # NOTE: Split normalization removed. With explicit T1/T2 modeling, the
    # split sum (swim + T1 + bike + T2 + run) should closely match the total
    # model. Any residual gap is diagnostic, not something to paper over.

    sigma_swim = df["sigma_swim"].values
    sigma_t1 = df["sigma_t1"].values
    sigma_bike = df["sigma_bike"].values
    sigma_t2 = df["sigma_t2"].values
    sigma_run = df["sigma_run"].values

    # Pack effect parameters
    if pack_params is None:
        pack_params = PackEffectParams()

    # Pack effect scaling: when split models include pack features (default),
    # the predicted bike time already partially accounts for drafting. Scale
    # down sim pack effects to avoid double-counting. When models are trained
    # with exclude_pack_features=True ("ability-only"), use full pack effects.
    pack_effect_scale = 1.0
    if bundle_metadata:
        if bundle_metadata.get("exclude_pack_features", False):
            pack_effect_scale = 1.0  # Ability-only models: full sim pack effects
            logger.info("Pack effect scale: 1.0 (ability-only models, no double-counting)")
        else:
            # Standard models include pack features — reduce sim pack adjustment
            # to only model the deviation from typical behavior, not the full effect.
            # Scale of 0.5 means sim adds half the pack effect on top of what's
            # already captured in the predicted bike time.
            pack_effect_scale = 0.5
            logger.info("Pack effect scale: 0.5 (models include pack features, reducing double-count)")

    # Pre-compute baseline pack effect from predicted swim + T1 times.
    # In reality, pack formation at bike mount depends on swim exit + T1.
    # This is the pack effect "baked into" the predictions. We subtract it
    # so the simulation only adds the DEVIATION from baseline.
    pred_bike_entry = pred_swim + pred_t1
    if use_pack_effects:
        baseline_pack_effect = continuous_gap_bike_effect(pred_bike_entry, pack_params)
    else:
        baseline_pack_effect = np.zeros(n_athletes)

    # MVN noise model: pre-compute per-athlete scaled covariance
    if use_mvn:
        # Empirical sigmas from the covariance diagonal
        empirical_sigmas = np.sqrt(np.diag(cov_matrix))  # shape (5,)
        # Per-athlete scale factors: ratio of athlete sigma to empirical sigma per split
        # This preserves heteroscedasticity while using learned correlations
        athlete_sigmas = np.column_stack([
            sigma_swim, sigma_t1, sigma_bike, sigma_t2, sigma_run
        ])  # shape (n_athletes, 5)
        # Scale factors per athlete per split
        athlete_scale = athlete_sigmas / empirical_sigmas[np.newaxis, :]  # (n_athletes, 5)
        logger.info(
            f"Using MVN noise model with learned 5×5 covariance matrix "
            f"(empirical sigmas: {', '.join(f'{s:.1f}' for s in empirical_sigmas)})"
        )
    else:
        athlete_scale = None  # not used

    # Legacy noise decomposition (used when no covariance matrix available)
    form_std = np.sqrt(form_share)   # Fraction of std from shared form
    noise_std = np.sqrt(1.0 - form_share)  # Fraction from split-specific noise

    # Track outcomes
    rank_matrix = np.zeros((n_sims, n_athletes), dtype=np.int32)
    total_matrix = np.zeros((n_sims, n_athletes), dtype=np.float64)
    # Track pack membership: how often each athlete ends up in the front pack
    front_pack_count = np.zeros(n_athletes, dtype=np.int32)
    # Track bike pack adjustment magnitudes
    pack_effect_sum = np.zeros(n_athletes, dtype=np.float64)
    # Track per-split simulation sums for diagnostics
    split_sums = {s: np.zeros(n_athletes, dtype=np.float64) for s in ["swim", "t1", "bike", "t2", "run"]}
    # Track pack count distribution per sim
    pack_count_histogram = np.zeros(n_athletes + 1, dtype=np.int64)  # index = n_packs

    merge_label = ""
    if merge_params is not None:
        merge_label = f", dynamic_merge=True, breakaway_bias={breakaway_bias:.1f}"

    swim_mode_label = f", swim_mode={'percentile' if use_pct_swim else 'legacy'}"
    logger.info(
        f"Running {n_sims} causal-chain simulations for {n_athletes} athletes "
        f"(pack_effects={use_pack_effects}, pack_scale={pack_effect_scale:.1f}, "
        f"bonus={pack_params.front_pack_bonus_sec:.1f}s, "
        f"penalty={pack_params.chase_penalty_sec:.1f}s{merge_label}, "
        f"t1t2_modeled={'yes' if has_t1 and has_t2 else 'fallback'}"
        f"{swim_mode_label})"
    )

    for sim in range(n_sims):
        if use_mvn:
            # ── MVN NOISE MODEL ──
            # Sample correlated noise from the learned covariance matrix.
            # The covariance naturally encodes split correlations (e.g., fast
            # swimmers tend to have fast T1s). No separate form_factor needed.
            raw_noise = rng.multivariate_normal(
                np.zeros(5), cov_matrix, size=n_athletes
            )  # shape (n_athletes, 5) — columns: swim, t1, bike, t2, run
            # Scale per athlete to preserve heteroscedasticity
            noise = raw_noise * athlete_scale  # (n_athletes, 5)

            sim_swim = pred_swim + noise[:, 0]
            sim_t1 = pred_t1 + noise[:, 1]
            # bike noise applied after pack effect below
            bike_noise_mvn = noise[:, 2]
            sim_t2_noise = noise[:, 3]
            sim_run_noise = noise[:, 4]
        else:
            # ── LEGACY NOISE MODEL (form_factor + independent noise) ──
            # Shared form factor: good day / bad day, correlated across splits
            form_factor = rng.normal(0, 1, size=n_athletes)

            swim_form = form_factor * sigma_swim * form_std
            swim_noise = rng.normal(0, sigma_swim * noise_std)
            sim_swim = pred_swim + swim_form + swim_noise

            t1_form = form_factor * sigma_t1 * form_std
            t1_noise = rng.normal(0, sigma_t1 * noise_std)
            sim_t1 = pred_t1 + t1_form + t1_noise

        # ── PERCENTILE SWIM OVERRIDE ──
        # When using percentile-based swim, replace sim_swim with times derived
        # from sampled percentiles and learned gap distributions. This produces
        # more realistic swim exit orderings and gap structures.
        if use_pct_swim:
            # Sample noisy swim exit percentiles using Beta noise (bounded 0-1)
            # Concentration parameter controls noise magnitude:
            # higher = tighter around predicted percentile
            concentration = 20.0  # Controls Beta distribution spread
            alpha = pred_swim_pct * concentration
            beta_param = (1.0 - pred_swim_pct) * concentration
            # Clamp to valid Beta parameters (>0)
            alpha = np.clip(alpha, 0.1, concentration)
            beta_param = np.clip(beta_param, 0.1, concentration)
            sampled_pct = rng.beta(alpha, beta_param)

            # Convert percentiles to gap-to-leader using learned distribution
            sampled_gaps = _percentile_to_gap(sampled_pct, swim_gap_dist)

            # Convert gaps to absolute swim times:
            # Use the predicted leader time (fastest predicted swimmer) as anchor
            leader_swim_time = pred_swim.min()
            sim_swim = leader_swim_time + sampled_gaps

        # ── PACK EFFECT ON BIKE (swim + T1 → bike entry order → pack formation) ──
        sim_bike_entry = sim_swim + sim_t1
        bike_pack_adjustment = np.zeros(n_athletes)
        if use_pack_effects:
            if merge_params is not None:
                sim_pack_effect = apply_pack_merges(
                    sim_bike_entry, pack_params, merge_params, rng, breakaway_bias
                )
            else:
                sim_pack_effect = continuous_gap_bike_effect(sim_bike_entry, pack_params)
            bike_pack_adjustment = (sim_pack_effect - baseline_pack_effect) * pack_effect_scale

        # Track pack statistics
        if use_pack_effects:
            front_pack_count += (sim_pack_effect <= 0).astype(np.int32)
            pack_effect_sum += bike_pack_adjustment

        if use_mvn:
            # ── MVN: bike, T2, run with correlated noise ──
            sim_bike = pred_bike + bike_noise_mvn + bike_pack_adjustment
            sim_t2 = pred_t2 + sim_t2_noise
            sim_run = pred_run + sim_run_noise
        else:
            # ── LEGACY: bike, T2, run with form + independent noise ──
            bike_form = form_factor * sigma_bike * form_std
            bike_noise = rng.normal(0, sigma_bike * noise_std)
            sim_bike = pred_bike + bike_form + bike_noise + bike_pack_adjustment

            t2_form = form_factor * sigma_t2 * form_std
            t2_noise = rng.normal(0, sigma_t2 * noise_std)
            sim_t2 = pred_t2 + t2_form + t2_noise

            run_form = form_factor * sigma_run * form_std
            run_noise = rng.normal(0, sigma_run * noise_std)
            sim_run = pred_run + run_form + run_noise

        # ── TOTAL (all 5 segments) ──
        sim_total = sim_swim + sim_t1 + sim_bike + sim_t2 + sim_run
        total_matrix[sim, :] = sim_total

        # Track per-split sums for diagnostics
        split_sums["swim"] += sim_swim
        split_sums["t1"] += sim_t1
        split_sums["bike"] += sim_bike
        split_sums["t2"] += sim_t2
        split_sums["run"] += sim_run

        # Track pack count per simulation
        if use_pack_effects:
            # Count distinct pack effects (unique values = number of packs)
            n_packs = len(np.unique(np.round(sim_pack_effect, 1)))
            if n_packs < len(pack_count_histogram):
                pack_count_histogram[n_packs] += 1

        # Rank (1 = fastest)
        ranks = np.argsort(np.argsort(sim_total)) + 1
        rank_matrix[sim, :] = ranks

    # Compute pack statistics
    sim_pack_stats = {
        "sim_front_pack_pct": front_pack_count / n_sims,
        "sim_avg_pack_effect": pack_effect_sum / n_sims,
    }

    # Build simulation diagnostics
    split_means = {s: vals / n_sims for s, vals in split_sums.items()}
    diagnostics = _build_sim_diagnostics(
        df, pred_swim, pred_t1, pred_bike, pred_t2, pred_run,
        split_means, front_pack_count / n_sims, pack_count_histogram, n_sims,
    )

    return _aggregate_results(df, rank_matrix, total_matrix, sim_pack_stats, diagnostics)


def _build_sim_diagnostics(
    df: pd.DataFrame,
    pred_swim: np.ndarray,
    pred_t1: np.ndarray,
    pred_bike: np.ndarray,
    pred_t2: np.ndarray,
    pred_run: np.ndarray,
    split_means: dict[str, np.ndarray],
    sim_front_pack_pct: np.ndarray,
    pack_count_histogram: np.ndarray,
    n_sims: int,
) -> dict:
    """
    Build simulation diagnostics comparing sim means to predictions.

    Returns a dict with:
    - per_split_bias: mean(sim_split - pred_split) per split
    - front_pack_rate_comparison: sim vs historical front_pack_rate
    - pack_count_distribution: how many packs formed per sim
    """
    diagnostics = {}

    # Per-split bias: how much the simulation mean deviates from predictions
    preds = {"swim": pred_swim, "t1": pred_t1, "bike": pred_bike, "t2": pred_t2, "run": pred_run}
    per_split_bias = {}
    for split_name, pred_vals in preds.items():
        sim_mean = split_means[split_name]
        bias = float(np.mean(sim_mean - pred_vals))
        per_split_bias[split_name] = bias
    diagnostics["per_split_bias"] = per_split_bias

    total_bias = sum(per_split_bias.values())
    diagnostics["total_bias"] = total_bias

    # Front pack rate: sim vs historical
    if "front_pack_rate" in df.columns:
        hist_fpr = df["front_pack_rate"].fillna(0.5).values
        # Compute correlation, guarding against zero-variance arrays
        corr = 0.0
        if len(hist_fpr) > 2 and np.std(hist_fpr) > 1e-9 and np.std(sim_front_pack_pct) > 1e-9:
            corr = float(np.corrcoef(hist_fpr, sim_front_pack_pct)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        diagnostics["front_pack_rate_comparison"] = {
            "hist_mean": float(np.mean(hist_fpr)),
            "sim_mean": float(np.mean(sim_front_pack_pct)),
            "correlation": corr,
        }

    # Pack count distribution (top entries)
    nonzero = np.nonzero(pack_count_histogram)[0]
    if len(nonzero) > 0:
        pack_dist = {int(k): int(pack_count_histogram[k]) for k in nonzero}
        avg_packs = sum(k * v for k, v in pack_dist.items()) / n_sims
        diagnostics["avg_packs_per_sim"] = float(avg_packs)
        diagnostics["pack_count_distribution"] = pack_dist

    # Log summary
    bias_str = ", ".join(f"{k}={v:+.1f}s" for k, v in per_split_bias.items())
    logger.info(f"Sim diagnostics — per-split bias: {bias_str}, total={total_bias:+.1f}s")
    if "front_pack_rate_comparison" in diagnostics:
        fpc = diagnostics["front_pack_rate_comparison"]
        logger.info(
            f"  Front pack rate: hist={fpc['hist_mean']:.2f}, sim={fpc['sim_mean']:.2f}, "
            f"corr={fpc['correlation']:.3f}"
        )
    if "avg_packs_per_sim" in diagnostics:
        logger.info(f"  Avg packs per sim: {diagnostics['avg_packs_per_sim']:.1f}")

    return diagnostics


def _run_monte_carlo_total(
    df: pd.DataFrame,
    n_sims: int,
    rng: np.random.Generator,
    use_pack_effects: bool,
) -> pd.DataFrame:
    """
    Fallback simulation using pred_total_sec when split predictions are unavailable.

    Uses the same form + noise structure but operates on total time directly.
    Pack effects use the legacy front_pack_rate-based binary sampling.
    """
    n_athletes = len(df)

    pred_total = df["pred_total_sec"].values.astype(np.float64)
    sigma_total = df["sigma_total"].values
    front_pack_rate = (
        df["front_pack_rate"].fillna(0.5).values
        if "front_pack_rate" in df.columns
        else np.full(n_athletes, 0.5)
    )

    # Use default pack constants for fallback mode
    pack_bonus = -25.0
    pack_penalty = 15.0

    rank_matrix = np.zeros((n_sims, n_athletes), dtype=np.int32)
    total_matrix = np.zeros((n_sims, n_athletes), dtype=np.float64)

    logger.info(
        f"Running {n_sims} total-time simulations for {n_athletes} athletes "
        "(fallback mode, no causal chain)"
    )

    for sim in range(n_sims):
        form_delta = rng.normal(0, sigma_total * 0.5)
        noise = rng.normal(0, sigma_total * 0.5)

        pack_effect = np.zeros(n_athletes)
        if use_pack_effects:
            front_pack = rng.random(n_athletes) < front_pack_rate
            pack_effect = np.where(front_pack, pack_bonus, pack_penalty)

        sim_total = pred_total + form_delta + noise + pack_effect
        total_matrix[sim, :] = sim_total

        ranks = np.argsort(np.argsort(sim_total)) + 1
        rank_matrix[sim, :] = ranks

    return _aggregate_results(df, rank_matrix, total_matrix)


def _aggregate_results(
    df: pd.DataFrame,
    rank_matrix: np.ndarray,
    total_matrix: np.ndarray,
    sim_pack_stats: dict | None = None,
    diagnostics: dict | None = None,
) -> pd.DataFrame:
    """Aggregate simulation outcomes into probability and interval columns."""
    logger.info("Aggregating simulation results")

    result_df = df.copy()

    # Probability calculations
    result_df["prob_win"] = (rank_matrix == 1).mean(axis=0)
    result_df["prob_podium"] = (rank_matrix <= 3).mean(axis=0)
    result_df["prob_top5"] = (rank_matrix <= 5).mean(axis=0)
    result_df["prob_top10"] = (rank_matrix <= 10).mean(axis=0)
    result_df["prob_top20"] = (rank_matrix <= 20).mean(axis=0)

    # Rank statistics
    result_df["expected_rank"] = rank_matrix.mean(axis=0)
    result_df["rank_p10"] = np.percentile(rank_matrix, 10, axis=0).astype(int)
    result_df["rank_p50"] = np.percentile(rank_matrix, 50, axis=0).astype(int)
    result_df["rank_p90"] = np.percentile(rank_matrix, 90, axis=0).astype(int)

    # Time statistics
    result_df["total_p10"] = np.percentile(total_matrix, 10, axis=0)
    result_df["total_p50"] = np.percentile(total_matrix, 50, axis=0)
    result_df["total_p90"] = np.percentile(total_matrix, 90, axis=0)

    # Mean total time across simulations — better aggregation than mean-rank
    # because it preserves time-domain signal (rank averaging is lossy and
    # regresses strong predictions toward mid-field)
    result_df["mean_total_sec"] = total_matrix.mean(axis=0)
    result_df["mean_time_rank"] = result_df["mean_total_sec"].rank(method="min").astype(int)

    # Pack statistics from simulation
    if sim_pack_stats:
        for col, vals in sim_pack_stats.items():
            result_df[col] = vals

    # Store diagnostics as DataFrame attribute for downstream access
    if diagnostics:
        result_df.attrs["sim_diagnostics"] = diagnostics

    # Sort by mean simulated time (preserves more signal than mean-rank)
    result_df = result_df.sort_values("mean_total_sec").reset_index(drop=True)

    top3_wins = result_df["prob_win"].nlargest(3).values
    logger.info(f"Simulation complete. Top 3 win probabilities: {top3_wins}")

    return result_df


# ── Output Formatting ────────────────────────────────────────────────


def format_simulation_output(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format simulation output for display/export.

    Returns a clean DataFrame with key columns for reporting.
    """
    from .utils_time import seconds_to_hms

    output_df = sim_df.copy()

    # Format probabilities as percentages
    for col in ["prob_win", "prob_podium", "prob_top5", "prob_top10", "prob_top20"]:
        if col in output_df.columns:
            output_df[col + "_pct"] = (output_df[col] * 100).round(1)

    # Format time intervals as HH:MM:SS
    for col in ["total_p10", "total_p50", "total_p90"]:
        if col in output_df.columns:
            output_df[col + "_hms"] = output_df[col].apply(
                lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None
            )

    # Format sim pack stats as percentage
    if "sim_front_pack_pct" in output_df.columns:
        output_df["sim_front_pack_pct_fmt"] = (output_df["sim_front_pack_pct"] * 100).round(0).astype(int)

    # Select display columns — focused on probabilities and ranking,
    # not individual split predictions (those are inputs to the sim, not outputs)
    display_cols = [
        "predicted_rank",
        "mean_time_rank",
        "athlete_full_name",
        "athlete_country_name",
        "prob_win_pct",
        "prob_podium_pct",
        "prob_top5_pct",
        "prob_top10_pct",
        "expected_rank",
        "total_p50_hms",
        "pred_total_hms",
    ]

    available_cols = [c for c in display_cols if c in output_df.columns]

    result = output_df[available_cols].copy()
    result = result.rename(columns={
        "predicted_rank": "Det. Rank",
        "mean_time_rank": "Sim Rank",
        "athlete_full_name": "Athlete",
        "athlete_country_name": "Country",
        "prob_win_pct": "Win %",
        "prob_podium_pct": "Podium %",
        "prob_top5_pct": "Top5 %",
        "prob_top10_pct": "Top10 %",
        "expected_rank": "E[Rank]",
        "sim_front_pack_pct_fmt": "Front Pk %",
        "total_p50_hms": "Sim Median",
        "pred_total_hms": "Det. Total",
    })

    return result


def print_sim_diagnostics(sim_df: pd.DataFrame) -> None:
    """
    Print simulation diagnostics stored in sim_df.attrs['sim_diagnostics'].

    Call this after run_monte_carlo() to see per-split bias, front pack rate
    comparison, and pack formation statistics.
    """
    diag = sim_df.attrs.get("sim_diagnostics")
    if not diag:
        print("  No simulation diagnostics available.")
        return

    print("\n--- Simulation Diagnostics ---")

    # Per-split bias
    if "per_split_bias" in diag:
        print("\n  Per-Split Bias (sim mean - prediction):")
        for split, bias in diag["per_split_bias"].items():
            print(f"    {split:>5s}: {bias:+.2f}s")
        print(f"    {'total':>5s}: {diag.get('total_bias', 0):+.2f}s")

    # Front pack rate comparison
    if "front_pack_rate_comparison" in diag:
        fpc = diag["front_pack_rate_comparison"]
        print(f"\n  Front Pack Rate:")
        print(f"    Historical mean:  {fpc['hist_mean']:.3f}")
        print(f"    Simulation mean:  {fpc['sim_mean']:.3f}")
        print(f"    Correlation:      {fpc['correlation']:.3f}")

    # Pack count distribution
    if "avg_packs_per_sim" in diag:
        print(f"\n  Pack Formation:")
        print(f"    Avg packs/sim: {diag['avg_packs_per_sim']:.1f}")
        if "pack_count_distribution" in diag:
            dist = diag["pack_count_distribution"]
            total_sims = sum(dist.values())
            # Show top 5 most common pack counts
            sorted_counts = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
            for n_packs, count in sorted_counts:
                pct = count / total_sims * 100
                print(f"    {n_packs} packs: {pct:.1f}%")
