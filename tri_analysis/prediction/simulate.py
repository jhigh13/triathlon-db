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
) -> PackEffectParams | None:
    """
    Look up distance-specific pack params from bundle metadata.

    Falls back to the overall pack params if no distance-specific params exist.
    Returns None if the distance is non-drafting (middle/long).
    """
    if distance_category is None:
        # Fall back to overall params
        d = bundle_metadata.get("pack_effect_params")
        return pack_params_from_dict(d) if d else None

    dist_key = distance_category.lower().strip()
    if dist_key == "olympic":
        dist_key = "standard"

    # Non-drafting distances: no pack effects
    if dist_key not in DRAFT_LEGAL_DISTANCES:
        logger.info(f"Distance '{dist_key}' is non-drafting, disabling pack effects")
        return None

    # Try distance-specific params first
    by_distance = bundle_metadata.get("pack_effect_params_by_distance", {})
    if dist_key in by_distance:
        params = pack_params_from_dict(by_distance[dist_key])
        # Apply distance-specific gap threshold
        params.max_gap_sec = DISTANCE_PACK_GAP_SEC.get(dist_key, 3.0)
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
        params.max_gap_sec = DISTANCE_PACK_GAP_SEC.get(dist_key, 3.0)
        params.distance_category = dist_key
        logger.info(
            f"No distance-specific pack params for '{dist_key}', "
            f"using overall (gap overridden to {params.max_gap_sec:.1f}s)"
        )
        return params

    return None


def get_distance_merge_params(
    bundle_metadata: dict,
    distance_category: str | None,
) -> MergeParams | None:
    """
    Look up distance-specific merge params from bundle metadata.

    Falls back to overall merge params. Returns None for non-drafting distances.
    """
    if distance_category is None:
        d = bundle_metadata.get("merge_params")
        return merge_params_from_dict(d) if d else None

    dist_key = distance_category.lower().strip()
    if dist_key == "olympic":
        dist_key = "standard"

    if dist_key not in DRAFT_LEGAL_DISTANCES:
        return None

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
) -> pd.DataFrame:
    """
    Add per-athlete, per-split uncertainty (sigma) columns.

    Uses per-athlete std_total_sec_24m if available, then derives per-split
    sigmas proportionally (swim ~15s, bike ~45s, run ~30s baseline).
    Scales all sigmas by a distance-specific multiplier (sprint = less noise).

    Args:
        pred_df: DataFrame with predictions
        sigma_total_col: Column name for athlete's historical std dev
        default_sigma: Default sigma if not available
        distance_category: Race distance ('sprint', 'standard', etc.) for
                          scaling uncertainty. None defaults to standard.

    Returns:
        DataFrame with sigma_total, sigma_swim, sigma_bike, sigma_run added
    """
    df = pred_df.copy()

    # Distance multiplier: sprint races have less absolute variance
    dist_mult = 1.0
    if distance_category:
        dist_mult = DISTANCE_SIGMA_MULTIPLIER.get(
            distance_category.lower().strip(), 1.0
        )
        if dist_mult != 1.0:
            logger.info(
                f"Distance-specific uncertainty: {distance_category} → "
                f"sigma multiplier={dist_mult:.2f}"
            )

    if sigma_total_col in df.columns:
        df["sigma_total"] = df[sigma_total_col].fillna(default_sigma)
        df["sigma_total"] = df["sigma_total"].clip(30, 300)
    else:
        df["sigma_total"] = default_sigma

    # Apply distance multiplier
    df["sigma_total"] = df["sigma_total"] * dist_mult

    # Scale per-split sigmas relative to athlete's total uncertainty
    ratio = df["sigma_total"] / (DEFAULT_SIGMA_TOTAL * dist_mult)
    df["sigma_swim"] = DEFAULT_SIGMA_SWIM * dist_mult * ratio
    df["sigma_t1"] = DEFAULT_SIGMA_T1 * dist_mult * ratio
    df["sigma_bike"] = DEFAULT_SIGMA_BIKE * dist_mult * ratio
    df["sigma_t2"] = DEFAULT_SIGMA_T2 * dist_mult * ratio
    df["sigma_run"] = DEFAULT_SIGMA_RUN * dist_mult * ratio

    return df


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
                    Higher = more correlation across splits.
        distance_category: Race distance ('sprint', 'standard', etc.) for
                          scaling uncertainty. None defaults to standard.

    Returns:
        DataFrame with original columns plus:
        - prob_win, prob_podium, prob_top5, prob_top10, prob_top20
        - expected_rank, rank_p10, rank_p50, rank_p90
        - total_p10, total_p50, total_p90 (seconds)
    """
    if pred_df.empty:
        logger.warning("Empty pred_df, returning empty DataFrame")
        return pred_df.copy()

    df = estimate_uncertainty(pred_df, distance_category=distance_category)
    n_athletes = len(df)
    rng = np.random.default_rng(random_state)

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

    # Pre-compute baseline pack effect from predicted swim + T1 times.
    # In reality, pack formation at bike mount depends on swim exit + T1.
    # This is the pack effect "baked into" the predictions. We subtract it
    # so the simulation only adds the DEVIATION from baseline.
    pred_bike_entry = pred_swim + pred_t1
    if use_pack_effects:
        baseline_pack_effect = continuous_gap_bike_effect(pred_bike_entry, pack_params)
    else:
        baseline_pack_effect = np.zeros(n_athletes)

    # Variance decomposition
    form_std = np.sqrt(form_share)   # Fraction of std from shared form
    noise_std = np.sqrt(1.0 - form_share)  # Fraction from split-specific noise

    # Track outcomes
    rank_matrix = np.zeros((n_sims, n_athletes), dtype=np.int32)
    total_matrix = np.zeros((n_sims, n_athletes), dtype=np.float64)
    # Track pack membership: how often each athlete ends up in the front pack
    front_pack_count = np.zeros(n_athletes, dtype=np.int32)
    # Track bike pack adjustment magnitudes
    pack_effect_sum = np.zeros(n_athletes, dtype=np.float64)

    merge_label = ""
    if merge_params is not None:
        merge_label = f", dynamic_merge=True, breakaway_bias={breakaway_bias:.1f}"

    logger.info(
        f"Running {n_sims} causal-chain simulations for {n_athletes} athletes "
        f"(pack_effects={use_pack_effects}, form_share={form_share:.0%}, "
        f"bonus={pack_params.front_pack_bonus_sec:.1f}s, "
        f"penalty={pack_params.chase_penalty_sec:.1f}s{merge_label}, "
        f"t1t2_modeled={'yes' if has_t1 and has_t2 else 'fallback'})"
    )

    for sim in range(n_sims):
        # Shared form factor: good day / bad day, correlated across splits
        # Positive = slower (bad day), Negative = faster (good day)
        form_factor = rng.normal(0, 1, size=n_athletes)

        # ── SWIM ──
        swim_form = form_factor * sigma_swim * form_std
        swim_noise = rng.normal(0, sigma_swim * noise_std)
        sim_swim = pred_swim + swim_form + swim_noise

        # ── T1 (transition 1: swim-to-bike) ──
        # T1 varies per athlete and affects pack formation at bike mount
        t1_form = form_factor * sigma_t1 * form_std
        t1_noise = rng.normal(0, sigma_t1 * noise_std)
        sim_t1 = pred_t1 + t1_form + t1_noise

        # ── PACK EFFECT ON BIKE (swim + T1 → bike entry order → pack formation) ──
        # Use swim + T1 for pack formation: this is when athletes enter the bike course
        sim_bike_entry = sim_swim + sim_t1
        bike_pack_adjustment = np.zeros(n_athletes)
        if use_pack_effects:
            if merge_params is not None:
                # Dynamic merging: chase packs can absorb solo riders ahead
                sim_pack_effect = apply_pack_merges(
                    sim_bike_entry, pack_params, merge_params, rng, breakaway_bias
                )
            else:
                # Static: one-shot from bike entry gaps
                sim_pack_effect = continuous_gap_bike_effect(sim_bike_entry, pack_params)
            # Center around baseline so mean matches prediction
            bike_pack_adjustment = sim_pack_effect - baseline_pack_effect

        # Track pack statistics
        if use_pack_effects:
            # Front pack = athletes with negative or near-zero pack effect (getting draft benefit)
            front_pack_count += (sim_pack_effect <= 0).astype(np.int32)
            pack_effect_sum += bike_pack_adjustment

        # ── BIKE (with pack effect) ──
        bike_form = form_factor * sigma_bike * form_std
        bike_noise = rng.normal(0, sigma_bike * noise_std)
        sim_bike = pred_bike + bike_form + bike_noise + bike_pack_adjustment

        # ── T2 (transition 2: bike-to-run) ──
        t2_form = form_factor * sigma_t2 * form_std
        t2_noise = rng.normal(0, sigma_t2 * noise_std)
        sim_t2 = pred_t2 + t2_form + t2_noise

        # ── RUN (individual effort, no pack dynamics) ──
        run_form = form_factor * sigma_run * form_std
        run_noise = rng.normal(0, sigma_run * noise_std)
        sim_run = pred_run + run_form + run_noise

        # ── TOTAL (all 5 segments) ──
        sim_total = sim_swim + sim_t1 + sim_bike + sim_t2 + sim_run
        total_matrix[sim, :] = sim_total

        # Rank (1 = fastest)
        ranks = np.argsort(np.argsort(sim_total)) + 1
        rank_matrix[sim, :] = ranks

    # Compute pack statistics
    sim_pack_stats = {
        "sim_front_pack_pct": front_pack_count / n_sims,
        "sim_avg_pack_effect": pack_effect_sum / n_sims,
    }

    return _aggregate_results(df, rank_matrix, total_matrix, sim_pack_stats)


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

    # Pack statistics from simulation
    if sim_pack_stats:
        for col, vals in sim_pack_stats.items():
            result_df[col] = vals

    # Sort by expected rank
    result_df = result_df.sort_values("expected_rank").reset_index(drop=True)

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
        "athlete_full_name",
        "athlete_country_name",
        "prob_win_pct",
        "prob_podium_pct",
        "prob_top5_pct",
        "prob_top10_pct",
        "expected_rank",
        "sim_front_pack_pct_fmt",
        "total_p50_hms",
        "pred_total_hms",
    ]

    available_cols = [c for c in display_cols if c in output_df.columns]

    result = output_df[available_cols].copy()
    result = result.rename(columns={
        "predicted_rank": "Det. Rank",
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
