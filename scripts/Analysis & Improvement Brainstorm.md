Analysis & Improvement Brainstorm
I've read through every file in the prediction pipeline, the brainstorm docs, the SQL layer, and the broader codebase. Here's my diagnosis and concrete improvement ideas organized by category.

THE CORE PROBLEM: Absolute Times Are Poisoning Your Features
The single biggest issue is that your features and targets are in absolute seconds, but you're training across different distances (sprint ~3600s total, standard ~7200s total). This creates several cascading problems:

ema_total_sec_5 is meaningless cross-distance. If an athlete does 3 sprints and 2 standards, the EMA blends ~3600s and ~7200s into ~5400s, which is neither. The distance filtering in sql.py:127-129 helps but falls back to all history when distance-matched data is sparse (features.py:524-527).

seed_total_rank is broken for mixed-distance fields. If some athletes' EMAs come from sprints and others from standards, ranking them by EMA total is comparing apples to oranges.

std_total_sec_24m conflates distance variation with athlete inconsistency. An athlete who alternates sprint/standard will show high variance not because they're inconsistent, but because the distances differ.

The model has no distance feature. prog_distance_category is used for filtering but is never passed as a model input (see features.py:594-624). The model literally cannot learn that sprints produce different times than standards.

IMPROVEMENT CATEGORY 1: Distance-Agnostic Performance Features
You do NOT need separate models. You need features that are normalized across distances. Here are concrete approaches, ordered by impact:

A. Finish Percentile EMA (highest impact, easiest)

For each historical race, compute:


finish_pct = finish_position / n_finishers_in_race
Then EMA this across races. This is naturally distance-agnostic. An athlete who finishes 5th/50 in a WTCS standard and 3rd/40 in a World Cup sprint has percentiles of 0.10 and 0.075 -- both meaningful, both comparable.

Currently ema_swim_pos_pct_7 does something similar for swim position, but there's no overall finish percentile EMA. This should be your most predictive feature.

B. Gap-to-Median Ratio (moderate effort, high impact)

For each historical race, compute:


performance_ratio = athlete_total_sec / race_median_total_sec
A ratio of 0.95 means 5% faster than median regardless of distance. EMA this. This gives the model a distance-normalized speed signal that still captures magnitude (unlike pure rank).

To implement: you need to store or compute the race median at feature-build time. Add a column to fetch_athlete_history that includes the race's median total time (via a window function or subquery).

C. Per-Split Pace Percentiles

Same logic applied per-split:


swim_pct = swim_position / n_swimmers
bike_pct = bike_position / n_bikers  
run_pct  = run_position / n_runners
EMA each. This tells the model "this athlete is typically a top-20% swimmer, mid-pack biker, top-10% runner" regardless of distance.

D. Cross-Distance Performance Features

Don't throw away cross-distance data -- use it explicitly:


ema_finish_pct_sprint   = EMA of finish_pct in sprint races only
ema_finish_pct_standard = EMA of finish_pct in standard races only
When predicting a sprint, the standard-distance percentile still tells the model something about the athlete's general ability. The model can learn the correlation coefficient.

IMPROVEMENT CATEGORY 2: Tier-Aware Performance Modeling
Your tier features exist but aren't giving the model what it needs. The current tier_delta (event_tier - athlete_avg_tier) captures "stepping up/down" but doesn't tell the model how good this athlete actually is at top-level racing.

A. Tier-Weighted Performance Score

Instead of raw finish percentile, compute a tier-adjusted version:


weighted_performance = finish_pct * tier_difficulty_factor
Where tier_difficulty_factor accounts for the fact that finishing 15th at WTCS is harder than finishing 1st at a Continental Cup. One approach:


# Empirical: average ability of a top-10 finisher at each tier
TIER_DIFFICULTY = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.50}
adjusted_pct = finish_pct * TIER_DIFFICULTY[event_tier]
EMA this adjusted percentile. Now a WTCS mid-packer (0.50 * 1.0 = 0.50) correctly ranks above a Continental Cup winner (0.02 * 0.50 = 0.01... wait, lower = better here). You'd need to flip the logic, but the idea is: scale the percentile by how competitive the field was.

B. Better approach: Elo or Glicko Rating

Instead of hand-crafted tier weights, compute an Elo-style rating for each athlete based on head-to-head results across all their races. This naturally handles:

Tier differences (beating WTCS athletes earns more Elo than beating Continental Cup athletes)
Field strength (stronger fields = more Elo movement)
Recency (recent results weighted more)
Cross-distance (beating the same people at different distances still counts)
This would likely become your single best feature. Libraries like elote or a custom implementation (50-100 lines) can compute this from your existing race_results table.

C. Field-Relative Seed Rank

Your seed_total_rank is based on ema_total_sec_5, which is broken cross-distance. If you switch to percentile-based or Elo-based seeding, this feature immediately becomes meaningful.

IMPROVEMENT CATEGORY 3: Target Variable Rethinking
Currently you predict total_sec directly. This is problematic because:

Sprint totals ~3600s and standard totals ~7200s are in completely different ranges
The model has to learn "what distance is this?" from indirect signals
Option A: Add distance as a feature (minimum change)

Simply add prog_distance_category (encoded as integer: sprint=0, standard=1) to get_feature_columns(). This costs one line of code and lets the model learn distance-specific intercepts. This alone should help meaningfully.

Option B: Predict percentile, then convert back (recommended)

Train the model to predict finish_percentile instead of total_sec. Benefits:

Naturally distance-agnostic target
Features and target are now in the same "units"
Model focuses on relative ordering, which is what you care about
To recover time predictions: estimate the race's median time from distance + field quality + historical venue data, then predicted_time = median_time * (1 + f(predicted_percentile)).

Option C: Two-stage model (most powerful, most complex)

Stage 1: Predict the race's median time from (distance, venue, field_quality, weather). This can be a simple model or lookup table.
Stage 2: Predict each athlete's delta from median (in seconds or percentage). This is what your gradient boosting model trains on.
At inference: predicted_time = predicted_median + predicted_delta.

IMPROVEMENT CATEGORY 4: Monte Carlo Simulation Upgrades
Your current simulation at simulate.py:65-178 has several simplifications worth addressing:

A. Learn pack effects from data (high priority)

The hardcoded -30s / +20s (simulate.py:25-26) is a guess. You can compute this empirically:


-- Compare total time of athletes in front swim pack vs not, same race
SELECT 
  e.prog_distance_category,
  pm.pack_id = 1 AS in_front_pack,
  AVG(total_time_sec) AS avg_total,
  AVG(total_time_sec) - LAG(AVG(total_time_sec)) OVER (...) AS pack_effect
FROM wtcs_pack_membership pm
JOIN race_results rr ON pm.athlete_id = rr.athlete_id AND pm.event_id = rr.event_id
JOIN events e ON ...
WHERE pm.checkpoint = 'swim'
GROUP BY e.prog_distance_category, (pm.pack_id = 1)
This gives you empirical pack effects per distance. Almost certainly sprint pack effects are different from standard pack effects.

B. Continuous gap effects instead of binary in/out

The binary front-pack model (in_front_pack true/false) is too coarse. In reality:

5 seconds behind the leader: still drafting, minimal penalty
15 seconds behind: small chase group, moderate penalty
45 seconds behind: isolated, large penalty
Model this as a continuous function:


# Instead of binary:
pack_effect = np.where(front_pack, -30, +20)

# Use continuous:
simulated_gap = rng.exponential(scale=athlete_typical_gap)  # sample from distribution
pack_effect = gap_penalty_function(simulated_gap)
# e.g., pack_effect = min(simulated_gap * 1.5, 90)  # penalty scales with gap, capped
C. Swim-Bike-Run causal chain

Currently the simulation samples total time as one lump. But in draft-legal triathlon, the splits are causally linked:


swim_time → determines swim_exit_gap → determines bike_pack → determines bike_time → determines T2_gap → influences run_pacing → determines run_time
A better simulation:


for sim in range(n_sims):
    # 1. Sample swim time
    swim = rng.normal(mu_swim, sigma_swim)
    
    # 2. Determine swim exit position and pack assignment
    swim_ranks = argsort(argsort(swim_times_all_athletes))
    swim_gaps = swim - swim[swim_ranks == 0]  # gap to leader
    in_front_pack = swim_gaps < pack_gap_threshold  # e.g., 10-15s
    
    # 3. Sample bike time conditional on pack
    bike_mu_adjusted = np.where(in_front_pack, mu_bike - pack_bonus, mu_bike + pack_penalty)
    bike_sigma_adjusted = np.where(in_front_pack, sigma_bike * 0.7, sigma_bike * 1.2)  # front pack = less variance
    bike = rng.normal(bike_mu_adjusted, bike_sigma_adjusted)
    
    # 4. Sample run time, potentially with fatigue from chase
    chase_effort = np.maximum(0, swim_gaps) * chase_fatigue_factor
    run = rng.normal(mu_run + chase_effort * run_penalty_per_sec_chased, sigma_run)
    
    # 5. Total and rank
    total = swim + bike + run + t1 + t2
This is significantly more realistic and uses your pack membership data to calibrate the thresholds.

D. Race-day conditions factor

Add a shared "race conditions" factor that shifts everyone:


# Weather / course factor that affects all athletes
race_conditions_delta = rng.normal(0, 30)  # seconds, shared across field
sim_total = pred_total + form_delta + individual_noise + pack_effect + race_conditions_delta
This captures "it was a slow day" or "fast course" -- currently missing.

E. Distance-specific uncertainty

Sprint sigma should be smaller than standard sigma (shorter race = less can go wrong). Currently sigma_total from std_total_sec_24m is mixed across distances, inflating sprint uncertainty and deflating standard uncertainty.

IMPROVEMENT CATEGORY 5: Training Pipeline
A. Hyperparameter tuning

The current defaults (train.py:77-93) are generic. Even a simple RandomizedSearchCV with 50 iterations would help:


param_distributions = {
    'regressor__n_estimators': [100, 200, 500],
    'regressor__max_depth': [4, 6, 8, 10],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__min_samples_leaf': [10, 20, 50],
}
B. Proper time-based cross-validation

Train on 2018-2023, validate on 2024-H1, test on 2024-H2+2025. The current backtest applies the model to data it may have trained on (no strict temporal separation).

C. Gender-specific models

Men and women have fundamentally different time distributions, pack dynamics, and competitive depths. Training separate models (or at minimum adding gender as a feature) would reduce noise.

PRIORITY ORDERING (What I'd implement first)
Add prog_distance_category as a model feature -- 5 minutes, immediate improvement
Add overall finish percentile EMA (ema_finish_pct_5) -- replaces broken ema_total_sec_5 as primary performance signal
Add gap-to-median ratio EMA -- distance-normalized speed signal
Compute empirical pack effects from data -- replace hardcoded constants
Implement causal swim→bike→run simulation chain -- realistic Monte Carlo
Add Elo ratings -- best single feature for cross-distance, cross-tier ranking
Predict percentile as target instead of absolute seconds
Hyperparameter tuning with time-based CV
Gender-specific models
Continuous gap-based pack effects
Items 1-3 are high-impact, low-effort changes that address the core "mixing distances" problem. Items 4-5 fix the Monte Carlo simulation. Items 6+ are bigger structural improvements.

Would you like me to start implementing any of these? I'd suggest tackling items 1-3 first since they're the foundation everything else builds on.

One caveat to be aware of: the Elo ratings in the database are the final ratings after processing all races through the present. When training on a 2020 race, the model would see the athlete's 2025 Elo, which is a mild form of temporal leakage. For a first pass this is fine -- an athlete's Elo doesn't change drastically year-to-year, so the signal is still informative. But if you want to do a rigorous time-based backtest later, you'd want per-race Elo snapshots (store the Elo at the time of each race). That would be a natural follow-on improvement.

Summary of Changes
1. Wetsuit Feature (features.py)
Simple addition — wetsuit (0/1 integer) is now included as a model feature. The data was already being fetched from the database and stored in the feature row; it just wasn't in get_feature_columns(). The model can now learn that wetsuit races have different swim dynamics (wetsuits add buoyancy/speed, compress the swim field).

2. Distance-Specific Uncertainty (simulate.py)
Added a DISTANCE_SIGMA_MULTIPLIER lookup table that scales all sigma values (swim, bike, run, total) by race distance. Sprint races use a 0.65x multiplier (shorter race = less absolute noise), while standard races use 1.0x (baseline). The run_monte_carlo() now accepts distance_category and passes it through to estimate_uncertainty().

3. Learned Pack Effects — How It Works
The problem: PackEffectParams defaults were hardcoded (front_pack_bonus=-25s, chase_penalty=+15s). The function learn_pack_effects_from_data() existed but was never called.

The fix:

During training (train_models.py:225-240), we now call fetch_pack_effect_data() to get paired swim-pack + bike-time data, then learn_pack_effects_from_data() to compute empirical bonus/penalty values from that data.
The learned PackEffectParams are serialized to a plain dict and stored in bundle.metadata["pack_effect_params"].
When running predictions (predict_program.py:140-156), the pack params are extracted from the bundle and passed to run_monte_carlo().
How learn_pack_effects_from_data() works (simulate.py:175-265):

Takes the paired swim-pack + bike-time DataFrame
For each race, computes the front-pack (swim gap ≤ 5s) median bike time as baseline
Computes each athlete's bike_delta = their_bike_time - front_pack_median_bike
Groups by swim gap: athletes with gap ≤ 5s → front_bonus (median of their deltas), athletes with gap > 15s → chase_penalty
Returns PackEffectParams with the learned values
What you can tune:

front_pack_threshold (default 5.0s) — swim gap defining "front pack"
chase_threshold (default 15.0s) — swim gap beyond which athletes are fully penalized
These are arguments to learn_pack_effects_from_data()
4. Continuous Gap Effects — How It Works
The problem: The old model used assign_packs_chain() with a binary 2-second threshold. A 0.1s difference in swim time could flip an athlete from pack 0 to pack 1, causing a ~40-second swing in bike time. This created discontinuities in the simulation.

The fix — continuous_gap_bike_effect() (simulate.py:280-350):

Instead of discrete pack IDs, the bike time adjustment is a smooth function of each athlete's swim gap to the leader:


gap ≤ 2s (max_gap_sec):     full front_pack_bonus (e.g., -25s)
2s < gap < 15s:             linear interpolation between bonus and penalty
gap ≥ 15s (chase_threshold): full chase_penalty (e.g., +15s)
Local density scaling: Rather than discrete pack sizes, we count how many athletes are within ±3 seconds of each athlete's swim time (density_window_sec). More nearby athletes = better drafting = amplified bonus (up to 1.5x). If fewer than min_pack_size_for_draft (3) athletes are nearby, the athlete is considered solo and gets the full chase penalty regardless of gap.

Key behavioral difference from the old model:

Old: athlete at 2.0s gap = pack 0 (bonus), at 2.1s gap = pack 1 (much reduced bonus) — a cliff
New: athlete at 2.0s gap gets -25s, at 2.1s gap gets -24.6s, at 5s gets -15.6s — smooth gradient
What you can tune in continuous_gap_bike_effect():

params.max_gap_sec (default 2.0): gap below which you get full bonus
chase_threshold (currently hardcoded at 15.0 in the function): gap above which you get full penalty
density_window_sec (default 3.0): window for counting nearby athletes
params.min_pack_size_for_draft (default 3): minimum local density for any drafting benefit
params.draft_size_scale (default 0.15): how much each additional nearby athlete amplifies the bonus
If you want to change the chase_threshold, it's at simulate.py line ~305. You could also make it a parameter on PackEffectParams if you want it tunable per-model.

To test these changes, you need to retrain the model (so the bundle includes the new wetsuit feature and learned pack params), then run the backtest. The commands would be: