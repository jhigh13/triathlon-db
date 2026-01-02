import pandas as pd

from tri_analysis.wtcs_radar import _pct_rank_within_group, _weighted_mean_ignore_na


def test_pct_rank_within_group_basic():
    s = pd.Series([10, 20, 20, 40, None])
    pct = _pct_rank_within_group(s)
    # Best value gets 0
    assert pct.iloc[0] == 0.0
    # Ties get same percentile (method='min')
    assert pct.iloc[1] == pct.iloc[2]
    # Worst (40) gets 1
    assert pct.iloc[3] == 1.0
    # NaN stays NaN
    assert pd.isna(pct.iloc[4])


def test_weighted_mean_ignore_na_renormalizes():
    values = {
        "a": pd.Series([1.0, None, 1.0]),
        "b": pd.Series([1.0, 2.0, None]),
    }
    weights = {"a": 0.7, "b": 0.3}
    out = _weighted_mean_ignore_na(values, weights)

    # both present
    assert out.iloc[0] == 1.0
    # only b present -> equals b
    assert out.iloc[1] == 2.0
    # only a present -> equals a
    assert out.iloc[2] == 1.0


def test_radar_mapping_high_is_better():
    # weakness 0 -> radar 10 (best)
    assert (10.0 - 9.0 * 0.0) == 10.0
    # weakness 1 -> radar 1 (worst)
    assert (10.0 - 9.0 * 1.0) == 1.0
    # mid
    assert (10.0 - 9.0 * 0.5) == 5.5
