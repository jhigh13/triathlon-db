import numpy as np

from tri_analysis.wtcs_pack_metrics import assign_pack_ids_chain


def test_assign_pack_ids_chain_empty():
    out = assign_pack_ids_chain([], max_gap_to_prev_sec=2)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_assign_pack_ids_chain_all_same_pack_when_gaps_le_threshold():
    # gaps: 0, 2, 1 => all <= 2
    elapsed = [100, 100, 102, 103]
    out = assign_pack_ids_chain(elapsed, max_gap_to_prev_sec=2)
    assert out.tolist() == [0, 0, 0, 0]


def test_assign_pack_ids_chain_new_pack_when_gap_gt_threshold():
    # gaps: 2 (same), 3 (new pack), 0 (same as pack 1)
    elapsed = [100, 102, 105, 105]
    out = assign_pack_ids_chain(elapsed, max_gap_to_prev_sec=2)
    assert out.tolist() == [0, 0, 1, 1]


def test_assign_pack_ids_chain_exact_threshold_does_not_split():
    elapsed = [100, 102, 104]
    out = assign_pack_ids_chain(elapsed, max_gap_to_prev_sec=2)
    assert out.tolist() == [0, 0, 0]
