import math
import numpy as np
import pytest

from dvq.statistical import (
    deng_entropy_short_sequence,
    unique_window_ratio,
    normalize_distribution,
    kmer_distribution_from_sequences,
    js_divergence_from_distributions,
    wasserstein_from_distributions,
    kl_divergence_from_distributions,
    compute_kl_distances,
    compute_jaccard_distances,
)


def test_normalize_distribution():
    raw = {"AAAA": 2.0, "AAAT": 2.0, "CCCC": 0.0, "GGGG": -1.0}
    norm = normalize_distribution(raw)
    assert set(norm.keys()) == {"AAAA", "AAAT"}
    assert math.isclose(norm["AAAA"], 0.5)
    assert math.isclose(norm["AAAT"], 0.5)


def test_kmer_distribution_boundary_preservation():
    # Two short sequences "AAAA" and "TTTT" with k=3 should only contain "AAA" and "TTT",
    # never boundary-spanning k-mers like "AAT" or "ATT".
    seqs = ["AAAA", "TTTT"]
    dist = kmer_distribution_from_sequences(seqs, k=3)
    assert set(dist.keys()) == {"AAA", "TTT"}
    assert math.isclose(dist["AAA"], 0.5)
    assert math.isclose(dist["TTT"], 0.5)


def test_distributions_identical_zero_distance():
    seqs = ["ACGTACGTACGT", "GCTAGCTAGCTA", "TTAAAGGCCCTT"]
    dist = kmer_distribution_from_sequences(seqs, k=4)
    assert math.isclose(js_divergence_from_distributions(dist, dist), 0.0, abs_tol=1e-9)
    assert math.isclose(wasserstein_from_distributions(dist, dist), 0.0, abs_tol=1e-7)
    kl_res = compute_kl_distances(dist, dist)
    assert kl_res["symmetric_kl"] < 1e-5


def test_weighted_amplification_over_unweighted():
    pop_a = ["ACGTACGTACGTACGT", "TCGATCGATCGATCGA", "AAAACCCCGGGGTTTT"]
    pop_b = ["ACGTACGTACGTACGT", "TCGATCGATCGATCGA", "AAAACCCCGGGGTTTT", "CCCCCCCCCCCCCCCC"]
    weights_a = [10.0, 10.0, 10.0]
    weights_b = [1.0, 1.0, 1.0, 200.0]

    dist_a_unw = kmer_distribution_from_sequences(pop_a, k=4)
    dist_b_unw = kmer_distribution_from_sequences(pop_b, k=4)

    dist_a_w = kmer_distribution_from_sequences(pop_a, k=4, weights=weights_a)
    dist_b_w = kmer_distribution_from_sequences(pop_b, k=4, weights=weights_b)

    js_unw = js_divergence_from_distributions(dist_a_unw, dist_b_unw)
    js_w = js_divergence_from_distributions(dist_a_w, dist_b_w)
    assert 0.0 <= js_unw < js_w <= 1.0

    w_unw = wasserstein_from_distributions(dist_a_unw, dist_b_unw)
    w_w = wasserstein_from_distributions(dist_a_w, dist_b_w)
    assert w_w > w_unw > 0.0


def test_short_sequence_deng_entropy_and_uwr():
    seq_diverse = "ACGTAGCTAGCTAGCTGACTACGT"
    seq_repeat = "AAAAAAAAAAAAAAAAAAAAAAAA"
    ent_div = deng_entropy_short_sequence(seq_diverse, chunk_size=3)
    ent_rep = deng_entropy_short_sequence(seq_repeat, chunk_size=3)
    assert ent_div > ent_rep >= 0.0
    assert unique_window_ratio(seq_diverse, chunk_size=3) > unique_window_ratio(seq_repeat, chunk_size=3)
    assert np.isnan(deng_entropy_short_sequence("AC", chunk_size=3))


def test_compute_jaccard_distances():
    seqs_a = ["ACGTACGTACGTACGT", "GCTAGCTAGCTAGCTA"]
    jacc = compute_jaccard_distances(seqs_a, seqs_a, k_sizes=(3, 6))
    assert math.isclose(jacc["jaccard_sim_k3"], 1.0)
    assert math.isclose(jacc["jaccard_dist_k3"], 0.0)
