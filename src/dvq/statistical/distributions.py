"""Multi-sequence k-mer distribution and divergence functions.

Extends DVQ's single-sequence statistical metrics (Jensen-Shannon divergence,
Hamming-cost Wasserstein optimal transport, Kullback-Leibler / Jeffreys
divergence, and MinHash Jaccard similarity) to arbitrary collections of DNA/RNA
sequences (e.g., multi-FASTA gene families, metagenomic contigs, viral
quasispecies, batches of generated synthetic sequences, or immune repertoires)
with optional sequence abundance weighting.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .wasserstein_distance import _build_cost_matrix, _get_top_kmers, _solve_emd2

try:
    from sourmash import MinHash as _MinHash
except ImportError:
    _MinHash = None


def normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    """Remove nonpositive or nonfinite mass and rescale k-mer frequencies to sum to 1."""
    clean = {
        k: float(v)
        for k, v in distribution.items()
        if np.isfinite(float(v)) and float(v) > 0
    }
    total = float(sum(clean.values()))
    if total <= 0:
        raise ValueError("Distribution has no positive mass.")
    return {k: v / total for k, v in clean.items()}


def kmer_distribution_from_sequences(
    sequences: Iterable[str],
    k: int = 6,
    weights: Iterable[float] | None = None,
    normalize_per_sequence: bool = True,
) -> dict[str, float]:
    """Build a pooled k-mer probability distribution across a collection of sequences
    while preserving individual sequence boundaries.

    Unlike concatenating sequences (which introduces artificial junction k-mers
    at sequence boundaries), each sequence is windowed independently.

    Parameters
    ----------
    sequences : Iterable[str]
        Collection of nucleotide sequences (e.g., multi-FASTA records, contigs,
        synthetic generated sequences, or receptor sequences).
    k : int, default=6
        k-mer length.
    weights : Iterable[float] | None, default=None
        Optional abundance, copy-number, or read-depth weight per sequence.
        If None, all sequences receive equal weight (1.0).
    normalize_per_sequence : bool, default=True
        If True, each sequence's k-mer counts are normalized by its window count
        before multiplying by its weight, preventing longer sequences from
        contributing disproportionately more probability mass.

    Returns
    -------
    dict[str, float]
        Normalized probability distribution over observed k-mers.
    """
    sequences_list = list(sequences)
    weights_list = [1.0] * len(sequences_list) if weights is None else list(weights)

    if len(sequences_list) != len(weights_list):
        raise ValueError("sequences and weights must have the same length.")

    pooled_counts: Counter = Counter()

    for sequence, weight in zip(sequences_list, weights_list):
        if not isinstance(sequence, str):
            continue
        sequence = sequence.strip().upper()
        if len(sequence) < k:
            continue
        if not set(sequence).issubset({"A", "C", "G", "T"}):
            continue
        weight_f = float(weight)
        if not np.isfinite(weight_f) or weight_f <= 0:
            continue

        kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
        if not kmers:
            continue

        sequence_counts = Counter(kmers)

        if normalize_per_sequence:
            n_windows = float(len(kmers))
            for kmer, count in sequence_counts.items():
                pooled_counts[kmer] += weight_f * count / n_windows
        else:
            for kmer, count in sequence_counts.items():
                pooled_counts[kmer] += weight_f * count

    return normalize_distribution(pooled_counts)


# Alias for backward compatibility with immune repertoire notebooks
calculate_repertoire_kmer_distribution = kmer_distribution_from_sequences


def js_divergence_from_distributions(
    distribution_p: dict[str, float],
    distribution_q: dict[str, float],
) -> float:
    """Jensen-Shannon divergence (in bits, bounded in [0, 1]) between two precomputed distributions."""
    distribution_p = normalize_distribution(distribution_p)
    distribution_q = normalize_distribution(distribution_q)

    keys = set(distribution_p) | set(distribution_q)
    mixture = {
        key: 0.5 * distribution_p.get(key, 0.0) + 0.5 * distribution_q.get(key, 0.0)
        for key in keys
    }

    kl_p_m = sum(
        p * np.log2(p / mixture[key])
        for key, p in distribution_p.items()
        if p > 0
    )
    kl_q_m = sum(
        q * np.log2(q / mixture[key])
        for key, q in distribution_q.items()
        if q > 0
    )
    return float(0.5 * (kl_p_m + kl_q_m))


def wasserstein_from_distributions(
    distribution_p: dict[str, float],
    distribution_q: dict[str, float],
    max_kmers: int = 512,
) -> float:
    """Earth Mover's (Wasserstein) distance between two k-mer distributions using Hamming ground cost."""
    distribution_p = normalize_distribution(distribution_p)
    distribution_q = normalize_distribution(distribution_q)

    selected_kmers = _get_top_kmers(distribution_p, distribution_q, max_kmers=max_kmers)
    if len(selected_kmers) == 0:
        raise ValueError("No k-mers selected for Wasserstein calculation.")

    vector_p = np.array([distribution_p.get(kmer, 0.0) for kmer in selected_kmers], dtype=float)
    vector_q = np.array([distribution_q.get(kmer, 0.0) for kmer in selected_kmers], dtype=float)

    if vector_p.sum() <= 0 or vector_q.sum() <= 0:
        raise ValueError("One distribution has no mass in the selected vocabulary.")

    vector_p /= vector_p.sum()
    vector_q /= vector_q.sum()

    cost_matrix = _build_cost_matrix(selected_kmers)
    return _solve_emd2(vector_p, vector_q, cost_matrix)


def kl_divergence_from_distributions(
    P: dict[str, float],
    Q: dict[str, float],
    epsilon: float = 1e-9,
) -> float:
    """Directional Kullback-Leibler divergence KL(P || Q) in bits with epsilon smoothing on Q."""
    P = normalize_distribution(P)
    Q = normalize_distribution(Q)

    all_keys = set(P) | set(Q)
    Q_smoothed = {k: Q.get(k, 0.0) + epsilon for k in all_keys}
    total_q = sum(Q_smoothed.values())
    Q_smoothed = {k: v / total_q for k, v in Q_smoothed.items()}

    return float(
        sum(p * np.log2(p / Q_smoothed[k]) for k, p in P.items() if p > 0)
    )


def compute_kl_distances(
    P: dict[str, float],
    Q: dict[str, float],
    label: str = "",
    epsilon: float = 1e-9,
) -> dict[str, float]:
    """Compute directional KL(P||Q), KL(Q||P), and symmetric Jeffreys divergence KL(P||Q) + KL(Q||P)."""
    results = {}
    try:
        kl_fwd = kl_divergence_from_distributions(P, Q, epsilon=epsilon)
        kl_rev = kl_divergence_from_distributions(Q, P, epsilon=epsilon)
        sym_kl = kl_fwd + kl_rev

        results[f"kl_early_to_late{label}"] = round(kl_fwd, 6)
        results[f"kl_late_to_early{label}"] = round(kl_rev, 6)
        results[f"symmetric_kl{label}"] = round(sym_kl, 6)
    except Exception:
        results[f"kl_early_to_late{label}"] = np.nan
        results[f"kl_late_to_early{label}"] = np.nan
        results[f"symmetric_kl{label}"] = np.nan

    return results


def compute_jaccard_distances(
    seqs_a: Sequence[str],
    seqs_b: Sequence[str],
    k_sizes: Sequence[int] = (1, 3, 6, 7, 20),
    n_hashes: int = 1000,
) -> dict[str, float]:
    """Compute MinHash (or exact set) Jaccard similarity and distance (1 - similarity) across k-mer sizes."""
    results = {}
    for k in k_sizes:
        try:
            if _MinHash is not None:
                mh_a = _MinHash(n=n_hashes, ksize=k)
                mh_b = _MinHash(n=n_hashes, ksize=k)

                for seq in seqs_a:
                    mh_a.add_sequence(seq)
                for seq in seqs_b:
                    mh_b.add_sequence(seq)

                sim = round(float(mh_a.similarity(mh_b)), 6)
            else:
                set_a = {
                    s[i : i + k]
                    for s in seqs_a
                    if isinstance(s, str) and len(s) >= k
                    for i in range(len(s) - k + 1)
                }
                set_b = {
                    s[i : i + k]
                    for s in seqs_b
                    if isinstance(s, str) and len(s) >= k
                    for i in range(len(s) - k + 1)
                }
                union = len(set_a | set_b)
                sim = round(len(set_a & set_b) / union, 6) if union > 0 else np.nan
            results[f"jaccard_sim_k{k}"] = sim
            results[f"jaccard_dist_k{k}"] = round(1.0 - sim, 6) if np.isfinite(sim) else np.nan
        except Exception:
            results[f"jaccard_sim_k{k}"] = np.nan
            results[f"jaccard_dist_k{k}"] = np.nan

    return results
