# Wasserstein distance between DNA sequences.
# Sequences are represented as k-mer frequency distributions.
# The ground metric between k-mers is Hamming distance, reflecting the true biological
# cost of substitution between nucleotides.

import numpy as np
from typing import Dict, List
from collections import Counter

try:
    import ot as _ot
except ImportError:
    _ot = None


def _solve_emd2(a: np.ndarray, b: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Solve exact Earth Mover's Distance using POT (ot.emd2) or SciPy HiGHS LP solver."""
    if _ot is not None:
        return float(_ot.emd2(a, b, cost_matrix))
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    n = len(a)
    if n == 0 or np.allclose(a, b):
        return 0.0
    c = cost_matrix.ravel()
    # 2n equality constraints: row sums = a, col sums = b (drop last redundant constraint)
    A_eq = lil_matrix((2 * n - 1, n * n), dtype=float)
    for i in range(n):
        A_eq[i, i * n : (i + 1) * n] = 1.0
    for j in range(n - 1):
        A_eq[n + j, j :: n] = 1.0
    b_eq = np.concatenate([a, b[:-1]])
    res = linprog(c, A_eq=A_eq.tocsc(), b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"EMD linear program failed: {res.message}")
    return float(max(0.0, res.fun))


def _calculate_kmer_distribution(seq: str, k: int) -> Dict[str, float]:
    """
    Calculate the k-mer frequency distribution of a DNA sequence.

    Parameters:
    seq (str): the input DNA sequence.
    k (int): the k-mer size.

    Returns:
    Dict[str, float]: a dictionary mapping each k-mer to its relative frequency.

    Raises:
    ValueError: if the sequence is shorter than k.
    """
    seq_len = len(seq)
    if seq_len < k:
        raise ValueError(f"Sequence length ({seq_len}) is shorter than k ({k}).")

    kmers = [seq[i:i + k] for i in range(seq_len - k + 1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    return {kmer: count / total for kmer, count in counts.items()}


def _hamming_distance(a: str, b: str) -> int:
    """
    Compute the Hamming distance between two equal-length k-mers,
    it counts the number of positions at which the nucleotides differ,
    representing the minimum number of substitutions to transform one k-mer
    into the other.

    Parameters:
    a (str): first k-mer.
    b (str): second k-mer.

    Returns:
    int: the Hamming distance between a and b.
    """
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _build_cost_matrix(kmers: List[str]) -> np.ndarray:
    """
    Build a pairwise Hamming distance matrix over a vocabulary of k-mers.
    This matrix serves as the ground metric for the Wasserstein computation,
    encoding the biological cost of moving probability mass between any two k-mers.

    Parameters:
    kmers (List[str]): the list of k-mers forming the shared vocabulary.

    Returns:
    np.ndarray: a symmetric (nxn) matrix of pairwise Hamming distances.
    """
    n = len(kmers)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _hamming_distance(kmers[i], kmers[j])
            cost[i, j] = d
            cost[j, i] = d
    return cost


def _get_top_kmers(
    dist_P: Dict[str, float],
    dist_Q: Dict[str, float],
    max_kmers: int
) -> List[str]:
    """
    Select the union of the top most frequent k-mers from two distributions,
    capped at max_kmers total. This keeps the cost matrix tractable for large k.

    Each distribution contributes up to max_kmers // 2 of its most frequent
    k-mers, and the union is taken before capping. This ensures neither sequence
    dominates the shared vocabulary.

    Parameters:
    dist_P (Dict[str, float]): k-mer distribution of sequence P.
    dist_Q (Dict[str, float]): k-mer distribution of sequence Q.
    max_kmers (int): Maximum number of k-mers to retain.

    Returns:
    List[str]: Sorted list of selected k-mers.
    """
    top_p = sorted(dist_P, key=dist_P.get, reverse=True)[:max_kmers // 2]
    top_q = sorted(dist_Q, key=dist_Q.get, reverse=True)[:max_kmers // 2]
    return sorted(set(top_p) | set(top_q))[:max_kmers]


def wasserstein_distance(
    seq_P: str,
    seq_Q: str,
    k: int = 7,
    max_kmers: int = 512
) -> float:
    """
    Calculate the Wasserstein distance between two
    DNA sequences based on their k-mer frequency distributions.

    Parameters:
    seq_P (str): first DNA sequence.
    seq_Q (str): second DNA sequence.
    k (int): k-mer size (default: 7).
    max_kmers (int): maximum vocabulary size for the cost matrix (default: 512).

    Returns:
    float: Wasserstein distance between the two sequences.

    Raises:
    ValueError: If either sequence produces an empty distribution over the
                selected k-mers after vocabulary reduction.
    """
    dist_P = _calculate_kmer_distribution(seq_P, k)
    dist_Q = _calculate_kmer_distribution(seq_Q, k)

    all_kmers = _get_top_kmers(dist_P, dist_Q, max_kmers)

    a = np.array([dist_P.get(kmer, 0.0) for kmer in all_kmers])
    b = np.array([dist_Q.get(kmer, 0.0) for kmer in all_kmers])

    # Renormalise after vocabulary reduction to ensure valid probability distributions
    a_sum, b_sum = a.sum(), b.sum()
    if a_sum == 0 or b_sum == 0:
        raise ValueError(
            "One or both sequences produced an empty distribution over the "
            "selected k-mers. Try increasing max_kmers or reducing k."
        )
    a /= a_sum
    b /= b_sum

    cost_matrix = _build_cost_matrix(all_kmers)

    return _solve_emd2(a, b, cost_matrix)


def wasserstein_distance_matrix(
    sequences: List[str],
    k: int = 7,
    max_kmers: int = 512
) -> np.ndarray:
    """
    Compute a pairwise Wasserstein distance matrix for a list of DNA sequences.

    Parameters:
    sequences (List[str]): a list of DNA sequence strings.
    k (int): the k-mer size (default: 7).
    max_kmers (int): maximum vocabulary size for the cost matrix (default: 512).

    Returns:
    np.ndarray: a symmetric (n x n) distance matrix.
    """
    n = len(sequences)
    distributions = [_calculate_kmer_distribution(seq, k) for seq in sequences]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            all_kmers = _get_top_kmers(distributions[i], distributions[j], max_kmers)

            a = np.array([distributions[i].get(kmer, 0.0) for kmer in all_kmers])
            b = np.array([distributions[j].get(kmer, 0.0) for kmer in all_kmers])
            a /= a.sum()
            b /= b.sum()

            cost_matrix = _build_cost_matrix(all_kmers)
            dist = _solve_emd2(a, b, cost_matrix)

            matrix[i, j] = dist
            matrix[j, i] = dist

    return matrix
