from collections import defaultdict
from tqdm import tqdm
import numpy as np
import logging

try:
    from sourmash import MinHash
except ImportError:
    MinHash = None

logging.basicConfig(level=logging.INFO)


def _create_mini_hash_of_a_sequence(seq: str, minihash):
    """Create a MinHash of a sequence (or list of sequences)."""
    if isinstance(seq, list):
        for s in seq:
            minihash.add_sequence(s)
    else:
        minihash.add_sequence(seq)
    return minihash


def _compare_two_sequences_and_return_similarity(
    seq, seq2, k: int, n: int
) -> float:
    """Calculate similarity (Jaccard) of two sequences with given k and number of hashes."""
    if MinHash is not None:
        mh1 = MinHash(n=n, ksize=k)
        mh2 = MinHash(n=n, ksize=k)
        mh1 = _create_mini_hash_of_a_sequence(seq, mh1)
        mh2 = _create_mini_hash_of_a_sequence(seq2, mh2)
        return round(mh1.similarity(mh2), 5)

    seqs1 = seq if isinstance(seq, list) else [seq]
    seqs2 = seq2 if isinstance(seq2, list) else [seq2]
    set1 = {s[i:i + k] for s in seqs1 if len(s) >= k for i in range(len(s) - k + 1)}
    set2 = {s[i:i + k] for s in seqs2 if len(s) >= k for i in range(len(s) - k + 1)}
    union = len(set1 | set2)
    return round(len(set1 & set2) / union, 5) if union > 0 else 0.0


def average_kmer_jaccard_similarity(
    seq,
    seq2,
    number_of_hashes: int = 20000,
    k_sizes: list = [1, 3, 7, 20]
) -> float:
    """
    Calculate the average Jaccard similarity for two sequence inputs
    across multiple k-mer sizes. Also logs the individual similarity
    for each k-mer size.
    """
    if not isinstance(seq, list):
        seq = [seq]
    if not isinstance(seq2, list):
        seq2 = [seq2]

    similarities = []
    for k in tqdm(k_sizes):
        sim = _compare_two_sequences_and_return_similarity(seq, seq2, k, number_of_hashes)
        logging.info(f"K = {k}, similarity = {sim:.5f}")
        similarities.append(sim)

    avg_similarity = round(float(np.mean(similarities)), 3)
    logging.info(f"Average similarity across k={k_sizes} is {avg_similarity:.3f}")
    return avg_similarity
