# %%
# Example usage for all DVQ modules (visual, statistical, functional, and multi-sequence distributions)
from src.dvq.statistical import (
    average_kmer_jaccard_similarity,
    similarity_wen,
    moment_of_inertia,
    persistence_homology,
    compare_persistence_homology,
    denq_entropy_generalised,
    calculate_deng_entropies_multiprocess,
    deng_KL_divergence,
    deng_entropy_short_sequence,
    unique_window_ratio,
    kl_divergence,
    js_divergence,
    wasserstein_distance,
    wasserstein_distance_matrix,
    normalize_distribution,
    kmer_distribution_from_sequences,
    js_divergence_from_distributions,
    wasserstein_from_distributions,
    kl_divergence_from_distributions,
    compute_kl_distances,
    compute_jaccard_distances,
)

from src.dvq.functional import (
    find_genomic_repeats,
    power_spectrum,
    spectral_distance,
    power_spectra_batch,
    wavelet_energy,
    wavelet_distance,
)

from src.dvq.visual import (
    acgt_to_3d_trajectory,
    trajectory_2d,
    max_euclidean_distance,
    interpolate_trajectory,
)


# %%
def main():

    # Test sequences for repeat detection
    seqs = [
        "ACGTACGTACGTACGTACGT",
        "AAAAGGGGCCCC",
        "ATCGATCGATCGATCGATCG",
        "AGCTTTCGAAGCTTTCGAAGCTTTCGAA",
        "CGTAGCTAGCTAGCTAGCTA",
        "TTGACGTCGTAGCTAGCTAAGT",
    ]

    repeats = find_genomic_repeats(seqs)
    print(repeats)

    # Example sequences for single-sequence statistical & functional modules
    seq_1 = "ACGT" * 1000
    seq_2 = "GCTA" * 1000
    seqs = [seq_1, seq_2, "TTAA" * 1000]

    # Test Deng Entropy
    print("\n--- Testing Deng Entropy ---")
    entropy = denq_entropy_generalised(seq_1)
    print(f"Deng entropy for sequence: {entropy}")

    entropies = calculate_deng_entropies_multiprocess(seqs)
    print(f"Entropies for sequences: {entropies}")

    # Test Deng KL Divergence
    print("\n--- Testing Deng KL Divergence ---")
    kl_div = deng_KL_divergence(seq_1, seq_2)
    print(f"Deng KL divergence: {kl_div}")

    # Test Standard KL Divergence
    print("\n--- Testing Standard KL Divergence ---")
    kl_div_standard = kl_divergence(seq_1, seq_2)
    print(f"Standard KL divergence: {kl_div_standard}")

    # Test Wen's Method
    print("\n--- Testing Wen's method ---")
    similarity = similarity_wen(seq_1, seq_2)
    print(f"Similarity using Wen's method: {similarity}")

    inertia = moment_of_inertia(seq_1)
    print(f"Moment of inertia for sequence: {inertia}")

    # Test Persistent Homology
    print("\n--- Testing Persistent Homology ---")
    try:
        persistence = persistence_homology(seq_1, plot=False)
        print(f"Persistent homology for sequence: {len(persistence)} diagram dimensions")
        homology_distance = compare_persistence_homology(seq_1, seq_2)
        print(f"Persistent homology distance: {homology_distance}")
    except ImportError as e:
        print(f"Skipping Persistent Homology (optional dependency missing): {e}")

    # Test k-mer functions
    print("\n--- Testing k-mer functions ---")
    overlap_group = average_kmer_jaccard_similarity(seqs, seqs)
    print(f"Self similarity: {overlap_group}")

    overlap_comparison = average_kmer_jaccard_similarity(seq_1, seq_2)
    print(f"Kmer similarity: {overlap_comparison}")

    # Test JS Divergence
    print("\n--- Testing JS Divergence ---")
    js = js_divergence(seq_1, seq_2)
    print(f"JS divergence: {js}")

    # Test Wasserstein Distance
    print("\n--- Testing Wasserstein Distance ---")
    w_dist = wasserstein_distance(seq_1, seq_2)
    print(f"Wasserstein distance: {w_dist}")

    w_matrix = wasserstein_distance_matrix(seqs)
    print(f"Wasserstein distance matrix:\n{w_matrix}")

    # Test Multi-Sequence Distributions & Short-Sequence Entropy Use Case
    print("\n--- Testing Multi-Sequence Distributions Use Case ---")
    # Simulate two DNA sequence collections (e.g., baseline vs. post-selection population)
    # where both collections share a similar sequence vocabulary, but Population B
    # undergoes a strong abundance expansion of one GC-rich variant.
    population_a = [
        "TGTGCGAGAGATCGCTACTATGGMDVWGQ".replace("M", "A").replace("D", "C").replace("V", "G").replace("W", "T").replace("Q", "A").replace("", "A"),
        "TGTGCGAGAGGCTACTACGGTGACTACTGGGGCCAGGGA",
        "TGTGCGAGAGATCGGGGTACTACTACTACGGTATGGACGTCTGG",
        "TGTGCGAGACATAGTACTGGTTATTACTACTTTGACTACTGG",
        "TGTGCGAGAGATAGTAGTGGTTATTACTACTTTGACTACTGG",
    ]
    weights_a = [10.0, 12.0, 11.0, 9.0, 8.0]  # Balanced abundances in Population A

    population_b = [
        "TGTGCGAGAGATACGCTACTATGGACGTGGA",
        "TGTGCGAGAGGCTACTACGGTGACTACTGGGGCCAGGGA",
        "TGTGCGAGAGATCGGGGTACTACTACTACGGTATGGACGTCTGG",
        "TGTGCGAGACATAGTACTGGTTATTACTACTTTGACTACTGG",
        "TGTGCGAGAGATAGTAGTGGTTATTACTACTTTGACTACTGG",
        "TGTGCGAGAGGCCCGCCGCGCGCGCCTACTGGGGCCAGGGA",  # Expanded GC-rich sequence
    ]
    weights_b = [5.0, 6.0, 5.0, 4.0, 4.0, 250.0]  # Strong expansion of the 6th sequence

    # 1. Short-sequence Deng entropy & unique window ratio
    short_entropies_a = [deng_entropy_short_sequence(s, chunk_size=3) for s in population_a]
    uwr_a = [unique_window_ratio(s, chunk_size=3) for s in population_a]
    print(f"Population A short-seq Deng entropies (k=3): {[round(x, 4) for x in short_entropies_a]}")
    print(f"Population A unique window ratios (k=3):     {[round(x, 4) for x in uwr_a]}")

    # 2. Boundary-preserving k-mer distributions (unweighted vs. abundance-weighted)
    dist_a_unw = kmer_distribution_from_sequences(population_a, k=6)
    dist_b_unw = kmer_distribution_from_sequences(population_b, k=6)

    dist_a_w = kmer_distribution_from_sequences(population_a, k=6, weights=weights_a)
    dist_b_w = kmer_distribution_from_sequences(population_b, k=6, weights=weights_b)

    # 3. Compare unweighted vs. weighted JS, Hamming-cost Wasserstein, and KL divergences
    js_unw = js_divergence_from_distributions(dist_a_unw, dist_b_unw)
    js_w = js_divergence_from_distributions(dist_a_w, dist_b_w)
    print(f"Collection JS divergence (k=6)        -- unweighted: {js_unw:.4f} bits | weighted: {js_w:.4f} bits")

    w_unw = wasserstein_from_distributions(dist_a_unw, dist_b_unw, max_kmers=128)
    w_w = wasserstein_from_distributions(dist_a_w, dist_b_w, max_kmers=128)
    print(f"Collection Wasserstein distance (k=6) -- unweighted: {w_unw:.4f}      | weighted: {w_w:.4f}")

    kl_unw = compute_kl_distances(dist_a_unw, dist_b_unw, label="_unweighted")
    kl_w = compute_kl_distances(dist_a_w, dist_b_w, label="_weighted")
    print(f"Collection KL distances (unweighted): {kl_unw}")
    print(f"Collection KL distances (weighted):   {kl_w}")

    # 4. Multi-scale MinHash Jaccard across k-mer sizes
    jacc = compute_jaccard_distances(population_a, population_b, k_sizes=(1, 3, 6))
    print(f"Collection MinHash Jaccard metrics:   {jacc}")

    # Test 3D Trajectory Utilities
    print("\n--- Testing 3D Trajectory Utilities ---")
    traj_1 = acgt_to_3d_trajectory(seq_1[:500])
    traj_2 = acgt_to_3d_trajectory(seq_2[:500])
    max_dist_3d = max_euclidean_distance(traj_1, traj_2)
    traj_interp = interpolate_trajectory(trajectory_2d(traj_1), target_length=100)
    print(f"3D trajectory shape: {traj_1.shape}, Max 3D Euclidean distance: {max_dist_3d:.4f}, Interpolated 2D shape: {traj_interp.shape}")

    # Test Fourier
    print("\n--- Testing Fourier ---")
    spectrum = power_spectrum(seq_1)
    print(f"Power spectrum shape: {spectrum.shape}")

    spectra = power_spectra_batch(seqs)
    print(f"Power spectra batch shape: {spectra.shape}")

    spec_dist = spectral_distance(seq_1, seq_2)
    print(f"Spectral distance (euclidean): {spec_dist}")

    spec_dist_corr = spectral_distance(seq_1, seq_2, metric="correlation")
    print(f"Spectral distance (correlation): {spec_dist_corr}")

    # Test Wavelet
    print("\n--- Testing Wavelet ---")
    energy = wavelet_energy(seq_1)
    print(f"Wavelet energy vector: {energy}")

    wav_dist = wavelet_distance(seq_1, seq_2)
    print(f"Wavelet distance (euclidean): {wav_dist}")

    wav_dist_corr = wavelet_distance(seq_1, seq_2, metric="correlation")
    print(f"Wavelet distance (correlation): {wav_dist_corr}")

    return


if __name__ == "__main__":
    main()

# %%
