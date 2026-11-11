# DVQ: Visual, Statistical, and Spectral Representation of Nucleotide Sequences

**DVQ** is a lightweight, community-extensible Python library for alignment-free visualization, statistical comparison, and signal-processing analysis of DNA/RNA sequences across scales—from short sequences and multi-FASTA collections to complete viral and microbial genomes.

---

## Installation

```bash
pip install dvq
```

For development installation from source:

```bash
git clone https://github.com/hssn-20/dvq.git
cd dvq
pip install -e ".[test]"
```

---

## Core Modules & Functionality

DVQ is organized into three modular subpackages (`dvq.visual`, `dvq.statistical`, and `dvq.functional`) with a flat top-level convenience namespace:

### 1. `dvq.visual` — Geometric & Spatial Representations

- **2D & 3D Cumulative Vector Trajectories (`plot_2d_sequences`, `plot_2d_comparison`, `plot_3d_sequences`, `acgt_to_3d_trajectory`, `max_euclidean_distance`, `interpolate_trajectory`)**:
  Maps nucleotides (`A, T/U, G, C`) onto non-degenerate 2D or 3D unit vectors (following Yau et al., 2003) and computes cumulative sum trajectories along the sequence.
  ```python
  import dvq

  seq1 = "ATCGATCGATCG"
  seq2 = "GCTAGCTAGCTA"
  dvq.plot_2d_comparison([[seq1], [seq2]], labels=["Strain A", "Strain B"])
  traj3d = dvq.acgt_to_3d_trajectory(seq1)
  ```
- **Chaos Game Representation (`CGR`, `FCGR`, `GenerateFCGR`)**:
  Bijective fractal encoding (`CGR.encode` / `CGR.decode`) and $2^k \times 2^k$ Frequency Chaos Game Representation (`FCGR`) matrices.
  ```python
  fcgr_gen = dvq.GenerateFCGR(destination_folder="img", kmer=5)
  matrix = fcgr_gen.from_seq("ATCGATCGATCG")
  ```
- **ColorSquare (`generate_color_square`)**:
  Clockwise spiral grid mapping nucleotides to RGB color channels for rapid visual detection of compositional skews and tandem repeats.
  ```python
  dvq.generate_color_square("ATCG" * 250, save_path="color_square.png")
  ```

---

### 2. `dvq.statistical` — Information-Theoretic, Optimal Transport & Multi-Sequence Distribution Metrics

- **Generalized Deng Entropy (`denq_entropy_generalised`, `deng_entropy_short_sequence`, `calculate_deng_entropies_multiprocess`, `deng_KL_divergence`, `unique_window_ratio`)**:
  Generalizes Shannon entropy by weighting sliding-window k-mer probabilities by combinatorial subset cardinality ($2^{|X_i|} - 1$). Includes window-normalized, `lru_cache`-accelerated `deng_entropy_short_sequence` tailored for short nucleotide sequences (15–100 nt).
- **Biological-Cost Wasserstein Distance (`wasserstein_distance`, `wasserstein_distance_matrix`)**:
  Computes exact Earth Mover's Distance (`POT` / `ot.emd2`) over top-$N$ k-mer distributions using pairwise **Hamming distance** as the ground substitution cost.
- **Jensen–Shannon & Kullback–Leibler Divergences (`js_divergence`, `kl_divergence`)**:
  Symmetric bounded (`[0, 1]` bit) JS divergence and directional KL divergence over sliding-window chunk distributions.
- **Multi-Sequence / Collection Distribution Functions (`kmer_distribution_from_sequences`, `wasserstein_from_distributions`, `js_divergence_from_distributions`, `kl_divergence_from_distributions`, `compute_kl_distances`, `compute_jaccard_distances`)**:
  Aggregate arbitrary collections of sequences (e.g., multi-FASTA gene families, metagenomic contigs, viral quasispecies, synthetic sequence batches, or immune repertoires) into boundary-preserving k-mer distributions with optional abundance `weights` and length-bias normalization (`normalize_per_sequence=True`).
  ```python
  dist_a = dvq.kmer_distribution_from_sequences(seqs_a, k=6, weights=counts_a)
  dist_b = dvq.kmer_distribution_from_sequences(seqs_b, k=6, weights=counts_b)

  w_dist = dvq.wasserstein_from_distributions(dist_a, dist_b, max_kmers=512)
  js_div = dvq.js_divergence_from_distributions(dist_a, dist_b)
  ```
- **Multi-Scale MinHash Jaccard Similarity (`average_kmer_jaccard_similarity`)**:
  Fast `sourmash` MinHash sketch overlap averaged across multiple k-mer resolutions (default $k \in \{1, 3, 7, 20\}$).
- **Wen's Method (`similarity_wen`, `moment_of_inertia`) & Persistent Homology (`persistence_homology`, `compare_persistence_homology`)**:
  Context-weighted 4D Fermat spiral moments of inertia and Vietoris–Rips persistence diagram sliced Wasserstein distance (`ripser` + `persim`).

---

### 3. `dvq.functional` — Spectral Signal Processing & Repeat Analysis

- **Fast Fourier Transform Power Spectra (`power_spectrum`, `spectral_distance`, `power_spectra_batch`)**:
  Converts 4-channel binary indicator encodings into length-normalized, z-scored frequency power spectra (`n_bins=512`) and compares periodicities via Euclidean or Pearson correlation distance.
- **Discrete Wavelet Transform Energy (`wavelet_energy`, `wavelet_distance`)**:
  Computes multi-resolution Daubechies (`db4`) wavelet decomposition energies across $L=6$ scales to capture localized compositional shifts.
- **Exact Tandem Repeat Detection (`find_genomic_repeats`)**:
  Identifies exact tandem repeats via `pytrf.GTRFinder` and extracts dominant motif-repeat pairs (useful both for biological satellite analysis and for auditing pathological repeat collapse in generative DNA language models).
  ```python
  df_repeats, top_motifs = dvq.find_genomic_repeats("ATGATGATGATGATGATG", top_n=5)
  ```

---

## Community Extensibility & Contributing

**DVQ is designed as an open, community-driven toolkit.** Because each subpackage (`dvq.visual`, `dvq.statistical`, `dvq.functional`) follows a simple functional contract—accepting standard nucleotide strings or sequence iterables and returning NumPy arrays, Matplotlib figures, or scalar distances—researchers can easily contribute new sequence encodings, alignment-free distance metrics, or visual diagnostics via pull requests at [https://github.com/hssn-20/dvq](https://github.com/hssn-20/dvq).

---

## License

Apache License 2.0.
