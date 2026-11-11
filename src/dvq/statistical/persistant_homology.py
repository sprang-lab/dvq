import matplotlib.pyplot as plt
import numpy as np

try:
    import persim
    import ripser
except ImportError:
    persim = None
    ripser = None


NUCLEOTIDE_MAPPING = {
    'a': np.array([1, 0, 0, 0]),
    'c': np.array([0, 1, 0, 0]),
    'g': np.array([0, 0, 1, 0]),
    't': np.array([0, 0, 0, 1]),
    'A': np.array([1, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0]),
    'G': np.array([0, 0, 1, 0]),
    'T': np.array([0, 0, 0, 1])
}


def encode_nucleotide_to_vector(nucleotide):
    return NUCLEOTIDE_MAPPING[nucleotide]


def chaos_4d_representation(dna_sequence):
    points = [encode_nucleotide_to_vector(dna_sequence[0])]
    for nucleotide in dna_sequence[1:]:
        vector = encode_nucleotide_to_vector(nucleotide)
        next_point = 0.5 * (points[-1] + vector)
        points.append(next_point)
    return np.array(points)


def persistence_homology(dna_sequence, multi=False, plot=False, sample_rate=7):
    if ripser is None or persim is None:
        raise ImportError("ripser and persim are required for persistence_homology.")
    if multi:
        c4dr_points = np.array([chaos_4d_representation(sequence) for sequence in dna_sequence])
        dgm_dna = [ripser.ripser(points[::sample_rate], maxdim=1)['dgms'] for points in c4dr_points]
        if plot:
            persim.plot_diagrams([dgm[1] for dgm in dgm_dna], labels=[f'sequence {i}' for i in range(len(dna_sequence))])
    else:
        c4dr_points = chaos_4d_representation(dna_sequence)
        dgm_dna = ripser.ripser(c4dr_points[::sample_rate], maxdim=1)['dgms']
        if plot:
            persim.plot_diagrams(dgm_dna[1])
    return dgm_dna


def compare_persistence_homology(dna_sequence1, dna_sequence2):
    if ripser is None or persim is None:
        raise ImportError("ripser and persim are required for compare_persistence_homology.")
    dgm_dna1 = persistence_homology(dna_sequence1)
    dgm_dna2 = persistence_homology(dna_sequence2)
    distance = persim.sliced_wasserstein(dgm_dna1[1], dgm_dna2[1])
    return distance
