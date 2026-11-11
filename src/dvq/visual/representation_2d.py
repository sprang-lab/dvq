from typing import Dict, Mapping, Optional, Sequence
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from tqdm import tqdm
from pathlib import Path

from collections import defaultdict
from itertools import product

# using a faster style for plotting
mplstyle.use('fast')

# Mapping of nucleotides to float coordinates
mapping_easy = {
    'A': np.array([0.5, -0.8660254037844386]),
    'T': np.array([0.5, 0.8660254037844386]),
    'G': np.array([0.8660254037844386, -0.5]),
    'C': np.array([0.8660254037844386, 0.5]),
    'N': np.array([0, 0])
}

# 3D nucleotide encoding (x, y, z) following Yau et al. (2003).
# The z axis increments by 1 per basepair so trajectories never self-intersect.
BASE_TO_VEC_3D: Dict[str, tuple[float, float, float]] = {
    "A": (0.0, 1.0, 1.0),
    "T": (0.0, -1.0, 1.0),
    "U": (0.0, -1.0, 1.0),
    "G": (1.0, 0.0, 1.0),
    "C": (-1.0, 0.0, 1.0),
}

# coordinates for x+iy
Coord = namedtuple("Coord", ["x","y"])

# coordinates for a CGR encoding
CGRCoords = namedtuple("CGRCoords", ["N","x","y"])

# coordinates for each nucleotide in the 2d-plane
DEFAULT_COORDS = dict(A=Coord(1,1),C=Coord(-1,1),G=Coord(-1,-1),T=Coord(1,-1))

# Function to convert a DNA sequence to a list of coordinates
def _dna_to_coordinates(dna_sequence, mapping):
    dna_sequence = dna_sequence.upper()
    coordinates = np.array([mapping.get(nucleotide, mapping['N']) for nucleotide in dna_sequence])
    return coordinates

# Function to create the cumulative sum of a list of coordinates
def _get_cumulative_coords(mapped_coords):
    cumulative_coords = np.cumsum(mapped_coords, axis=0)
    return cumulative_coords

# Function to take a list of DNA sequences and plot them in a single figure
def plot_2d_sequences(dna_sequences, mapping=mapping_easy, single_sequence=False):
    fig, ax = plt.subplots()
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        ax.plot(*cumulative_coords.T)
    plt.show()

# Function to plot a comparison of DNA sequences
def plot_2d_comparison(dna_sequences_grouped, labels, mapping=mapping_easy):
    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dna_sequences_grouped)))
    for count, (dna_sequences, color) in enumerate(zip(dna_sequences_grouped, colors)):
        for dna_sequence in dna_sequences:
            mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
            cumulative_coords = _get_cumulative_coords(mapped_coords)
            ax.plot(*cumulative_coords.T, color=color, label=labels[count])
    # Only show unique labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()


def acgt_to_3d_trajectory(seq: str) -> np.ndarray:
    """Map an ACGT(U) string to a 3D cumulative trajectory of shape (len(seq), 3)."""
    vecs = np.array(
        [BASE_TO_VEC_3D.get(ch.upper(), (0.0, 0.0, 1.0)) for ch in seq],
        dtype=np.float32,
    )
    return np.cumsum(vecs, axis=0)


def trajectory_2d(traj3d: np.ndarray) -> np.ndarray:
    """Project a (..., L, 3) trajectory down to (..., L, 2) by dropping the z axis."""
    return traj3d[..., :2]


def max_euclidean_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """Maximum pointwise Euclidean distance between two trajectories of matching shape."""
    if traj_a.shape != traj_b.shape:
        raise ValueError(f"Trajectory shapes disagree: {traj_a.shape} vs {traj_b.shape}")
    dists = np.linalg.norm(traj_a - traj_b, axis=-1)
    return float(dists.max())


def interpolate_trajectory(traj: np.ndarray, target_length: int = 1000) -> np.ndarray:
    """Linearly interpolate a (L, D) trajectory to a fixed length along its first axis."""
    if traj.ndim != 2:
        raise ValueError(f"Expected (L, D), got shape {traj.shape}")
    L = traj.shape[0]
    if L == target_length:
        return traj
    xs_old = np.linspace(0.0, 1.0, L)
    xs_new = np.linspace(0.0, 1.0, target_length)
    cols = [np.interp(xs_new, xs_old, traj[:, d]) for d in range(traj.shape[1])]
    return np.stack(cols, axis=-1).astype(traj.dtype, copy=False)


def plot_3d_sequences(
    sequences: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    output_path: Optional[str | Path] = None,
    label_colors: Optional[Mapping[str, str]] = None,
    max_points: int = 5000,
    title: str = "3D Sequence Representation",
    show: bool = True,
):
    """Plot per-sequence 3D cumulative trajectories (basepair, GC, TAU axes) colored by group label."""
    if labels is None:
        labels = ["Sequence"] * len(sequences)
    if label_colors is None:
        unique = sorted(set(labels))
        cmap = plt.get_cmap("tab10")
        label_colors = {lbl: cmap(i % 10) for i, lbl in enumerate(unique)}

    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    seen_labels = set()
    for seq, lbl in zip(sequences, labels):
        traj = acgt_to_3d_trajectory(seq)
        if len(traj) > max_points:
            idx = np.linspace(0, len(traj) - 1, max_points, dtype=np.int64)
            traj = traj[idx]
        plot_lbl = lbl if lbl not in seen_labels else None
        seen_labels.add(lbl)
        ax.plot(
            traj[:, 2],
            traj[:, 0],
            traj[:, 1],
            color=label_colors[lbl],
            alpha=0.55,
            lw=0.7,
            label=plot_lbl,
        )

    ax.set_xlabel("Basepair axis")
    ax.set_ylabel("GC axis")
    ax.set_zlabel("TAU axis")
    ax.set_title(title)
    if len(seen_labels) > 1:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# Class to create a Chaos Game Representation
class CGR:
    "Chaos Game Representation for DNA"
    def __init__(self, coords: Optional[Dict[chr,tuple]]=None):
        self.nucleotide_coords = DEFAULT_COORDS if coords is None else coords
        self.cgr_coords = CGRCoords(0,0,0)

    def nucleotide_by_coords(self,x,y):
        "Get nucleotide by coordinates (x,y)"
        # filter nucleotide by coordinates
        filtered = dict(filter(lambda item: item[1] == Coord(x,y), self.nucleotide_coords.items()))

        return list(filtered.keys())[0]

    def forward(self, nucleotide: str):
        "Compute next CGR coordinates"
        x = (self.cgr_coords.x + self.nucleotide_coords.get(nucleotide).x)/2
        y = (self.cgr_coords.y + self.nucleotide_coords.get(nucleotide).y)/2

        # update cgr_coords
        self.cgr_coords = CGRCoords(self.cgr_coords.N+1,x,y)

    def backward(self,):
        "Compute last CGR coordinates. Current nucleotide can be inferred from (x,y)"
        # get current nucleotide based on coordinates
        n_x,n_y = self.coords_current_nucleotide()
        nucleotide = self.nucleotide_by_coords(n_x,n_y)

        # update coordinates to the previous one
        x = 2*self.cgr_coords.x - n_x
        y = 2*self.cgr_coords.y - n_y

        # update cgr_coords
        self.cgr_coords = CGRCoords(self.cgr_coords.N-1,x,y)

        return nucleotide

    def coords_current_nucleotide(self,):
        x = 1 if self.cgr_coords.x>0 else -1
        y = 1 if self.cgr_coords.y>0 else -1
        return x,y

    def encode(self, sequence: str):
        "From DNA sequence to CGR"
        # reset starting position to (0,0,0)
        self.reset_coords()
        for nucleotide in sequence:
            self.forward(nucleotide)
        return self.cgr_coords

    def reset_coords(self,):
        self.cgr_coords = CGRCoords(0,0,0)

    def decode(self, N:int, x:int, y:int)->str:
        "From CGR to DNA sequence"
        self.cgr_coords = CGRCoords(N,x,y)

        # decoded sequence
        sequence = []

        # Recover the entire genome
        while self.cgr_coords.N>0:
            nucleotide = self.backward()
            sequence.append(nucleotide)
        return "".join(sequence[::-1])

# Class to create a Frequency Chaos Game Representation
class FCGR(CGR):
    """Frequency matrix CGR
    an (2**k x 2**k) 2D representation will be created for a
    n-long sequence.
    - k represents the k-mer.
    - 2**k x 2**k = 4**k the total number of k-mers (sequences of length k)
    - pixel value correspond to the value of the frequency for each k-mer
    """

    def __init__(self, k: int,):
        super().__init__()
        self.k = k # k-mer representation
        self.kmers = list("".join(kmer) for kmer in product("ACGT", repeat=self.k))
        self.kmer2pixel = self.kmer2pixel_position()

    def __call__(self, sequence: str):
        "Given a DNA sequence, returns an array with his frequencies in the same order as FCGR"
        self.count_kmers(sequence)

        # Create an empty array to save the FCGR values
        array_size = int(2**self.k)
        freq_matrix = np.zeros((array_size,array_size))

        # Assign frequency to each box in the matrix
        for kmer, freq in self.freq_kmer.items():
            pos_x, pos_y = self.kmer2pixel[kmer]
            freq_matrix[int(pos_x)-1,int(pos_y)-1] = freq
        return freq_matrix

    def count_kmer(self, kmer):
        if "N" not in kmer:
            self.freq_kmer[kmer] += 1

    def count_kmers(self, sequence: str):
        self.freq_kmer = defaultdict(int)
        # representativity of kmers
        last_j = len(sequence) - self.k + 1
        kmers  = (sequence[i:(i+self.k)] for i in range(last_j))
        # count kmers in a dictionary
        list(self.count_kmer(kmer) for kmer in kmers)

    def kmer_probabilities(self, sequence: str):
        self.probabilities = defaultdict(float)
        N=len(sequence)
        for key, value in self.freq_kmer.items():
            self.probabilities[key] = float(value) / (N - self.k + 1)

    def pixel_position(self, kmer: str):
        "Get pixel position in the FCGR matrix for a k-mer"

        coords = self.encode(kmer)
        N,x,y = coords.N, coords.x, coords.y

        # Coordinates from [-1,1]² to [1,2**k]²
        np_coords = np.array([(x + 1)/2, (y + 1)/2]) # move coordinates from [-1,1]² to [0,1]²
        np_coords *= 2**self.k # rescale coordinates from [0,1]² to [0,2**k]²
        x,y = np.ceil(np_coords) # round to upper integer

        # Turn coordinates (cx,cy) into pixel (px,py) position
        # px = 2**k-cy+1, py = cx
        return 2**self.k-int(y)+1, int(x)

    def kmer2pixel_position(self,):
        kmer2pixel = dict()
        for kmer in self.kmers:
            kmer2pixel[kmer] = self.pixel_position(kmer)
        return kmer2pixel

# Class to generate FCGR from a list of fasta files
class GenerateFCGR:

    def __init__(self, destination_folder: Path = "img", kmer: int = 5, ):
        self.destination_folder = Path(destination_folder)
        self.kmer = kmer
        self.fcgr = FCGR(kmer)
        self.counter = 0 # count number of time a sequence is converted to fcgr

        # Create destination folder if needed
        self.destination_folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, list_fasta,):

        for fasta in tqdm(list_fasta, desc="Generating FCGR"):
            self.from_fasta(fasta)

    def from_seq(self, seq: str):
        "Get FCGR from a sequence"
        seq = self.preprocessing(seq)
        chaos = self.fcgr(seq)
        self.counter +=1
        return chaos

    def reset_counter(self,):
        self.counter=0

    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in seq:
          if letter not in "ATCG":
            seq = seq.replace(letter,"N")
        return seq
