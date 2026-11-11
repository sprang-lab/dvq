# Wavelet-based representation and comparison of biological sequences.
# Each sequence is encoded as four binary indicator signals (one per nucleotide)
# and decomposed using the Discrete Wavelet Transform (DWT) with a Daubechies
# mother wavelet. The representation is a z-score normalised energy vector
# summarising the signal's activity at each decomposition level (scale).
import numpy as np
from typing import Literal

try:
    import pywt
except ImportError:
    pywt = None

# Nucleotide channels for binary indicator encoding
NUCLEOTIDE_CHARS = [ord('A'), ord('C'), ord('G'), ord('T')]

# Default mother wavelet: Daubechies order 4.
DEFAULT_WAVELET = 'db4'

# Default number of decomposition levels.
DEFAULT_N_LEVELS = 6


def _binary_indicator_encoding(seq: str) -> np.ndarray:
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    matrix = np.zeros((4, len(arr)), dtype=float)
    for i, char_code in enumerate(NUCLEOTIDE_CHARS):
        matrix[i] = (arr == char_code)
    return matrix


def _level_energy(coefficients: np.ndarray) -> float:
    return float(np.sum(coefficients ** 2))


def _normalise_energy_vector(energy_vector: np.ndarray) -> np.ndarray:
    mean = np.mean(energy_vector)
    std = np.std(energy_vector)
    if std == 0:
        return np.zeros_like(energy_vector)
    return (energy_vector - mean) / std


def _haar_wavedec_fallback(signal: np.ndarray, n_levels: int):
    """Pure-NumPy multi-level Haar DWT fallback when PyWavelets is not installed."""
    approx = signal.astype(float)
    details = []
    for _ in range(n_levels):
        if len(approx) < 2:
            break
        if len(approx) % 2 == 1:
            approx = np.append(approx, approx[-1])
        even = approx[0::2]
        odd = approx[1::2]
        detail = (even - odd) / np.sqrt(2.0)
        approx = (even + odd) / np.sqrt(2.0)
        details.append(detail)
    # Return [approximation, detail_N, ..., detail_1] matching pywt.wavedec order
    return [approx] + details[::-1]


def wavelet_energy(
    seq: str,
    wavelet: str = DEFAULT_WAVELET,
    n_levels: int = DEFAULT_N_LEVELS,
    normalise: bool = True
) -> np.ndarray:
    if not seq:
        raise ValueError("Sequence must not be empty.")

    encoded = _binary_indicator_encoding(seq)
    combined_energy = np.zeros(n_levels + 1, dtype=float)

    for channel in encoded:
        if pywt is not None:
            max_levels = pywt.dwt_max_level(len(channel), wavelet)
            actual_levels = min(n_levels, max_levels)
            coeffs = pywt.wavedec(channel, wavelet, level=actual_levels)
        else:
            coeffs = _haar_wavedec_fallback(channel, n_levels)

        details = coeffs[1:][::-1]
        approximation = coeffs[0]

        for level_idx, detail_coeffs in enumerate(details):
            combined_energy[level_idx] += _level_energy(detail_coeffs)

        combined_energy[n_levels] += _level_energy(approximation)

    if normalise:
        combined_energy = _normalise_energy_vector(combined_energy)

    return combined_energy


def wavelet_distance(
    seq_P: str,
    seq_Q: str,
    metric: Literal["euclidean", "correlation"] = "euclidean",
    wavelet: str = DEFAULT_WAVELET,
    n_levels: int = DEFAULT_N_LEVELS
) -> float:
    ep = wavelet_energy(seq_P, wavelet=wavelet, n_levels=n_levels, normalise=True)
    eq = wavelet_energy(seq_Q, wavelet=wavelet, n_levels=n_levels, normalise=True)

    if metric == "euclidean":
        return float(np.linalg.norm(ep - eq))

    elif metric == "correlation":
        corr = np.corrcoef(ep, eq)[0, 1]
        corr = np.clip(corr, -1.0, 1.0)
        return float(1.0 - corr)

    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose 'euclidean' or 'correlation'."
        )
