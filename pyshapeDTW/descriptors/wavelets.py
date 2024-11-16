from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pywt

from .base import BaseDescriptor


@dataclass
class DWTParams:
    """Parameters for Discrete Wavelet Transform descriptor."""

    num_levels: int = 3
    """Number of decomposition levels."""

    wavelet: str = "haar"
    """Wavelet to use (default: Haar wavelet as in original code)."""

    mode: str = "symmetric"
    """Signal extension mode for dealing with boundaries."""


class DWT(BaseDescriptor):
    """Discrete Wavelet Transform descriptor."""

    def __init__(self, params: DWTParams | None = None) -> None:
        """Initialize DWT descriptor."""
        self.params = params or DWTParams()

    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute DWT descriptor for subsequence."""
        # Validate input
        seq = self._validate_input(subsequence)

        # Handle multivariate case
        if seq.shape[1] > 1:
            return self._compute_multivariate_dwt(seq)

        # Get sequence length and pad to power of 2 if needed
        seq = seq.flatten()
        padded_seq = self._pad_to_power_of_2(seq)

        # Compute wavelet transform
        coeffs = pywt.wavedec(
            padded_seq,
            wavelet=self.params.wavelet,
            level=self.params.num_levels,
            mode=self.params.mode,
        )

        # Convert coefficients list to array
        return np.concatenate(coeffs)

    def _pad_to_power_of_2(
        self, seq: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Pad sequence to nearest power of 2 length."""
        length = len(seq)

        # Find next power of 2
        next_pow2 = int(2 ** np.ceil(np.log2(length)))

        if next_pow2 == length:
            return seq

        # Pad with mean value
        pad_length = next_pow2 - length
        return np.pad(
            seq, (0, pad_length), mode="constant", constant_values=np.mean(seq)
        )

    def _compute_multivariate_dwt(
        self, seq: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Handle multivariate sequences."""
        all_coeffs = []
        for dim in range(seq.shape[1]):
            dim_seq = seq[:, dim]
            dim_coeffs = self(dim_seq.reshape(-1, 1))
            all_coeffs.append(dim_coeffs)
        return np.concatenate(all_coeffs)

    def decomposition_lengths(self, seq_length: int) -> list[int]:
        """Get lengths of coefficients at each decomposition level."""
        # Get padded length
        padded_length = int(2 ** np.ceil(np.log2(seq_length)))

        # Use PyWavelets to get correct coefficient lengths
        dummy_seq = np.zeros(padded_length)
        coeffs = pywt.wavedec(
            dummy_seq,
            wavelet=self.params.wavelet,
            level=self.params.num_levels,
            mode=self.params.mode,
        )

        return [len(c) for c in coeffs]
