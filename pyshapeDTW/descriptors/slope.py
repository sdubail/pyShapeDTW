from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyshapeDTW.descriptors.base import BaseDescriptor


@dataclass
class SlopeParams:
    """Parameters for Slope descriptor."""

    seg_num: int = 5
    """Number of segments to divide the sequence into."""

    scale: float = 1
    """Scale factor for x-axis in slope computation."""


class Slope(BaseDescriptor):
    """Slope descriptor for time series shape representation.

    Divides sequence into equal-length segments and computes slope of
    least squares line fit in each segment. The descriptor is y-shift invariant
    as it only captures shape through local slopes.
    """

    def __init__(self, params: SlopeParams | None = None) -> None:
        """Initialize Slope descriptor."""
        self.params = params or SlopeParams()

    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute Slope descriptor for subsequence.

        Args:
            subsequence: Time series subsequence (n_samples, n_features)

        Returns:
            descriptor: Slope descriptor (n_segments * n_features,)
        """
        # Validate input
        seq = self._validate_input(subsequence)
        length, n_features = seq.shape

        # Handle multivariate case by treating each dimension separately
        slopes = []
        for dim in range(n_features):
            dim_slopes = self._compute_slopes(seq[:, dim])
            slopes.append(dim_slopes)

        # Concatenate slopes from all dimensions
        return np.concatenate(slopes)

    def _compute_slopes(self, seq: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute slopes for each segment.

        Args:
            seq: 1D sequence

        Returns:
            slopes: Array of slopes for each segment
        """
        length = len(seq)
        seg_len = length // self.params.seg_num

        # Handle case where sequence is too short
        if seg_len < 2:  # Need at least 2 points per segment
            raise ValueError(
                f"Sequence length {length} too short for {self.params.seg_num} segments"
            )

        slopes = np.zeros(self.params.seg_num, dtype=np.float64)

        for i in range(self.params.seg_num):
            start_idx = i * seg_len
            # For last segment, include any remaining points
            end_idx = start_idx + seg_len if i < self.params.seg_num - 1 else length

            # X values for this segment
            x = np.arange(end_idx - start_idx) * self.params.scale
            y = seq[start_idx:end_idx]

            # Use polyfit to get slope
            slopes[i] = np.polyfit(x, y, 1)[0]

        return slopes
