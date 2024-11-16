from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .base import BaseDescriptor


@dataclass
class PAAParams:
    """Parameters for PAA (Piecewise Aggregate Approximation) descriptor."""

    seg_num: int = 10
    """Number of segments to divide the sequence into."""

    seg_len: int | None = None
    """Length of each segment. If None, computed from seg_num."""

    priority: str = "seg_num"
    """Which parameter to prioritize ('seg_num' or 'seg_len')."""


class PAA(BaseDescriptor):
    """Piecewise Aggregate Approximation descriptor."""

    def __init__(self, params: PAAParams | None = None) -> None:
        """Initialize PAA descriptor."""
        self.params = params or PAAParams()

    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute PAA descriptor for subsequence.

        Args:
            subsequence: Time series subsequence (n_samples, n_features)

        Returns:
            descriptor: PAA descriptor (n_segments, n_features)
        """
        # Validate input
        seq = self._validate_input(subsequence)
        length, n_features = seq.shape

        if length == 0:
            raise ValueError("Input should not be empty")

        # If sequence is shorter than requested segments, use full sequence
        if self.params.priority == "seg_num" and length <= self.params.seg_num:
            return seq

        # Determine segmentation parameters
        seg_num, seg_len = self._compute_segment_params(length)

        # Compute segment boundaries
        if seg_num * seg_len < length:
            # Last segment gets remaining points
            seg_lengths = np.array(
                [seg_len] * (seg_num - 1) + [length - seg_len * (seg_num - 1)]
            )
        elif seg_num * seg_len == length:
            seg_lengths = np.array([seg_len] * seg_num)
        else:
            # Adjust last segment length
            seg_lengths = np.array(
                [seg_len] * (seg_num - 1) + [seg_len - (seg_len * seg_num - length)]
            )

        # Compute segment boundaries
        boundaries = np.concatenate(([0], np.cumsum(seg_lengths)))

        # Compute mean for each segment and feature
        means = np.zeros((seg_num, n_features), dtype=np.float64)
        for i in range(seg_num):
            start = boundaries[i]
            end = boundaries[i + 1]
            means[i] = np.mean(seq[start:end], axis=0)
        return means.squeeze()

    def _compute_segment_params(self, length: int) -> tuple[int, int]:
        """Compute number of segments and segment length.

        Args:
            length: Length of input sequence

        Returns:
            seg_num: Number of segments
            seg_len: Length of each segment
        """
        if self.params.priority == "seg_num":
            # Use minimum of sequence length and requested segments
            seg_num = min(self.params.seg_num, length)
            seg_len = length // seg_num

        else:  # priority == 'seg_len'
            if self.params.seg_len is None:
                raise ValueError("seg_len must be specified when priority is 'seg_len'")

            seg_len = min(self.params.seg_len, length)
            seg_num = max(1, length // seg_len)

        # Ensure we always have at least one point per segment
        if seg_len < 1:
            seg_len = 1
            seg_num = length

        return seg_num, seg_len
