from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyshapeDTW.descriptors.base import BaseDescriptor


@dataclass
class HOG1DParams:
    """Parameters for HOG1D descriptor."""

    n_bins: int = 8
    """Number of orientation bins."""

    cells: tuple[int, int] = (1, 25)
    """Cell dimensions [height, width]."""

    overlap: int = 0
    """Overlap between cells."""

    scale: float = 0.1
    """Scale factor for gradient computation."""

    signed: bool = True
    """Whether to use signed gradients (-π/2 to π/2) or unsigned (0 to π/2)."""


class HOG1D(BaseDescriptor):
    """1D Histogram of Oriented Gradients descriptor.

    Implementation of HOG descriptor adapted for 1D time series, used as
    the primary shape descriptor in the ShapeDTW paper.
    """

    def __init__(self, params: HOG1DParams | None = None) -> None:
        """Initialize HOG1D descriptor.

        Args:
            params: Parameters for HOG1D computation
        """
        self.params = params or HOG1DParams()

        # Precompute angle bins
        if self.params.signed:
            self.angles = np.linspace(
                -np.pi / 2, np.pi / 2, self.params.n_bins + 1, dtype=np.float64
            )
        else:
            self.angles = np.linspace(
                0, np.pi / 2, self.params.n_bins + 1, dtype=np.float64
            )

        # Compute center angles for interpolation
        self.center_angles = (self.angles[:-1] + self.angles[1:]) / 2

    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute HOG1D descriptor for subsequence.

        Args:
            subsequence: Time series subsequence

        Returns:
            descriptor: HOG1D descriptor vector
        """
        # Validate input
        seq = self._validate_input(subsequence)

        # Compute gradients
        grads, angles = self._compute_gradients(seq)

        # Compute histogram for each cell
        cell_width = self.params.cells[1]
        stride = cell_width - self.params.overlap

        n_cells = len(range(0, len(grads) - cell_width + 1, stride))
        histograms = np.zeros((n_cells, self.params.n_bins), dtype=np.float64)

        for i, start in enumerate(range(0, len(grads) - cell_width + 1, stride)):
            cell_grads = grads[start : start + cell_width]
            cell_angles = angles[start : start + cell_width]

            histograms[i] = self._compute_cell_histogram(cell_grads, cell_angles)

        return histograms.flatten()

    def _compute_gradients(
        self, seq: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute gradients and their orientations.

        Args:
            seq: Input sequence

        Returns:
            magnitudes: Gradient magnitudes
            angles: Gradient angles
        """
        # Compute centered gradients
        grads = np.gradient(seq.flatten()) / self.params.scale

        # Compute angles - use 1 as dx since we're dealing with time series
        angles = np.arctan2(grads, 1)

        if not self.params.signed:
            angles = np.abs(angles)

        return np.abs(grads), angles

    def _compute_cell_histogram(
        self, grads: npt.NDArray[np.float64], angles: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute orientation histogram for one cell.

        Args:
            grads: Gradient magnitudes in cell
            angles: Gradient angles in cell

        Returns:
            histogram: Orientation histogram
        """
        histogram = np.zeros(self.params.n_bins, dtype=np.float64)

        # For each gradient in the cell
        for grad, angle in zip(grads, angles, strict=False):
            # Find which bin it belongs to
            bin_idx = np.searchsorted(self.angles, angle) - 1

            # Handle edge cases
            if bin_idx >= self.params.n_bins:
                bin_idx = self.params.n_bins - 1
            elif bin_idx < 0:
                bin_idx = 0

            # Linear interpolation between neighboring bins
            if bin_idx == 0:
                # First bin
                weight = np.cos(self.center_angles[0] - angle)
                histogram[0] += grad * weight
                histogram[1] += grad * (1 - weight)
            elif bin_idx == self.params.n_bins - 1:
                # Last bin
                weight = np.cos(angle - self.center_angles[-1])
                histogram[-2] += grad * (1 - weight)
                histogram[-1] += grad * weight
            else:
                # Middle bins
                angle_diff = angle - self.center_angles[bin_idx]
                if abs(angle_diff) > abs(angle - self.center_angles[bin_idx + 1]):
                    weight = np.cos(self.center_angles[bin_idx + 1] - angle)
                    histogram[bin_idx] += grad * (1 - weight)
                    histogram[bin_idx + 1] += grad * weight
                else:
                    weight = np.cos(self.center_angles[bin_idx - 1] - angle)
                    histogram[bin_idx - 1] += grad * weight
                    histogram[bin_idx] += grad * (1 - weight)

        # Normalize histogram
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram /= norm

        return histogram
