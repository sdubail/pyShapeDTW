from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyshapeDTW.descriptors.base import BaseDescriptor


@dataclass
class HOG1DParams:
    """Parameters for HOG1D descriptor."""

    n_bins: int = 8
    cells: tuple[int, int] = (1, 25)  # height, width
    overlap: int = 0
    scale: float = 0.1
    signed: bool = True


class HOG1D(BaseDescriptor):
    def __init__(self, params: HOG1DParams | None = None):
        self.params = params or HOG1DParams()

        # Precompute angle bins
        if self.params.signed:
            self.angles = np.linspace(-np.pi / 2, np.pi / 2, self.params.n_bins + 1)
        else:
            self.angles = np.linspace(0, np.pi / 2, self.params.n_bins + 1)

        # Center angles for cosine interpolation
        self.center_angles = (self.angles[:-1] + self.angles[1:]) / 2

    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Validate input
        seq = self._validate_input(subsequence)

        # Handle multivariate case
        if seq.shape[1] > 1:
            return self._compute_multivariate_hog1d(seq)

        # Compute gradients and angles
        grads, angles = self._compute_gradients(seq)

        # Compute histograms for each cell
        cell_width = self.params.cells[1]
        stride = cell_width - self.params.overlap

        n_cells = len(range(0, len(grads) - cell_width + 1, stride))
        histograms = np.zeros((n_cells, self.params.n_bins))

        for i, start in enumerate(range(0, len(grads) - cell_width + 1, stride)):
            cell_grads = grads[start : start + cell_width]
            cell_angles = angles[start : start + cell_width]
            histograms[i] = self._compute_cell_histogram(cell_grads, cell_angles)

        return histograms.flatten()

    def _compute_gradients(
        self, seq: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute gradients exactly as in MATLAB implementation."""
        seq_flat = seq.flatten()
        dx = self.params.scale  # Fixed dx scale

        if len(seq_flat) < 3:
            # Handle short sequences
            dy = np.diff(seq_flat)
            grads = np.abs(dy / dx)
            angles = np.arctan2(dy, dx)
            return np.pad(grads, (0, 1)), np.pad(angles, (0, 1))

        # Compute centered differences
        dy = seq_flat[2:] - seq_flat[:-2]
        grads = np.abs(dy / (2 * dx))  # Magnitude
        angles = np.arctan2(dy, 2 * dx)  # Angle

        # Pad first and last gradient
        grads = np.pad(grads, (1, 1), mode="edge")
        angles = np.pad(angles, (1, 1), mode="edge")

        if not self.params.signed:
            angles = np.abs(angles)

        return grads, angles

    def _compute_cell_histogram(
        self, grads: npt.NDArray[np.float64], angles: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute histogram for one cell using cosine interpolation."""
        histogram = np.zeros(self.params.n_bins)

        for grad, angle in zip(grads, angles, strict=False):
            # Find which bin the angle falls into
            bin_idx = np.searchsorted(self.angles, angle) - 1

            # Handle edge cases
            if bin_idx >= self.params.n_bins:
                bin_idx = self.params.n_bins - 1
            elif bin_idx < 0:
                bin_idx = 0

            # Cosine interpolation
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
                # Middle bins - find closest center angle
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

    def _compute_multivariate_hog1d(
        self, seq: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Handle multivariate sequences.

        Args:
            seq: Multivariate input sequence (n_samples, n_features)

        Returns:
            descriptors: Concatenated HOG1D descriptors for each dimension
        """
        all_coeffs = []
        for dim in range(seq.shape[1]):
            dim_seq = seq[:, dim]
            dim_coeffs = self(dim_seq.reshape(-1, 1))
            all_coeffs.append(dim_coeffs)
        return np.concatenate(all_coeffs)
