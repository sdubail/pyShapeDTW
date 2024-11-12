from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def wpath2mat(p: np.ndarray) -> np.ndarray:
    """Convert warping path indices to matrix form.
    Direct translation of wpath2mat.m

    Args:
        p: Array of indices

    Returns:
        wMat: Warping matrix where each row represents the warping weights.
              wMat[i, p[i]] = 1 indicates alignment point
    """
    # Get final index
    len_signal = int(np.max(p)) + 1
    len_path = len(p)

    # Create sparse warping matrix
    wMat = np.zeros((len_path, len_signal))

    # Fill with ones at warping indices
    for i in range(len_path):
        wMat[i, int(p[i])] = 1

    return wMat


def apply_warping(signal: np.ndarray, warp_mat: np.ndarray) -> np.ndarray:
    """Apply warping matrix to signal.

    Args:
        signal: Time series to warp (n_samples, n_features)
        warp_mat: Warping matrix from wpath2mat

    Returns:
        warped_signal: Warped version of input signal
    """
    if len(signal.shape) == 1:
        signal = signal.reshape(-1, 1)

    # Apply warping matrix
    warped = warp_mat @ signal

    if warped.shape[1] == 1:
        warped = warped.flatten()

    return warped


def get_warped_segments(signal: np.ndarray, path: np.ndarray) -> list[np.ndarray]:
    """Extract warped segments based on warping path.

    Args:
        signal: Input time series
        path: Warping path as array of indices

    Returns:
        segments: List of warped segments
    """
    segments = []
    current_idx = path[0]
    current_segment = [signal[current_idx]]

    for idx in path[1:]:
        if idx == current_idx:
            # Extension of current segment
            current_segment.append(signal[idx])
        else:
            # New segment
            segments.append(np.array(current_segment))
            current_idx = idx
            current_segment = [signal[idx]]

    # Add final segment
    segments.append(np.array(current_segment))

    return segments


def stretch_path(path: np.ndarray, source_len: int, target_len: int) -> np.ndarray:
    """Stretch/compress warping path to match target length.

    Args:
        path: Original warping path indices
        source_len: Original length
        target_len: Desired length

    Returns:
        stretched_path: Stretched/compressed warping path
    """
    # Create interpolation points
    orig_points = np.linspace(0, 1, len(path))
    new_points = np.linspace(0, 1, target_len)

    # Interpolate path
    stretched_path = np.interp(new_points, orig_points, path)

    return np.round(stretched_path).astype(int)


def get_warping_amount(
    path: np.ndarray, normalize: bool = True
) -> Union[float, np.ndarray]:
    """Compute amount of warping applied at each point.

    Args:
        path: Warping path indices
        normalize: Whether to normalize by sequence length

    Returns:
        amounts: Array of warping amounts or total normalized warping amount
    """
    # Compute warping as difference from diagonal path
    diagonal = np.linspace(0, len(path) - 1, len(path))
    amounts = np.abs(path - diagonal)

    if normalize:
        return np.sum(amounts) / len(path)
    return amounts


def validate_warping_path(path: np.ndarray, len1: int, len2: int) -> bool:
    """Validate a warping path meets DTW constraints.

    Args:
        path: Warping path as Nx2 array of indices
        len1: Length of first sequence
        len2: Length of second sequence

    Returns:
        valid: Whether path is valid
    """
    if len(path.shape) != 2 or path.shape[1] != 2:
        return False

    # Check boundary conditions
    if not (path[0, 0] == 0 and path[0, 1] == 0):
        return False
    if not (path[-1, 0] == len1 - 1 and path[-1, 1] == len2 - 1):
        return False

    # Check monotonicity and continuity
    for i in range(1, len(path)):
        diff = path[i] - path[i - 1]
        # Steps must be 0 or 1 in each dimension
        if np.any(diff < 0) or np.any(diff > 1):
            return False
        # Must move in at least one dimension
        if np.all(diff == 0):
            return False

    return True


class WarpingPath:
    """Class to handle warping paths with useful methods."""

    def __init__(self, indices1: np.ndarray, indices2: np.ndarray):
        """Initialize with matching indices from DTW.

        Args:
            indices1: Indices for first sequence
            indices2: Indices for second sequence
        """
        self.path = np.column_stack((indices1, indices2))

    @property
    def indices1(self) -> np.ndarray:
        """Get indices for first sequence."""
        return self.path[:, 0]

    @property
    def indices2(self) -> np.ndarray:
        """Get indices for second sequence."""
        return self.path[:, 1]

    def to_matrix(self) -> np.ndarray:
        """Convert to warping matrix form."""
        return wpath2mat(self.indices1)

    def apply(self, signal: np.ndarray, sequence: int = 1) -> np.ndarray:
        """Apply warping to signal.

        Args:
            signal: Signal to warp
            sequence: Which sequence indices to use (1 or 2)
        """
        indices = self.indices1 if sequence == 1 else self.indices2
        warp_mat = wpath2mat(indices)
        return apply_warping(signal, warp_mat)

    def get_segments(self, signal: np.ndarray, sequence: int = 1) -> list[np.ndarray]:
        """Get warped segments.

        Args:
            signal: Signal to segment
            sequence: Which sequence indices to use (1 or 2)
        """
        indices = self.indices1 if sequence == 1 else self.indices2
        return get_warped_segments(signal, indices)

    def get_warping_amount(self, normalize: bool = True) -> Union[float, np.ndarray]:
        """Compute warping amount."""
        return get_warping_amount(self.indices1, normalize)

    def is_valid(self, len1: int, len2: int) -> bool:
        """Check if warping path is valid."""
        return validate_warping_path(self.path, len1, len2)

    def plot(self, ax=None):
        """Plot warping path."""
        if ax is None:
            ax = plt.gca()
        ax.plot(self.indices1, self.indices2, "-o")
        ax.set_xlabel("Sequence 1")
        ax.set_ylabel("Sequence 2")
        ax.grid(True)
