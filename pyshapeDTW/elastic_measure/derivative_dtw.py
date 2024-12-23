from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from dtw import dtw

from pyshapeDTW.elastic_measure.base_dtw import dtw_fast, dtw_locality
from pyshapeDTW.elastic_measure.warping import wpath2mat


class DerivativeDTW:
    def __init__(
        self, metric: str = "euclidean", step_pattern: str = "symmetric2"
    ) -> None:
        """
        Initialize DTW parameters.

        Args:
            metric: Distance metric to use for DTW computation
            step_pattern: Step pattern for DTW warping path
        """
        self.metric: str = metric
        self.step_pattern: str = step_pattern

    def _validate_input(
        self, sequence1: npt.NDArray[np.float64], sequence2: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Validate input time series and ensure 2D shape.

        Args:
            sequence1: First time series
            sequence2: Second time series

        Returns:
            Tuple of validated sequences reshaped to (n,1) if 1D
        """
        if (
            sequence1 is None
            or sequence2 is None
            or len(sequence1) == 0
            or len(sequence2) == 0
        ):
            raise ValueError("Input two univariate/multivariate time series instances.")

        # Only reshape if 1D, preserve original shape if already 2D
        sequence1_2d = sequence1.reshape(-1, 1) if sequence1.ndim == 1 else sequence1
        sequence2_2d = sequence2.reshape(-1, 1) if sequence2.ndim == 1 else sequence2

        len1, dims1 = sequence1_2d.shape
        len2, dims2 = sequence2_2d.shape

        if len1 < dims1 or len2 < dims2:
            raise ValueError(
                "Each dimension of time series should be organized column-wise."
            )

        if dims1 != dims2:
            raise ValueError("Two time series should have the same dimensions.")

        return sequence1_2d, sequence2_2d

    def calcKeoghGradient1D(
        self, sequence: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Calculate gradient as defined by Keogh, 2001.

        Gradient formula:
            grad(q_i) = [(q_i - q_{i-1}) + (q_{i+1} - q_{i-1}) / 2] / 2

        Args:
            sequence: 1D numpy array representing the time series.

        Returns:
            seq_grad: 1D numpy array of gradients, same length as input sequence.
        """
        if (
            sequence is None
            or not isinstance(sequence, np.ndarray)
            or len(sequence.shape) != 1
        ):
            raise ValueError(
                "Please input a univariate time series as a 1D numpy array."
            )

        # Pad the sequence at both ends
        seq_pad: npt.NDArray[np.float64] = np.pad(sequence, (1, 1), mode="edge")

        # Calculate gradients
        seq_grad: npt.NDArray[np.float64] = (
            (seq_pad[1:-1] - seq_pad[:-2]) + (seq_pad[2:] - seq_pad[:-2]) / 2
        ) / 2

        return seq_grad

    def calcKeoghGradient(
        self, sequence: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Usable for both univariate and multivariate time series.

        Args:
            sequence: mxn numpy array
                    m -- number of time stamps
                    n -- number of dimensions
                    Generally, m >> n; if not, a warning is raised.

        Returns:
            grads: mxn numpy array of gradients, one for each dimension.
        """
        if sequence is None or not isinstance(sequence, np.ndarray):
            raise ValueError(
                "Input must be a univariate/multivariate time series instance as a numpy array."
            )

        len_seq, dims = sequence.shape
        if len_seq < dims:
            raise ValueError(
                "Each dimension of the time series should be organized column-wise."
            )

        grads: list[npt.NDArray[np.float64]] = []

        for i in range(dims):
            # Compute gradient for each dimension using calcKeoghGradient1D
            grad = self.calcKeoghGradient1D(sequence[:, i])
            grads.append(grad)

        # Combine gradients for all dimensions column-wise
        return np.column_stack(grads)

    def _compute_aligned_distance(
        self,
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
        match: npt.NDArray[np.int64],
    ) -> float:
        """Compute Euclidean distance between sequences aligned by warping path.

        Args:
            p: First sequence
            q: Second sequence
            match: Warping path indices

        Returns:
            distance: Euclidean distance between aligned sequences
        """
        # Convert matching indices to warping matrices
        wp: npt.NDArray[np.float64] = wpath2mat(match[:, 0])
        wq: npt.NDArray[np.float64] = wpath2mat(match[:, 1])

        # Apply warping and compute Euclidean distance
        warped_p: npt.NDArray[np.float64] = wp @ p
        warped_q: npt.NDArray[np.float64] = wq @ q

        return float(np.sqrt(np.sum((warped_p - warped_q) ** 2)))

    def __call__(
        self, sequence1: npt.NDArray[np.float64], sequence2: npt.NDArray[np.float64]
    ) -> tuple[float, float, npt.NDArray[np.int64]]:
        """
        Perform derivative DTW computation.

        Args:
            sequence1: First time series (m1 x n numpy array)
            sequence2: Second time series (m2 x n numpy array)

        Returns:
            Tuple containing:
                dDerivative: DTW distance using shape descriptors
                dRaw: Distance between aligned raw sequences
                match: Optimal warping path as indices array
        """
        # Validate input and ensure 2D shape
        sequence1, sequence2 = self._validate_input(sequence1, sequence2)

        # 1. Calculate derivatives
        grads_p = self.calcKeoghGradient(sequence1)
        grads_q = self.calcKeoghGradient(sequence2)

        # 2. Run DTW
        alignment = dtw(
            grads_p,
            grads_q,
            dist_method=self.metric,
            step_pattern=self.step_pattern,
            keep_internals=True,
        )

        # Extract matching indices
        match: npt.NDArray[np.int64] = np.column_stack(
            (alignment.index1, alignment.index2)
        ).astype(np.int64)

        # Cumulative distances along the warping path
        dDerivative = self._compute_aligned_distance(grads_p, grads_q, match)
        dRaw = self._compute_aligned_distance(sequence1, sequence2, match)

        return dDerivative, dRaw, match
