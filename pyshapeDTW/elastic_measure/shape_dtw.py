from typing import Protocol

import numpy as np
import numpy.typing as npt
from dtw import dtw

from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.elastic_measure.warping import wpath2mat


class ShapeDTW:
    """Shape-based Dynamic Time Warping implementation."""

    def __init__(self, seqlen: int, metric: str = "euclidean") -> None:
        """Initialize Shape-DTW.

        Args:
            seqlen: Length of subsequences used for shape descriptors
            metric: Distance metric for comparing shape descriptors
        """
        self.seqlen: int = seqlen
        self.metric: str = metric

    def __call__(
        self,
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
        descriptor: BaseDescriptor,
    ) -> tuple[float, float, int, npt.NDArray[np.int64]]:
        """Compute Shape-DTW between sequences.

        Args:
            p: First sequence
            q: Second sequence
            descriptor: Shape descriptor to use

        Returns:
            raw_distance: Distance between aligned raw sequences
            shape_distance: DTW distance using shape descriptors
            path_length: Length of optimal warping path
            match: Optimal warping path as indices array
        """
        return self.compute(p, q, descriptor)

    def compute(
        self,
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
        descriptor: BaseDescriptor,
    ) -> tuple[float, float, int, npt.NDArray[np.int64]]:
        """Compute Shape-DTW between sequences.

        Args:
            p: First sequence
            q: Second sequence
            descriptor: Shape descriptor to use

        Returns:
            raw_distance: Distance between aligned raw sequences
            shape_distance: DTW distance using shape descriptors
            path_length: Length of optimal warping path
            match: Optimal warping path as indices array
        """
        # Ensure sequences are 2D
        p_arr: npt.NDArray[np.float64] = np.asarray(p, dtype=np.float64)
        q_arr: npt.NDArray[np.float64] = np.asarray(q, dtype=np.float64)
        if p_arr.ndim == 1:
            p_arr = p_arr.reshape(-1, 1)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(-1, 1)

        # Compute shape descriptors
        p_desc: npt.NDArray[np.float64] = self._compute_shape_descriptors(
            p_arr, descriptor
        )
        q_desc: npt.NDArray[np.float64] = self._compute_shape_descriptors(
            q_arr, descriptor
        )

        # Run DTW with shape descriptors
        alignment = dtw(p_desc, q_desc, dist_method=self.metric, keep_internals=True)

        # Extract matching indices
        match: npt.NDArray[np.int64] = np.column_stack(
            (alignment.index1, alignment.index2)
        ).astype(np.int64)

        # Compute distance between aligned raw sequences
        aligned_distance: float = self._compute_aligned_distance(p_arr, q_arr, match)

        return (aligned_distance, float(alignment.distance), len(match), match)

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

    def _compute_shape_descriptors(
        self, seq: npt.NDArray[np.float64], descriptor: BaseDescriptor
    ) -> npt.NDArray[np.float64]:
        """Compute shape descriptors for sequence.

        Args:
            seq: Input sequence
            descriptor: Shape descriptor

        Returns:
            descriptors: Array of shape descriptors
        """
        # Sample subsequences
        subsequences: npt.NDArray[np.float64] = self._sample_subsequences(seq)

        # Compute descriptor for each subsequence
        descriptors: list[npt.NDArray[np.float64]] = []
        for i in range(len(subsequences)):
            desc = descriptor(subsequences[i])
            descriptors.append(desc)

        return np.array(descriptors, dtype=np.float64)

    def _sample_subsequences(
        self, seq: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Sample subsequences of length seqlen centered at each point.

        Args:
            seq: Input sequence

        Returns:
            subsequences: Array of subsequences
        """
        n_samples: int = len(seq)
        half_len: int = self.seqlen // 2

        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)

        subsequences: list[npt.NDArray[np.float64]] = []
        for i in range(n_samples):
            # Handle boundary conditions
            start: int = max(0, i - half_len)
            end: int = min(n_samples, i + half_len + 1)

            # Extract subsequence
            subseq: npt.NDArray[np.float64] = seq[start:end]

            # Pad if needed
            if len(subseq) <= self.seqlen:
                pad_left: int = max(0, half_len - i)
                pad_right: int = max(0, half_len - (n_samples - (i + 1)))

                pad_width = [(pad_left, pad_right)] + [(0, 0)] * (seq.ndim - 1)

                subseq = np.pad(subseq, pad_width, mode="edge")

            subsequences.append(subseq)

        return np.array(subsequences, dtype=np.float64)


class ShapeDTWMulti(ShapeDTW):
    """Multivariate version of Shape-DTW."""

    def _compute_shape_descriptors(
        self, seq: npt.NDArray[np.float64], descriptor: BaseDescriptor
    ) -> npt.NDArray[np.float64]:
        """Compute shape descriptors handling multiple dimensions.

        Args:
            seq: Input multivariate sequence
            descriptor: Shape descriptor

        Returns:
            descriptors: Array of concatenated shape descriptors
        """
        # Handle each dimension separately then concatenate
        descriptors: list[npt.NDArray[np.float64]] = []
        for dim in range(seq.shape[1]):
            dim_seq = seq[:, dim : dim + 1]
            dim_desc = super()._compute_shape_descriptors(dim_seq, descriptor)
            descriptors.append(dim_desc)

        return np.concatenate(descriptors, axis=1)
