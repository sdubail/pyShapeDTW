from typing import Optional, Tuple

import numpy as np
from dtw import dtw


def dtw_fast(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Fast DTW implementation using dtw-python package.

    Args:
        p: First sequence (n_samples1, n_features)
        q: Second sequence (n_samples2, n_features)

    Returns:
        dist: DTW distance
        match: Matching indices array of shape (n_matches, 2)
    """
    alignment = dtw(p, q, distance_only=False)
    return alignment.distance, np.column_stack((alignment.index1, alignment.index2))


def dtw_locality(
    s: np.ndarray, t: np.ndarray, w: float | None = None
) -> tuple[float, np.ndarray, int, np.ndarray]:
    """DTW with locality constraint using dtw-python package.

    Args:
        s: First sequence
        t: Second sequence
        w: Window size for locality constraint

    Returns:
        dist: DTW distance
        dMat: Distance matrix
        lPath: Length of matching path
        match: Matching indices array
    """
    window_args = {"window_size": int(w)} if w is not None else {}

    alignment = dtw(
        s,
        t,
        window_type="sakoechiba" if w is not None else "none",
        window_args=window_args,
        keep_internals=True,
    )

    match = np.column_stack((alignment.index1, alignment.index2))

    return (alignment.distance, alignment.costMatrix, len(match), match)
