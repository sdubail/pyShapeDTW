from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ScaleParams:
    """Parameters for smooth curve scaling."""

    len: int
    max_derivative: float = 1.0
    n_nest: int = 10  # Controls smoothness


@dataclass
class StretchParams:
    """Parameters for temporal stretching."""

    percentage: float = 0.15  # Percentage of points to stretch
    amount: int = 2  # Maximum stretch amount


def simulate_smooth_curve(params: ScaleParams) -> npt.NDArray[np.float64]:
    """Generate a smooth scaling curve.
    Translation of simulateSmoothCurve.m

    Args:
        params: Parameters for curve generation

    Returns:
        curve: Smooth scaling curve
    """
    # Generate initial random curve using sin
    dt = np.sin(np.random.randn(params.len)) * params.max_derivative

    # Nested integration for smoothing
    for _ in range(params.n_nest):
        cum_dt = np.cumsum(dt)

        # Normalize to [-2π, 2π] range if needed
        if (max(cum_dt) - min(cum_dt)) > 2 * np.pi:
            cum_dt = (cum_dt - min(cum_dt)) / np.ptp(cum_dt) * (2 * np.pi) + min(cum_dt)

        dt = np.sin(cum_dt) * params.max_derivative

    return np.cumsum(dt)


def stretching_ts(
    length: int, params: StretchParams
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Create warped version of time series by stretching segments.
    Translation of stretchingTS.m

    Args:
        length: Length of original time series
        params: Parameters for stretching

    Returns:
        simulated_idx: Indices after stretching
        match: Ground truth alignment
    """
    # Number of points to stretch
    n_pts = round(params.percentage * length)

    # Randomly select points to stretch
    rng = np.random.default_rng()
    idx_pts = rng.choice(length, size=n_pts, replace=False)
    amounts = rng.integers(1, params.amount + 1, size=n_pts)

    # Create stretching map
    stretches = np.zeros(length)
    stretches[idx_pts] = amounts

    # Create matching indices
    simulated_idx = []
    match = []
    cnt_len = 0

    for i in range(length):
        stretch = int(stretches[i])
        if stretch == 0:
            cnt_len += 1
            simulated_idx.append(i)
            match.append([i, cnt_len - 1])
        else:
            for _ in range(stretch):
                cnt_len += 1
                simulated_idx.append(i)
                match.append([i, cnt_len - 1])

    return np.array(simulated_idx), np.array(match)


def scale_time_series(
    ts: npt.NDArray[np.float64],
    scale_curve: npt.NDArray[np.float64],
    scale_range: tuple[float, float] = (0.4, 1.0),
) -> npt.NDArray[np.float64]:
    """Scale time series using smooth curve.

    Args:
        ts: Time series to scale
        scale_curve: Scaling curve
        scale_range: Min/max scaling factors

    Returns:
        scaled: Scaled time series
    """
    # Normalize scale curve to desired range
    normalized = (scale_curve - min(scale_curve)) / np.ptp(scale_curve)
    scale = normalized * (scale_range[1] - scale_range[0]) + scale_range[0]

    return scale * ts
