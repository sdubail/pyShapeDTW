import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pyshapeDTW.elastic_measure.warping import wpath2mat


def plot_elastic_matching(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    match: npt.NDArray[np.int64],
    shift: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot elastic matching between two sequences.
    Translation of plotElasticMatching.m

    Args:
        s: First sequence
        t: Second sequence
        match: Matching indices array
        shift: Whether to shift first sequence up for visualization
        ax: Matplotlib axes to plot on

    Returns:
        ax: The axes object with the plot
    """
    if ax is None:
        ax = plt.gca()

    # Apply vertical shift if requested
    if shift:
        shift_amount = max(t) - min(t)
        s = s + shift_amount
    else:
        shift_amount = 0

    # Plot sequences
    ax.plot(s, "r-o", linewidth=2, markersize=4, label="Query")
    ax.plot(t, "k-s", linewidth=2, markersize=4, label="Reference")

    # Plot matching lines
    for i, j in match:
        ax.plot([i, j], [s[i], t[j]], "b-", alpha=0.3)

    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_warped_ts(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    match: npt.NDArray[np.int64],
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot both elastic matching and warped sequences.
    Translation of plotWarpedTS.m

    Args:
        s: First sequence
        t: Second sequence
        match: Matching indices array
        fig: Matplotlib figure to plot on

    Returns:
        fig: The figure with the plots
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))

    # Plot 1: Elastic matching
    ax1 = fig.add_subplot(211)
    plot_elastic_matching(s, t, match, shift=True, ax=ax1)
    ax1.set_title("Elastic Matching")

    # Plot 2: Warped sequences overlay
    ax2 = fig.add_subplot(212)

    # Get warped sequences
    wp = wpath2mat(match[:, 0])
    wq = wpath2mat(match[:, 1])

    warped_s = wp @ s
    warped_t = wq @ t

    # Plot warped sequences
    ax2.plot(warped_s, "r-", label="Query (warped)", linewidth=2)
    ax2.plot(warped_t, "k--", label="Reference (warped)", linewidth=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("Warped Sequences")

    fig.tight_layout()
    return fig
