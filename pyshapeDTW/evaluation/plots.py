import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

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


def plot_alignment_eval(results_df: pd.DataFrame) -> plt.Figure:
    """Plot alignment errors vs stretch percentage."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in results_df["method"].unique():
        method_data = results_df[results_df["method"] == method]

        means = method_data.groupby("stretch_pct")["error"].mean()
        stds = method_data.groupby("stretch_pct")["error"].std()

        ax.plot(means.index, means.values, marker="o", label=method)
        ax.fill_between(
            means.index,
            means.values - stds.values,
            means.values + stds.values,
            alpha=0.2,
        )

    ax.set_xlabel("Stretch Percentage")
    ax.set_ylabel("Alignment Error")
    ax.set_title("Alignment Error vs Stretch Percentage")
    ax.legend()
    ax.grid(True)

    return fig


def plot_classification_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Plot shapeDTW vs DTW classification accuracies.

    Args:
        results_df: DataFrame containing classification results

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get DTW accuracies
    dtw_results = results_df[results_df["method"] == "DTW"]

    # Plot each shapeDTW variant
    shape_methods = [m for m in results_df["method"].unique() if m != "DTW"]

    for method in shape_methods:
        shape_results = results_df[results_df["method"] == method]
        merged = pd.merge(
            dtw_results, shape_results, on="dataset", suffixes=("_dtw", "_shape")
        )

        ax.scatter(
            merged["accuracy_dtw"], merged["accuracy_shape"], label=method, alpha=0.6
        )

    # Add diagonal line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, zorder=0)

    ax.set_xlabel("DTW Accuracy")
    ax.set_ylabel("ShapeDTW Accuracy")
    ax.set_title("Classification Performance Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig
