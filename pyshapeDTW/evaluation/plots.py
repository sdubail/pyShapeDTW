import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyshapeDTW.elastic_measure.warping import wpath2mat

# Set global plotting style
plt.style.use("default")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica"]
rcParams["font.size"] = 15
rcParams["axes.linewidth"] = 3
rcParams["lines.linewidth"] = 3
rcParams["grid.linewidth"] = 1.5
rcParams["xtick.major.width"] = 2
rcParams["ytick.major.width"] = 2


def plot_elastic_matching(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    match: npt.NDArray[np.int64],
    shift: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot elastic matching between two sequences with correspondence lines."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Apply vertical shift if requested
    if shift:
        shift_amount = max(t) - min(t)
        s = s + shift_amount
    else:
        shift_amount = 0

    # Plot sequences
    ax.plot(s, "r-o", markersize=6, label="Query")
    ax.plot(t, "k-s", markersize=6, label="Reference")

    # Plot matching lines
    for i, j in match:
        ax.plot([i, j], [s[i], t[j]], "b-", alpha=0.3, linewidth=1)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=25, frameon=False)

    if title:
        ax.set_title(title, fontsize=25, pad=15)

    # Style ticks
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    return ax


def plot_warped_sequences(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    match: npt.NDArray[np.int64],
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot time-warped sequences overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Get warped sequences
    wp = wpath2mat(match[:, 0])
    wq = wpath2mat(match[:, 1])

    warped_s = wp @ s
    warped_t = wq @ t

    # Plot warped sequences
    ax.plot(warped_s, "r-", label="Query (warped)")
    ax.plot(warped_t, "k--", label="Reference (warped)")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=25, frameon=False)

    if title:
        ax.set_title(title, fontsize=25, pad=15)

    # Style ticks
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    return ax


def plot_alignment_path(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    matches: dict[str, npt.NDArray[np.int64]],
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot alignment paths from different methods."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 15))

    # Set up plot limits
    ax.set_xlim(0, len(s))
    ax.set_ylim(0, len(t))

    # Plot diagonal line for reference
    diag = np.linspace(0, min(len(s), len(t)))
    ax.plot(diag, diag, "k--", alpha=0.3, linewidth=2)

    # Colors for different methods
    colors = {"GT": "k", "DTW": "m", "dDTW": "b", "shapeDTW": "r"}

    # Plot each alignment path
    for method, match in matches.items():
        if "shapeDTW" in method:
            method = "shapeDTW"
        color = colors.get(method, "gray")
        ax.plot(match[:, 0], match[:, 1], "-", label=method, color=color)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Original", fontsize=25, labelpad=10)
    ax.set_ylabel("Simulated", fontsize=25, labelpad=10)
    ax.legend(fontsize=15, frameon=False)

    if title:
        ax.set_title(title, fontsize=25, pad=15)
    else:
        ax.set_title("Alignment Path", fontsize=25, pad=15)

    # Make plot square
    ax.set_aspect("equal")

    # Style ticks
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    return ax


def plot_all_alignments(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    matches: dict[str, npt.NDArray[np.int64]],
    figsize: tuple[int, int] = (15, 15),
) -> Figure:
    """Create combined plot with all three visualizations."""
    fig = plt.figure(figsize=figsize)

    # Elastic matching plot
    ax1 = plt.subplot(311)
    plot_elastic_matching(s, t, matches["DTW"], ax=ax1, title="Elastic Matching DTW")

    ax2 = plt.subplot(312)
    key_method = None
    for key in matches.keys():
        if "shapeDTW" in key:
            key_method = key

    if key_method is not None:
        plot_elastic_matching(
            s, t, matches[key_method], ax=ax2, title=f"Elastic Matching {key_method}"
        )

    # Alignment paths plot
    ax3 = plt.subplot(313)
    plot_alignment_path(s, t, matches, ax=ax3, title="Alignment Paths")

    plt.tight_layout()
    return fig


def plot_alignment_eval(results_df: pd.DataFrame) -> plt.Figure:
    """Plot alignment errors vs stretch percentage."""
    fig, ax = plt.subplots(figsize=(15, 10))

    for method in results_df["method"].unique():
        if "HOG1D" not in method:
            method_data = results_df[results_df["method"] == method]

            means = method_data.groupby("stretch_pct")["error"].mean()
            stds = method_data.groupby("stretch_pct")["error"].std()

            ax.plot(
                means.index,
                means.values,
                marker="o",
                markersize=8,
                label=method,
                linewidth=3,
            )
            # ax.fill_between(
            #     means.index,
            #     means.values - stds.values,
            #     means.values + stds.values,
            #     alpha=0.05,
            # )

    ax.set_xlabel("Stretch Percentage", fontsize=25, labelpad=10)
    ax.set_ylabel("Mean Absolute Deviation", fontsize=25, labelpad=10)
    # ax.set_title("Alignment Error vs Stretch Percentage", fontsize=25, pad=15)
    ax.legend(fontsize=20, frameon=False)
    ax.grid(True, alpha=0.3)

    # Style ticks
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    return fig


def plot_classification_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Plot shapeDTW vs DTW classification accuracies, one subplot per descriptor."""
    # Get DTW accuracies
    dtw_results = results_df[results_df["method"] == "DTW"]

    # Identify shapeDTW methods (descriptors)
    shape_methods = [m for m in results_df["method"].unique() if m != "DTW"]
    num_descriptors = len(shape_methods)

    # Define figure size and subplots layout
    fig_width_per_plot = 10  # Width of each subplot
    fig = plt.figure(figsize=(fig_width_per_plot * num_descriptors, 10))

    for i, method in enumerate(shape_methods):
        ax = fig.add_subplot(1, num_descriptors, i + 1)

        shape_results = results_df[results_df["method"] == method]
        merged = pd.merge(
            dtw_results, shape_results, on="dataset", suffixes=("_dtw", "_shape")
        )

        ax.scatter(
            merged["accuracy_dtw"],
            merged["accuracy_shape"],
            alpha=0.6,
            s=100,
            color="blue",
        )

        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.6, zorder=0)

        ax.set_xlabel("DTW Accuracy", fontsize=35, labelpad=10)
        ax.set_ylabel("ShapeDTW Accuracy", fontsize=35, labelpad=10)
        ax.set_title(method, fontsize=35, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=35, frameon=False)

        # Style ticks
        ax.tick_params(width=2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig
