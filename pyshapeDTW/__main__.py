from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from dtw import dtw

from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.descriptors.hog1d import HOG1D, HOG1DParams
from pyshapeDTW.descriptors.paa import PAA, PAAParams
from pyshapeDTW.elastic_measure.shape_dtw import ShapeDTW
from pyshapeDTW.simulation.transforms import (
    ScaleParams,
    StretchParams,
    scale_time_series,
    simulate_smooth_curve,
    stretching_ts,
)
from pyshapeDTW.visualization.plots import plot_warped_ts

app = typer.Typer()

DESCRIPTORS: dict[str, BaseDescriptor] = {
    "hog1d": HOG1D(HOG1DParams(cells=(1, 25))),
    "paa": PAA(PAAParams(seg_num=4)),
}


def generate_test_sequences(
    length: int = 200,
    scale_range: tuple[float, float] = (0.4, 1.0),
    stretch_percentage: float = 0.15,
    stretch_amount: int = 2,
    noise_level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore
    """Generate test sequences with known alignment."""
    # Generate base sequence
    t = np.linspace(0, 4 * np.pi, length)
    seq = np.sin(t) + 0.5 * np.sin(3 * t)

    # Generate scaling
    scale_params = ScaleParams(len=length)
    scale = simulate_smooth_curve(scale_params)
    scaled_seq = scale_time_series(seq, scale, scale_range)

    # Apply stretching
    stretch_params = StretchParams(percentage=stretch_percentage, amount=stretch_amount)
    sim_idx, gt_align = stretching_ts(length, stretch_params)
    transformed = scaled_seq[sim_idx]

    # Add noise if requested
    if noise_level > 0:
        transformed += np.random.normal(0, noise_level, transformed.shape)

    return seq, transformed, gt_align


def compute_alignment_error(
    pred_align: np.ndarray,  # type:ignore
    gt_align: np.ndarray,  # type:ignore
    seq_len1: int,
    seq_len2: int,
) -> float:
    """Compute alignment error using warping matrices.

    Args:
        pred_align: Predicted alignment path
        gt_align: Ground truth alignment path
        seq_len1: Length of first sequence
        seq_len2: Length of second sequence
    """
    # Convert alignments to binary matrices
    pred_matrix = np.zeros((seq_len1, seq_len2))
    gt_matrix = np.zeros((seq_len1, seq_len2))

    # Fill matrices
    for i, j in pred_align:
        if i < seq_len1 and j < seq_len2:  # Guard against out of bounds
            pred_matrix[i, j] = 1

    for i, j in gt_align:
        gt_matrix[i, j] = 1

    # Compute error as sum of absolute differences
    error = np.sum(np.abs(pred_matrix - gt_matrix))

    # Normalize by path length
    error = error / len(gt_align)

    return error


@app.command("sim-alignment")
def compare_alignments(
    descriptor_arg: str = typer.Option("hog1d", help="Descriptor to use for shapeDTW"),
    length: int = typer.Option(200, help="Length of base sequence"),
    scale_min: float = typer.Option(0.4, help="Minimum scaling factor"),
    scale_max: float = typer.Option(1.0, help="Maximum scaling factor"),
    stretch_pct: float = typer.Option(0.15, help="Percentage of points to stretch"),
    stretch_amt: int = typer.Option(2, help="Maximum stretch amount"),
    noise: float = typer.Option(0.0, help="Noise level to add"),
    save_path: Path | None = typer.Option(None, help="Path to save plots"),
    show: bool = typer.Option(True, help="Whether to show plots"),
) -> None:
    """Compare DTW and ShapeDTW alignments on synthetic data."""
    # Generate test sequences
    orig, transformed, gt_align = generate_test_sequences(
        length=length,
        scale_range=(scale_min, scale_max),
        stretch_percentage=stretch_pct,
        stretch_amount=stretch_amt,
        noise_level=noise,
    )

    # Run standard DTW
    alignment = dtw(orig, transformed)
    dtw_match = np.column_stack((alignment.index1, alignment.index2))

    # Run ShapeDTW
    descriptor = DESCRIPTORS[descriptor_arg]
    sdtw = ShapeDTW(seqlen=20)
    _, _, _, sdtw_match = sdtw(orig, transformed, descriptor)

    # Create plots
    fig_dtw = plt.figure(figsize=(15, 12))
    fig_shape = plt.figure(figsize=(15, 12))

    # Plot DTW results
    plot_warped_ts(orig, transformed, dtw_match, fig_dtw)

    # Plot ShapeDTW results
    plot_warped_ts(orig, transformed, sdtw_match, fig_shape)

    plt.tight_layout()

    dtw_error = compute_alignment_error(
        dtw_match, gt_align, len(orig), len(transformed)
    )
    sdtw_error = compute_alignment_error(
        sdtw_match, gt_align, len(orig), len(transformed)
    )

    typer.echo("\nSequence Lengths:")
    typer.echo(f"Original: {len(orig)}")
    typer.echo(f"Transformed: {len(transformed)}")
    typer.echo("\nWarping Path Lengths:")
    typer.echo(f"Ground Truth: {len(gt_align)}")
    typer.echo(f"DTW: {len(dtw_match)}")
    typer.echo(f"ShapeDTW: {len(sdtw_match)}")
    typer.echo("\nAlignment Errors:")
    typer.echo(f"DTW: {dtw_error:.4f}")
    typer.echo(f"ShapeDTW: {sdtw_error:.4f}")

    # Save if requested
    if save_path is not None:
        plt.savefig(save_path)

    # Show if requested
    if show:
        plt.show()


if __name__ == "__main__":
    app()
