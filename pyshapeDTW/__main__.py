from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from dtw import dtw

from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.descriptors.hog1d import HOG1D, HOG1DParams
from pyshapeDTW.descriptors.paa import PAA, PAAParams
from pyshapeDTW.descriptors.wavelets import DWT, DWTParams
from pyshapeDTW.elastic_measure.shape_dtw import ShapeDTW
from pyshapeDTW.evaluation.alignment import (
    AlignmentEvalConfig,
    AlignmentEvaluator,
    ScaleParams,
    StretchParams,
    compute_alignment_error,
    scale_time_series,
    simulate_smooth_curve,
    stretching_ts,
)
from pyshapeDTW.evaluation.classification import (
    ClassificationEvalConfig,
    ClassificationEvaluator,
)
from pyshapeDTW.evaluation.plots import (
    plot_alignment_eval,
    plot_classification_comparison,
    plot_warped_ts,
)

app = typer.Typer()

DESCRIPTORS: dict[str, BaseDescriptor] = {
    "hog1d": HOG1D(HOG1DParams(cells=(1, 25))),
    "paa": PAA(PAAParams(seg_num=4)),
    "dwt": DWT(DWTParams()),
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


@app.command("sim-alignment")
def simulate_alignments(
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
        plt.savefig(f"pyshapeDTW/results/{save_path}")

    # Show if requested
    if show:
        plt.show()


@app.command("ucr-alignment")
def ucr_alignment(
    dataset_names: list[str] | None = None, n_pairs_per_dataset: int = 5
) -> None:
    ### Plenty of additionnal arguments to add (streching, scaling...)

    if dataset_names is None:
        dataset_names = ["GunPoint", "ECG200", "Coffee"]

    config = AlignmentEvalConfig(
        dataset_names=dataset_names,
        n_pairs_per_dataset=n_pairs_per_dataset,
        results_dir=Path("pyshapeDTW/results"),
        descriptors={"HOG1D": HOG1D(), "Wavelets": DWT()},
    )

    evaluator = AlignmentEvaluator(config)
    results_df = evaluator.run_evaluation()

    # Plot and save results
    fig = plot_alignment_eval(results_df)
    fig.savefig(config.results_dir / "alignment_results.png")
    results_df.to_csv(config.results_dir / "alignment_results.csv", index=False)


@app.command("ucr-classification")
def ucr_classification(
    dataset_names: list[str] | None = None,
) -> None:
    """Run classification evaluation on UCR datasets comparing DTW and shapeDTW."""
    if dataset_names is None:
        dataset_names = ["GunPoint", "ECG200", "Coffee"]  # Example datasets

    # Setup descriptors with parameters from paper
    descriptors = {
        "HOG1D": HOG1D(
            HOG1DParams(
                n_bins=8,
                cells=(1, 25),  # Two non-overlapping intervals
                overlap=0,
                scale=0.1,
            )
        ),
        "PAA": PAA(PAAParams(seg_num=5)),  # 5 equal-length intervals
        "DWT": DWT(DWTParams()),  # Default params as in paper
    }

    # Configure evaluation
    config = ClassificationEvalConfig(
        dataset_names=dataset_names,
        descriptors=descriptors,
        seqlen=30,  # As specified in paper
        results_dir=Path("pyshapeDTW/results"),
    )

    # Run evaluation
    evaluator = ClassificationEvaluator(config)
    results_df = evaluator.run_evaluation()

    # Plot results
    fig = plot_classification_comparison(results_df)

    results_df.to_csv(config.results_dir / "classification_results.csv", index=False)
    fig.savefig(config.results_dir / "classification_comparison.png")


if __name__ == "__main__":
    # start CLI app # ####
    app()
