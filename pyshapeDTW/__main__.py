import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from dtw import dtw
from tqdm import tqdm

from pyshapeDTW.data.ucr import UCRDataset
from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.descriptors.hog1d import HOG1D, HOG1DParams
from pyshapeDTW.descriptors.paa import PAA, PAAParams
from pyshapeDTW.descriptors.slope import Slope, SlopeParams
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
    plot_all_alignments,
    plot_classification_comparison,
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
    dataset_file: Path = typer.Option(
        "pyshapeDTW/data/todo_datasets.csv",
        help="File containing dataset names",
    ),
    n_pairs_per_dataset: int = typer.Option(
        5, help="Number of pairs to sample per dataset"
    ),
    stretch_min: float = typer.Option(0.1, help="Minimum stretch percentage"),
    stretch_max: float = typer.Option(0.5, help="Maximum stretch percentage"),
    stretch_steps: int = typer.Option(5, help="Number of stretch percentage steps"),
    stretch_amt: int = typer.Option(2, help="Maximum stretch amount"),
    scale_min: float = typer.Option(0.4, help="Minimum scaling factor"),
    scale_max: float = typer.Option(1.0, help="Maximum scaling factor"),
    results_dir: Path = typer.Option(
        Path("pyshapeDTW/results"), help="Directory to save results"
    ),
) -> None:
    """Compare DTW and ShapeDTW alignments on UCR datasets."""
    # if dataset_names is None:
    #     dataset_names = ["GunPoint", "ECG200", "Coffee"]
    dataset_names = pd.read_csv(dataset_file, header=0)["dataset"].to_list()
    typer.echo(f"Looking at {len(dataset_names)} datasets from {dataset_file}")

    # Create stretch percentages list
    stretch_percentages = np.linspace(stretch_min, stretch_max, stretch_steps).tolist()

    # Create results directory if it doesn't exist
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup descriptors with parameters from paper
    descriptors: dict[str, BaseDescriptor] = {
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
        "Slope": Slope(SlopeParams()),
    }

    config = AlignmentEvalConfig(
        dataset_names=dataset_names,
        n_pairs_per_dataset=n_pairs_per_dataset,
        stretch_percentages=stretch_percentages,
        stretch_amount=stretch_amt,
        scale_range=(scale_min, scale_max),
        results_dir=results_dir,
        descriptors=descriptors,
    )

    # Run evaluation
    evaluator = AlignmentEvaluator(config)
    results_df, best_alignments = evaluator.run_evaluation()

    # Save numerical results
    results_df.to_csv(results_dir / "alignment_results_full_MAD.csv", index=False)

    # Save best alignment samples using pickle
    with open(results_dir / "best_alignments_full_MAD.pkl", "wb") as f:
        pickle.dump(best_alignments, f)

    # Create summary of best alignments
    summary_rows = [
        {
            "dataset": sample.dataset,
            "error_gap": sample.error_gap,
            "stretch_pct": sample.stretch_pct,
            "descriptor": sample.descriptor_name,
            "orig_len": len(sample.original),
            "transform_len": len(sample.transformed),
        }
        for sample in best_alignments
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "best_alignments_summary_full_MAD.csv", index=False)

    # Plot overall results
    fig = plot_alignment_eval(results_df)
    fig.savefig(results_dir / "alignment_results_full_MAD.png")

    # Create alignment_samples subfolder if it doesn't exist
    samples_dir = results_dir / "alignment_samples_MAD"
    samples_dir.mkdir(exist_ok=True)

    # Get indices of top 5 error gaps
    top_5_indices = summary_df.sort_values("error_gap", ascending=False).index[:5]

    # Plot the 5 alignments with largest error gaps
    for i, idx in enumerate(top_5_indices):
        alignment = best_alignments[idx]

        # Create matches dictionary for each method
        matches = {
            # "GT": alignment.gt_match,
            "DTW": alignment.dtw_match,
        }
        if alignment.shapedtw_match is not None:
            matches[f"shapeDTW-{alignment.descriptor_name}"] = alignment.shapedtw_match

        # Create combined plot
        fig = plot_all_alignments(
            alignment.original, alignment.transformed, matches, figsize=(15, 15)
        )

        # Save plot with error gap info in filename
        error_gap = summary_df.loc[idx, "error_gap"]
        fig.savefig(samples_dir / f"alignment_sample_{i+1}_gap_{error_gap:.4f}.png")
        plt.close(fig)

    # Print summary
    typer.echo(f"\nResults saved to: {samples_dir}")
    typer.echo("\nBest alignment gaps per dataset:")
    for _, row in summary_df.sort_values("error_gap", ascending=False).iterrows():
        typer.echo(
            f"{row['dataset']}: {row['error_gap']:.4f} "
            f"(stretch={row['stretch_pct']:.2f}, descriptor={row['descriptor']})"
        )


@app.command("ucr-classification")
def ucr_classification(
    dataset_file: Path = typer.Option(
        "pyshapeDTW/data/todo_datasets.csv",
        help="File containing dataset names",
    ),
) -> None:
    """Run classification evaluation on UCR datasets comparing DTW and shapeDTW."""

    # dataset_names = ["CinCECGtorso"]  # ["GunPoint", "ECG200", "Coffee"]  # Example datasets
    # dataset_names = np.loadtxt(dataset_file, dtype=str).tolist()
    dataset_names = pd.read_csv(dataset_file, header=0)["dataset"].to_list()
    typer.echo(f"Looking at {len(dataset_names)} datasets from {dataset_file}")

    # Setup descriptors with parameters from paper
    descriptors: dict[str, BaseDescriptor] = {
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
        "Slope": Slope(SlopeParams()),
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
    results_df = evaluator.run_evaluation(
        path=config.results_dir / "classification_results_full_incremental.csv"
    )

    # Plot results
    fig = plot_classification_comparison(results_df)

    results_df.to_csv(
        config.results_dir / "classification_results_full.csv", index=False
    )
    fig.savefig(config.results_dir / "classification_comparison_full.png")


@app.command("test-datasets")
def test_dataset_availability(
    dataset_file: Path = typer.Option(
        "pyshapeDTW/data/ucr_datasets.txt", help="File containing dataset names"
    ),
    output_file: Path = typer.Option(
        "pyshapeDTW/data/available_datasets.txt", help="Output file for results"
    ),
    show_sizes: bool = typer.Option(True, help="Show dataset sizes"),
) -> None:
    """Test which datasets from the list can be loaded successfully."""
    # Load dataset names from file
    dataset_names = np.loadtxt(dataset_file, dtype=str).tolist()
    typer.echo(f"Testing {len(dataset_names)} datasets...\n")

    # Initialize UCR dataset loader
    ucr = UCRDataset()

    available = []
    unavailable = []
    dataset_info = []

    # Try loading each dataset
    for name in tqdm(dataset_names):
        try:
            # Try to load both train and test splits
            X_train, y_train = ucr.load_dataset(name, "train")
            X_test, y_test = ucr.load_dataset(name, "test")
            available.append(name)

            # Get dataset size info
            n_train = len(X_train)
            n_test = len(X_test)
            seq_length = X_train.shape[1]
            total_points = (n_train + n_test) * seq_length
            dataset_info.append(
                {
                    "name": name,
                    "n_train": n_train,
                    "n_test": n_test,
                    "seq_length": seq_length,
                    "total_points": total_points,
                }
            )
        except Exception as e:
            unavailable.append((name, str(e)))

    # Print results
    typer.echo("\nResults:")

    if show_sizes:
        # Sort by total number of points
        dataset_info.sort(key=lambda x: x["total_points"], reverse=True)
        typer.echo("\nDataset Sizes (sorted by total points):")
        for info in dataset_info:
            typer.echo(
                f"  {info['name']:<30} "
                f"Train: {info['n_train']:>5}, "
                f"Test: {info['n_test']:>5}, "
                f"Length: {info['seq_length']:>5}, "
                f"Total Points: {info['total_points']:>10,}"
            )

    # Write available datasets with info to file
    with open(output_file, "w") as f:
        # Write header
        f.write("name\tn_train\tn_test\tseq_length\ttotal_points\n")
        # Write data
        for info in dataset_info:
            f.write(
                f"{info['name']}\t{info['n_train']}\t{info['n_test']}\t{info['seq_length']}\t{info['total_points']}\n"
            )

    # Print summary
    typer.echo(f"\nAvailable datasets ({len(available)}/{len(dataset_names)}):")
    for name in available:
        typer.echo(f"  ✓ {name}")

    if unavailable:
        typer.echo(f"\nUnavailable datasets ({len(unavailable)}/{len(dataset_names)}):")
        for name, error in unavailable:
            typer.echo(f"  ✗ {name}: {error}")

    # Also write unavailable datasets to separate file
    with open(output_file.with_suffix(".errors.txt"), "w") as f:
        for name, error in unavailable:
            f.write(f"{name}\t{error}\n")

    typer.echo(
        f"\nResults saved to '{output_file}' and '{output_file.with_suffix('.errors.txt')}'"
    )


if __name__ == "__main__":
    # start CLI app # ####
    app()
