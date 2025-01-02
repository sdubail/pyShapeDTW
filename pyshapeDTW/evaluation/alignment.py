import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from dtw import dtw
from tqdm import tqdm

from pyshapeDTW.data.ucr import UCRDataset
from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.descriptors.hog1d import HOG1D
from pyshapeDTW.elastic_measure.derivative_dtw import DerivativeDTW
from pyshapeDTW.elastic_measure.shape_dtw import ShapeDTW
from pyshapeDTW.evaluation.plots import plot_alignment_eval


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
    if ts.ndim > 1 and scale_curve.ndim == 1:
        scale = scale.reshape(-1, 1)
    return scale * ts


def compute_mean_absolute_deviation(
    pred_align: np.ndarray,  # shape (N, 2) containing (i,j) index pairs
    gt_align: np.ndarray,  # shape (M, 2) containing (i,j) index pairs
    seq_len1: int,  # Length of reference sequence
) -> float:
    """
    Compute mean absolute deviation between predicted and ground truth DTW alignments.

    As defined in the paper:
    - Computes the area between two warping paths
    - Normalizes by the length of the reference sequence
    """
    if len(gt_align) == 0 or len(pred_align) == 0:
        raise ValueError("Empty alignment passed")

    # Sort alignments by reference sequence index (i)
    pred_align = pred_align[pred_align[:, 0].argsort()]
    gt_align = gt_align[gt_align[:, 0].argsort()]

    # Convert to paths by getting target sequence indices (j)
    pred_path = pred_align[:, 1]
    gt_path = gt_align[:, 1]

    # Interpolate paths to have values at every reference sequence index
    ref_indices = np.arange(seq_len1)

    # Interpolate predicted path
    pred_path_interp = np.interp(ref_indices, pred_align[:, 0], pred_path)

    # Interpolate ground truth path
    gt_path_interp = np.interp(ref_indices, gt_align[:, 0], gt_path)

    # Compute area between paths as sum of absolute differences
    area = np.sum(np.abs(pred_path_interp - gt_path_interp))

    # Normalize by reference sequence length
    mad = area / seq_len1

    return mad


def compute_alignment_error(
    pred_align: np.ndarray, gt_align: np.ndarray, seq_len1: int, seq_len2: int
) -> float:
    """Compute error between predicted and ground truth alignments."""

    if len(gt_align) == 0:
        raise ValueError("Empty alignment passed")

    # Convert alignments to binary matrices
    pred_matrix = np.zeros((seq_len1, seq_len2))
    gt_matrix = np.zeros((seq_len1, seq_len2))

    # Fill matrices
    for i, j in pred_align:
        if i < seq_len1 and j < seq_len2:
            pred_matrix[i, j] = 1

    for i, j in gt_align:
        gt_matrix[i, j] = 1

    # Compute error as normalized sum of absolute differences
    error = np.sum(np.abs(pred_matrix - gt_matrix))
    error = error / len(gt_align)

    return error


@dataclass
class BestAlignmentSample:
    """Stores best alignment sample data for a dataset."""

    dataset: str
    error_gap: float  # Difference between DTW and ShapeDTW error
    original: npt.NDArray[np.float64]
    transformed: npt.NDArray[np.float64]
    dtw_match: npt.NDArray[np.int64]
    shapedtw_match: npt.NDArray[np.int64] | None
    stretch_pct: float
    descriptor_name: str | None


@dataclass
class AlignmentEvalConfig:
    """Configuration for alignment evaluation."""

    dataset_names: list[str]
    n_pairs_per_dataset: int = 10
    stretch_percentages: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    stretch_amount: int = 2
    scale_range: tuple[float, float] = (0.4, 1.0)
    max_derivative: float = 1.0
    n_nest: int = 10
    descriptors: dict[str, BaseDescriptor] = field(
        default_factory=lambda: {
            "HOG1D": HOG1D(),
        }
    )
    seqlen: int = 30
    results_dir: Path = Path("pyshapeDTW/results")


class AlignmentEvaluator:
    """Framework for evaluating alignment methods on UCR datasets."""

    def __init__(self, config: AlignmentEvalConfig):
        self.config = config
        self.ucr = UCRDataset()

    def _compare_alignments(
        self,
        original: np.ndarray,
        transformed: np.ndarray,
        gt_align: np.ndarray,
        dataset_name: str,
        stretch_pct: float,
    ) -> tuple[list[dict[str, Any]], BestAlignmentSample]:
        """Compare different alignment methods and return results and potential best sample."""
        results = []

        # Standard DTW
        alignment = dtw(original, transformed)
        dtw_match = np.column_stack((alignment.index1, alignment.index2))
        dtw_error = compute_mean_absolute_deviation(  # compute_alignment_error(
            dtw_match,
            gt_align,
            len(original),  # , len(transformed)
        )

        results.append(
            {
                "dataset": dataset_name,
                "method": "DTW",
                "stretch_pct": stretch_pct,
                "error": dtw_error,
            }
        )

        # DerivativeDTW
        dedtw = DerivativeDTW()
        _, _, dedtw_match = dedtw(original, transformed)
        dedtw_error = compute_mean_absolute_deviation(  # compute_alignment_error(
            dedtw_match,
            gt_align,
            len(original),  # , len(transformed)
        )

        results.append(
            {
                "dataset": dataset_name,
                "method": "DerivativeDTW",
                "stretch_pct": stretch_pct,
                "error": dedtw_error,
            }
        )

        # Track best ShapeDTW result for this comparison
        best_shapedtw_error = float("inf")
        best_shapedtw_match = None
        best_descriptor = None

        # ShapeDTW with different descriptors
        sdtw = ShapeDTW(seqlen=self.config.seqlen)
        for desc_name, descriptor in self.config.descriptors.items():
            _, _, _, sdtw_match = sdtw(original, transformed, descriptor)
            sdtw_error = compute_mean_absolute_deviation(  # compute_alignment_error(
                sdtw_match,
                gt_align,
                len(original),  # , len(transformed)
            )

            results.append(
                {
                    "dataset": dataset_name,
                    "method": f"ShapeDTW-{desc_name}",
                    "stretch_pct": stretch_pct,
                    "error": sdtw_error,
                }
            )

            if sdtw_error < best_shapedtw_error:
                best_shapedtw_error = sdtw_error
                best_shapedtw_match = sdtw_match
                best_descriptor = desc_name

        # Calculate error gap
        error_gap = dtw_error - best_shapedtw_error

        # Create alignment sample
        sample = BestAlignmentSample(
            dataset=dataset_name,
            error_gap=error_gap,
            original=original,
            transformed=transformed,
            dtw_match=dtw_match,
            shapedtw_match=best_shapedtw_match,
            stretch_pct=stretch_pct,
            descriptor_name=best_descriptor,
        )

        return results, sample

    def evaluate_dataset(
        self, dataset_name: str
    ) -> tuple[list[dict[str, Any]], BestAlignmentSample]:
        """Evaluate alignment methods on a single dataset."""
        results: list[dict[str, Any]] = []
        X, _ = self.ucr.load_dataset(dataset_name, normalize=True)  # type:ignore

        # Track best sample for this dataset
        best_sample: BestAlignmentSample | None = None

        indices = np.random.choice(
            len(X), len(X), replace=False
        )  # in the end we don't use self.n_pairs_per_dataset but the whole dataset
        # if using it, replace (len(X), len(X)) with (len(X), self.config.n_pairs_per_dataset)

        for idx in tqdm(indices, desc=f"Processing {dataset_name}"):
            original = X[idx]

            # Generate scaling curve
            scale_params = ScaleParams(
                len=len(original),
                max_derivative=self.config.max_derivative,
                n_nest=self.config.n_nest,
            )
            scale = simulate_smooth_curve(scale_params)

            # Apply scaling
            scaled = scale_time_series(original, scale, self.config.scale_range)

            for stretch_pct in self.config.stretch_percentages:
                # Apply stretching to scaled sequence
                params = StretchParams(
                    percentage=stretch_pct, amount=self.config.stretch_amount
                )
                sim_idx, gt_align = stretching_ts(len(scaled), params)
                transformed = scaled[sim_idx]

                # Compare alignments and get potential best sample
                new_results, sample = self._compare_alignments(
                    original, transformed, gt_align, dataset_name, stretch_pct
                )

                # Update results
                results.extend(new_results)

                # Update best sample if we have a larger error gap
                if best_sample is None or sample.error_gap > best_sample.error_gap:
                    best_sample = sample
        if best_sample is None:
            raise ValueError(f"Best sample is None for dataset {dataset_name}")
        return results, best_sample

    def run_evaluation(self) -> tuple[pd.DataFrame, list[BestAlignmentSample]]:
        """Run evaluation on all configured datasets with incremental saving."""
        # Create results directory files
        results_path = self.config.results_dir / "alignment_results_incremental_MAD.csv"
        alignments_path = (
            self.config.results_dir / "best_alignments_incremental_MAD.pkl"
        )

        # Load existing results if any
        if results_path.exists():
            existing_results = pd.read_csv(results_path)
        else:
            existing_results = pd.DataFrame(
                columns=["dataset", "method", "stretch_pct", "error"]
            )
        # Load existing best alignments if any
        if alignments_path.exists():
            with open(alignments_path, "rb") as f:
                best_alignments: list[BestAlignmentSample] = pickle.load(f)
        else:
            best_alignments = []
        # Get already processed datasets
        processed_datasets = existing_results["dataset"].unique()
        for dataset in tqdm(self.config.dataset_names, desc="Processing datasets"):
            if dataset in processed_datasets:
                print(f"Skipping {dataset} - already processed")
                continue

            try:
                print(f"\nEvaluating {dataset}")
                results, best_sample = self.evaluate_dataset(dataset)

                # Update results file
                new_results = pd.concat(
                    [existing_results, pd.DataFrame(results)], ignore_index=True
                )
                new_results.to_csv(results_path, index=False)

                # Update existing results for next iteration
                existing_results = new_results

                # Update best alignments
                best_alignments.append(best_sample)
                with open(alignments_path, "wb") as f:
                    pickle.dump(best_alignments, f)

            except Exception as e:
                print(f"Error while evaluating dataset {dataset}: {e!s}")
                continue

        return pd.DataFrame(existing_results), best_alignments
