from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

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


@pytest.fixture
def basic_config(descriptor: Any) -> AlignmentEvalConfig:
    """Create basic evaluation configuration."""
    return AlignmentEvalConfig(
        dataset_names=["Dataset1"],
        n_pairs_per_dataset=2,
        stretch_percentages=[0.1, 0.2],
        stretch_amount=2,
        scale_range=(0.4, 1.0),
        max_derivative=1.0,
        n_nest=10,
        descriptors={"MockDescriptor": descriptor},
        seqlen=10,  # Match mock data length
        results_dir=Path(
            "dummy_path"
        ),  # Path doesn't matter as we mock file operations
    )


@pytest.fixture
def sample_alignment_data() -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]
):
    """Create a sample pair of sequences with known alignment."""
    # Use small sequence for testing
    orig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Scale sequence
    scale_params = ScaleParams(len=len(orig), max_derivative=1.0, n_nest=10)
    scale = simulate_smooth_curve(scale_params)
    scaled_seq = scale_time_series(orig, scale)

    # Stretch sequence
    stretch_params = StretchParams(percentage=0.15, amount=2)
    sim_idx, gt_align = stretching_ts(len(orig), stretch_params)
    simulated_seq = scaled_seq[sim_idx]

    return orig, simulated_seq, gt_align


def test_compute_alignment_error_empty() -> None:
    """Test compute_alignment_error with empty alignment."""
    with pytest.raises(ValueError, match="Empty alignment passed"):
        compute_alignment_error(np.array([[0, 0]]), np.array([]), 2, 2)


def test_compute_alignment_error_matching() -> None:
    """Test compute_alignment_error with matching alignments."""
    pred_align = np.array([[0, 0], [1, 1]])
    gt_align = np.array([[0, 0], [1, 1]])
    error = compute_alignment_error(pred_align, gt_align, 2, 2)
    assert error == 0.0


def test_compute_alignment_error_mismatch() -> None:
    """Test compute_alignment_error with mismatched alignments."""
    pred_align = np.array([[0, 0], [1, 1]])
    gt_align = np.array([[0, 1], [1, 0]])
    error = compute_alignment_error(pred_align, gt_align, 2, 2)
    assert error > 0.0


def test_scale_time_series() -> None:
    """Test scaling of time series."""
    seq = np.array([1.0, 2.0, 12.0])
    scale = np.array([0.5, 1.0, 1.5])
    scaled = scale_time_series(seq, scale)
    assert scaled.shape == seq.shape
    assert np.all(scaled[:-1] != seq[:-1])  # Values should be different except for last


def test_stretching_ts() -> None:
    """Test time series stretching."""
    length = 5
    params = StretchParams(percentage=0.2, amount=2)
    sim_idx, gt_align = stretching_ts(length, params)

    assert len(sim_idx) > length  # Should be stretched
    assert gt_align.shape[1] == 2  # Alignment should be pairs
    assert np.all(np.diff(gt_align[:, 0]) >= 0)  # Monotonic increase
    assert np.all(np.diff(gt_align[:, 1]) >= 0)  # Monotonic increase


class TestAlignmentEvaluator:
    def test_compare_alignments(
        self,
        basic_config: AlignmentEvalConfig,
        sample_alignment_data: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
        ],
    ) -> None:
        """Test comparison of different alignment methods."""
        evaluator = AlignmentEvaluator(basic_config)
        orig, transformed, gt_align = sample_alignment_data

        results, best_sample = evaluator._compare_alignments(
            orig, transformed, gt_align, "TestDataset", 0.15
        )

        # Check results structure
        assert isinstance(results, list)
        assert len(results) == 3  # DTW, DerivativeDTW, and ShapeDTW-MockDescriptor

        # Verify best sample
        assert best_sample.dataset == "TestDataset"
        assert best_sample.stretch_pct == 0.15
        assert best_sample.descriptor_name in basic_config.descriptors.keys()

    def test_evaluate_dataset(
        self, basic_config: AlignmentEvalConfig, mock_ucr: Any
    ) -> None:
        """Test dataset evaluation with mocked data."""
        evaluator = AlignmentEvaluator(basic_config)

        X_train, _, _, _ = mock_ucr.return_value.load_dataset("Dataset1")
        results, best_sample = evaluator.evaluate_dataset("Dataset1")

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert all("error" in r for r in results)
        assert isinstance(best_sample, type(best_sample))
        assert best_sample.dataset == "Dataset1"

    def test_run_evaluation(
        self,
        basic_config: AlignmentEvalConfig,
        mock_ucr: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test full evaluation run without file I/O."""
        # Mock file operations
        saved_results: list[pd.DataFrame] = []
        saved_samples: list[list[Any]] = []

        def mock_save_results(df: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
            saved_results.append(df.copy())

        def mock_save_samples(
            f: Any, samples: list[Any], *args: Any, **kwargs: Any
        ) -> None:
            saved_samples.append(samples.copy())

        # Patch DataFrame.to_csv and pickle.dump
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_save_results)
        monkeypatch.setattr("pickle.dump", mock_save_samples)

        evaluator = AlignmentEvaluator(basic_config)

        # Run evaluation
        results_df, best_samples = evaluator.run_evaluation()

        # Check DataFrame structure
        assert isinstance(results_df, pd.DataFrame)
        assert set(results_df.columns) == {"dataset", "method", "stretch_pct", "error"}

        # Check best samples
        assert isinstance(best_samples, list)
        assert len(best_samples) > 0

    def test_multivariate_alignment(
        self, basic_config: AlignmentEvalConfig, mock_ucr: Any
    ) -> None:
        """Test alignment with multivariate data."""
        evaluator = AlignmentEvaluator(basic_config)

        # Use multivariate dataset
        X_train, _, _, _ = mock_ucr.return_value.load_dataset("MultivariateDataset")
        results, best_sample = evaluator.evaluate_dataset("MultivariateDataset")

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert all("error" in r for r in results)
        assert isinstance(best_sample, type(best_sample))
        assert best_sample.dataset == "MultivariateDataset"
