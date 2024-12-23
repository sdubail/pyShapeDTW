from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pyshapeDTW.descriptors.base import BaseDescriptor
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
def base_sequence() -> npt.NDArray[np.float64]:
    """Generate base test sequence."""
    t = np.linspace(0, 4 * np.pi, 200)
    return np.sin(t) + 0.5 * np.sin(3 * t)


def create_transformed_pair(
    seq: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Create a transformed pair of sequences with known alignment."""
    # Generate smooth scaling curve
    scale_params = ScaleParams(len=len(seq), max_derivative=1.0, n_nest=10)
    scale = simulate_smooth_curve(scale_params)

    # Scale sequence
    scaled_seq = scale_time_series(seq, scale)

    # Stretch sequence
    stretch_params = StretchParams(percentage=0.15, amount=2)
    sim_idx, gt_align = stretching_ts(len(seq), stretch_params)
    simulated_seq = scaled_seq[sim_idx]

    return seq, simulated_seq, gt_align


def test_transformation_consistency(base_sequence: npt.NDArray[np.float64]) -> None:
    """Test that transformations preserve key properties."""
    orig, transformed, align = create_transformed_pair(base_sequence)

    # Length checks
    assert len(transformed) >= len(orig)

    # Alignment checks
    assert align[:, 0].max() == len(orig) - 1
    assert align[:, 1].max() == len(transformed) - 1
    assert np.all(np.diff(align[:, 0]) >= 0)  # Monotonic
    assert np.all(np.diff(align[:, 1]) >= 0)


def test_scaling_properties() -> None:
    """Test properties of scaling transformation."""
    params = ScaleParams(len=100)
    scale = simulate_smooth_curve(params)

    # Should be smooth
    assert np.all(np.abs(np.diff(scale, 2)) < 1.0)

    # Test scaling range
    seq = np.ones(100)
    scaled = scale_time_series(seq, scale, scale_range=(0.5, 2.0))
    assert np.all(scaled >= 0.4)
    assert np.all(scaled <= 2.1)  # Small margin for numerical error


def test_stretching_properties() -> None:
    """Test properties of stretching transformation."""
    params = StretchParams(percentage=0.2, amount=3)
    sim_idx, align = stretching_ts(100, params)

    # Check monotonicity
    assert np.all(np.diff(align[:, 0]) >= 0)
    assert np.all(np.diff(align[:, 1]) >= 0)

    # Check stretching amount
    unique_mappings = len(np.unique(align[:, 0]))
    assert unique_mappings == 100  # All original points should be mapped
    assert len(align) > 100  # Sequence should be stretched


def test_compute_alignment_error() -> None:
    """Test alignment error computation."""
    # Test with identical alignments
    pred_align = np.array([[0, 0], [1, 1], [2, 2]])
    gt_align = np.array([[0, 0], [1, 1], [2, 2]])
    error = compute_alignment_error(pred_align, gt_align, 3, 3)
    assert error == 0.0

    # Test with different alignments
    pred_align = np.array([[0, 0], [1, 1], [2, 2]])
    gt_align = np.array([[0, 0], [1, 2], [2, 1]])
    error = compute_alignment_error(pred_align, gt_align, 3, 3)
    assert error > 0.0


@pytest.fixture
def basic_config(descriptor: BaseDescriptor) -> AlignmentEvalConfig:
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
        results_dir=Path("test_results"),
    )


class TestAlignmentEvaluator:
    def test_init(self, basic_config: AlignmentEvalConfig) -> None:
        """Test evaluator initialization."""
        evaluator = AlignmentEvaluator(basic_config)
        assert evaluator.config == basic_config
        assert len(evaluator.results) == 0

    def test_compare_alignments(self, basic_config: AlignmentEvalConfig) -> None:
        """Test comparison of different alignment methods."""
        evaluator = AlignmentEvaluator(basic_config)

        # Create test sequences with known pattern
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformed = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
        gt_align = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

        results = evaluator._compare_alignments(
            original, transformed, gt_align, "TestDataset", 0.1
        )

        # Check results structure
        assert isinstance(results, list)
        assert (
            len(results) == 3
        )  # DTW and ShapeDTW-MockDescriptor / DerivativeDTW excluded for now, pass from 2 to 3 when added back

        # Check result contents
        methods = {"DTW", "DerivativeDTW", "ShapeDTW-MockDescriptor"}
        for result in results:
            assert isinstance(result, dict)
            assert set(result.keys()) == {"dataset", "method", "stretch_pct", "error"}
            assert result["dataset"] == "TestDataset"
            assert result["method"] in methods
            assert result["stretch_pct"] == 0.1
            assert isinstance(result["error"], int | float)
            assert result["error"] >= 0.0

    def test_evaluate_dataset(self, basic_config: AlignmentEvalConfig) -> None:
        """Test evaluation of a single dataset."""
        evaluator = AlignmentEvaluator(basic_config)

        # Run evaluation
        evaluator.evaluate_dataset("Dataset1")

        # Check results were generated
        assert len(evaluator.results) > 0
        expected_results = (
            basic_config.n_pairs_per_dataset
            * len(basic_config.stretch_percentages)
            * 3  # DTW + ShapeDTW-MockDescriptor / DerivativeDTW excluded for now, pass from 2 to 3 when added back
        )
        assert len(evaluator.results) == expected_results

    def test_run_evaluation(self, basic_config: AlignmentEvalConfig) -> None:
        """Test full evaluation run."""
        evaluator = AlignmentEvaluator(basic_config)

        # Run evaluation
        results_df = evaluator.run_evaluation()

        # Check DataFrame structure
        assert isinstance(results_df, pd.DataFrame)
        assert set(results_df.columns) == {"dataset", "method", "stretch_pct", "error"}

        # Check contents
        assert len(results_df) > 0
        assert all(results_df["dataset"].isin(basic_config.dataset_names))
        assert all(results_df["stretch_pct"].isin(basic_config.stretch_percentages))
        assert all(
            results_df["method"].isin(
                ["DTW", "DerivativeDTW", "ShapeDTW-MockDescriptor"]
            )
        )
