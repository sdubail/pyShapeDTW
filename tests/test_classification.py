from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.descriptors.hog1d import HOG1D
from pyshapeDTW.evaluation.classification import (
    ClassificationEvalConfig,
    ClassificationEvaluator,
)


@pytest.fixture
def test_sequences() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Create test sequences."""
    # Create 5 sequences of length 10, with 2 features
    X = np.random.randn(5, 10, 2)
    y = np.array([0, 0, 1, 1, 0])
    return X, y


@pytest.fixture
def basic_config(descriptor: BaseDescriptor) -> ClassificationEvalConfig:
    """Create basic evaluation configuration."""
    return ClassificationEvalConfig(
        dataset_names=["GunPoint"],  # Using actual UCR dataset
        descriptors={"MockDescriptor": descriptor},
        seqlen=10,  # Shorter for testing
        results_dir=Path("test_results"),
    )


class TestClassificationEvaluator:
    """Tests for classification evaluation pipeline."""

    def test_init(self, basic_config: ClassificationEvalConfig) -> None:
        """Test evaluator initialization."""
        evaluator = ClassificationEvaluator(basic_config)
        assert evaluator.config == basic_config
        assert len(evaluator.results) == 0
        assert evaluator.config.seqlen == 10

    def test_compute_descriptors(
        self,
        basic_config: ClassificationEvalConfig,
        test_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]],
    ) -> None:
        """Test descriptor computation."""
        evaluator = ClassificationEvaluator(basic_config)
        X, _ = test_sequences

        descriptors = evaluator.compute_descriptors(
            X, basic_config.descriptors["MockDescriptor"]
        )

        assert len(descriptors) == len(X)
        assert all(isinstance(d, np.ndarray) for d in descriptors)
        # Each descriptor should be computed for every timestep
        assert all(len(d) == len(x) for d, x in zip(descriptors, X))

    def test_nearest_neighbor(
        self,
        basic_config: ClassificationEvalConfig,
        test_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]],
    ) -> None:
        """Test nearest neighbor classification."""
        evaluator = ClassificationEvaluator(basic_config)
        X, y = test_sequences

        # Use first 3 sequences as training, test with the 4th
        train_data = list(X[:3])
        test_instance = X[3]
        y_train = y[:3]

        pred = evaluator.nearest_neighbor(train_data, test_instance, y_train)
        assert isinstance(pred, int | np.integer)
        assert pred in y_train  # Prediction should be one of training labels

    def test_evaluate_dataset(self, basic_config: ClassificationEvalConfig) -> None:
        """Test dataset evaluation."""
        evaluator = ClassificationEvaluator(basic_config)

        # Test DTW
        dtw_results = evaluator.evaluate_dataset("GunPoint")
        assert isinstance(dtw_results, dict)
        assert "accuracy" in dtw_results
        assert "dataset" in dtw_results
        assert dtw_results["dataset"] == "GunPoint"
        assert 0 <= dtw_results["accuracy"] <= 1

        # Test shapeDTW
        shape_results = evaluator.evaluate_dataset(
            "GunPoint", basic_config.descriptors["MockDescriptor"]
        )
        assert isinstance(shape_results, dict)
        assert "accuracy" in shape_results
        assert "dataset" in shape_results
        assert shape_results["dataset"] == "GunPoint"
        assert 0 <= shape_results["accuracy"] <= 1

    def test_run_evaluation(self, basic_config: ClassificationEvalConfig) -> None:
        """Test full evaluation run."""
        evaluator = ClassificationEvaluator(basic_config)
        results_df = evaluator.run_evaluation()

        # Check DataFrame structure
        assert set(results_df.columns) == {"dataset", "method", "accuracy"}

        # Check number of results
        assert (
            len(results_df) == len(basic_config.descriptors) + 1
        )  # DTW + shapeDTW variants

        # Check methods
        expected_methods = {"DTW"} | {
            f"ShapeDTW-{name}" for name in basic_config.descriptors
        }
        assert set(results_df["method"]) == expected_methods

        # Check values
        assert all(results_df["dataset"] == "GunPoint")
        assert all(results_df["accuracy"].between(0, 1))

    def test_multivariate_handling(
        self, basic_config: ClassificationEvalConfig
    ) -> None:
        """Test handling of multivariate time series."""
        evaluator = ClassificationEvaluator(basic_config)

        # Create multivariate test data
        X = np.random.randn(5, 10, 3)  # 5 sequences, length 10, 3 features
        descriptors = evaluator.compute_descriptors(
            X, basic_config.descriptors["MockDescriptor"]
        )

        assert len(descriptors) == len(X)
        # Shape descriptors should preserve temporal dimension
        assert all(len(d) == len(x) for d, x in zip(descriptors, X))
