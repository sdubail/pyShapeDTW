from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyshapeDTW.evaluation.classification import (
    ClassificationEvalConfig,
    ClassificationEvaluator,
)


@pytest.fixture
def basic_config(descriptor: Any) -> ClassificationEvalConfig:
    """Create basic evaluation configuration."""
    return ClassificationEvalConfig(
        dataset_names=["Dataset1"],
        descriptors={"MockDescriptor": descriptor},
        seqlen=10,  # Match mock data length
        results_dir=Path("dummy_path"),  # Won't be used due to mocking
        n_jobs=None,  # Disable parallel processing in tests
        batch_size=2,
    )


class TestClassificationEvaluator:
    """Tests for classification evaluation pipeline."""

    def test_compute_descriptors(
        self, basic_config: ClassificationEvalConfig, mock_ucr: Any
    ) -> None:
        """Test descriptor computation."""
        evaluator = ClassificationEvaluator(basic_config)
        X_train, _, _, _ = mock_ucr.return_value.load_dataset("Dataset1")

        descriptors = evaluator.compute_descriptors(
            X_train, basic_config.descriptors["MockDescriptor"]
        )

        assert len(descriptors) == len(X_train)
        assert all(isinstance(d, np.ndarray) for d in descriptors)
        assert all(len(d) == len(x) for d, x in zip(descriptors, X_train))

    def test_evaluate_dataset(
        self, basic_config: ClassificationEvalConfig, mock_ucr: Any
    ) -> None:
        """Test dataset evaluation with both DTW and ShapeDTW."""
        evaluator = ClassificationEvaluator(basic_config)
        data = mock_ucr.return_value.load_dataset("Dataset1")

        # Test basic DTW
        results = evaluator.evaluate_dataset("TestDataset", data)
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert 0 <= float(results["accuracy"]) <= 1

        # Test ShapeDTW
        results = evaluator.evaluate_dataset(
            "TestDataset", data, basic_config.descriptors["MockDescriptor"]
        )
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert 0 <= float(results["accuracy"]) <= 1

    def test_run_evaluation(
        self,
        basic_config: ClassificationEvalConfig,
        mock_ucr: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test full evaluation run with mocked data loading."""
        evaluator = ClassificationEvaluator(basic_config)

        # Mock DataFrame.to_csv
        saved_data: list[pd.DataFrame] = []

        def mock_to_csv(df: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
            saved_data.append(df.copy())

        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

        # Run evaluation
        results = evaluator.run_evaluation(Path("dummy_results.csv"))

        # Check results structure
        assert isinstance(results, pd.DataFrame)
        assert set(results.columns) == {"dataset", "method", "accuracy"}

        # Check methods
        methods = set(results["method"])
        expected_methods = {"DTW", "ShapeDTW-MockDescriptor"}
        assert methods == expected_methods

        # Verify accuracies
        assert all(0 <= acc <= 1 for acc in results["accuracy"])

    def test_multivariate(
        self, basic_config: ClassificationEvalConfig, mock_ucr: Any
    ) -> None:
        """Test handling of multivariate data."""
        evaluator = ClassificationEvaluator(basic_config)

        # Use multivariate dataset from mock
        data = mock_ucr.return_value.load_dataset("MultivariateDataset")
        X_train, y_train, X_test, y_test = data

        # Test that everything works with multivariate data
        results = evaluator.evaluate_dataset(
            "MultivariateDataset", [X_train, y_train, X_test, y_test]
        )

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert 0 <= float(results["accuracy"]) <= 1
