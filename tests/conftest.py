from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture(autouse=True)
def mock_ucr() -> Generator[MagicMock, None, None]:
    """Mock UCR_UEA_datasets to avoid downloads."""
    # Create synthetic data
    univariate_data = (
        np.random.randn(2, 10, 1),  # X_train
        np.array([0, 1]),  # y_train
        np.random.randn(2, 10, 1),  # X_test
        np.array([1, 0]),  # y_test
    )

    multivariate_data = (
        np.random.randn(2, 10, 3),  # X_train
        np.array([0, 1]),  # y_train
        np.random.randn(2, 10, 3),  # X_test
        np.array([1, 0]),  # y_test
    )

    def mock_load_dataset(name: str) -> Any:
        if name == "MultivariateDataset":
            return multivariate_data
        return univariate_data

    with patch("pyshapeDTW.data.ucr.UCR_UEA_datasets") as mock_class:
        # Setup mock methods
        mock_class.return_value.list_datasets.return_value = [
            "Dataset1",
            "Dataset2",
            "MultivariateDataset",
        ]
        mock_class.return_value.list_cached_datasets.return_value = ["Dataset1"]
        mock_class.return_value.load_dataset.side_effect = mock_load_dataset

        yield mock_class


class MockDescriptor:
    """Simple mock descriptor for testing."""

    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns mean and std of subsequence as descriptor."""
        return np.array([np.mean(subsequence), np.std(subsequence)], dtype=np.float64)


@pytest.fixture
def descriptor() -> MockDescriptor:
    """Fixture providing mock descriptor."""
    return MockDescriptor()
