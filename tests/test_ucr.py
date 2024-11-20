from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pyshapeDTW.data.ucr import UCRDataset


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


def test_list_datasets() -> None:
    """Test dataset listing."""
    ucr = UCRDataset()
    datasets = ucr.list_datasets()
    assert len(datasets) == 3
    assert "Dataset1" in datasets
    assert "MultivariateDataset" in datasets


def test_list_cached() -> None:
    """Test cached dataset listing."""
    ucr = UCRDataset()
    cached = ucr.list_cached_datasets()
    assert len(cached) == 1
    assert "Dataset1" in cached


def test_load_univariate() -> None:
    """Test loading univariate dataset."""
    ucr = UCRDataset()
    X, y = ucr.load_dataset("Dataset1", split="train")

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert X.ndim == 2  # Should remove singleton dimension
    assert X.shape == (2, 10)


def test_load_multivariate() -> None:
    """Test loading multivariate dataset."""
    ucr = UCRDataset()
    X, y = ucr.load_dataset("MultivariateDataset", split="train")

    assert X.ndim == 3
    assert X.shape == (2, 10, 3)


@pytest.mark.parametrize("split", ["train", "test"])
def test_splits(split: str) -> None:
    """Test different dataset splits."""
    ucr = UCRDataset()
    X, y = ucr.load_dataset("Dataset1", split=split)
    assert len(X) == 2
    assert len(y) == 2  # type:ignore


def test_normalization() -> None:
    """Test z-normalization."""
    ucr = UCRDataset()

    # With normalization
    X_norm, _ = ucr.load_dataset("Dataset1", normalize=True)
    assert np.allclose(X_norm.mean(axis=1), 0, atol=1e-10)
    assert np.allclose(X_norm.std(axis=1), 1, atol=1e-10)

    # Without normalization
    X_raw, _ = ucr.load_dataset("Dataset1", normalize=False)
    assert not np.allclose(X_raw.mean(axis=1), 0)


def test_dataframe_conversion() -> None:
    """Test conversion to DataFrame."""
    ucr = UCRDataset()

    # Test univariate
    df, _ = ucr.load_dataset("Dataset1", return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    assert "dataset" in df.columns
    assert df.shape[1] == 12  # 10 timesteps + label + dataset

    # Test multivariate
    df, _ = ucr.load_dataset("MultivariateDataset", return_df=True)
    assert isinstance(df, pd.DataFrame)
    # Should have columns for each variable at each timestep
    assert df.shape[1] == 32  # 3 vars * 10 timesteps + label + dataset


def test_invalid_split() -> None:
    """Test error handling for invalid split."""
    ucr = UCRDataset()
    with pytest.raises(ValueError):
        ucr.load_dataset("Dataset1", split="invalid")


def test_cache_control(mock_ucr: MagicMock) -> None:
    """Test cache control through tslearn."""
    # With cache
    ucr1 = UCRDataset(use_cache=True)
    ucr1.load_dataset("Dataset1")

    # Without cache
    ucr2 = UCRDataset(use_cache=False)
    ucr2.load_dataset("Dataset1")

    # Should have created two different loaders
    assert mock_ucr.call_count == 2
    assert mock_ucr.call_args_list[0][1]["use_cache"] is True
    assert mock_ucr.call_args_list[1][1]["use_cache"] is False


def test_multivariate_normalization() -> None:
    """Test normalization for multivariate data."""
    ucr = UCRDataset()
    X, _ = ucr.load_dataset("MultivariateDataset", normalize=True)

    # Each variable should be normalized independently
    for i in range(X.shape[2]):
        assert np.allclose(X[:, :, i].mean(axis=1), 0, atol=1e-10)
        assert np.allclose(X[:, :, i].std(axis=1), 1, atol=1e-10)
