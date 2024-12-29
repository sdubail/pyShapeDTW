from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pyshapeDTW.data.ucr import UCRDataset


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


def test_multivariate_normalization() -> None:
    """Test normalization for multivariate data."""
    ucr = UCRDataset()
    X, _ = ucr.load_dataset("MultivariateDataset", normalize=True)  # type:ignore

    # Each variable should be normalized independently
    for i in range(X.shape[2]):
        assert np.allclose(X[:, :, i].mean(axis=1), 0, atol=1e-10)
        assert np.allclose(X[:, :, i].std(axis=1), 1, atol=1e-10)
