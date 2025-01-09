import os
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets


class UCRDataset:
    """Handler for UCR Time Series Classification datasets using tslearn."""

    def __init__(
        self, use_cache: bool = True, cache_dir: str = "~/tslearn_cache"
    ) -> None:
        """Initialize dataset handler.

        Args:
            use_cache: Whether to use tslearn's cached datasets
        """
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TSLEARN_CACHE_DIR"] = cache_dir
        self.loader = UCR_UEA_datasets(use_cache=use_cache)
        self.loader = UCR_UEA_datasets(use_cache=use_cache)

    def list_datasets(self) -> list[str]:
        """Get list of all available UCR datasets."""
        return self.loader.list_datasets()

    def list_cached_datasets(self) -> list[str]:
        """Get list of locally cached datasets."""
        return self.loader.list_cached_datasets()

    def load_dataset(
        self,
        dataset_name: str,
        split: Literal["train"] | Literal["test"] | Literal["all"] = "train",
        normalize: bool = True,
        return_df: bool = False,
    ) -> (
        tuple[npt.NDArray[np.float64] | pd.DataFrame, npt.NDArray[np.int64] | None]
        | tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.int64],
            npt.NDArray[np.float64],
            npt.NDArray[np.int64],
        ]
    ):
        """Load a UCR dataset.

        Args:
            dataset_name: Name of the dataset
            split: One of 'train', 'test', or 'all'
            normalize: Whether to z-normalize the time series
            return_df: If True, return as pandas DataFrame (only for train/test splits)

        Returns:
            If split is 'train' or 'test':
                X: Time series data if return_df is False, else time series and label as pd.DataFrame
                y: Labels if return_df is False, else None
            If split is 'all':
                X_train: Training time series data
                y_train: Training labels
                X_test: Test time series data
                y_test: Test labels
        """
        # Validate split
        if split not in ["train", "test", "all"]:
            raise ValueError("split must be one of 'train', 'test', or 'all'")

        # Load data using tslearn's cache
        print("Loading dataset...")
        X_train, y_train, X_test, y_test = self.loader.load_dataset(dataset_name)
        if X_train is None:
            raise ValueError(f"Failed to load dataset {dataset_name}")
        print("Done loading.")
        # Process data based on split
        if split == "all":
            # Remove singleton dimension if univariate
            if X_train.shape[2] == 1:
                X_train = X_train.squeeze(axis=2)
                X_test = X_test.squeeze(axis=2)

            # Z-normalize if requested
            if normalize:
                X_train = self._z_normalize(X_train)
                X_test = self._z_normalize(X_test)

            return (
                X_train.astype(np.float64),
                y_train.astype(np.int64),
                X_test.astype(np.float64),
                y_test.astype(np.int64),
            )
        else:
            # Select split
            X = X_train if split == "train" else X_test
            y = y_train if split == "train" else y_test

            # Remove singleton dimension if univariate
            if X.shape[2] == 1:
                X = X.squeeze(axis=2)

            # Z-normalize if requested
            if normalize:
                X = self._z_normalize(X)

            # Convert to DataFrame if requested
            if return_df:
                return self._to_dataframe(X, y, dataset_name), None

            return X.astype(np.float64), y.astype(np.int64)

    @staticmethod
    def _z_normalize(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Z-normalize time series."""
        if X.ndim == 2:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)
        else:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)

        std[std == 0] = 1.0
        return (X - mean) / std

    @staticmethod
    def _to_dataframe(
        X: npt.NDArray[np.float64], y: npt.NDArray[np.int64], dataset_name: str
    ) -> pd.DataFrame:
        """Convert arrays to DataFrame."""
        if X.ndim == 2:
            df = pd.DataFrame(X)
            df.columns = [f"time_{i}" for i in range(X.shape[1])]
        else:
            dfs = []
            for i in range(X.shape[2]):
                df_i = pd.DataFrame(X[:, :, i])
                df_i.columns = [f"var{i}_time_{j}" for j in range(X.shape[1])]
                dfs.append(df_i)
            df = pd.concat(dfs, axis=1)

        df["label"] = y
        df["dataset"] = dataset_name
        return df
