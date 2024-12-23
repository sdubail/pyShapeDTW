from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from dtw import dtw
from tqdm import tqdm

from pyshapeDTW.data.ucr import UCRDataset
from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.elastic_measure.shape_dtw import ShapeDTW


@dataclass
class ClassificationEvalConfig:
    """Configuration for classification evaluation."""

    dataset_names: list[str]
    descriptors: dict[str, BaseDescriptor]
    seqlen: int = 30  # Length of subsequences as mentioned in paper
    results_dir: Path = Path("results")


class ClassificationEvaluator:
    """Framework for evaluating classification performance on UCR datasets."""

    def __init__(self, config: ClassificationEvalConfig):
        """Initialize evaluator with configuration."""
        self.config = config
        self.ucr = UCRDataset()
        self.results: list[dict[str, Any]] = []

    def compute_descriptors(
        self, X: npt.NDArray[np.float64], descriptor: BaseDescriptor
    ) -> list[npt.NDArray[np.float64]]:
        """Compute shape descriptors for all sequences.

        Args:
            X: Array of time series (n_samples, n_timesteps, n_features)
            descriptor: Shape descriptor to use

        Returns:
            descriptors: List of descriptor arrays for each sequence
        """
        # Create ShapeDTW instance just to use its descriptor computation
        sdtw = ShapeDTW(seqlen=self.config.seqlen)
        return [sdtw._compute_shape_descriptors(x, descriptor) for x in X]

    def nearest_neighbor(
        self,
        train_data: list[npt.NDArray[np.float64]],
        test_instance: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.int64],
    ) -> int:
        """Find nearest neighbor class using DTW.

        Args:
            train_data: Training sequences
            test_instance: Test sequence
            y_train: Training labels

        Returns:
            predicted_class: Predicted class label
        """
        distances = []
        for train_seq in train_data:
            alignment = dtw(test_instance, train_seq)
            distances.append(alignment.distance)

        nearest_idx = np.argmin(distances)
        return y_train[nearest_idx]

    def evaluate_dataset(
        self, dataset_name: str, descriptor: BaseDescriptor | None = None
    ) -> dict[str, float | str]:
        """Evaluate classification performance on a single dataset.

        Args:
            dataset_name: Name of dataset to evaluate
            descriptor: Optional shape descriptor for shapeDTW

        Returns:
            metrics: Dictionary containing accuracy
        """
        # Load data
        X_train, y_train = self.ucr.load_dataset(dataset_name, "train")
        if y_train is None:
            raise ValueError(f"Dataset {dataset_name} doesn't contain labels")
        X_test, y_test = self.ucr.load_dataset(dataset_name, "test")

        if descriptor is not None:
            # Compute descriptors for shapeDTW
            train_seqs = self.compute_descriptors(X_train, descriptor)
            test_seqs = self.compute_descriptors(X_test, descriptor)
        else:
            # Use raw sequences for DTW
            train_seqs = list(X_train)
            test_seqs = list(X_test)

        # Classify test instances
        y_pred = []

        for test_seq in tqdm(test_seqs, desc=f"Processing {dataset_name}"):
            pred = self.nearest_neighbor(train_seqs, test_seq, y_train)
            y_pred.append(pred)

        # Compute accuracy
        accuracy = np.mean(np.array(y_pred) == y_test)

        return {"dataset": dataset_name, "accuracy": accuracy}

    def run_evaluation(self) -> pd.DataFrame:
        """Run evaluation on all configured datasets.

        Returns:
            results_df: DataFrame containing results
        """
        results = []

        for dataset in self.config.dataset_names:
            # Evaluate DTW
            dtw_results = self.evaluate_dataset(dataset)
            dtw_results["method"] = "DTW"
            results.append(dtw_results)

            # Evaluate shapeDTW with each descriptor
            for desc_name, descriptor in self.config.descriptors.items():
                shape_results = self.evaluate_dataset(dataset, descriptor)
                shape_results["method"] = f"ShapeDTW-{desc_name}"
                results.append(shape_results)

        return pd.DataFrame(results)
