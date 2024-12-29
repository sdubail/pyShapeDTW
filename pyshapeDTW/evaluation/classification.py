import time
from concurrent.futures import ProcessPoolExecutor
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


def lb_keogh(seq1: npt.NDArray[np.float64], seq2: npt.NDArray[np.float64]) -> float:
    """Compute LB_Keogh lower bound for DTW distance for multivariate sequences.

    Args:
        seq1: First sequence (n_timestamps, n_features)
        seq2: Second sequence (n_timestamps, n_features)

    Returns:
        lb: Lower bound distance
    """
    # Ensure sequences are 2D
    if seq1.ndim == 1:
        seq1 = seq1.reshape(-1, 1)
    if seq2.ndim == 1:
        seq2 = seq2.reshape(-1, 1)

    window = max(len(seq1) // 10, 1)  # 10% window
    n_timestamps, n_features = seq2.shape

    # Compute envelope for each dimension
    upper = np.zeros_like(seq2)
    lower = np.zeros_like(seq2)

    for i in range(n_timestamps):
        start_idx = max(0, i - window)
        end_idx = min(n_timestamps, i + window + 1)
        window_data = seq2[start_idx:end_idx]
        upper[i] = np.max(window_data, axis=0)
        lower[i] = np.min(window_data, axis=0)

    # Compute LB_Keogh per dimension
    above_upper = seq1 > upper
    below_lower = seq1 < lower

    squared_diffs = np.zeros_like(seq1)
    squared_diffs[above_upper] = (seq1[above_upper] - upper[above_upper]) ** 2
    squared_diffs[below_lower] = (seq1[below_lower] - lower[below_lower]) ** 2

    # Sum over both timestamps and features
    lb = np.sqrt(np.sum(squared_diffs))
    return lb


def compute_distances_direct(
    x1: npt.NDArray[np.float64],
    x2_list: list[npt.NDArray[np.float64]],
) -> tuple[list[int], list[float]]:
    """Compute DTW distances directly for one sequence against many."""
    indices, distances = [], []
    best_dist = float("inf")

    for j, x2 in enumerate(x2_list):
        # # Try LB_Keogh bound first ?
        # lb = lb_keogh(x1, x2)
        # if lb >= best_dist:
        #     continue

        # Compute DTW if needed
        dist = dtw(x1, x2, distance_only=True).distance
        if dist < best_dist:
            best_dist = dist
        indices.append(j)
        distances.append(dist)

    return indices, distances


def compute_batch_distances(
    args: tuple[
        list[int], list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]
    ],
) -> tuple[list[int], list[int], list[float]]:
    """Compute DTW distances for a batch of sequences (for parallel processing)."""
    batch_indices, X1, X2 = args
    rows, cols, distances = [], [], []

    for i in batch_indices:
        cols_i, distances_i = compute_distances_direct(X1[i], X2)
        rows.extend([i] * len(cols_i))
        cols.extend(cols_i)
        distances.extend(distances_i)

    return rows, cols, distances


def compute_dtw_distance_matrix(
    X1: list[npt.NDArray[np.float64]],
    X2: list[npt.NDArray[np.float64]],
    n_jobs: int | None = None,
    batch_size: int = 100,
) -> npt.NDArray[np.float64]:
    """Efficiently compute DTW distance matrix between two sets of sequences.

    Args:
        X1: First set of sequences
        X2: Second set of sequences
        n_jobs: Number of parallel jobs. If None or -1, use direct computation
        batch_size: Size of batches for parallel processing

    Returns:
        distances: Distance matrix (n_samples1, n_samples2)
    """
    n1, n2 = len(X1), len(X2)
    distances = np.full((n1, n2), np.inf)

    if n_jobs is None or n_jobs == -1:
        # Direct computation
        # start_time = time.time()
        for i in tqdm(range(n1), desc="Computing DTW distances"):
            # for i in range(n1):
            cols, dists = compute_distances_direct(X1[i], X2)
            for j, d in zip(cols, dists, strict=False):
                distances[i, j] = d
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"DTW distances computation time : {execution_time}")
    else:
        # Parallel computation
        batch_indices = [
            list(range(i, min(i + batch_size, n1))) for i in range(0, n1, batch_size)
        ]
        args = [(indices, X1, X2) for indices in batch_indices]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(compute_batch_distances, arg) for arg in args]

            for future in tqdm(futures, desc="Computing DTW distances"):
                rows, cols, dists = future.result()
                for i, j, d in zip(rows, cols, dists, strict=False):
                    distances[i, j] = d

    return distances


def efficient_nearest_neighbor(
    train_data: list[npt.NDArray[np.float64]],
    test_data: list[npt.NDArray[np.float64]],
    y_train: npt.NDArray[np.int64],
    n_jobs: int | None = None,
    batch_size: int = 100,
) -> npt.NDArray[np.int64]:
    """Find nearest neighbors using DTW with LB_Keogh optimization."""

    # Compute DTW distances efficiently
    distances = compute_dtw_distance_matrix(
        test_data,
        train_data,
        n_jobs=n_jobs,
        batch_size=batch_size,
    )

    # Find nearest neighbors
    nearest_indices = np.argmin(distances, axis=1)
    predictions = y_train[nearest_indices]

    return predictions


@dataclass
class ClassificationEvalConfig:
    """Configuration for classification evaluation."""

    dataset_names: list[str]
    descriptors: dict[str, BaseDescriptor]
    seqlen: int = 30
    results_dir: Path = Path("results")
    n_jobs: int | None = None  # Number of parallel jobs
    batch_size: int = 100  # Batch size for processing


class ClassificationEvaluator:
    """Framework for evaluating classification performance on UCR datasets."""

    def __init__(self, config: ClassificationEvalConfig):
        self.config = config
        self.ucr = UCRDataset()

    def compute_descriptors(
        self, X: npt.NDArray[np.float64], descriptor: BaseDescriptor
    ) -> list[npt.NDArray[np.float64]]:
        """Compute shape descriptors for all sequences."""
        sdtw = ShapeDTW(seqlen=self.config.seqlen)
        return [sdtw._compute_shape_descriptors(x, descriptor) for x in X]

    def evaluate_dataset(
        self,
        dataset_name: str,
        data: list[npt.NDArray[np.float64]],
        descriptor: BaseDescriptor | None = None,
    ) -> dict[str, float | str]:
        """Evaluate classification performance on a single dataset."""
        X_train, y_train, X_test, y_test = data
        # X_test, y_test = (
        #     np.array([X_test[0]]),
        #     np.array([y_test[0]]),
        # )
        if descriptor is not None:
            # Compute descriptors for shapeDTW
            # start_time = time.time()
            train_seqs = self.compute_descriptors(X_train, descriptor)
            test_seqs = self.compute_descriptors(X_test, descriptor)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print(f"Descriptor computation time : {execution_time}")
        else:
            # Use raw sequences for DTW
            train_seqs = list(X_train)
            test_seqs = list(X_test)
        # Use optimized nearest neighbor implementation
        y_pred = efficient_nearest_neighbor(
            train_seqs,
            test_seqs,
            y_train,
            n_jobs=self.config.n_jobs,
            batch_size=self.config.batch_size,
        )

        # Compute accuracy
        accuracy = np.mean(y_pred == y_test)

        return {"dataset": dataset_name, "accuracy": accuracy}

    def run_evaluation(self, path: Path) -> pd.DataFrame:
        """Run evaluation on all configured datasets."""
        path.parent.mkdir(parents=True, exist_ok=True)

        for dataset in tqdm(self.config.dataset_names, desc="Processing datasets"):
            try:
                # Load existing results
                if path.exists():
                    existing_results = pd.read_csv(path)
                else:
                    existing_results = pd.DataFrame(
                        columns=["dataset", "method", "accuracy"]
                    )

                print(f"Evaluating {dataset}\n")
                # start_time = time.time()
                X_train, y_train, X_test, y_test = self.ucr.load_dataset(dataset, "all")  # type:ignore
                # end_time = time.time()
                # execution_time = end_time - start_time
                # print(f"Loading time: {execution_time} seconds")
                if y_train is None:
                    raise ValueError(f"Dataset {dataset} doesn't contain labels")
                # Evaluate DTW if not already done
                if not (
                    (existing_results["dataset"] == dataset)
                    & (existing_results["method"] == "DTW")
                ).any():
                    print("Method : simple DTW ...")
                    dtw_results = self.evaluate_dataset(
                        dataset,
                        [X_train, y_train, X_test, y_test],  # type:ignore
                    )
                    print("\n")
                    dtw_results["method"] = "DTW"
                    new_df = pd.concat(
                        [existing_results, pd.DataFrame([dtw_results])],
                        ignore_index=True,
                    )
                    new_df.to_csv(path, index=False)
                    existing_results = new_df

                # Evaluate shapeDTW with each descriptor
                for desc_name, descriptor in self.config.descriptors.items():
                    method_name = f"ShapeDTW-{desc_name}"
                    if not (
                        (existing_results["dataset"] == dataset)
                        & (existing_results["method"] == method_name)
                    ).any():
                        print(f"Method: {method_name} ...")
                        shape_results = self.evaluate_dataset(
                            dataset,
                            [X_train, y_train, X_test, y_test],  # type:ignore
                            descriptor,
                        )
                        print("\n")
                        shape_results["method"] = method_name
                        new_df = pd.concat(
                            [existing_results, pd.DataFrame([shape_results])],
                            ignore_index=True,
                        )
                        new_df.to_csv(path, index=False)
                        existing_results = new_df

            except Exception as e:
                print(f"Error while evaluating dataset {dataset}: {e!s}")

        return pd.DataFrame(existing_results)
