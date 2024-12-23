from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from dtw import dtw
from numpy.testing import assert_array_almost_equal

from pyshapeDTW.elastic_measure.derivative_dtw import DerivativeDTW
from pyshapeDTW.elastic_measure.warping import wpath2mat


@pytest.fixture
def test_sequences() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Fixture for test sequences
    sequence1: npt.NDArray[np.float64] = np.random.rand(10, 2)
    sequence2: npt.NDArray[np.float64] = np.random.rand(10, 2)
    return sequence1, sequence2


@pytest.fixture
def ddtw() -> DerivativeDTW:
    return DerivativeDTW()


def test_init() -> None:
    ddtw = DerivativeDTW(metric="euclidean", step_pattern="symmetric2")
    assert ddtw.metric == "euclidean"
    assert ddtw.step_pattern == "symmetric2"


def test_validate_input_valid_sequences(
    ddtw: DerivativeDTW,
    test_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    sequence1, sequence2 = test_sequences
    # Should not raise any exception
    ddtw._validate_input(sequence1, sequence2)


def test_validate_input_invalid_sequences(ddtw: DerivativeDTW) -> None:
    with pytest.raises(ValueError):
        ddtw._validate_input(None, None)  # type: ignore

    with pytest.raises(ValueError):
        ddtw._validate_input(
            np.random.rand(10, 2), np.random.rand(5, 3)
        )  # Mismatched dimensions

    with pytest.raises(ValueError):
        ddtw._validate_input(
            np.random.rand(2, 10), np.random.rand(2, 10)
        )  # Rows < Columns


def test_calc_keogh_gradient_1d(ddtw: DerivativeDTW) -> None:
    sequence: npt.NDArray[np.float64] = np.arange(10, dtype=np.float64)
    result1: npt.NDArray[np.float64] = np.ones(10, dtype=np.float64)
    result1[0] = 0.25
    result1[-1] = 0.75
    grad1: npt.NDArray[np.float64] = ddtw.calcKeoghGradient1D(sequence)

    sequence2: npt.NDArray[np.float64] = np.array([1, 3, 2, 5], dtype=np.float64)
    result2: npt.NDArray[np.float64] = np.array([0.5, 1.25, 0, 2.25], dtype=np.float64)
    grad2: npt.NDArray[np.float64] = ddtw.calcKeoghGradient1D(sequence2)

    assert grad1.shape == sequence.shape
    assert isinstance(grad1, np.ndarray)
    assert_array_almost_equal(grad1, result1)
    assert_array_almost_equal(grad2, result2)

    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient1D(None)  # type: ignore

    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient1D(np.random.rand(10, 2))  # Not 1D


def test_calc_keogh_gradient_multivariate(
    ddtw: DerivativeDTW,
    test_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    sequence1, _ = test_sequences
    grads: npt.NDArray[np.float64] = ddtw.calcKeoghGradient(sequence1)

    sequence2: npt.NDArray[np.float64] = np.tile(
        np.array([1, 3, 2, 5], dtype=np.float64), (2, 1)
    ).T
    result2: npt.NDArray[np.float64] = np.tile(
        np.array([0.5, 1.25, 0, 2.25], dtype=np.float64), (2, 1)
    ).T
    grad2: npt.NDArray[np.float64] = ddtw.calcKeoghGradient(sequence2)

    assert grads.shape == sequence1.shape
    assert isinstance(grads, np.ndarray)
    assert_array_almost_equal(grad2, result2)

    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient(None)  # type: ignore

    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient(np.random.rand(2, 10))  # Rows < Columns


def test_compute_aligned_distance(ddtw: DerivativeDTW) -> None:
    sequence1: npt.NDArray[np.float64] = np.random.rand(10, 1)
    alignment: Any = dtw(sequence1, sequence1, keep_internals=True)
    match: npt.NDArray[np.int64] = np.column_stack(
        (alignment.index1, alignment.index2)
    ).astype(np.int64)
    dist: float = ddtw._compute_aligned_distance(sequence1, sequence1, match)

    assert isinstance(dist, float)
    assert np.isclose(dist, 0.0)


def test_call(
    ddtw: DerivativeDTW,
    test_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    sequence1, _ = test_sequences
    dDerivative: float
    dRaw: float
    match: npt.NDArray[np.int64]
    dDerivative, dRaw, match = ddtw(sequence1, sequence1)

    match_exact: npt.NDArray[np.int64] = np.tile(
        np.arange(10, dtype=np.int64), (2, 1)
    ).T

    assert isinstance(dDerivative, float)
    assert isinstance(dRaw, float)
    assert isinstance(match, np.ndarray)
    assert match.shape[1] == 2
    assert match.shape[0] > 0
    assert np.isclose(dDerivative, 0.0)
    assert np.isclose(dRaw, 0.0)
    assert_array_almost_equal(match, match_exact)
