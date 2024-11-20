import numpy as np
import pytest
from dtw import dtw

from pyshapeDTW.elastic_measure.derivative_dtw import (
    DerivativeDTW,  # Replace with the actual module name
)
from pyshapeDTW.elastic_measure.warping import wpath2mat


@pytest.fixture
def test_sequences():
    # Fixture for test sequences
    sequence1 = np.random.rand(10, 2)
    sequence2 = np.random.rand(10, 2)
    return sequence1, sequence2


def test_init_valid_sequences(test_sequences):
    sequence1, sequence2 = test_sequences
    ddtw = DerivativeDTW(sequence1, sequence2)
    assert np.array_equal(ddtw.sequence1, sequence1)
    assert np.array_equal(ddtw.sequence2, sequence2)


def test_init_invalid_sequences():
    with pytest.raises(ValueError):
        DerivativeDTW(None, None)

    with pytest.raises(ValueError):
        DerivativeDTW(
            np.random.rand(10, 2), np.random.rand(5, 3)
        )  # Mismatched dimensions

    with pytest.raises(ValueError):
        DerivativeDTW(np.random.rand(2, 10), np.random.rand(2, 10))  # Rows < Columns


def test_calc_keogh_gradient_1d():
    ddtw = DerivativeDTW(np.random.rand(10, 2), np.random.rand(10, 2))
    sequence = np.arange(10)
    result1 = np.ones(10)
    result1[0] = 0.25
    result1[-1] = 0.75
    grad1 = ddtw.calcKeoghGradient1D(sequence)

    sequence2 = np.array([1, 3, 2, 5])
    result2 = np.array([0.5, 1.25, 0, 2.25])
    grad2 = ddtw.calcKeoghGradient1D(sequence2)

    assert grad1.shape == sequence.shape
    assert isinstance(grad1, np.ndarray)
    assert (grad1 == result1).all()
    assert (((grad2 - result2) ** 2).sum()) < 10**-6

    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient1D(None)

    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient1D(np.random.rand(10, 2))  # Not 1D


def test_calc_keogh_gradient_multivariate(test_sequences):
    sequence1, _ = test_sequences
    ddtw = DerivativeDTW(sequence1, sequence1)
    grads = ddtw.calcKeoghGradient(sequence1)

    sequence2 = np.tile(np.array([1, 3, 2, 5]), (2, 1)).T
    result2 = np.tile(np.array([0.5, 1.25, 0, 2.25]), (2, 1)).T
    grad2 = ddtw.calcKeoghGradient(sequence2)

    assert grads.shape == sequence1.shape
    assert isinstance(grads, np.ndarray)
    assert (((grad2 - result2) ** 2).sum()) < 10**-6
    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient(None)

    with pytest.raises(ValueError):
        ddtw.calcKeoghGradient(np.random.rand(2, 10))  # Rows < Columns


def test_compute_aligned_distance(test_sequences):
    sequence1 = np.random.rand(10, 1)
    ddtw = DerivativeDTW(sequence1, sequence1)
    alignment = dtw(sequence1, sequence1, keep_internals=True)
    match = np.column_stack((alignment.index1, alignment.index2)).astype(np.int64)
    dist = ddtw._compute_aligned_distance(sequence1, sequence1, match)
    assert isinstance(dist, float)
    assert (dist) ** 2 < 10**-6


def test_compute(test_sequences):
    sequence1, _ = test_sequences
    ddtw = DerivativeDTW(sequence1, sequence1)
    dDerivative, dRaw, match = ddtw.compute()
    match_exact = np.tile(np.arange(10), (2, 1)).T
    assert isinstance(dDerivative, float)
    assert isinstance(dRaw, float)
    assert isinstance(match, np.ndarray)
    assert match.shape[1] == 2
    assert match.shape[0] > 0
    assert (dDerivative**2) < 10**-6
    assert (dDerivative**2) < 10**-6
    assert np.sum((match - match_exact) ** 2) < 10**-6
