import numpy as np
import pytest
from numpy.typing import NDArray

from pyshapeDTW.elastic_measure.base_dtw import dtw_fast, dtw_locality


@pytest.fixture
def simple_sequences() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fixture providing simple test sequences."""
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([1, 2, 3, 3, 4, 5], dtype=np.float64)
    return x, y


def test_dtw_fast(
    simple_sequences: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Test basic DTW functionality."""
    x, y = simple_sequences

    dist, match = dtw_fast(x, y)

    # Test basic properties
    assert isinstance(dist, float | np.floating)
    assert dist >= 0
    assert len(match) >= max(len(x), len(y))
    assert match[0, 0] == 0 and match[0, 1] == 0  # Start at beginning
    assert match[-1, 0] == len(x) - 1 and match[-1, 1] == len(y) - 1  # End at end


@pytest.mark.parametrize("window_size", [None, 2, 3])
def test_dtw_locality(
    simple_sequences: tuple[NDArray[np.float64], NDArray[np.float64]],
    window_size: int | None,
) -> None:
    """Test DTW with locality constraint for different window sizes."""
    x, y = simple_sequences

    dist, dMat, lPath, match = dtw_locality(x, y, window_size)

    # Test basic properties
    assert isinstance(dist, float | np.floating)
    assert dist >= 0
    assert dMat.shape == (len(x), len(y))
    assert isinstance(lPath, int | np.integer)
    assert lPath >= max(len(x), len(y))

    # Test window constraint if specified
    if window_size is not None:
        for i, j in match:
            assert abs(i - j) <= window_size


@pytest.mark.parametrize("shape", [(10,), (10, 1), (10, 3)])
def test_different_input_shapes(shape: tuple[int, ...]) -> None:
    """Test DTW with different input shapes."""
    # Create random sequences
    x = np.random.rand(*shape)
    y = np.random.rand(*shape)

    # Test both functions
    dist_fast, match_fast = dtw_fast(x, y)
    dist_local, _, _, match_local = dtw_locality(x, y)

    assert isinstance(dist_fast, float | np.floating)
    assert dist_fast >= 0
    assert match_fast.shape[1] == 2
    assert match_local.shape[1] == 2


def test_invalid_inputs() -> None:
    """Test error handling for invalid inputs."""
    x = np.array([1, 2, 3], dtype=np.float64)
    y = np.array([[1, 2], [3, 4]], dtype=np.float64)  # Different dimensions

    with pytest.raises(ValueError):
        dtw_fast(x, y)

    with pytest.raises(ValueError):
        dtw_locality(x, y)
