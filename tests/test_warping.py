from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from pyshapeDTW.elastic_measure.warping import (
    WarpingPath,
    apply_warping,
    get_warped_segments,
    wpath2mat,
)


@pytest.fixture
def simple_path() -> NDArray[np.int_]:
    """Fixture providing a simple warping path."""
    return np.array([0, 1, 1, 2], dtype=np.int_)


@pytest.fixture
def simple_signal() -> NDArray[np.float64]:
    """Fixture providing a simple test signal."""
    return np.array([1, 2, 3], dtype=np.float64)


@pytest.fixture
def warping_path() -> WarpingPath:
    """Fixture providing a WarpingPath instance."""
    indices1 = np.array([0, 1, 1, 2], dtype=np.int_)
    indices2 = np.array([0, 1, 2, 3], dtype=np.int_)
    return WarpingPath(indices1, indices2)


def test_wpath2mat(simple_path: NDArray[np.int_]) -> None:
    """Test warping matrix creation."""
    wmat = wpath2mat(simple_path)

    assert wmat.shape == (4, 3)
    assert np.all(np.sum(wmat, axis=1) == 1)  # One alignment per row
    assert wmat[0, 0] == 1
    assert np.all(wmat[1:3, 1] == 1)
    assert wmat[3, 2] == 1


@pytest.mark.parametrize("signal_type", ["1d", "2d"])
def test_apply_warping(
    simple_path: NDArray[np.int_], signal_type: Literal["1d", "2d"]
) -> None:
    """Test warping application for different signal types."""
    if signal_type == "1d":
        signal: NDArray[np.float64] = np.array([1, 2, 3], dtype=np.float64)
        expected: NDArray[np.float64] = np.array([1, 2, 2, 3], dtype=np.float64)
    else:
        signal = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)
        expected = np.array([[1, 10], [2, 20], [2, 20], [3, 30]], dtype=np.float64)

    wmat = wpath2mat(simple_path)
    warped = apply_warping(signal, wmat)

    assert np.array_equal(warped, expected)


def test_get_warped_segments(
    simple_signal: NDArray[np.float64], simple_path: NDArray[np.int_]
) -> None:
    """Test warped segment extraction."""
    segments = get_warped_segments(simple_signal, simple_path)

    assert len(segments) == 3
    assert np.array_equal(segments[0], [1])
    assert np.array_equal(segments[1], [2, 2])
    assert np.array_equal(segments[2], [3])


class TestWarpingPath:
    """Tests for WarpingPath class."""

    def test_properties(self, warping_path: WarpingPath) -> None:
        """Test basic properties."""
        assert len(warping_path.indices1) == 4
        assert len(warping_path.indices2) == 4
        assert warping_path.path.shape == (4, 2)

    def test_to_matrix(self, warping_path: WarpingPath) -> None:
        """Test conversion to matrix form."""
        wmat = warping_path.to_matrix()
        assert wmat.shape == (4, 3)
        assert np.all(np.sum(wmat, axis=1) == 1)

    def test_apply(
        self, warping_path: WarpingPath, simple_signal: NDArray[np.float64]
    ) -> None:
        """Test warping application."""
        warped1 = warping_path.apply(simple_signal, sequence=1)
        assert len(warped1) == 4

    def test_get_segments(
        self, warping_path: WarpingPath, simple_signal: NDArray[np.float64]
    ) -> None:
        """Test segment extraction."""
        segments = warping_path.get_segments(simple_signal)
        assert len(segments) == 3

    def test_warping_amount(self, warping_path: WarpingPath) -> None:
        """Test warping amount computation."""
        amount = warping_path.get_warping_amount()
        assert isinstance(amount, float)
        assert amount >= 0

        amounts = warping_path.get_warping_amount(normalize=False)
        assert len(amounts) == len(warping_path.indices1)

    @pytest.mark.parametrize(
        "len1,len2,expected",
        [
            (3, 4, True),  # Valid lengths
            (2, 4, False),  # Invalid first length
            (3, 3, False),  # Invalid second length
        ],
    )
    def test_is_valid(
        self, warping_path: WarpingPath, len1: int, len2: int, expected: bool
    ) -> None:
        """Test path validation."""
        assert warping_path.is_valid(len1, len2) == expected


@pytest.mark.parametrize(
    "invalid_path",
    [
        np.array([[0, 0], [2, 1], [1, 2]], dtype=np.int_),  # Non-monotonic
        np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int_),  # Doesn't start at (0,0)
        np.array([[0, 0], [1, 1]], dtype=np.int_),  # Doesn't reach end
    ],
)
def test_invalid_paths(invalid_path: NDArray[np.int_]) -> None:
    """Test validation of invalid paths."""
    wpath = WarpingPath(invalid_path[:, 0], invalid_path[:, 1])
    assert not wpath.is_valid(4, 4)
