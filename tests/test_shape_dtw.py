import numpy as np
import numpy.typing as npt
import pytest

from pyshapeDTW.descriptors.base import BaseDescriptor
from pyshapeDTW.elastic_measure.shape_dtw import ShapeDTW, ShapeDTWMulti


@pytest.fixture
def simple_sequences() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fixture providing simple test sequences."""
    t: npt.NDArray[np.float64] = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
    p: npt.NDArray[np.float64] = np.sin(t)
    q: npt.NDArray[np.float64] = np.sin(t + np.pi / 4)  # Phase shifted sine
    return p, q


@pytest.fixture
def shape_dtw() -> ShapeDTW:
    """Fixture providing ShapeDTW instance."""
    return ShapeDTW(seqlen=20)


@pytest.fixture
def multi_sequences() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fixture providing multivariate test sequences."""
    t: npt.NDArray[np.float64] = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
    p: npt.NDArray[np.float64] = np.column_stack((np.sin(t), np.cos(t))).astype(
        np.float64
    )
    q: npt.NDArray[np.float64] = np.column_stack(
        (np.sin(t + np.pi / 4), np.cos(t + np.pi / 4))
    ).astype(np.float64)
    return p, q


def test_shape_dtw_init() -> None:
    """Test ShapeDTW initialization."""
    sdtw = ShapeDTW(seqlen=20, metric="euclidean")
    assert sdtw.seqlen == 20
    assert sdtw.metric == "euclidean"


def test_basic_shape_dtw(
    shape_dtw: ShapeDTW,
    simple_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    descriptor: BaseDescriptor,
) -> None:
    """Test basic ShapeDTW functionality."""
    p, q = simple_sequences

    raw_dist, shape_dist, path_len, match = shape_dtw(p, q, descriptor)

    assert isinstance(raw_dist, float)
    assert isinstance(shape_dist, float)
    assert raw_dist >= 0
    assert shape_dist >= 0
    assert path_len >= max(len(p), len(q))
    assert match.shape[1] == 2
    assert match.dtype == np.int64


def test_shape_descriptors(
    shape_dtw: ShapeDTW,
    simple_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    descriptor: BaseDescriptor,
) -> None:
    """Test shape descriptor computation."""
    p, q = simple_sequences

    # Get descriptors
    p_desc = shape_dtw._compute_shape_descriptors(p.reshape(-1, 1), descriptor)
    q_desc = shape_dtw._compute_shape_descriptors(q.reshape(-1, 1), descriptor)

    assert p_desc.dtype == np.float64
    assert q_desc.dtype == np.float64
    assert p_desc.shape[0] == len(p)
    assert q_desc.shape[0] == len(q)
    assert p_desc.shape == q_desc.shape


def test_subsequence_sampling(shape_dtw: ShapeDTW) -> None:
    """Test subsequence sampling."""
    # Create test sequence
    seq: npt.NDArray[np.float64] = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64
    ).reshape(-1, 1)

    subsequences = shape_dtw._sample_subsequences(seq)

    assert subsequences.dtype == np.float64
    assert len(subsequences) == len(seq)
    assert all(
        len(subseq) == shape_dtw.seqlen
        if shape_dtw.seqlen % 2 == 1
        else shape_dtw.seqlen + 1  # window always centered on sample
        for subseq in subsequences
    )

    # Test padding at boundaries
    assert np.allclose(subsequences[0][: shape_dtw.seqlen // 2], seq[0])
    assert np.allclose(subsequences[-1][-shape_dtw.seqlen // 2 :], seq[-1])


@pytest.mark.parametrize("metric", ["euclidean", "cityblock", "cosine"])
def test_different_metrics(
    simple_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    descriptor: BaseDescriptor,
    metric: str,
) -> None:
    """Test different distance metrics."""
    p, q = simple_sequences
    sdtw = ShapeDTW(seqlen=20, metric=metric)

    raw_dist, shape_dist, path_len, match = sdtw(p, q, descriptor)

    assert isinstance(raw_dist, float)
    assert isinstance(shape_dist, float)
    assert raw_dist >= 0
    assert shape_dist >= 0


def test_multivariate_shape_dtw(
    multi_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    descriptor: BaseDescriptor,
) -> None:
    """Test multivariate ShapeDTW."""
    p, q = multi_sequences
    sdtw = ShapeDTWMulti(seqlen=20)

    raw_dist, shape_dist, path_len, match = sdtw(p, q, descriptor)

    assert isinstance(raw_dist, float)
    assert isinstance(shape_dist, float)
    assert raw_dist >= 0
    assert shape_dist >= 0
    assert path_len >= max(len(p), len(q))
    assert match.dtype == np.int64


@pytest.mark.parametrize("seqlen", [10, 20, 30])
def test_different_sequence_lengths(
    simple_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    descriptor: BaseDescriptor,
    seqlen: int,
) -> None:
    """Test different subsequence lengths."""
    p, q = simple_sequences
    sdtw = ShapeDTW(seqlen=seqlen)

    raw_dist, shape_dist, path_len, match = sdtw(p, q, descriptor)

    assert isinstance(raw_dist, float)
    assert isinstance(shape_dist, float)
    assert raw_dist >= 0
    assert shape_dist >= 0


def test_aligned_distance(
    shape_dtw: ShapeDTW,
    simple_sequences: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    """Test computation of aligned sequence distance."""
    p, q = simple_sequences
    # Create simple matching path
    match: npt.NDArray[np.int64] = np.column_stack(
        (np.arange(len(p)), np.arange(len(p)))
    ).astype(np.int64)

    distance = shape_dtw._compute_aligned_distance(
        p.reshape(-1, 1), q.reshape(-1, 1), match
    )

    assert isinstance(distance, float)
    assert distance >= 0
    # When sequences aligned one-to-one, should equal Euclidean distance
    assert np.allclose(distance, np.sqrt(np.sum((p - q) ** 2)))


def test_input_validation(shape_dtw: ShapeDTW, descriptor: BaseDescriptor) -> None:
    """Test handling of different input formats."""
    # 1D input
    p: npt.NDArray[np.float64] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    q: npt.NDArray[np.float64] = np.array([2.0, 3.0, 4.0], dtype=np.float64)

    raw_dist, shape_dist, path_len, match = shape_dtw(p, q, descriptor)
    assert isinstance(raw_dist, float)

    # 2D input
    p_2d: npt.NDArray[np.float64] = p.reshape(-1, 1)
    q_2d: npt.NDArray[np.float64] = q.reshape(-1, 1)

    raw_dist_2d, shape_dist_2d, path_len_2d, match_2d = shape_dtw(
        p_2d, q_2d, descriptor
    )
    assert isinstance(raw_dist_2d, float)

    # Results should be the same
    assert np.allclose(raw_dist, raw_dist_2d)
    assert np.allclose(shape_dist, shape_dist_2d)
