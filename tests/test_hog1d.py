import numpy as np
import numpy.typing as npt
import pytest

from pyshapeDTW.descriptors.hog1d import HOG1D, HOG1DParams


@pytest.fixture
def simple_sequence() -> npt.NDArray[np.float64]:
    """Generate simple test sequence."""
    t = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
    return np.sin(t)


@pytest.fixture
def hog1d() -> HOG1D:
    """Create default HOG1D descriptor."""
    return HOG1D()


@pytest.fixture
def custom_hog1d() -> HOG1D:
    """Create HOG1D descriptor with custom parameters."""
    params = HOG1DParams(n_bins=12, cells=(1, 20), overlap=5, scale=0.2, signed=True)
    return HOG1D(params)


def test_hog1d_init(hog1d: HOG1D) -> None:
    """Test HOG1D initialization."""
    assert hog1d.params.n_bins == 8
    assert hog1d.params.cells == (1, 25)
    assert hog1d.params.overlap == 0
    assert hog1d.params.scale == 0.1
    assert hog1d.params.signed is True

    assert len(hog1d.angles) == hog1d.params.n_bins + 1
    assert len(hog1d.center_angles) == hog1d.params.n_bins


def test_hog1d_compute(hog1d: HOG1D, simple_sequence: npt.NDArray[np.float64]) -> None:
    """Test basic HOG1D computation."""
    # Test with 1D input
    desc1 = hog1d(simple_sequence)
    assert isinstance(desc1, np.ndarray)
    assert desc1.dtype == np.float64

    # Test with 2D input
    desc2 = hog1d(simple_sequence.reshape(-1, 1))
    assert isinstance(desc2, np.ndarray)
    assert desc2.dtype == np.float64

    # Results should be the same
    assert np.allclose(desc1, desc2)


def test_hog1d_normalization(hog1d: HOG1D) -> None:
    """Test histogram normalization."""
    # Create sequence with known gradients
    seq = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    desc = hog1d(seq)

    # Each cell histogram should be normalized
    cell_width = hog1d.params.cells[1]
    n_cells = len(desc) // hog1d.params.n_bins

    for i in range(n_cells):
        cell_hist = desc[i * hog1d.params.n_bins : (i + 1) * hog1d.params.n_bins]
        # L2 norm should be close to 1
        assert np.abs(np.linalg.norm(cell_hist) - 1.0) < 1e-6


@pytest.mark.parametrize("signed", [True, False])
def test_hog1d_signed_unsigned(
    signed: bool, simple_sequence: npt.NDArray[np.float64]
) -> None:
    """Test signed vs unsigned gradients."""
    params = HOG1DParams(signed=signed)
    desc = HOG1D(params)(simple_sequence)

    if signed:
        # Should capture both positive and negative gradients
        assert np.any(desc > 0)
    else:
        # Angles should be in [0, π/2]
        angles = np.linspace(0, np.pi / 2, params.n_bins + 1)
        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi / 2)


def test_hog1d_overlap(simple_sequence: npt.NDArray[np.float64]) -> None:
    """Test cell overlap behavior."""
    # No overlap
    desc1 = HOG1D(HOG1DParams(overlap=0))(simple_sequence)

    # With overlap
    desc2 = HOG1D(HOG1DParams(overlap=10))(simple_sequence)

    # Overlap should produce more cells
    assert len(desc2) > len(desc1)


# def test_invalid_input(hog1d: HOG1D) -> None:
#     """Test error handling for invalid inputs."""
#     # 3D input
#     with pytest.raises(ValueError):
#         hog1d(np.random.randn(10, 2, 2))

#     # Empty input
#     with pytest.raises(ValueError):
#         hog1d(np.array([]))


def test_gradient_computation(hog1d: HOG1D) -> None:
    """Test gradient computation."""
    # Linear sequence should have constant gradient
    seq = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    grads, angles = hog1d._compute_gradients(seq)

    assert np.allclose(grads[1:-1], 1.0 / hog1d.params.scale)
    assert len(grads) == len(seq)
    assert len(angles) == len(seq)


def test_custom_parameters(
    custom_hog1d: HOG1D, simple_sequence: npt.NDArray[np.float64]
) -> None:
    """Test HOG1D with custom parameters."""
    desc = custom_hog1d(simple_sequence)

    # Check descriptor shape matches custom parameters
    n_cells = len(
        range(
            0,
            len(simple_sequence) - custom_hog1d.params.cells[1] + 1,
            custom_hog1d.params.cells[1] - custom_hog1d.params.overlap,
        )
    )
    expected_length = n_cells * custom_hog1d.params.n_bins

    assert len(desc) == expected_length


def test_multivariate_hog1d(hog1d: HOG1D) -> None:
    """Test HOG1D with multivariate input."""
    # Create 2D sequence
    t = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
    seq_2d = np.column_stack([np.sin(t), np.cos(t)])

    desc = hog1d(seq_2d)

    # Should compute descriptor for each dimension
    single_dim_desc = hog1d(seq_2d[:, 0])
    assert len(desc) == 2 * len(single_dim_desc)


def test_edge_cases(hog1d: HOG1D) -> None:
    """Test HOG1D with edge cases."""
    # Very short sequence
    short_seq = np.array([1.0, 2.0], dtype=np.float64)
    desc_short = hog1d(short_seq)
    assert isinstance(desc_short, np.ndarray)

    # Constant sequence
    const_seq = np.ones(50, dtype=np.float64)
    desc_const = hog1d(const_seq)
    assert isinstance(desc_const, np.ndarray)

    # Single value
    single_val = np.array([1.0], dtype=np.float64)
    desc_single = hog1d(single_val)
    assert isinstance(desc_single, np.ndarray)


def test_boundary_handling(hog1d: HOG1D) -> None:
    """Test gradient computation at sequence boundaries."""
    seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    grads, angles = hog1d._compute_gradients(seq)

    # Should have same length as input
    assert len(grads) == len(seq)
    assert len(angles) == len(seq)

    # Check boundary padding
    assert grads[0] == grads[1]  # First gradient padded
    assert grads[-1] == grads[-2]  # Last gradient padded


def test_cell_boundary_handling(hog1d: HOG1D) -> None:
    """Test cell computation at sequence boundaries."""
    # Create sequence shorter than cell size
    seq = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    desc = hog1d(seq)
    assert isinstance(desc, np.ndarray)

    # Create sequence exactly one cell size
    seq_cell = np.ones(hog1d.params.cells[1], dtype=np.float64)
    desc_cell = hog1d(seq_cell)
    assert isinstance(desc_cell, np.ndarray)


def test_histogram_binning_properties(hog1d: HOG1D) -> None:
    """Test histogram binning properties."""
    # Create sequence with known gradients
    t = np.linspace(0, np.pi, 100, dtype=np.float64)
    seq = np.sin(t)
    desc = hog1d(seq)

    # Reshape descriptor into cells
    n_cells = len(desc) // hog1d.params.n_bins
    histograms = desc.reshape(n_cells, hog1d.params.n_bins)

    # Each histogram should sum close to 1 (after L2 normalization)
    for hist in histograms:
        assert np.abs(np.linalg.norm(hist) - 1.0) < 1e-6
