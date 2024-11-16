import numpy as np
import numpy.typing as npt
import pytest

from pyshapeDTW.descriptors.paa import PAA, PAAParams


@pytest.fixture
def simple_sequence() -> npt.NDArray[np.float64]:
    """Create simple test sequence."""
    return np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64
    )


@pytest.fixture
def paa_default() -> PAA:
    """Create default PAA descriptor."""
    return PAA()


@pytest.fixture
def paa_custom() -> PAA:
    """Create PAA descriptor with custom parameters."""
    params = PAAParams(seg_num=5, seg_len=2, priority="seg_num")
    return PAA(params)


def test_paa_basic(paa_default: PAA, simple_sequence: npt.NDArray[np.float64]) -> None:
    """Test basic PAA functionality."""
    desc = paa_default(simple_sequence)

    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float64
    assert len(desc) == paa_default.params.seg_num

    # First segment should be mean of first few values
    assert np.isclose(
        desc[0],
        np.mean(simple_sequence[: len(simple_sequence) // paa_default.params.seg_num]),
    )


def test_paa_segment_number(simple_sequence: npt.NDArray[np.float64]) -> None:
    """Test different numbers of segments."""
    for n_segments in [2, 5, 10]:
        paa = PAA(PAAParams(seg_num=n_segments))
        desc = paa(simple_sequence)
        assert len(desc) == n_segments


def test_paa_segment_length(simple_sequence: npt.NDArray[np.float64]) -> None:
    """Test segment length priority."""
    paa = PAA(PAAParams(seg_len=2, priority="seg_len"))
    desc = paa(simple_sequence)

    expected_segments = len(simple_sequence) // 2
    assert len(desc) == expected_segments


def test_paa_edge_cases() -> None:
    """Test PAA with edge cases."""
    paa = PAA(PAAParams(seg_num=3))

    # Single value
    single = np.array([1.0])
    desc_single = paa(single)
    assert len(desc_single) == 1
    assert desc_single[0] == 1.0

    # Two values
    double = np.array([1.0, 2.0])
    desc_double = paa(double)
    assert len(desc_double) <= 2

    # Empty sequence should raise error
    with pytest.raises(ValueError):
        paa(np.array([]))


def test_paa_invariants(
    paa_default: PAA, simple_sequence: npt.NDArray[np.float64]
) -> None:
    """Test PAA invariants."""
    desc = paa_default(simple_sequence)

    # Mean of descriptor should approximate mean of sequence
    assert np.isclose(np.mean(desc), np.mean(simple_sequence), rtol=1e-1)

    # All segments should be represented
    assert len(desc) == paa_default.params.seg_num


def test_paa_2d_input(paa_custom: PAA) -> None:
    """Test PAA with 2D input."""
    seq_2d = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float64,
    )
    desc = paa_custom(seq_2d)

    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float64
    # Should preserve number of features
    assert desc.shape[1] == seq_2d.shape[1]
    # Should have requested number of segments (or less if sequence is shorter)
    assert desc.shape[0] == min(paa_custom.params.seg_num, len(seq_2d))

    # Test if means are computed correctly for each dimension
    first_seg_mean = np.mean(seq_2d[: len(seq_2d) // paa_custom.params.seg_num], axis=0)
    assert np.allclose(desc[0], first_seg_mean)


def test_paa_multivariate(paa_default: PAA) -> None:
    """Test PAA with multivariate sequences."""
    # Create multivariate sequence
    t = np.linspace(0, 2 * np.pi, 100)
    seq_multi = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])

    desc = paa_default(seq_multi)

    assert desc.shape[1] == 3  # Should preserve number of dimensions
    assert desc.shape[0] == paa_default.params.seg_num

    # Test if each dimension is handled correctly
    for dim in range(3):
        desc_single = paa_default(seq_multi[:, dim : dim + 1])
        assert np.allclose(desc[:, dim], desc_single)


def test_paa_invariants_multivariate(paa_default: PAA) -> None:
    """Test PAA invariants with multivariate data."""
    # Create multivariate sequence
    seq_multi = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=np.float64,
    )

    desc = paa_default(seq_multi)

    # Mean along time should be preserved for each dimension
    assert np.allclose(np.mean(desc, axis=0), np.mean(seq_multi, axis=0), rtol=1e-1)


if __name__ == "__main__":

    def paa_visualization_multivariate() -> None:
        """Test PAA visualization for multivariate data."""
        # Generate sample multivariate sequence
        t = np.linspace(0, 2 * np.pi, 100)
        seq = np.column_stack([np.sin(t), np.cos(t)])

        paa = PAA(PAAParams(seg_num=10))
        desc = paa(seq)

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot first dimension
        ax1.plot(seq[:, 0], "b-", label="Original", alpha=0.6)
        seg_len = len(seq) // len(desc)
        for i, mean_val in enumerate(desc[:, 0]):
            start = i * seg_len
            end = start + seg_len if i < len(desc) - 1 else len(seq)
            ax1.plot([start, end - 1], [mean_val, mean_val], "r-", linewidth=2)
        ax1.set_title("First Dimension")
        ax1.grid(True)

        # Plot second dimension
        ax2.plot(seq[:, 1], "b-", label="Original", alpha=0.6)
        for i, mean_val in enumerate(desc[:, 1]):
            start = i * seg_len
            end = start + seg_len if i < len(desc) - 1 else len(seq)
            ax2.plot([start, end - 1], [mean_val, mean_val], "r-", linewidth=2)
        ax2.set_title("Second Dimension")
        ax2.grid(True)

        plt.tight_layout()

        # Save or show based on test environment
        try:
            plt.savefig("../plots/paa_multivariate_visualization_test.png")
        except:
            plt.close()

    # Run visualization test
    paa_visualization_multivariate()
