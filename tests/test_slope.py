import numpy as np
import numpy.typing as npt
import pytest

from pyshapeDTW.descriptors.slope import Slope, SlopeParams


@pytest.fixture
def simple_sequence() -> npt.NDArray[np.float64]:
    """Create simple test sequence with known slope."""
    x = np.linspace(0, 100, 100, dtype=np.float64)
    return 2 * x + 1  # Line with slope 2


@pytest.fixture
def slope_default() -> Slope:
    """Create default Slope descriptor."""
    return Slope()


def test_slope_init() -> None:
    """Test Slope descriptor initialization."""
    slope = Slope()
    assert slope.params.seg_num == 5  # Default as per paper
    assert slope.params.scale == 1.0  # Default natural scale

    custom = Slope(SlopeParams(seg_num=10, scale=0.5))
    assert custom.params.seg_num == 10
    assert custom.params.scale == 0.5


def test_slope_basic(
    slope_default: Slope, simple_sequence: npt.NDArray[np.float64]
) -> None:
    """Test basic slope descriptor computation."""
    # Test with 1D input
    desc1 = slope_default(simple_sequence)
    assert isinstance(desc1, np.ndarray)
    assert desc1.dtype == np.float64

    # Test with 2D input
    desc2 = slope_default(simple_sequence.reshape(-1, 1))
    assert isinstance(desc2, np.ndarray)
    assert desc2.dtype == np.float64

    # Results should be the same
    assert np.allclose(desc1, desc2)
    # For a straight line, all slopes should be approximately equal
    assert np.allclose(desc1, 2.0, rtol=1e-1)


def test_slope_multivariate(slope_default: Slope) -> None:
    """Test slope descriptor with multivariate input."""
    # Create 2D sequence with different slopes
    x = np.linspace(0, 100, 100)
    seq_2d = np.column_stack([2 * x, -3 * x])  # Slopes: 2 and -3

    desc = slope_default(seq_2d)

    assert len(desc) == 2 * slope_default.params.seg_num
    # First half should be around 2, second half around -3
    assert np.allclose(desc[: slope_default.params.seg_num], 2.0, rtol=1e-1)
    assert np.allclose(desc[slope_default.params.seg_num :], -3.0, rtol=1e-1)


def test_slope_shift_invariance(slope_default: Slope) -> None:
    """Test y-shift invariance of slope descriptor."""
    x = np.linspace(0, 100, 100)
    seq1 = 2 * x + 1
    seq2 = 2 * x + 10  # Same sequence shifted up by 9

    desc1 = slope_default(seq1)
    desc2 = slope_default(seq2)

    assert np.allclose(desc1, desc2)


@pytest.mark.parametrize("length", [10, 50, 100])
def test_different_lengths(slope_default: Slope, length: int) -> None:
    """Test slope descriptor with different sequence lengths."""
    seq = np.linspace(0, 1, length)
    desc = slope_default(seq)

    # Should always return number of segments * number of dimensions
    expected_length = slope_default.params.seg_num
    assert len(desc) == expected_length


def test_short_sequence(slope_default: Slope) -> None:
    """Test handling of sequences shorter than number of segments."""
    seq = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="too short"):
        slope_default(seq)


def test_constant_sequence(slope_default: Slope) -> None:
    """Test slope descriptor with constant sequence."""
    seq = np.ones(100)
    desc = slope_default(seq)

    assert np.allclose(desc, 0.0)  # All slopes should be zero


def test_noisy_sequence(slope_default: Slope) -> None:
    """Test slope descriptor with noisy sequence."""
    x = np.linspace(0, 100, 100)
    noise = np.random.normal(0, 10, 100)
    seq = 2 * x + noise  # Noisy line with slope 2

    desc = slope_default(seq)

    # Mean slope should be close to 2 despite noise
    assert np.abs(np.mean(desc) - 2.0) < 0.5


if __name__ == "__main__":
    # Run visualization test
    def visualization() -> None:
        """Visualize how the slope descriptor segments and fits a sequence."""
        import matplotlib.pyplot as plt

        # Generate test sequence
        x = np.linspace(0, 100, 100)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)

        # Compute slopes
        slope = Slope()
        desc = slope._compute_slopes(y)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, "b-", label="Original", alpha=0.6)

        # Plot segment boundaries and fitted lines
        seg_len = len(y) // slope.params.seg_num
        for i in range(slope.params.seg_num):
            start_idx = i * seg_len
            end_idx = min(start_idx + seg_len, len(y))

            # Plot segment
            x_seg = x[start_idx:end_idx]
            y_seg = y[start_idx:end_idx]

            # Plot fitted line
            x_fit = np.array([x_seg[0], x_seg[-1]])
            y_mean = np.mean(y_seg)
            y_fit = desc[i] * slope.params.scale * (x_fit - x_fit[0]) + y_mean
            plt.plot(x_fit, y_fit, "r-", linewidth=2)

            # Plot segment boundaries
            plt.axvline(x=x[start_idx], color="g", linestyle="--", alpha=0.5)

        plt.axvline(x=x[-1], color="g", linestyle="--", alpha=0.5)
        plt.title("Slope Descriptor Segmentation and Fitting")
        plt.legend()
        plt.grid(True)

        try:
            plt.savefig("tests/plots/slope_visualization_test.png")
        except:
            plt.close()

    visualization()
