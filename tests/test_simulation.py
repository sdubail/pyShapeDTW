import numpy as np
import numpy.typing as npt
import pytest

from pyshapeDTW.descriptors.hog1d import HOG1D, HOG1DParams
from pyshapeDTW.elastic_measure.shape_dtw import ShapeDTW
from pyshapeDTW.simulation.transforms import (
    ScaleParams,
    StretchParams,
    scale_time_series,
    simulate_smooth_curve,
    stretching_ts,
)


@pytest.fixture
def base_sequence() -> npt.NDArray[np.float64]:
    """Generate base test sequence."""
    t = np.linspace(0, 4 * np.pi, 200)
    return np.sin(t) + 0.5 * np.sin(3 * t)


@pytest.fixture
def shape_dtw() -> ShapeDTW:
    """Create ShapeDTW instance."""
    return ShapeDTW(seqlen=20)


@pytest.fixture
def hog1d() -> HOG1D:
    """Create HOG1D descriptor."""
    params = HOG1DParams(cells=(1, 25), n_bins=20, scale=0.1)
    return HOG1D(params)


def create_transformed_pair(
    seq: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Create a transformed pair of sequences with known alignment."""
    # Generate smooth scaling curve
    scale_params = ScaleParams(len=len(seq), max_derivative=1.0, n_nest=10)
    scale = simulate_smooth_curve(scale_params)

    # Scale sequence
    scaled_seq = scale_time_series(seq, scale)

    # Stretch sequence
    stretch_params = StretchParams(percentage=0.15, amount=2)
    sim_idx, gt_align = stretching_ts(len(seq), stretch_params)
    simulated_seq = scaled_seq[sim_idx]

    return seq, simulated_seq, gt_align


def alignment_error(
    gt_align: npt.NDArray[np.int64], pred_align: npt.NDArray[np.int64]
) -> float:
    """Compute alignment error between ground truth and predicted alignment."""
    # Convert to warping curves
    gt_curve = np.zeros((max(gt_align[:, 0]) + 1, max(gt_align[:, 1]) + 1))
    pred_curve = np.zeros_like(gt_curve)

    for i, j in gt_align:
        gt_curve[i, j] = 1
    for i, j in pred_align:
        pred_curve[i, j] = 1

    return np.sum(np.abs(gt_curve - pred_curve))


def test_transformation_consistency(base_sequence: npt.NDArray[np.float64]) -> None:
    """Test that transformations preserve key properties."""
    orig, transformed, align = create_transformed_pair(base_sequence)

    # Length checks
    assert len(transformed) >= len(orig)

    # Alignment checks
    assert align[:, 0].max() == len(orig) - 1
    assert align[:, 1].max() == len(transformed) - 1
    assert np.all(np.diff(align[:, 0]) >= 0)  # Monotonic
    assert np.all(np.diff(align[:, 1]) >= 0)


def test_scaling_properties() -> None:
    """Test properties of scaling transformation."""
    params = ScaleParams(len=100)
    scale = simulate_smooth_curve(params)

    # Should be smooth
    assert np.all(np.abs(np.diff(scale, 2)) < 1.0)

    # Test scaling range
    seq = np.ones(100)
    scaled = scale_time_series(seq, scale, scale_range=(0.5, 2.0))
    assert np.all(scaled >= 0.4)
    assert np.all(scaled <= 2.1)  # Small margin for numerical error


def test_stretching_properties() -> None:
    """Test properties of stretching transformation."""
    params = StretchParams(percentage=0.2, amount=3)
    sim_idx, align = stretching_ts(100, params)

    # Check monotonicity
    assert np.all(np.diff(align[:, 0]) >= 0)
    assert np.all(np.diff(align[:, 1]) >= 0)

    # Check stretching amount
    unique_mappings = len(np.unique(align[:, 0]))
    assert unique_mappings == 100  # All original points should be mapped
    assert len(align) > 100  # Sequence should be stretched


if __name__ == "__main__":
    # def test_shape_dtw_alignment(
    #     base_sequence: npt.NDArray[np.float64], shape_dtw: ShapeDTW, hog1d: HOG1D
    # ) -> None:
    #     """Test ShapeDTW's ability to recover true alignment."""
    #     # Create transformed pair
    #     orig, transformed, gt_align = create_transformed_pair(base_sequence)

    #     # Run Shape-DTW
    #     _, _, _, pred_align = shape_dtw(orig, transformed, hog1d)

    #     # Compute alignment error
    #     error = alignment_error(gt_align, pred_align)

    #     # Error should be reasonable (threshold determined empirically)
    #     assert error / len(gt_align) < 0.5

    # @pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.2])
    # def test_noise_robustness(
    #     base_sequence: npt.NDArray[np.float64],
    #     shape_dtw: ShapeDTW,
    #     hog1d: HOG1D,
    #     noise_level: float,
    # ) -> None:
    #     """Test robustness to noise."""
    #     # Create transformed pair
    #     orig, transformed, gt_align = create_transformed_pair(base_sequence)

    #     # Add noise
    #     noisy = transformed + np.random.normal(0, noise_level, transformed.shape)

    #     # Run Shape-DTW
    #     _, _, _, pred_align = shape_dtw(orig, noisy, hog1d)

    #     # Compute alignment error
    #     error = alignment_error(gt_align, pred_align)

    #     # Error should increase gracefully with noise
    #     assert error / len(gt_align) < 0.5 + noise_level

    # Run some visual tests
    import matplotlib.pyplot as plt

    # Generate example
    t = np.linspace(0, 4 * np.pi, 200)
    seq = np.sin(t) + 0.5 * np.sin(3 * t)
    orig, transformed, gt_align = create_transformed_pair(seq)

    # Run Shape-DTW
    sdtw = ShapeDTW(seqlen=20)
    params = HOG1DParams(cells=(1, 25), n_bins=20, scale=0.1)
    hog1d_des = HOG1D(params)
    _, _, _, pred_align = sdtw(orig, transformed, hog1d_des)

    error = alignment_error(gt_align, pred_align)

    # Plot original vs transformed
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(orig, label="Original")
    plt.plot(transformed, label="Transformed")
    plt.legend()
    plt.title("Sequences")

    # Plot alignments
    plt.subplot(122)
    plt.plot(gt_align[:, 0], gt_align[:, 1], "b-", label="Ground Truth")
    plt.plot(
        pred_align[:, 0], pred_align[:, 1], "r--", label=f"Predicted - error = {error}"
    )
    plt.legend()
    plt.title("Alignments")
    plt.show()
