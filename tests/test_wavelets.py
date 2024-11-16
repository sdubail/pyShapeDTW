import numpy as np
import numpy.typing as npt
import pytest
import pywt

from pyshapeDTW.descriptors.wavelets import DWT, DWTParams


@pytest.fixture
def simple_sequence() -> npt.NDArray[np.float64]:
    """Create simple test sequence."""
    t = np.linspace(0, 1, 128, dtype=np.float64)
    return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)


@pytest.fixture
def dwt_default() -> DWT:
    """Create default DWT descriptor."""
    return DWT()


@pytest.fixture
def dwt_custom() -> DWT:
    """Create DWT descriptor with custom parameters."""
    params = DWTParams(num_levels=4, wavelet="db4", mode="symmetric")
    return DWT(params)


def test_dwt_basic(dwt_default: DWT, simple_sequence: npt.NDArray[np.float64]) -> None:
    """Test basic DWT functionality."""
    desc = dwt_default(simple_sequence)

    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float64

    # Check if coefficients capture frequency components
    # Low frequency coefficients should have larger magnitude
    low_freq_coeffs = desc[: len(desc) // 2]
    high_freq_coeffs = desc[len(desc) // 2 :]
    assert np.mean(np.abs(low_freq_coeffs)) > np.mean(np.abs(high_freq_coeffs))


def test_padding(dwt_default: DWT) -> None:
    """Test sequence padding to power of 2."""
    # Test sequence of length 100
    seq = np.random.randn(100)
    padded = dwt_default._pad_to_power_of_2(seq)

    # Should be padded to 128
    assert len(padded) == 128
    # Original sequence should be preserved
    assert np.all(padded[:100] == seq)
    # Padding should be mean value
    assert np.allclose(padded[100:], np.mean(seq))


@pytest.mark.parametrize("length", [32, 60, 128, 250])
def test_different_lengths(dwt_default: DWT, length: int) -> None:
    """Test DWT with different sequence lengths."""
    # Create sequence
    seq = np.random.randn(length)
    desc = dwt_default(seq.reshape(-1, 1))

    # Get expected lengths from PyWavelets
    padded_length = int(2 ** np.ceil(np.log2(length)))
    dummy_seq = np.zeros(padded_length)
    expected_coeffs = pywt.wavedec(
        dummy_seq,
        wavelet=dwt_default.params.wavelet,
        level=dwt_default.params.num_levels,
        mode=dwt_default.params.mode,
    )
    expected_total_length = sum(len(c) for c in expected_coeffs)

    assert len(desc) == expected_total_length

    # Test decomposition_lengths method
    computed_lengths = dwt_default.decomposition_lengths(length)
    assert sum(computed_lengths) == len(desc)


def test_multivariate(dwt_default: DWT) -> None:
    """Test DWT with multivariate sequences."""
    # Create 2D sequence
    t = np.linspace(0, 1, 128)
    seq_2d = np.column_stack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 10 * t)])

    desc = dwt_default(seq_2d)

    # Should have coefficients for both dimensions
    single_dim_length = len(dwt_default(seq_2d[:, 0]))
    assert len(desc) == 2 * single_dim_length


def test_reconstruction_error(
    dwt_default: DWT, simple_sequence: npt.NDArray[np.float64]
) -> None:
    """Test reconstruction error using wavelets."""
    # Get coefficients
    coeffs = dwt_default(simple_sequence)

    # Reconstruct using PyWavelets
    padded_seq = dwt_default._pad_to_power_of_2(simple_sequence)
    wavelet = pywt.Wavelet(dwt_default.params.wavelet)

    # Split coefficients back into levels
    lengths = dwt_default.decomposition_lengths(len(simple_sequence))
    split_coeffs = []
    start = 0
    for length in lengths:
        split_coeffs.append(coeffs[start : start + length])
        start += length

    # Reconstruct
    reconstructed = pywt.waverec(split_coeffs, wavelet)

    # Check reconstruction error (should be small)
    error = np.mean((padded_seq - reconstructed[: len(padded_seq)]) ** 2)
    assert error < 1e-10


def test_different_wavelets(simple_sequence: npt.NDArray[np.float64]) -> None:
    """Test different wavelet types."""
    wavelets = ["haar", "db2", "sym2", "coif1"]

    for wavelet in wavelets:
        dwt = DWT(DWTParams(wavelet=wavelet))
        desc = dwt(simple_sequence)
        assert isinstance(desc, np.ndarray)


def test_energy_preservation(
    dwt_default: DWT, simple_sequence: npt.NDArray[np.float64]
) -> None:
    """Test if energy is approximately preserved in transform."""
    desc = dwt_default(simple_sequence)

    # Energy in time domain
    time_energy = np.sum(simple_sequence**2)
    # Energy in wavelet domain
    wavelet_energy = np.sum(desc**2)

    # Should be approximately equal (up to padding effects)
    assert np.abs(time_energy - wavelet_energy) / time_energy < 0.1


if __name__ == "__main__":
    # Run visualization test

    def visualization() -> None:
        # Generate sample sequence with multiple frequency components
        t = np.linspace(0, 1, 256)
        seq = (
            np.sin(2 * np.pi * 5 * t)  # Low frequency
            + 0.5 * np.sin(2 * np.pi * 20 * t)  # Medium frequency
            + 0.25 * np.sin(2 * np.pi * 50 * t)
        )  # High frequency

        # Compute DWT
        dwt = DWT(DWTParams(num_levels=4))
        coeffs = dwt(seq)

        # Plot original signal and coefficient levels
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot original signal
        ax1.plot(t, seq)
        ax1.set_title("Original Signal")
        ax1.grid(True)

        # Plot wavelet coefficients
        lengths = dwt.decomposition_lengths(len(seq))
        start = 0

        # Use a basic colormap that's guaranteed to exist
        colors = colormaps["tab10"](np.linspace(0, 1, len(lengths)))

        for i, length in enumerate(lengths):
            coeff_slice = coeffs[start : start + length]
            ax2.plot(
                np.arange(start, start + length),
                coeff_slice,
                color=colors[i],
                label=f'Level {i//2}{"A" if i%2==0 else "D"}',
            )
            start += length

        ax2.set_title("Wavelet Coefficients")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True)

        plt.tight_layout()

        try:
            plt.savefig("tests/plots/dwt_visualization_test.png", bbox_inches="tight")
        except:
            plt.close()

    visualization()
