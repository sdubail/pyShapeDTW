from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


class ShapeDescriptor(Protocol):
    """Protocol defining shape descriptor interface."""

    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute shape descriptor for a subsequence.

        Args:
            subsequence: Time series subsequence to describe

        Returns:
            descriptor: Shape descriptor vector
        """
        ...


class BaseDescriptor(ABC):
    """Abstract base class for shape descriptors."""

    @abstractmethod
    def __call__(self, subsequence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute shape descriptor for a subsequence."""
        pass

    def _validate_input(
        self, subsequence: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Validate and normalize input subsequence.

        Args:
            subsequence: Input subsequence

        Returns:
            normalized: Validated and reshaped subsequence
        """
        # Convert to numpy array if needed
        arr = np.asarray(subsequence, dtype=np.float64)

        # Ensure 2D
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        return arr
