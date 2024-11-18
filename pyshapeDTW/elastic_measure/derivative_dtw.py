import numpy as np
from pyshapeDTW.elastic_measure.warping import wpath2mat
from pyshapeDTW.elastic_measure.base_dtw import dtw_fast, dtw_locality
from dtw import dtw
import numpy.typing as npt





class DerivativeDTW:
    def __init__(self, sequence1, sequence2, metric: str = "euclidean", step_pattern="symmetric2"):
        """
        Initialize with two time series sequences.
        
        Args:
            sequence1: m1 x n numpy array
            sequence2: m2 x n numpy array
        """
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.metric: str = metric
        self.step_pattern: str = step_pattern
        self._validate_input()

    def _validate_input(self):
        """Validate input time series."""
        if self.sequence1 is None or self.sequence2 is None or len(self.sequence1) == 0 or len(self.sequence2) == 0:
            raise ValueError("Input two univariate/multivariate time series instances.")
        
        len1, dims1 = self.sequence1.shape
        len2, dims2 = self.sequence2.shape

        if len1 < dims1 or len2 < dims2:
            raise ValueError("Each dimension of time series should be organized column-wise.")

        if dims1 != dims2:
            raise ValueError("Two time series should have the same dimensions.")

    def calcKeoghGradient1D(self, sequence):
        """
        Calculate gradient as defined by Keogh, 2001.
        
        Gradient formula:
            grad(q_i) = [(q_i - q_{i-1}) + (q_{i+1} - q_{i-1}) / 2] / 2
        
        To ensure the gradient sequence has the same length as the original,
        the sequence is padded at both ends.
        
        Args:
            sequence: 1D numpy array representing the time series.
        
        Returns:
            seq_grad: 1D numpy array of gradients, same length as input sequence.
        """
        if sequence is None or not isinstance(sequence, np.ndarray) or len(sequence.shape) != 1:
            raise ValueError("Please input a univariate time series as a 1D numpy array.")
        
        # Pad the sequence at both ends
        seq_pad = np.pad(sequence, (1, 1), mode='edge')
        
        # Calculate gradients
        seq_grad = ((seq_pad[1:-1] - seq_pad[:-2]) +
                    (seq_pad[2:] - seq_pad[:-2]) / 2) / 2

        return seq_grad
    
    def calcKeoghGradient(self, sequence):
        """
        Usable for both univariate and multivariate time series.
        
        Args:
            sequence: mxn numpy array
                    m -- number of time stamps
                    n -- number of dimensions
                    Generally, m >> n; if not, a warning is raised.
        
        Returns:
            grads: mxn numpy array of gradients, one for each dimension.
        """
        if sequence is None or not isinstance(sequence, np.ndarray):
            raise ValueError("Input must be a univariate/multivariate time series instance as a numpy array.")

        len_seq, dims = sequence.shape
        if len_seq < dims:
            raise ValueError("Each dimension of the time series should be organized column-wise.")

        grads = []

        for i in range(dims):
            # Compute gradient for each dimension using calcKeoghGradient1D
            grad = self.calcKeoghGradient1D(sequence[:, i])
            grads.append(grad)

        # Combine gradients for all dimensions column-wise
        grads = np.column_stack(grads)
        return grads
    
    def _compute_aligned_distance(
        self,
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
        match: npt.NDArray[np.int64],
        ) -> float:
        """Compute Euclidean distance between sequences aligned by warping path.

        Args:
            p: First sequence
            q: Second sequence
            match: Warping path indices

        Returns:
            distance: Euclidean distance between aligned sequences
        """
        # Convert matching indices to warping matrices
        wp: npt.NDArray[np.float64] = wpath2mat(match[:, 0])
        wq: npt.NDArray[np.float64] = wpath2mat(match[:, 1])

        # Apply warping and compute Euclidean distance
        warped_p: npt.NDArray[np.float64] = wp @ p
        warped_q: npt.NDArray[np.float64] = wq @ q

        return float(np.sqrt(np.sum((warped_p - warped_q) ** 2)))
    
    def compute(self):
        """
        Perform derivative DTW computation.
        
        Returns:
            dDerivative: Cumulative distances along the warping path
                         (calculated among derivatives)
            dRaw: Cumulative distances along the warping path
                  (calculated among raw sequences)
            Match: px2 matrix, the warping path
            
        Returns:
            raw_distance: Distance between aligned raw sequences
            shape_distance: DTW distance using shape descriptors
            path_length: Length of optimal warping path
            match: Optimal warping path as indices array
            
        """
        p = self.sequence1
        q = self.sequence2

        # 1. Calculate derivatives
        grads_p = self.calcKeoghGradient(p)
        grads_q = self.calcKeoghGradient(q)

        # 2. Run DTW
        alignment = dtw(grads_p, grads_q, dist_method=self.metric, step_pattern=self.step_pattern, keep_internals=True)

        # Extract matching indices
        match: npt.NDArray[np.int64] = np.column_stack(
            (alignment.index1, alignment.index2)
        ).astype(np.int64)

        # Cumulative distances along the warping path (calculated among derivatives)
        dDerivative = self._compute_aligned_distance(grads_p, grads_q, match) # alignment.costMatrix 
    
        # compute distance using raw signals, instead of descriptor distances
        dRaw = self._compute_aligned_distance(p, q, match)
     
        return dDerivative, dRaw, match