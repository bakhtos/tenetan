import tensorly as tl
import numpy as np
from abc import ABC


class SupraAdjacencyMatrix(ABC):

    def __init__(self):
        self._supra = None
        self._jc = None
        self._mnc = None
        self._mlc = None
        self._cc = None
        self._orig_T = None
        self._orig_N = None

    def compute_centrality(self):
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self._supra)

        # Find the index of the largest eigenvalue
        max_eigenvalue_index = np.argmax(eigenvalues)

        # Get the corresponding eigenvector
        dominant_eigenvector = eigenvectors[:, max_eigenvalue_index]
        # Ensure it has a positive first element (or flip signs)
        if dominant_eigenvector[0] < 0:
            dominant_eigenvector = -dominant_eigenvector
        self._jc = np.reshape(dominant_eigenvector, shape=(self._orig_T, self._orig_N)).T
        self._mnc = np.sum(self._jc, axis=1)
        self._mlc = np.sum(self._jc, axis=0)
        self._cc = self._jc / self._mlc


    @property
    def joint_centrality(self):
        return self._jc

    @property
    def jc(self):
        return self._jc

    @property
    def marginal_node_centrality(self):
        return self._mnc

    @property
    def mnc(self):
        return self._mnc

    @property
    def marginal_layer_centrality(self):
        return self._mlc

    @property
    def mlc(self):
        return self._mlc

    @property
    def conditional_centrality(self):
        return self._cc

    @property
    def cc(self):
        return self._cc


class TaylorSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot, epsilon=1.0, time_coupling="FB", centrality_function=None):
        super().__init__()
        if centrality_function is None:
            centrality_function = lambda x: x
        N = snapshot.N
        T = snapshot.T
        self._orig_N = N
        self._orig_T = T
        snapshot = tl.to_numpy(snapshot._tensor)

        # Create an empty (N*T, N*T) matrix
        block_diag_matrix = np.zeros((N * T, N * T))

        # Get indices for diagonal blocks
        idx = np.arange(T) * N

        # Assign values using NumPy advanced indexing
        for t in range(T):
            block_diag_matrix[idx[t]:idx[t] + N, idx[t]:idx[t] + N] = epsilon*centrality_function(snapshot[:, :, t])
        if isinstance(time_coupling, str):
            diagonal_matrix = np.zeros((T,T))
            if "F" in time_coupling:
                diagonal_matrix += np.eye(T, k=1)
            if "B" in time_coupling:
                diagonal_matrix += np.eye(T, k=-1)
        else:
            diagonal_matrix = time_coupling
        block_diag_matrix += np.kron(diagonal_matrix, np.eye(N))

        self._supra = block_diag_matrix