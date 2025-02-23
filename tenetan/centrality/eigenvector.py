import tensorly as tl
import numpy as np


class SupraAdjacencyMatrix:
    pass


class TaylorSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot, epsilon=1.0, time_coupling = "FB", centrality_function = None):
        if centrality_function is None:
            centrality_function = lambda x: x
        N = snapshot.N
        T = snapshot.T
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

        print(block_diag_matrix)
        print(block_diag_matrix.shape)  # Should be (N*T, N*T)
        self._supra = block_diag_matrix