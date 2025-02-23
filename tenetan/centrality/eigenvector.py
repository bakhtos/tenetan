import tensorly as tl
import numpy as np


class SupraAdjacencyMatrix:

    def __init__(self, snapshot, time_coupling, epsilon=1.0, centrality_function=None):
        self._cc = None
        self._mlc = None
        self._mnc = None
        self._jc = None

        if centrality_function is None:
            centrality_function = lambda x: x
        N = snapshot.N
        T = snapshot.T
        NT = N*T
        self._orig_N = N
        self._orig_T = T
        self._orig_NT = NT
        self._snapshot = tl.to_numpy(snapshot.tensor)

        # Create an empty (N*T, N*T) matrix
        supra_adjacency = np.zeros((NT, NT))

        # Get indices for diagonal blocks
        idx = np.arange(T) * N

        # Assign values using NumPy advanced indexing
        for t in range(T):
            supra_adjacency[idx[t]:idx[t] + N, idx[t]:idx[t] + N] = epsilon*centrality_function(self._snapshot[:, :, t])
        if isinstance(time_coupling, str):
            diagonal_matrix = np.zeros((T,T))
            if "F" in time_coupling:
                diagonal_matrix += np.eye(T, k=1)
            if "B" in time_coupling:
                diagonal_matrix += np.eye(T, k=-1)
            diagonal_matrix = np.kron(diagonal_matrix, np.eye(N))
        elif time_coupling is None:
            diagonal_matrix = np.zeros((NT, NT))
        else:
            diagonal_matrix = time_coupling
        supra_adjacency += diagonal_matrix

        self._supra = supra_adjacency

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

    def __init__(self, snapshot, epsilon=1.0, centrality_function=None):
        super().__init__(snapshot, time_coupling='FB', centrality_function=centrality_function, epsilon=epsilon)
        del self._snapshot


class YinSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot, centrality_function=None):
        super().__init__(snapshot, time_coupling=None, centrality_function=centrality_function)

        N = self._orig_N
        T = self._orig_T
        NT = self._orig_NT
        # Compute C[j, t] for all j and t
        C_j_t = np.sum(self._snapshot[:, :, :-1] * self._snapshot[:, :, 1:], axis=0)  # Shape (N, T-1)
        # Initialize the big NT x NT matrix
        inter_layer_similarity = np.zeros((NT, NT))

        # Place C_j_t at off-diagonal (t, t+1) and (t+1, t) positions
        for t in range(T - 1):
            start_t = t * N
            start_t1 = (t + 1) * N

            inter_layer_similarity[start_t:start_t + N, start_t1:start_t1 + N] = np.diag(C_j_t[:, t])  # t to t+1
            inter_layer_similarity[start_t1:start_t1 + N, start_t:start_t + N] = np.diag(C_j_t[:, t])  # t+1 to t

        self._supra += inter_layer_similarity
        del self._snapshot
