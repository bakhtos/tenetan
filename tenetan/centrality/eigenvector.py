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
        if time_coupling is None:
            diagonal_matrix = np.zeros((NT, NT))
        elif isinstance(time_coupling, str):
            if time_coupling not in ["F", "B", "FB", "BF"]:
                raise ValueError("time_coupling must be one of the following: F, B, BF, FB")
            diagonal_matrix = np.zeros((T,T))
            if "F" in time_coupling:
                diagonal_matrix += np.eye(T, k=1)
            if "B" in time_coupling:
                diagonal_matrix += np.eye(T, k=-1)
            diagonal_matrix = np.kron(diagonal_matrix, np.eye(N))
        elif isinstance(time_coupling, np.ndarray):
            if time_coupling.shape == (T, T):
                diagonal_matrix = np.kron(time_coupling, np.eye(N))
            elif time_coupling.shape == (NT, NT):
                diagonal_matrix = time_coupling
            else:
                raise ValueError(f"Cannot use time_coupling of shape {time_coupling.shape}; must be either (T,T) or (NT, NT)")
        elif callable(time_coupling):
            diagonal_matrix = time_coupling(self._snapshot)
        else:
            raise ValueError("Time coupling must be a numpy.ndarray or string")
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

    def __init__(self, snapshot):
        super().__init__(snapshot, time_coupling=YinLayerSimilarity)


def YinLayerSimilarity(snapshot):
    N = snapshot.shape[0]
    T = snapshot.shape[2]
    NT = N*T

    # Compute C[j, t] for all j and t
    C_j_t = np.sum(snapshot[:, :, :-1] * snapshot[:, :, 1:], axis=0)  # Shape (N, T-1)
    # Initialize the big NT x NT matrix
    inter_layer_similarity = np.zeros((NT, NT))

    # Place C_j_t at off-diagonal (t, t+1) and (t+1, t) positions
    for t in range(T - 1):
        start_t = t * N
        start_t1 = (t + 1) * N

        inter_layer_similarity[start_t:start_t + N, start_t1:start_t1 + N] = np.diag(C_j_t[:, t])  # t to t+1
        inter_layer_similarity[start_t1:start_t1 + N, start_t:start_t + N] = np.diag(C_j_t[:, t])  # t+1 to t

    return inter_layer_similarity


class LiuSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, time_coupling=LiuLayerSimilarity, centrality_function=LiuCentralityMatrix)


def LiuLayerSimilarity(snapshot):
    N = snapshot.shape[0]
    T = snapshot.shape[2]
    NT = N*T
    # Compute numerator: sum over j of A[i,j,t] * A[i,j,t+1]
    numerator = np.sum(snapshot[:, :, :-1] * snapshot[:, :, 1:], axis=1)  # Shape (N, T-1)

    # Compute denominators separately
    denominator_t1 = np.sum(snapshot[:, :, 1:], axis=1)  # Sum over j for A[i,j,t+1] (Shape: N, T-1)
    denominator_t = np.sum(snapshot[:, :, :-1], axis=1)  # Sum over j for A[i,j,t] (Shape: N, T-1)

    # Avoid division by zero
    denominator_t1 = np.where(denominator_t1 == 0, 1e-10, denominator_t1)
    denominator_t = np.where(denominator_t == 0, 1e-10, denominator_t)

    # Compute S matrices
    S_t_t1 = numerator / denominator_t1  # S_{i,t,t+1} uses denominator of t+1
    S_t1_t = numerator / denominator_t  # S_{i,t+1,t} uses denominator of t

    # Initialize the big NT x NT matrix
    inter_layer_similarity = np.zeros((NT, NT))

    # Place S_t_t1 at (t, t+1) and S_t1_t at (t+1, t)
    for t in range(T - 1):
        start_t = t * N
        start_t1 = (t + 1) * N

        inter_layer_similarity[start_t:start_t + N, start_t1:start_t1 + N] = np.diag(S_t_t1[:, t])  # t → t+1
        inter_layer_similarity[start_t1:start_t1 + N, start_t:start_t + N] = np.diag(S_t1_t[:, t])  # t+1 → t
    return inter_layer_similarity

def LiuCentralityMatrix(A):
    # Compute degrees (count non-zero elements in each row)
    d = np.count_nonzero(A, axis=1)  # Shape (N,)

    # Compute W using broadcasting
    W = d[:, None] + d[None, :]  # Outer sum of degrees
    # Set values to zero where A is zero
    W[A == 0] = 0
    return W