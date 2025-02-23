from .centrality_function import LiuCentralityFunction
from .layer_similarity import YinLayerSimilarity, LiuLayerSimilarity

import tensorly as tl
import numpy as np


class SupraAdjacencyMatrix:

    def __init__(self, snapshot, inter_layer_similarity, centrality_function=None, epsilon=1.0):
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
        snapshot = tl.to_numpy(snapshot.tensor)

        # Set centrality (adjacency) matrices to the SCM
        supra_centrality = np.zeros((NT, NT))
        # Get indices for diagonal blocks
        idx = np.arange(T) * N
        # Assign values using NumPy advanced indexing
        for t in range(T):
            supra_centrality[idx[t]:idx[t] + N, idx[t]:idx[t] + N] = epsilon*centrality_function(snapshot[:, :, t])

        # Set the inter-layer similarity to the SCM
        if inter_layer_similarity is None:
            inter_layer_similarity = np.zeros((NT, NT))
        elif isinstance(inter_layer_similarity, str):
            if inter_layer_similarity not in ["F", "B", "FB", "BF", "C"]:
                raise ValueError("time_coupling must be one of the following: F, B, BF, FB, C")
            if inter_layer_similarity == 'C':
                # Create the block matrix using Kronecker product with identity matrix
                inter_layer_similarity = np.triu(np.ones((T, T)), k=1)
            else:
                ils = np.zeros((T,T))
                if "F" in inter_layer_similarity:
                    ils += np.eye(T, k=1)
                if "B" in inter_layer_similarity:
                    ils += np.eye(T, k=-1)
                inter_layer_similarity = ils
        elif callable(inter_layer_similarity):
            inter_layer_similarity = inter_layer_similarity(snapshot)

        if isinstance(inter_layer_similarity, np.ndarray):
            if inter_layer_similarity.shape == (T, T):
                ils = np.kron(inter_layer_similarity, np.eye(N))
            elif inter_layer_similarity.shape == (NT, NT):
                ils = inter_layer_similarity
            else:
                raise ValueError(f"Cannot use time_coupling of shape {inter_layer_similarity.shape}; must be either (T,T) or (NT, NT)")
        else:
            raise ValueError("Time coupling must be a numpy.ndarray, callable or string (or None)")
        supra_centrality += ils

        self._supra = supra_centrality

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
    def supracentrality_matrix(self):
        return self._supra


    @property
    def scm(self):
        return self._supra


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
        super().__init__(snapshot, inter_layer_similarity='FB', centrality_function=centrality_function,
                         epsilon=epsilon)


class YinSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=YinLayerSimilarity)


class LiuSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=LiuLayerSimilarity, centrality_function=LiuCentralityFunction)


