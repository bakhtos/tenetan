from .centrality_function import LiuCentralityFunction
from .layer_similarity import YinLayerSimilarity, LiuLayerSimilarity, HuangLayerSimilarity

import tensorly as tl
import numpy as np
from scipy.linalg import eigh, pinv

__all__ = ["SupraAdjacencyMatrix", "TaylorSupraMatrix", "LiuSupraMatrix", "YinSupraMatrix", "HuangSupraMatrix"]


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

        self._tac = None
        self._fom = None

    def zero_first_order_expansion(self, fom=True):
        """Compute the time-averaged centrality and first-order mover scores.
        """

        C =self._supra
        NT = self._orig_NT
        T = self._orig_T
        N = self._orig_N

        # Timed-averaged centralities (zeroth order expansion)
        # Step 1: Compute X^(1) using Eq. (4.14)
        X1 = np.zeros((N, N))
        sin_vector = []
        gamma1 = 0.0
        for t in range(T):
            sin_factor = np.sin(np.pi * (t + 1) / (T + 1))
            sin_vector.append(sin_factor)
            sin_factor = sin_factor ** 2
            gamma1 += sin_factor
            C_t = C[t * N:(t + 1) * N, t * N:(t + 1) * N]  # Extract N×N block at time t
            X1 += C_t * sin_factor  # Apply the weighting

        X1 /= gamma1  # Normalize

        # Step 2: Solve eigenvector equation X^(1) alpha = λ1 alpha
        eigenvalues, eigenvectors = eigh(X1)  # Compute all eigenvalues & eigenvectors
        lambda1 = eigenvalues[-1]  # Largest eigenvalue (last element)
        alpha = eigenvectors[:, -1]  # Corresponding eigenvector (last column)

        self._tac = alpha

        if not fom:
            return

        # First-order-mover scores (first order expansion)
        # Step 3: Compute X^(2) using Eq. (4.22)
        A = np.zeros((T, T))  # Inter-layer coupling matrix (undirected chain)
        for t in range(T - 1):
            A[t, t + 1] = A[t + 1, t] = 1  # Connect layers sequentially

        lambda0 = np.max(eigh(A, eigvals_only=True))  # Leading eigenvalue of A
        L0 = pinv(lambda0 * np.eye(T) - A)  # Compute (λ0 I - A)†
        L0_pinv = np.kron(L0, np.eye(N))  # Compute L_0 = (λ0 I - A)† ⊗ I

        G = np.zeros((NT, NT))
        for t in range(T):
            block = C[t * N:(t + 1) * N, t * N:(t + 1) * N]  # Extract diagonal block
            G[t * N:(t + 1) * N, t * N:(t + 1) * N] = block  # Place in G

        u = np.array(sin_vector) / np.sqrt(gamma1)
        U_matrix = np.zeros((NT, N))  # Store all u_i vectors
        for i in range(N):
            U_matrix[i * T:(i + 1) * T, i] = u  # Set u in the correct block

        X2 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                # Construct block vectors u_i and u_j
                u_i = U_matrix[:, i]  # Precomputed u_i
                u_j = U_matrix[:, j]  # Precomputed u_j

                X2[i, j] = u_i.T @ G @ L0_pinv @ G @ u_j

        # Step 4: Compute first-order correction beta
        lambda2 = alpha.T @ X2 @ alpha  # Compute λ2
        beta = np.linalg.solve(X1 - lambda1 * np.eye(N), (lambda2 * np.eye(N) - X2) @ alpha)

        # Step 5: Compute first-order mover scores
        v0 = U_matrix @ alpha  # Compute v0
        L0_G_v0 = L0_pinv @ G @ v0  # Compute L0† G v0
        mover_scores = np.sqrt(beta ** 2 + np.sum(L0_G_v0.reshape(N, T) ** 2, axis=1))  # Compute final scores

        self._fom = mover_scores

    @property
    def tac(self):
        return self._tac

    @property
    def time_averaged_centrality(self):
        return self._tac

    @property
    def fom(self):
        return self._fom

    @property
    def first_order_mover_scores(self):
        return self._fom

class YinSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=YinLayerSimilarity)


class LiuSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=LiuLayerSimilarity, centrality_function=LiuCentralityFunction)

class HuangSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=HuangLayerSimilarity)
