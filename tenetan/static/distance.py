import numpy as np

from .linalg import NormalizedLaplacian, Laplacian

__all__ = ["DeltaCon", "SpectralDistance", "NormalizedSpectralDistance", "GraphEditDistance"]

def DeltaCon(A1: np.ndarray, A2: np.ndarray, direction="in"):
    """
    Koutra, D., Shah, N., Vogelstein, J. T., Gallagher, B., & Faloutsos, C. (2016)
    Deltacon: Principled massive-graph similarity function with attribution.
    ACM Transactions on Knowledge Discovery from Data (TKDD), 10(3), 1-43.
    """
    D1 = np.diag(A1.sum(axis=0 if direction=="in" else 1))
    D2 = np.diag(A2.sum(axis=0 if direction=="in" else 1))
    N = A1.shape[0]

    eps_1 = 1 / (1 + np.max(D1))
    eps_2 = 1 / (1 + np.max(D2))

    S1 = np.linalg.inv(np.eye(N) + (eps_1 ** 2) * D1 - eps_1 * A1)
    S2 = np.linalg.inv(np.eye(N) + (eps_2 ** 2) * D2 - eps_2 * A2)

    # Matusita Distance
    return np.sqrt(np.sum(np.square(np.sqrt(S1) - np.sqrt(S2))))


def SpectralDistance(A1: np.ndarray, A2: np.ndarray, direction="in",
                     normalized_laplacian=False, n_eig=None):
    if n_eig is None:
        n_eig = A1.shape[0]
    L = NormalizedLaplacian if normalized_laplacian else Laplacian
    eig1 = np.linalg.eigvals(L(A1, direction=direction))
    eig1.sort()
    eig2 = np.linalg.eigvals(L(A2, direction=direction))
    eig2.sort()
    eig1 = np.flip(eig1)
    eig2 = np.flip(eig2)
    eig1 = eig1[:n_eig]
    eig2 = eig2[:n_eig]
    return np.linalg.norm(eig1-eig2)


def NormalizedSpectralDistance(A1: np.ndarray, A2: np.ndarray, direction="in",
                     normalized_laplacian=False, n_eig=None):
    if n_eig is None:
        n_eig = A1.shape[0]
    L = NormalizedLaplacian if normalized_laplacian else Laplacian
    eig1 = np.linalg.eigvals(L(A1, direction=direction))
    eig1.sort()
    eig2 = np.linalg.eigvals(L(A2, direction=direction))
    eig2.sort()
    eig1 = np.flip(eig1)
    eig2 = np.flip(eig2)
    eig1 = eig1[:n_eig]
    eig2 = eig2[:n_eig]
    m1 = np.sum(np.square(eig1))
    m2 = np.sum(np.square(eig2))
    m = max(m1, m2)
    return np.linalg.norm(eig1-eig2)/np.sqrt(m)


def GraphEditDistance(A1: np.ndarray, A2: np.ndarray):

    def nodes(A: np.ndarray):
        # Find rows and columns that are not all zeros
        row_nonzero = np.any(A != 0, axis=1)
        col_nonzero = np.any(A != 0, axis=0)

        # Find indices where either row or column has a non-zero element
        active_indices = np.where(row_nonzero | col_nonzero)[0]
        return len(active_indices)

    def edges(A: np.ndarray):
        return  np.count_nonzero(A)

    A1 = A1.astype(np.bool)
    A2 = A2.astype(np.bool)

    return (nodes(A1) + nodes(A2) - 2*nodes(A1&A2)
            + edges(A1) + edges(A2) - 2*edges(A1&A2))
