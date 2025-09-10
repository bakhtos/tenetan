import numpy as np

__all__ = ["Laplacian", "NormalizedLaplacian"]

def Laplacian(A: np.ndarray, direction="in") -> np.ndarray:
    D = A.sum(axis=0 if direction=="in" else 1)
    return  np.diag(D) - A


def NormalizedLaplacian(A: np.ndarray, direction="in") -> np.ndarray:
    # Degree vector
    D = A.sum(axis=0 if direction=="in" else 1)
    N = A.shape[0]

    # Inverse sqrt of degrees, avoid division by zero
    with np.errstate(divide='ignore'):
        inv_sqrt_deg = 1.0 / np.sqrt(D)
    inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0  # replace inf with 0 for isolated nodes

    # Diagonal matrix of inverse sqrt degrees
    D_inv_sqrt = np.diag(inv_sqrt_deg)

    # Normalized Laplacian
    return np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
