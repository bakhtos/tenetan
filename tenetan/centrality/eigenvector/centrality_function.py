import numpy as np

__all__ = ["LiuCentralityFunction"]

def LiuCentralityFunction(A):
    # Compute degrees (count non-zero elements in each row)
    d = np.count_nonzero(A, axis=1)  # Shape (N,)

    # Compute W using broadcasting
    W = d[:, None] + d[None, :]  # Outer sum of degrees
    # Set values to zero where A is zero
    W[A == 0] = 0
    return W
