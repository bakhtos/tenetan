import numpy as np

__all__ = ["LiuLayerSimilarity", "YinLayerSimilarity"]


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
