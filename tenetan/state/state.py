from tenetan.networks import SnapshotGraph
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree


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
                     normalized=False, n_eig=None):
    if n_eig is None:
        n_eig = A1.shape[0]
    L = NormalizedLaplacian if normalized else Laplacian
    eig1 = np.linalg.eigvals(L(A1, direction=direction))
    eig1.sort()
    eig2 = np.linalg.eigvals(L(A2, direction=direction))
    eig2.sort()
    eig1 = np.flip(eig1)
    eig2 = np.flip(eig2)
    eig1 = eig1[:n_eig]
    eig2 = eig2[:n_eig]
    return np.linalg.norm(eig1-eig2)

def DunnIndex(D: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0

    intra_dists = []
    inter_dists = []

    for i in unique_labels:
        members_i = np.where(labels == i)[0]
        if len(members_i) <= 1:
            intra_dists.append(0)
        else:
            intra = D[np.ix_(members_i, members_i)]
            intra_dists.append(np.max(intra))

        for j in unique_labels:
            if i < j:
                members_j = np.where(labels == j)[0]
                inter = D[np.ix_(members_i, members_j)]
                inter_dists.append(np.min(inter))

    max_intra = max(intra_dists)
    min_inter = min(inter_dists)

    return min_inter / max_intra if max_intra > 0 else 0


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


def MasudaHolme(G: SnapshotGraph, sim: callable = GraphEditDistance):

    N, T = G.N, G.T
    A = G.tensor
    distance_matrix = np.zeros((T,T))
    for t1, t2 in combinations(range(T), 2):
        A1 = A[:, :, t1]
        A2 = A[:, :, t2]
        distance_matrix[t1,t2] = distance_matrix[t2, t1] = sim(A1, A2)
    distance_vector = squareform(distance_matrix)
    try:
        linkage_matrix = linkage(distance_vector)
    except:
        return None, None, None, distance_matrix

    dunn_scores = np.zeros(T)
    labels = cut_tree(linkage_matrix)
    labels = np.flip(labels, axis=1)
    for C in range(1, T+1):
        dunn = DunnIndex(distance_matrix, labels[:, C-1])
        dunn_scores[C-1] = dunn

    best_C = int(np.argmax(dunn_scores))

    return best_C, labels, dunn_scores, distance_matrix


if __name__ == "__main__":

    G = SnapshotGraph()
    G.load_csv("../datasets/eg_taylor.csv",
               source="i", target="j", timestamp="t", weight="w",
               sort_vertices=True, sort_timestamps=True)
    distance_matrix, best_C, labels, dunn_scores = MasudaHolme(G, sim=SpectralDistance)
    print(best_C)
    print(dunn_scores)
    print(labels)
    print(distance_matrix)


