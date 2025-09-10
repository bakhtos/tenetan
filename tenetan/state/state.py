from tenetan.networks import SnapshotGraph
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree

from tenetan.static.distance import *


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
        return None, None, None, distance_matrix, None

    dunn_scores = np.zeros(T)
    labels = cut_tree(linkage_matrix)
    labels = np.flip(labels, axis=1)
    for C in range(1, T+1):
        dunn = DunnIndex(distance_matrix, labels[:, C-1])
        dunn_scores[C-1] = dunn

    best_C = int(np.argmax(dunn_scores))

    return best_C, labels, dunn_scores, distance_matrix, linkage_matrix


if __name__ == "__main__":

    G = SnapshotGraph()
    G.load_csv("../datasets/eg_taylor.csv",
               source="i", target="j", timestamp="t", weight="w",
               sort_vertices=True, sort_timestamps=True)
    best_C, labels, dunn_scores, distance_matrix, linkage_matrix = MasudaHolme(G, sim=SpectralDistance)
    print(best_C)
    print(dunn_scores)
    print(labels)
    print(distance_matrix)
    print(linkage_matrix)


