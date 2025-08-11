import numpy as np
from tenetan.networks import SnapshotGraph
from typing import Dict, List, Optional, Any
from collections import defaultdict

def DOMPLA(
    G: SnapshotGraph,
    T_max: int = 20,
    r: float = 0.10,
    q: float = 1e-3,
    inflation: float = 1.2,
    tol: float = 1e-6,
):
    """
    Dynamic Overlapping Multi-Label Propagation on a temporal network.

    Parameters
    ----------
    G :
        Adjacency tensor of shape (N, N, T). A[i, j, t] is the weight of edge i->j at time t.
        Can be directed or undirected; weights should be nonnegative.
    T_max : int
        Max MLPA iterations per snapshot.
    r : float
        Post-processing threshold for label retention. A node keeps all labels whose
        probability >= r * max_prob(node). Overlap occurs when multiple survive.
    q : float
        Conditional update threshold. A node updates only if L1 change >= q.
    inflation : float
        Inflation exponent (>1) to boost dominant labels (element-wise power).
    tol : float
        Convergence tolerance on global change of label probabilities.

    Returns
    -------
    communities : List[Dict[int, List[Any]]]
        For each t, a dict: {label -> [nodes]}.
        Labels are integers.
    node_labels : List[Dict[Any, List[int]]]
        For each t, a dict: {node -> [labels]}.
        Labels are integers.
    P : np.ndarray
        An (N, N, T) array of label probabilities: P[node, label, t].
    """

    if inflation <= 1.0:
        raise ValueError("Inflation must be above 1.0")

    # One MLPA round with optional warm start
    def MLPA(W: np.ndarray, P0: np.ndarray) -> np.ndarray:
        # P[node, label]; initialize with delta labels (or warm start)
        P = P0.copy()
        # ensure valid probabilities
        P[P < 0] = 0.0
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        P = P / row_sums

        order = np.argsort(-W.sum(axis=1))  # degree(desc) order
        for it in range(T_max):
            total_change = 0.0
            # one "visit" over nodes in degree order
            for x in order:
                # neighbors speak: aggregate neighbor label dists via W
                # (row-normalized W => convex combination)
                new_dist =  W[x] @ P if inflation is None else np.power(W[x] @ P, inflation)
                s = new_dist.sum()
                if s > 0:
                    new_dist /= s
                else:
                    # isolated node: keep its current memory (or its own label)
                    new_dist = P[x].copy()

                # conditional update (avoid tiny oscillations)
                delta = np.abs(new_dist - P[x]).sum()
                if delta >= q:
                    P[x] = new_dist
                    total_change += delta

            if total_change <= tol:
                break
        return P

    # Helper: convert label probabilities to overlapping communities
    def select_labels(P: np.ndarray):
        communities: Dict[int, List[Any]] = defaultdict(list)
        node_labels: Dict[Any, List[int]] = defaultdict(list)
        # keep labels >= r * max_prob for each node
        maxp = P.max(axis=1, keepdims=True)
        # avoid zeros to prevent all-drop
        maxp[maxp == 0.0] = 1.0
        keep = P >= (r * maxp)
        # Assign node to all kept labels
        for node_id, node in enumerate(G.vertices):
            labels = np.where(keep[node_id])[0]
            # safety: if nothing passes r (shouldn't happen), keep argmax
            if labels.size == 0:
                labels = np.array([int(np.argmax(P[node_id]))])
            for lbl in labels:
                communities[int(lbl)].append(node)
                node_labels[node].append(int(lbl))
        # remove empty communities
        communities = {lbl: nodes for lbl, nodes in communities.items() if len(nodes) > 0}

        return communities, node_labels

    A = G.tensor.copy()
    N, T = G.N, G.T

    communities: List[Dict[int, List[Any]]] = []
    node_labels: List[Dict[Any, List[int]]] = []

    # Normalize adjacency row-wise to obtain neighbor influence weights
    # add self-loops to every snapshot: broadcast I across the third axis
    A += np.eye(N)[:, :, None]
    # row sums per snapshot (keepdims to broadcast back on division)
    row_sums = A.sum(axis=1, keepdims=True)
    # avoid division by zero for isolated rows in any snapshot
    row_sums[row_sums == 0.0] = 1.0
    # row-normalize all snapshots at once
    A /= row_sums

    # Iterate over time
    P = np.zeros_like(A)
    for t in range(0, T):
        Pt = MLPA(A[:,:,t], P0=P[:,:,t-1] if t>0 else np.eye(N, dtype=float))
        communities_t, node_labels_t = select_labels(Pt)
        communities.append(communities_t)
        node_labels.append(node_labels_t)
        P[:,:,t] = Pt

    return communities, node_labels, P


# ---------- Example usage ----------
if __name__ == "__main__":
    N, T = 10, 3
    A = np.zeros((N, N, T), dtype=float)


    def add_undirected(A, u, v, w, t):
        A[u, v, t] += w
        A[v, u, t] += w


    # t=0: Two clear clusters, weak bridge 4-5
    for t in [0]:
        for u in [0, 1, 2, 3, 4]:
            for v in [0, 1, 2, 3, 4]:
                if u < v:
                    add_undirected(A, u, v, 2.0, t)
        for u in [5, 6, 7, 8, 9]:
            for v in [5, 6, 7, 8, 9]:
                if u < v:
                    add_undirected(A, u, v, 2.0, t)
        add_undirected(A, 4, 5, 0.2, t)

    # t=1: Strengthen 4's ties to C2
    A[:, :, 1] = A[:, :, 0]
    for v in [6, 7]:
        add_undirected(A, 4, v, 1.2, 1)
    for v in [0, 1]:
        A[4, v, 1] *= 0.5;
        A[v, 4, 1] *= 0.5

    # t=2: Make 5 overlap into C1, strengthen bridge
    A[:, :, 2] = A[:, :, 1]
    for v in [1, 2]:
        add_undirected(A, 5, v, 1.5, 2)
    add_undirected(A, 4, 5, 1.5, 2)

    # ------------------- Run DOMLPA -------------------
    G = SnapshotGraph(A)
    G.vertices = [str(i) for i in G.vertices]
    comms, nodes_t, P_seq = DOMPLA(
        G,
        T_max=35,
        r=0.4,
        q=1e-3,
        inflation=1.8,
        tol=1e-6
    )

    # Print results
    for t, C in enumerate(comms):
        print(f"\n=== t={t} communities (label -> nodes) ===")
        for lbl, nodes in sorted(C.items()):
            print(f"{lbl}: {sorted(nodes)}")
    print(nodes_t)