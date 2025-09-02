import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
from tenetan.networks import SnapshotGraph
from collections import defaultdict
from typing import List, Set, Dict, Any, Tuple

__all__ = ["StaticLouvain", "TemporalLouvain"]

def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    """Make labels contiguous (0..K-1) in a stable encounter order."""
    uniq = {}
    out = np.empty_like(labels, dtype=int)
    next_id = 0
    for i, c in enumerate(labels):
        if c not in uniq:
            uniq[c] = next_id
            next_id += 1
        out[i] = uniq[c]
    return out

def StaticLouvain(W: np.ndarray, threshold: float = 1e-07, resolution: float = 1.0, seed=None) -> np.ndarray:
    """
    Run Louvain community detection (Blondel et al.) on a weighted directed graph given by adjacency array W.

    Parameters
    ----------
    W: np.ndarray
        Adjacency (weight) matrix
    threshold: float
        Threshold parameter for networkx.algorithms.community.louvain_communities.
        Default 1e-07.
    resolution: float
        Resolution parameter for networkx.algorithms.community.louvain_communities.
        Default 1.0.
    seed: Any | None
        Random seed for networkx.algorithms.community.louvain_communities.
    """

    n = W.shape[0]
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    comms = louvain_communities(G, weight="weight", threshold=threshold, resolution=resolution, seed=seed)
    labels = np.empty(n, dtype=int)
    for cid, nodes in enumerate(comms):
        for v in nodes:
            labels[int(v)] = cid
    # return _renumber_labels(labels)
    return labels

def TemporalLouvain(
    G,
    *,
    louvain_threshold: float = 1e-07,
    louvain_resolution: float = 1.0,
    louvain_seed=None,
    change_tol: float = 0.0
):
    """
    Temporal Louvain algorithm for community detection by analysis of changed edges.
    He, J., & Chen, D. (2015). A fast algorithm for community detection in temporal network.
    Physica A: Statistical Mechanics and its Applications, 429, 87-94.

    Parameters
    ----------
    G : SnapshotGraph
    louvain_threshold: float
        Threshold parameter. Passed to NetworkX Louvain.
        Default 1e-07.
    louvain_resolution : float
        Louvain resolution (default 1.0). Passed to NetworkX Louvain.
        Default 1.0.
    louvain_seed : int or None
        Random seed for Louvain algorithm. Passed to NetworkX Louvain.
    change_tol : float
        Absolute tolerance for detecting a node's connection change between
        t-1 and t (any incident edge |Δ| > change_tol -> node considered "changed").
        Default 0.0, i.e. any change in connection or weight is considered a change edge.
        Increase to permit small variations in edge weights.

    Returns
    -------
    communities : List[Dict[int, List[Any]]]
        For each t, a dict: {label -> [nodes]}.
        Community labels are integers.
    node_labels : List[Dict[Any, int]]
        For each t, a dict: {node -> [label]}.
        Community labels are integers.
    labels: np.ndarray
        Shape (N, T), contains the label of node with index n at time t

    Notes
    -----
    Implements the two-step loop from He & Chen: (1) run Louvain on t=0;
    (2) for t>0, compress unchanged nodes (within their prev community) into
    supernodes to build a small graph, run Louvain there, then expand.
    """
    # --- Inputs & checks
    A = G.tensor
    N, T = G.N, G.T

    def W(t):
        return A[:, :, t]

    labels_t = []
    # First iteration t=0: static Louvain
    labels = StaticLouvain(A[:, :, 0], threshold=louvain_threshold,
                           resolution=louvain_resolution, seed=louvain_seed)
    labels_t.append(labels)

    # Temporal Louvain
    for t in range(1,T):
        Wt = W(t)
        prev_labels = labels_t[t-1]

        # Detect node changes between t-1 and t, up to a certain tolerance
        diff = np.abs(Wt - W(t - 1))
        changed = (diff > change_tol).any(axis=1)

        # Supernode construction per He & Chen:
        #   within each previous community, one supernode for all UNCHANGED nodes,
        #   and one supernode per CHANGED node.
        groups = []
        next_gid = 0
        for c in np.unique(prev_labels):
            idx = np.where(prev_labels == c)[0]
            # unchanged subset within this community
            u = idx[~changed[idx]]
            if u.size > 0:
                groups.append(u)
                next_gid += 1
            # each changed node becomes its own group
            for i in idx[changed[idx]]:
                groups.append(np.array([i], dtype=int))
                next_gid += 1

        K = len(groups)

        # Dense membership matrix (OK for moderate N; use sparse if needed)
        M = np.zeros((N, K), dtype=float)
        for k, nodes in enumerate(groups):
            M[nodes, k] = 1.0

        # Compressed adjacency
        B = M.T @ Wt @ M

        # Louvain on compressed graph
        group_labels = StaticLouvain(B, threshold=louvain_threshold,
                                     resolution=louvain_resolution, seed=louvain_seed)

        # Assign labels to original nodes
        labels = np.empty(N, dtype=int)
        for k, nodes in enumerate(groups):
            labels[nodes] = group_labels[k]
        # labels = _renumber_labels(labels)
        labels_t.append(labels)

    # Construct mappings from node to label and from label to list of nodes
    node_labels = []
    communities = []
    for labels in labels_t:
        node_labels_t = {}
        communities_t = defaultdict(list)
        for i, node in enumerate(G.vertices):
            label = int(labels[i])
            node_labels_t[node] = label
            communities_t[label].append(node)
        node_labels.append(node_labels_t)
        communities.append(communities_t)

    # (N,T) matrix of node labels
    labels = np.stack(labels_t, axis=1)

    return communities, node_labels, labels


def StepwiseLouvain(
    G,
    *,
    louvain_threshold: float = 1e-07,
    louvain_resolution: float = 1.0,
    louvain_seed=None,
    match_threshold: float = 0.3,
):
    """
    Stepwise (division + agglomeration) temporal community detection (He et al., 2017).
    Always treats snapshots as directed (if your data are undirected, the slices are symmetric).

    Parameters
    ----------
    G : SnapshotGraph
        Must provide:
          - tensor: np.ndarray of shape (N, N, T)
          - N: int
          - T: int
          - vertices: iterable of node ids (length N)
    louvain_threshold : float
        Passed to StaticLouvain (NetworkX Louvain threshold).
    louvain_resolution : float
        Passed to StaticLouvain (resolution).
    louvain_seed : Any or None
        Passed to StaticLouvain (seed).
    match_threshold : float
        Jaccard threshold θ to match step communities across t-1 → t (Eq. 1 in He et al., 2017).

    Returns
    -------
    communities : List[Dict[int, List[Any]]]
        For each t, {label -> [original node ids]}.
    node_labels : List[Dict[Any, int]]
        For each t, {node id -> label}.
    labels : np.ndarray
        Shape (N, T): label of node with index n at time t.
    dynamic_communities : List[np.ndarray]
        Each item is an (N, T) uint8 matrix; entry [n, t] == 1 if node n is in that
        dynamic community at time t, else 0.

    Notes
    -----
    Implements He et al. (2017) two-stage loop:
      • t=0: Louvain on full snapshot.
      • t>0: Division via ΔQ rule, then partition remaining nodes into modules (Louvain),
              Agglomeration on module graph (Louvain), Expand to node communities.
    Dynamic communities are formed by Jaccard matching of step communities across
    consecutive snapshots (Eq. 1; matched if ≥ θ).
    """
    N, T = G.N, G.T

    def A(t): return G.tensor[:, :, t]

    labels_t: List[np.ndarray] = []

    # ---- t=0: Louvain on full snapshot ----
    labels0 = StaticLouvain(A(0), threshold=louvain_threshold,
                            resolution=louvain_resolution, seed=louvain_seed)
    labels_t.append(labels0)

    # ---- t>0: stepwise detection (division → agglomeration) ----
    for t in range(1, T):
        At = A(t)
        prev_labels = labels_t[t - 1]

        # Precompute strengths and wG
        s = At.sum(axis=1) + At.sum(axis=0)  # in + out (directed)
        wG = At.sum()

        # Build ΔQ for all pairs (handle empty graph)
        if wG > 0:
            DQ = (At / wG) - (np.outer(s, s) / (2.0 * (wG ** 2)))
        else:
            DQ = np.full_like(At, -np.inf, dtype=float)

        # Neighbor mask: q is a neighbor of p if there's an edge p->q or q->p
        nbr_mask = (At > 0) | (At.T > 0)
        # never allow self as "neighbor"
        np.fill_diagonal(nbr_mask, False)

        # Apply the mask; non-neighbors get -inf so they won't win argmax
        DQ_masked = np.where(nbr_mask, DQ, -np.inf)

        # Best neighbor index for each p (over all q that are neighbors)
        best_q_for = DQ_masked.argmax(axis=1)  # shape (N,)

        # ---- Division stage ----
        modules: List[Set[int]] = []
        module_origin: List[int] = []  # which prev community a module came from
        removed_nodes: Set[int] = set()

        for i_prev, cid in enumerate(np.unique(prev_labels)):
            C_prev = np.where(prev_labels == cid)[0]
            C_prev_set = set(C_prev)

            # remove p if its best-ΔQ neighbor lies outside C_prev
            to_remove: Set[int] = set()
            for p in C_prev:
                best_q = int(best_q_for[p])
                # If node p has no neighbors, DQ_masked[p,:] will be all -inf; handle that:
                if not np.isfinite(DQ_masked[p, best_q]):
                    continue
                if best_q not in C_prev_set:
                    to_remove.add(p)

            remaining = C_prev_set - to_remove
            removed_nodes |= to_remove

            if remaining:
                idx = np.array(sorted(remaining), dtype=int)
                subW = At[np.ix_(idx, idx)]
                sub_labels = StaticLouvain(
                    subW,
                    threshold=louvain_threshold,
                    resolution=louvain_resolution,
                    seed=louvain_seed,
                )
                for cid in np.unique(sub_labels):
                    nodes = idx[np.where(sub_labels == cid)[0]]
                    modules.append(set(map(int, nodes.tolist())))
                    module_origin.append(i_prev)

        # removed nodes become singleton modules
        for v in removed_nodes:
            modules.append({int(v)})
            # provenance (prev community index)
            for i_prev, cid in enumerate(np.unique(prev_labels)):
                if prev_labels[v] == cid:
                    module_origin.append(i_prev)
                    break

        # No modules → fall back to plain Louvain on Wt
        if not modules:
            labels = StaticLouvain(At, threshold=louvain_threshold,
                                   resolution=louvain_resolution, seed=louvain_seed)
            labels_t.append(labels)
            continue

        # ---- Agglomeration stage ----
        K = len(modules)
        H = np.zeros((K, K), dtype=float)
        for i in range(K):
            I = np.fromiter(modules[i], dtype=int)
            for j in range(K):
                if i == j:
                    continue
                J = np.fromiter(modules[j], dtype=int)
                H[i, j] = At[np.ix_(I, J)].sum()

        mod_labels = StaticLouvain(
            H, threshold=louvain_threshold, resolution=louvain_resolution, seed=louvain_seed
        )

        labels = np.empty(N, dtype=int)
        for k, mod in enumerate(modules):
            labels[list(mod)] = int(mod_labels[k])
        labels_t.append(labels)

    # -------------------------------------------------------------------------
    # Build per-snapshot outputs identical to TemporalLouvain
    # -------------------------------------------------------------------------
    communities: List[Dict[int, List[Any]]] = []
    node_labels: List[Dict[Any, int]] = []
    for labels in labels_t:
        node_to_label: Dict[Any, int] = {}
        label_to_nodes: Dict[int, List[Any]] = defaultdict(list)
        for i, node in enumerate(G.vertices):
            lab = int(labels[i])
            node_to_label[node] = lab
            label_to_nodes[lab].append(node)
        node_labels.append(node_to_label)
        communities.append(label_to_nodes)

    labels_matrix = np.stack(labels_t, axis=1)  # (N, T)

    # -------------------------------------------------------------------------
    # Build dynamic communities: list of N×T binary matrices via Jaccard
    # (Eq. 1). Community at t matches a previous track if Jaccard ≥ θ.
    # Splits/merges are handled by cloning tracks for additional matches.
    # -------------------------------------------------------------------------
    # Step communities as sets per t
    step_sets: List[List[Set[int]]] = []
    for t in range(T):
        labs = labels_t[t]
        step_sets.append([set(np.where(labs == cid)[0]) for cid in np.unique(labs)])

    def jaccard(S: Set[int], T_: Set[int]) -> float:
        if not S and not T_:
            return 1.0
        inter = len(S & T_)
        if inter == 0:
            return 0.0
        union = len(S) + len(T_) - inter
        return inter / union

    # Each track holds: matrix (N×T uint8) and last set at t-1
    tracks: List[Tuple[np.ndarray, Set[int]]] = []

    # Initialize tracks from t=0
    for C in step_sets[0]:
        M = np.zeros((N, T), dtype=np.uint8)
        if len(C) > 0:
            M[list(C), 0] = 1
        tracks.append((M, C))

    # Progressively link t=1..T-1
    for t in range(1, T):
        curr = step_sets[t]
        # For matching, snapshot of previous tracks' last sets
        prev_sets = [S_prev for (_, S_prev) in tracks]

        # Keep which prev tracks got used (so unmatched ones simply have 0s at column t)
        used_prev = set()
        new_tracks: List[Tuple[np.ndarray, Set[int]]] = []

        for C in curr:
            # Find all prev tracks with Jaccard ≥ θ
            candidates = []
            for i_prev, S_prev in enumerate(prev_sets):
                jac = jaccard(S_prev, C)
                if jac >= match_threshold:
                    candidates.append((i_prev, jac))
            # No match → start a new track at t
            if not candidates:
                M = np.zeros((N, T), dtype=np.uint8)
                if len(C) > 0:
                    M[list(C), t] = 1
                new_tracks.append((M, C))
                continue

            # Sort by similarity (desc)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 1) Extend the best-matching existing track in-place
            best_idx = candidates[0][0]
            used_prev.add(best_idx)
            tracks[best_idx][0][list(C), t] = 1
            # update last set to C
            tracks[best_idx] = (tracks[best_idx][0], C)

            # 2) For additional matches (merges), CLONE that previous track and extend
            for i_prev, _ in candidates[1:]:
                used_prev.add(i_prev)
                M_clone = tracks[i_prev][0].copy()
                if len(C) > 0:
                    M_clone[list(C), t] = 1
                new_tracks.append((M_clone, C))

        # Unmatched previous tracks: just carry zeros at column t (implicitly already zeros)

        # Append any new tracks created this step
        tracks.extend(new_tracks)

    dynamic_communities = [M for (M, _) in tracks]

    return communities, node_labels, labels_matrix, dynamic_communities

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
    G.vertices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    comms, nl, l = TemporalLouvain(G)
    print(comms)
    print(nl)
    print(l)