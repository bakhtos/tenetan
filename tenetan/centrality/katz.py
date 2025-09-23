from collections import defaultdict
import math
from abc import ABC, abstractmethod


class DecayFunction(ABC):
    @abstractmethod
    def phi(self, delta_t: float) -> float:
        """Weight for an edge given time since last interaction (ϕ)."""
        pass


class ConstantDecay(DecayFunction):
    def __init__(self, beta: float):
        self.beta = beta

    def phi(self, delta_t: float) -> float:
        return self.beta


class FastDecayFunction(DecayFunction, ABC):
    @abstractmethod
    def decay_multiplier(self, delta_t: float) -> float:
        """Decay multiplier for aging centrality scores."""
        pass


class ExponentialDecay(FastDecayFunction):
    def __init__(self, beta: float, c: float):
        self.beta = beta
        self.c = c

    def phi(self, delta_t: float) -> float:
        return self.beta * math.exp(-self.c * delta_t)

    def decay_multiplier(self, delta_t: float) -> float:
        return math.exp(-self.c * delta_t)


class BaseTemporalKatzCentrality(ABC):
    @abstractmethod
    def update(self, u, v, t) -> float:
        pass


class FastTemporalKatzCentrality(BaseTemporalKatzCentrality):
    def __init__(self, decay_function: FastDecayFunction):
        """
        Parameters:
            decay_function: an object with methods
                - phi(delta_t): edge weight based on time elapsed
                - decay_multiplier(delta_t): multiplier for score aging
        """
        if not isinstance(decay_function, FastDecayFunction):
            raise TypeError("decay_function must implement FastDecayFunction interface.")
        self.decay = decay_function
        self.r = defaultdict(float)       # centrality scores
        self.t_last = defaultdict(float)  # last update time per node
        self.edge_data = defaultdict(list)  # v → list of (u, t, w_uv)

    def _decay_score(self, u, t_now):
        delta = t_now - self.t_last[u]
        if delta > 0:
            self.r[u] *= self.decay.decay_multiplier(delta)
            self.t_last[u] = t_now

    def update(self, u, v, t):
        """Ingest edge (u → v) at time t and return current scores of all known nodes, including zeros."""

        # Decay u
        self._decay_score(u, t)
        r_u = self.r[u]
        w_uv = 1 + r_u

        self.edge_data[v].append((u, t, w_uv))

        # Decay v and update its centrality
        self._decay_score(v, t)
        self.r[v] += w_uv * self.decay.phi(0)

        # Return current scores for all known nodes, including zero scores
        current_scores = {}
        for node in self.t_last:
            self._decay_score(node, t)
            current_scores[node] = self.r[node]  # could be zero

        return current_scores


class TemporalKatzCentrality(BaseTemporalKatzCentrality):
    def __init__(self, decay_function: DecayFunction):
        if not isinstance(decay_function, DecayFunction):
            raise TypeError("decay_function must implement DecayFunction interface.")
        self.decay = decay_function
        self.edge_data = defaultdict(list)  # v → list of (z, t_zv, w_zv)

    def update(self, u, v, t) -> float:
        """Update the graph with edge (u → v) at time t using general update rule (Eq. 11)."""

        # Compute r_u(t)
        r_u = sum(w_zu * self.decay.phi(t - t_zu) for z, t_zu, w_zu in self.edge_data[u])

        # Add new edge (u → v) with weight = 1 + r_u
        self.edge_data[v].append((u, t, (1 + r_u)))

        return self.query(v, t)

    def query(self, u, t) -> float:
        """Compute the centrality of node u at time t using Eq. (11)."""
        return sum(
            w_zv * self.decay.phi(t - t_zv)
            for z, t_zv, w_zv in self.edge_data[u]
            if t_zv < t
        )


class TruncatedTemporalKatzCentrality:
    def __init__(self, decay_function: DecayFunction, K):
        """
        decay_function: must have .phi(delta_t)
        K: maximum walk length to consider
        """
        if not isinstance(decay_function, DecayFunction):
            raise TypeError("decay_function must implement DecayFunction interface.")
        self.decay = decay_function
        self.K = K
        # edge_data[k][v] = list of (z, t_zv, w_zv) for paths of length k ending at v
        self.edge_data = [defaultdict(list) for _ in range(K + 1)]  # index 0 unused

    def update(self, u, v, t):
        """
        Add edge u → v at time t.
        Updates all truncated Katz scores up to length K.
        Returns centrality scores r_v^{[k]}(t) for all nodes.
        """
        # Walks of length 1: base case
        self.edge_data[1][v].append((u, t, 1.0))

        # Walks of length k: grow from walks of length (k-1) ending at u
        for k in range(2, self.K + 1):
            score = 0.0
            for z, t_zu, w_zu in self.edge_data[k - 1][u]:
                score += w_zu * self.decay.phi(t - t_zu)
            if score > 0:
                self.edge_data[k][v].append((u, t, score))

        # Recompute total centrality r_v^{[K]}(t) for all nodes
        scores = {}
        for node in set.union(*[set(e.keys()) for e in self.edge_data[1:]]):
            r = 0.0
            for k in range(1, self.K + 1):
                for z, t_zv, w_zv in self.edge_data[k][node]:
                    r += w_zv * self.decay.phi(t - t_zv)
            scores[node] = r

        return scores