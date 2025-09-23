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


class BaseTemporalKatzCentrality(ABC):
    @abstractmethod
    def update(self, u, v, t) -> float:
        pass


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


class TruncatedTemporalKatzCentrality(BaseTemporalKatzCentrality):
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

    def update(self, u, v, t) -> float:
        self.edge_data[1][v].append((u, t, 1.0))
        for k in range(2, self.K + 1):
            score = sum(
                w_zu * self.decay.phi(t - t_zu)
                for z, t_zu, w_zu in self.edge_data[k - 1][u]
                if t_zu < t
            )
            if score > 0:
                self.edge_data[k][v].append((u, t, score))
        return self.query(v, t)

    def query(self, u, t) -> float:
        r = 0.0
        for k in range(1, self.K + 1):
            for z, t_zv, w_zv in self.edge_data[k][u]:
                if t_zv < t:
                    r += w_zv * self.decay.phi(t - t_zv)
        return r


class LazyKatzCentrality:
    def __init__(self, beta: float, c: float):
        self.beta = beta
        self.c = c
        self.r = defaultdict(float)
        self.t_last = defaultdict(lambda: -math.inf)  # last updated time for each node

    def decay(self, delta_t: float) -> float:
        """Decay multiplier for elapsed time."""
        return math.exp(-self.c * delta_t)

    def update(self, u, v, t) -> float:
        """
        Update the centrality when edge (u → v) arrives at time t.
        Applies the lazy decay rule.
        Returns updated r[v](t).
        """

        # Apply lazy decay
        self.r[u] = self.r[u] * self.decay(t - self.t_last[u])
        self.r[v] = self.r[v] * self.decay(t - self.t_last[v]) + (1 + self.r[u]) * self.beta

        # Update timestamps
        self.t_last[u] = t
        self.t_last[v] = t

        return self.r[v]

    def query(self, u, t) -> float:
        """Return decayed score r[u](t) on demand."""
        return self.r[u] * self.decay(t - self.t_last[u])