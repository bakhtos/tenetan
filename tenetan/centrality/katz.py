from collections import defaultdict
import math
from abc import ABC, abstractmethod


class DecayFunction(ABC):
    @abstractmethod
    def phi(self, delta_t: float) -> float:
        """Weight for an edge given time since last interaction (ϕ)."""
        pass

    @abstractmethod
    def decay_multiplier(self, delta_t: float) -> float:
        """Decay multiplier for aging centrality scores."""
        pass


class ExponentialDecay(DecayFunction):
    def __init__(self, beta: float, c: float):
        self.beta = beta
        self.c = c

    def phi(self, delta_t: float) -> float:
        return self.beta * math.exp(-self.c * delta_t)

    def decay_multiplier(self, delta_t: float) -> float:
        return math.exp(-self.c * delta_t)


class ConstantDecay(DecayFunction):
    def __init__(self, beta: float):
        self.beta = beta

    def phi(self, delta_t: float) -> float:
        return self.beta

    def decay_multiplier(self, delta_t: float) -> float:
        return 1.0


class TemporalKatzCentrality:
    def __init__(self, decay_function: DecayFunction):
        """
        Parameters:
            decay_function: an object with methods
                - phi(delta_t): edge weight based on time elapsed
                - decay_multiplier(delta_t): multiplier for score aging
        """
        if not isinstance(decay_function, DecayFunction):
            raise TypeError("decay_function must implement DecayFunction interface.")
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