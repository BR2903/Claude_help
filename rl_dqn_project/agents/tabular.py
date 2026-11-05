"""Tabular Q-learning (Sutton & Barto, 2018, Sec. 6.5) on a discretized state space."""
from __future__ import annotations

import numpy as np

from common.schedules import LinearSchedule
from envs.discretizers import Discretizer


class TabularAgent:
    """Off-policy TD control:  Q(s,a) <- Q(s,a) + alpha [ R + gamma max_a' Q(s',a') - Q(s,a) ].

    * Behaviour policy: epsilon-greedy over Q (ties broken uniformly at random).
    * Target policy:    greedy over Q (that is why the target uses max, not the next action taken).
    * `init_value` sets the initial Q for every (s, a). With rewards <= 0 (MountainCar: -1 per step),
      init_value = 0 is OPTIMISTIC: every unvisited action looks better than any tried one,
      which drives systematic exploration even with little epsilon.
    """

    def __init__(
        self,
        discretizer: Discretizer,
        n_actions: int,
        seed: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100_000,
        init_value: float = 0.0,
    ):
        self.disc = discretizer
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps = LinearSchedule(eps_start, eps_end, eps_decay_steps)
        self.rng = np.random.default_rng(seed)
        self.Q = np.full((discretizer.n_states, n_actions), init_value, dtype=np.float64)
        self.visits = np.zeros(discretizer.n_states, dtype=np.int64)  # for coverage diagnostics
        self._td_abs_sum, self._n_updates = 0.0, 0

    # ---- acting ---------------------------------------------------------- #
    def _argmax_random_tie(self, q_row: np.ndarray) -> int:
        best = np.flatnonzero(q_row == q_row.max())
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))

    def act(self, obs: np.ndarray, t: int) -> int:
        """epsilon-greedy action used while TRAINING."""
        if self.rng.random() < self.eps(t):
            return int(self.rng.integers(self.n_actions))
        return self._argmax_random_tie(self.Q[self.disc(obs)])

    def greedy(self, obs: np.ndarray) -> int:
        """epsilon = 0 action used for EVALUATION (no learning happens here)."""
        return self._argmax_random_tie(self.Q[self.disc(obs)])

    # ---- learning -------------------------------------------------------- #
    def observe(self, obs, action, reward, next_obs, terminated: bool, truncated: bool, t: int) -> None:
        s, s2 = self.disc(obs), self.disc(next_obs)
        self.visits[s] += 1
        # terminated -> true terminal state, its value is 0 -> no bootstrap.
        # truncated (time limit) -> the state is NOT terminal, so we still bootstrap from s'.
        bootstrap = 0.0 if terminated else self.gamma * self.Q[s2].max()
        td_error = reward + bootstrap - self.Q[s, action]
        self.Q[s, action] += self.alpha * td_error
        self._td_abs_sum += abs(td_error)
        self._n_updates += 1

    def pop_stats(self) -> dict:
        """Averages since the last call (logged every `log_every` steps by the training loop)."""
        out = {"td_abs": self._td_abs_sum / max(self._n_updates, 1)}
        self._td_abs_sum, self._n_updates = 0.0, 0
        return out

    def coverage(self) -> float:
        """Fraction of discrete states visited at least once."""
        return float((self.visits > 0).mean())

    def save(self, path: str) -> None:
        np.savez_compressed(path, Q=self.Q, visits=self.visits, bins=np.array(self.disc.bins))
