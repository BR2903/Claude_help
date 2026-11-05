"""State-space binning for tabular RL on continuous-observation environments.

A discretizer maps a continuous observation vector to ONE integer state index,
so the Q-table can be a plain 2-D array Q[state, action].

How it works (per dimension d):
  * We keep the *interior* bin edges e_1 < e_2 < ... < e_{k-1} (k = number of bins).
  * np.searchsorted(edges, x) returns how many edges lie at or below x  ->  a bin id in [0, k-1].
  * Values outside [low, high] automatically fall into the first / last bin,
    so "clipping" is implicit: anything faster than the clip value is treated
    as "very fast" (the outermost bin).
  * The per-dimension bin ids are combined with np.ravel_multi_index
    (like reading digits of a mixed-radix number) into a single integer.

Live-modification hook (oral defense): pass `edges=` to give any dimension
hand-made, non-uniform boundaries, e.g. finer bins near the vertical for the
pole angle. See `cartpole_discretizer(angle_edges=...)`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# CartPole-v1 observation = [cart position, cart velocity, pole angle, pole angular velocity]
#
# Range justification (see report Sec. 3.1 and experiments/obs_ranges.py):
#   * position: the episode TERMINATES at |x| > 2.4, so [-2.4, 2.4] is the full useful range.
#   * angle:    the episode TERMINATES at |theta| > 12 deg = 0.2095 rad.
#   * velocities are unbounded (+-inf in observation_space). Empirically (random and trained
#     policies) |x_dot| stays below ~2 and |theta_dot| below ~3 in >99.9% of steps,
#     so we clip at 3.0 and 3.5. Using +-inf would make uniform bins impossible, and a much
#     wider range would waste almost every bin on values that never occur.
# --------------------------------------------------------------------------- #
CARTPOLE_LOW = np.array([-2.4, -3.0, -0.2095, -3.5])
CARTPOLE_HIGH = np.array([2.4, 3.0, 0.2095, 3.5])

# MountainCar-v0 observation = [position, velocity]; the space is already bounded.
MOUNTAINCAR_LOW = np.array([-1.2, -0.07])
MOUNTAINCAR_HIGH = np.array([0.6, 0.07])


class Discretizer:
    """Maps a continuous observation to a single integer state index."""

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        bins: int | Sequence[int],
        edges: dict[int, Sequence[float]] | None = None,
    ):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        n_dims = low.shape[0]
        if np.isscalar(bins):
            bins = [int(bins)] * n_dims
        bins = [int(b) for b in bins]
        assert len(bins) == n_dims, "one bin count per observation dimension"

        self.low, self.high = low, high
        self.edges: list[np.ndarray] = []
        for d in range(n_dims):
            if edges is not None and d in edges:
                # user-supplied interior edges (must be sorted) -> len(e) + 1 bins
                e = np.sort(np.asarray(edges[d], dtype=np.float64))
                bins[d] = len(e) + 1
            else:
                # k uniform bins on [low, high] -> k - 1 interior edges
                e = np.linspace(low[d], high[d], bins[d] + 1)[1:-1]
            self.edges.append(e)
        self.bins = tuple(bins)
        self.n_states = int(np.prod(self.bins))

    def bin_ids(self, obs: np.ndarray) -> tuple[int, ...]:
        """Per-dimension bin index of an observation (useful for debugging/plots)."""
        return tuple(int(np.searchsorted(e, x, side="right")) for e, x in zip(self.edges, obs))

    def __call__(self, obs: np.ndarray) -> int:
        """Single integer state index in [0, n_states)."""
        return int(np.ravel_multi_index(self.bin_ids(obs), self.bins))

    def describe(self) -> str:
        return f"Discretizer(bins={self.bins}, n_states={self.n_states})"


def cartpole_discretizer(bins: int | Sequence[int] = 10, angle_edges: Sequence[float] | None = None,
                         low=None, high=None) -> Discretizer:
    """CartPole discretizer. `angle_edges` overrides the pole-angle (dim 2) boundaries."""
    edges = {2: angle_edges} if angle_edges is not None else None
    return Discretizer(CARTPOLE_LOW if low is None else low,
                       CARTPOLE_HIGH if high is None else high, bins, edges)


def mountaincar_discretizer(bins: int | Sequence[int] = 20) -> Discretizer:
    return Discretizer(MOUNTAINCAR_LOW, MOUNTAINCAR_HIGH, bins)


def make_discretizer(spec: dict) -> Discretizer:
    """Build a discretizer from a JSON-serialisable spec (used by the experiment runner)."""
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "cartpole":
        return cartpole_discretizer(**spec)
    if kind == "mountaincar":
        return mountaincar_discretizer(**spec)
    if kind == "generic":
        return Discretizer(**spec)
    raise ValueError(f"unknown discretizer kind {kind!r}")
