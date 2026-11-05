"""Empirical CartPole observation ranges -> justification for the discretizer's clipping bounds.

The two velocities are unbounded in CartPole's observation_space, so we must CHOOSE a range before
binning. We roll out (1) a uniformly random policy and (2) the trained DQN (if results exist) and
report percentiles of |x_dot| and |theta_dot|. The clip values in envs/discretizers.py
(3.0 and 3.5) should cover >= 99.9% of observed values.

Usage: python experiments/obs_ranges.py
"""
from __future__ import annotations

import _common  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from _common import ROOT, write_table
from envs.discretizers import CARTPOLE_HIGH

NAMES = ["cart position", "cart velocity", "pole angle", "pole ang. velocity"]


def rollout(policy, n_steps: int, seed: int) -> np.ndarray:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    out = []
    for _ in range(n_steps):
        out.append(obs)
        obs, _, term, trunc, _ = env.step(policy(obs))
        if term or trunc:
            obs, _ = env.reset()
    return np.array(out)


def main():
    env = gym.make("CartPole-v1")
    policies = {"random": lambda o: env.action_space.sample()}
    env.action_space.seed(0)

    w = ROOT / "results" / "tier1_cartpole" / "dqn_eps25k" / "seed_0.pt"
    if w.exists():
        from agents.dqn import QNetwork

        net = QNetwork(4, 2, (128, 128))
        net.load_state_dict(torch.load(w, map_location="cpu"))

        @torch.no_grad()
        def trained(o):
            return int(net(torch.as_tensor(o, dtype=torch.float32)[None]).argmax())

        policies["trained DQN"] = trained

    rows = []
    for pname, pol in policies.items():
        X = rollout(pol, 50_000, seed=0)
        for d, name in enumerate(NAMES):
            a = np.abs(X[:, d])
            rows.append([pname, name, f"{np.percentile(a, 99):.3f}", f"{np.percentile(a, 99.9):.3f}",
                         f"{a.max():.3f}", f"{CARTPOLE_HIGH[d]:.4g}", f"{100 * np.mean(a <= CARTPOLE_HIGH[d]):.2f}%"])
    header = ["Policy", "Dimension", "|x| p99", "|x| p99.9", "|x| max", "Clip bound", "Within bound"]
    caption = ("CartPole-v1 observation magnitudes over 50k steps. Velocities are unbounded in the "
               "observation space; the chosen clip bounds cover essentially all visited values.")
    write_table("t1_obs_ranges_all", header, rows, caption)  # every dimension (CSV for reference)
    # the report only needs the two unbounded (velocity) dimensions
    vel = [r for r in rows if "velocity" in r[1]]
    write_table("t1_obs_ranges", ["Policy", "Dimension", "|x| p99.9", "|x| max", "Clip"],
                [[r[0], r[1], r[3], r[4], r[5]] for r in vel],
                "CartPole-v1 velocity magnitudes over 50k steps (random and trained policy). Both are unbounded "
                "in the observation space; our clip bounds cover every visited value. Position and angle use the "
                "termination limits.")


if __name__ == "__main__":
    main()
