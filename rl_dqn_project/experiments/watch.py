"""Watch a trained agent play (pygame window). Handy for the live defense.

  python experiments/watch.py --env LunarLander-v3                       # results/tier2_lunarlander/full/seed_0.pt
  python experiments/watch.py --env CartPole-v1                          # results/tier1_cartpole/dqn_eps25k/seed_0.pt
  python experiments/watch.py --env CartPole-v1 --tabular --bins 10      # tabular Q-table instead
  python experiments/watch.py --env LunarLander-v3 --config no_target --seed 2 --episodes 3
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from _common import ROOT
from agents.dqn import QNetwork
from envs.discretizers import cartpole_discretizer

DEFAULTS = {"CartPole-v1": ("tier1_cartpole", "dqn_eps25k", (128, 128)),
            "LunarLander-v3": ("tier2_lunarlander", "full", (256, 256)),
            "MountainCar-v0": ("tier1_mountaincar", "dqn", (128, 128))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1", choices=list(DEFAULTS))
    p.add_argument("--config", default=None, help="config folder name (default depends on env)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--tabular", action="store_true")
    p.add_argument("--bins", type=int, default=10)
    a = p.parse_args()

    exp, config, hidden = DEFAULTS[a.env]
    env = gym.make(a.env, render_mode="human")
    if a.tabular:
        config = a.config or f"tabular_bins{a.bins}"
        Q = np.load(ROOT / "results" / exp / config / f"seed_{a.seed}.npz")["Q"]
        disc = cartpole_discretizer(a.bins)

        def policy(o):
            return int(np.argmax(Q[disc(o)]))
    else:
        config = a.config or config
        dueling = config == "dueling"
        net = QNetwork(env.observation_space.shape[0], env.action_space.n, hidden, dueling)
        net.load_state_dict(torch.load(ROOT / "results" / exp / config / f"seed_{a.seed}.pt", map_location="cpu"))

        @torch.no_grad()
        def policy(o):
            return int(net(torch.as_tensor(o, dtype=torch.float32)[None]).argmax())

    for ep in range(a.episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ret += r
            done = term or trunc
        print(f"episode {ep}: return {ret:.1f}")
    env.close()


if __name__ == "__main__":
    main()
