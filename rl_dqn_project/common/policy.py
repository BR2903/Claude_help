"""Rebuild the greedy (epsilon = 0) policy of a finished run from its saved weights.

Everything needed is in the run's JSON spec (env, agent type, discretizer, network shape), so the
policy is exactly the one that was trained, whatever the config. Used by eval100.py and make_gifs.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


def load_policy(json_path: str | Path, n_obs: int, n_actions: int) -> Callable[[np.ndarray], int]:
    import torch

    json_path = Path(json_path)
    spec = json.loads(json_path.read_text())["spec"]
    if spec["agent"] == "tabular":
        from envs.discretizers import make_discretizer

        Q = np.load(json_path.with_suffix(".npz"))["Q"]
        disc = make_discretizer(spec["discretizer"])

        def policy(obs):
            # first-index tie-break: deterministic; ties only remain in never-visited states
            return int(np.argmax(Q[disc(obs)]))

        return policy

    from agents.dqn import QNetwork

    kw = spec["agent_kwargs"]
    net = QNetwork(n_obs, n_actions, kw.get("hidden", (128, 128)), kw.get("dueling", False))
    net.load_state_dict(torch.load(json_path.with_suffix(".pt"), map_location="cpu"))
    net.eval()

    @torch.no_grad()
    def policy(obs):
        return int(net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax())

    return policy
