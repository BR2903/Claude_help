"""Agent-agnostic training loop with periodic frozen-weight greedy evaluation.

Every agent implements:
    act(obs, t) -> action           epsilon-greedy, used for training
    greedy(obs) -> action           epsilon = 0, used for evaluation
    observe(obs, a, r, obs2, terminated, truncated, t)   learning step(s)
    pop_stats() -> dict             averaged diagnostics since the last call

The x-axis of everything we log is the CUMULATIVE NUMBER OF ENVIRONMENT STEPS `t`, never the
episode index: tabular and DQN agents produce episodes of very different lengths, so
"episodes" would be an unfair clock.
"""
from __future__ import annotations

import time
from typing import Callable

import gymnasium as gym
import numpy as np


def evaluate(agent, env: gym.Env, n_episodes: int, seed_base: int) -> list[float]:
    """Greedy (epsilon = 0) returns. No call to agent.observe -> the weights are frozen.

    The same `seed_base + k` start states are used at every checkpoint of a run, so changes in
    the evaluation curve reflect changes in the policy, not in the sampled initial states.
    """
    returns = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + k)
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(agent.greedy(obs))
            total += r
            done = terminated or truncated
        returns.append(total)
    return returns


def train(
    make_env: Callable[[], gym.Env],
    agent,
    total_steps: int,
    seed: int,
    eval_every: int = 5_000,
    eval_episodes: int = 10,
    log_every: int = 1_000,
) -> dict:
    env, eval_env = make_env(), make_env()
    obs, _ = env.reset(seed=seed)  # seeds the env RNG; later resets continue that stream
    env.action_space.seed(seed)
    eval_seed_base = 1_000_000 + 1_000 * seed  # disjoint from training seeds

    log = {
        "train_episode_end_step": [], "train_return": [], "train_length": [], "train_terminated": [],
        "eval_step": [], "eval_return_mean": [], "eval_return_std": [], "eval_returns": [],
        "stats_step": [], "stats": [],
    }

    def do_eval(t: int) -> None:
        rets = evaluate(agent, eval_env, eval_episodes, eval_seed_base)
        log["eval_step"].append(t)
        log["eval_return_mean"].append(float(np.mean(rets)))
        log["eval_return_std"].append(float(np.std(rets)))
        log["eval_returns"].append([float(x) for x in rets])

    wall0, cpu0 = time.perf_counter(), time.process_time()
    do_eval(0)  # performance of the untrained (random-init) policy
    ep_return, ep_len = 0.0, 0
    for t in range(1, total_steps + 1):
        action = agent.act(obs, t)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated, truncated, t)
        ep_return += reward
        ep_len += 1
        obs = next_obs

        if terminated or truncated:
            log["train_episode_end_step"].append(t)
            log["train_return"].append(float(ep_return))
            log["train_length"].append(ep_len)
            log["train_terminated"].append(bool(terminated))
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        if t % log_every == 0:
            log["stats_step"].append(t)
            log["stats"].append(agent.pop_stats())
        if t % eval_every == 0:
            do_eval(t)

    log["wall_seconds"] = time.perf_counter() - wall0
    log["cpu_seconds"] = time.process_time() - cpu0
    log["total_steps"] = total_steps
    log["n_episodes"] = len(log["train_return"])
    env.close()
    eval_env.close()
    return log
