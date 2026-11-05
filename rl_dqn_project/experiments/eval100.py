"""Final 100-episode greedy evaluation of every trained agent, against the official thresholds.

The project's success thresholds are defined as a mean return over 100 episodes (CartPole-v1 >= 475,
MountainCar-v0 >= -110, LunarLander-v3 >= 200). The learning curves use 10 greedy episodes per checkpoint;
this script loads the FINAL saved weights of every run (results/<exp>/<config>/seed_k.pt|.npz) and plays
100 greedy episodes (epsilon = 0, no learning) from start states disjoint from training and from the
checkpoint evaluations (reset seeds 2,000,000 + 1,000*seed + k).

Note: these are the weights at the END of training, not the best checkpoint, so a config that
forgets after its peak (Sec. "catastrophic forgetting") scores lower here than its best curve value.

Output: results/final_eval100.json (cache, per run) and results/tables/t_final100.csv/.tex
Usage:  python experiments/eval100.py [--force] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import _common  # noqa: F401
import numpy as np

from _common import ROOT, fmt, write_table
from common.stats import mean_ci

RESULTS = ROOT / "results"
CACHE = RESULTS / "final_eval100.json"
N_EPISODES = 100
EXPERIMENTS = {  # experiment -> (env id, success threshold over 100 episodes)
    "tier1_cartpole": ("CartPole-v1", 475.0),
    "tier1_mountaincar": ("MountainCar-v0", -110.0),
    "tier2_lunarlander": ("LunarLander-v3", 200.0),
}


def evaluate_run(json_path: str) -> tuple[str, list[float]]:
    import gymnasium as gym
    import torch

    from common.policy import load_policy

    torch.set_num_threads(1)
    spec = json.loads(open(json_path).read())["spec"]
    env = gym.make(spec["env_id"], **spec.get("env_kwargs", {}))
    policy = load_policy(json_path, env.observation_space.shape[0], env.action_space.n)
    base = 2_000_000 + 1_000 * spec["seed"]
    rets = []
    for k in range(N_EPISODES):
        obs, _ = env.reset(seed=base + k)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            total += r
            done = term or trunc
        rets.append(float(total))
    env.close()
    return json_path, rets


def rel(p) -> str:
    return os.path.relpath(p, RESULTS).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-evaluate runs that are already cached")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    cache = json.loads(CACHE.read_text()) if CACHE.exists() and not a.force else {}
    paths = [p for exp in EXPERIMENTS for p in sorted((RESULTS / exp).glob("*/seed_*.json"))]
    todo = [str(p) for p in paths if rel(p) not in cache]
    if todo:
        workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
        print(f"evaluating {len(todo)} runs x {N_EPISODES} episodes on {workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(evaluate_run, p) for p in todo]
            for i, f in enumerate(as_completed(futs), 1):
                p, rets = f.result()
                cache[rel(p)] = rets
                print(f"  [{i}/{len(todo)}] {rel(p)}: {np.mean(rets):.1f}", flush=True)
                CACHE.write_text(json.dumps(cache))  # incremental: a crash keeps finished runs

    rows = []
    for exp, (env_id, thr) in EXPERIMENTS.items():
        configs = sorted({k.split("/")[1] for k in cache if k.startswith(exp + "/")})
        for c in configs:
            means = np.array([np.mean(v) for k, v in cache.items() if k.startswith(f"{exp}/{c}/")])
            m, h = mean_ci(means)
            rows.append([env_id, c, fmt((m, h, len(means))), f"{means.max():.1f}",
                         f"{int((means >= thr).sum())}/{len(means)}", f"{thr:g}"])
    header = ["Env", "Config", "100-ep mean (final weights)", "Best seed", "Seeds ≥ threshold", "Threshold"]
    caption = (f"Final weights, {N_EPISODES} greedy episodes per seed on held-out start states. "
               "Mean $\\pm$ 95\\% t-CI over seeds of each seed's 100-episode mean; a seed passes when its own "
               "100-episode mean reaches the official threshold.")
    write_table("t_final100", header, rows, caption)
    # shorter version for the report: the main agent of each study (full table: t_final100)
    main_cfgs = {"tabular_bins10", "dqn_eps5k", "tabular_optimistic", "dqn", "full", "no_target", "no_replay"}
    write_table("t_final100_main", header, [r for r in rows if r[1] in main_cfgs],
                caption + " Main configurations only; every run is in results/tables/t\\_final100.csv.")


if __name__ == "__main__":
    main()
