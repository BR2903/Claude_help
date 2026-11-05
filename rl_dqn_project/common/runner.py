"""Runs (config x seed) jobs in parallel worker processes and caches each result as JSON.

Layout:  results/<experiment>/<config>/seed_<k>.json     training/eval logs + the exact spec
         results/<experiment>/<config>/seed_<k>.pt|.npz  final weights (DQN) / Q-table (tabular)

A run whose JSON already exists is skipped (use --force to recompute), so re-running an
experiment script after a crash only computes what is missing, and `--plot-only` is instant.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def run_single(spec: dict) -> dict:
    """Build env + agent from a JSON spec, train, save weights. Runs inside a worker process."""
    import gymnasium as gym
    import torch

    from common.loop import train
    from common.seeding import set_global_seeds

    torch.set_num_threads(1)  # one core per run; we parallelise across runs instead
    seed = spec["seed"]
    set_global_seeds(seed)
    env_kwargs = spec.get("env_kwargs", {})

    def make_env():
        return gym.make(spec["env_id"], **env_kwargs)

    probe = make_env()
    obs_dim, n_actions = probe.observation_space.shape[0], probe.action_space.n
    probe.close()

    if spec["agent"] == "tabular":
        from agents.tabular import TabularAgent
        from envs.discretizers import make_discretizer

        agent = TabularAgent(make_discretizer(spec["discretizer"]), n_actions, seed=seed, **spec["agent_kwargs"])
    elif spec["agent"] == "dqn":
        from agents.dqn import DQNAgent

        agent = DQNAgent(obs_dim, n_actions, seed=seed, **spec["agent_kwargs"])
    else:
        raise ValueError(spec["agent"])

    log = train(make_env, agent, spec["total_steps"], seed,
                eval_every=spec.get("eval_every", 5_000), eval_episodes=spec.get("eval_episodes", 10))
    if spec["agent"] == "tabular":
        log["state_coverage"] = agent.coverage()
        log["n_states"] = agent.disc.n_states
    else:
        log["n_updates"] = agent.n_updates
    agent.save(spec["weights_path"])
    log["spec"] = spec
    log["host"] = {"node": platform.node(), "cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
                   "torch": torch.__version__, "gymnasium": gym.__version__}
    return log


def _job(spec: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    log = run_single(spec)
    out = Path(spec["json_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(log))
    tmp.replace(out)  # atomic: a half-written file never looks like a finished run
    return spec["json_path"], time.perf_counter() - t0


def run_experiment(experiment: str, configs: dict[str, dict], seeds: list[int],
                   workers: int | None = None, force: bool = False) -> None:
    """configs: {config_name: spec-without-seed}. Each is run once per seed."""
    jobs = []
    for name, base in configs.items():
        for seed in seeds:
            d = RESULTS / experiment / name
            d.mkdir(parents=True, exist_ok=True)
            ext = ".npz" if base["agent"] == "tabular" else ".pt"
            spec = {**base, "seed": seed, "config": name, "experiment": experiment,
                    "json_path": str(d / f"seed_{seed}.json"), "weights_path": str(d / f"seed_{seed}{ext}")}
            if force or not Path(spec["json_path"]).exists():
                jobs.append(spec)
    if not jobs:
        print(f"[{experiment}] all {len(configs) * len(seeds)} runs cached; nothing to do.")
        return

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    # slowest jobs (most steps, DQN) first -> better packing across workers
    jobs.sort(key=lambda s: (s["agent"] != "dqn", -s["total_steps"]))
    print(f"[{experiment}] running {len(jobs)} jobs on {workers} workers ...", flush=True)
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_job, s): s for s in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            path, secs = fut.result()
            rel = Path(path).relative_to(RESULTS)
            print(f"  [{i}/{len(jobs)}] {rel}  ({secs:.0f}s, elapsed {time.perf_counter() - t0:.0f}s)", flush=True)


def load_runs(experiment: str, config: str) -> list[dict]:
    d = RESULTS / experiment / config
    return [json.loads(p.read_text()) for p in sorted(d.glob("seed_*.json"))]
