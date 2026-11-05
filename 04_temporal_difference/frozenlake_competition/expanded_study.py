# AI assistance: Codex drafted this sampled-experience comparison and executed it.
"""Bounded comparison of all three permitted families; no official seed tuning.

TD: 30,000 total transitions. MC: 30,000 episodes. Horizon remains the
previously approved 100 steps, not a newly established competition rule.
"""
import json
import time
from pathlib import Path
import numpy as np
from solution import DEFAULT, action, environment, evaluate, train

ROOT = Path(__file__).resolve().parent


def train_mc(config, seed, episodes=30000):
    start = time.perf_counter()
    env = environment()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    q = np.zeros((16, 4), dtype=np.float64)
    counts = np.zeros((16, 4), dtype=np.int64)
    state, _ = env.reset(seed=seed)
    steps = successes = 0
    for episode in range(episodes):
        if episode:
            state, _ = env.reset()
        fraction = min(episode / (episodes * config["decay_fraction"]), 1)
        epsilon = config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])
        trajectory = []
        while True:
            a = action(q, state, rng, epsilon)
            nxt, reward, terminated, truncated, _ = env.step(a)
            trajectory.append((state, a, float(reward)))
            steps += 1
            state = nxt
            if terminated or truncated:
                successes += int(reward == 1)
                break
        returns = [0.0] * len(trajectory)
        value = 0.0
        for t in range(len(trajectory)-1, -1, -1):
            value = trajectory[t][2] + config["gamma"] * value
            returns[t] = value
        seen = set()
        for (s, a, _), value in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s, a] += 1
            rate = 1.0 / counts[s, a] if config["mc_rate"] == "mean" else float(config["mc_rate"])
            q[s, a] += rate * (value - q[s, a])
    env.close()
    assert np.isfinite(q).all()
    return q, dict(seed=seed, transitions=steps, completed_episodes=episodes,
                   successes=successes, seconds=time.perf_counter()-start)


def train_candidate(config, seed):
    return train_mc(config, seed) if config["method"] == "mc" else train(config, seed)


def summaries(runs):
    rewards = np.array([r["evaluation"]["total_reward"] for r in runs])
    return dict(mean=float(rewards.mean()), std=float(rewards.std(ddof=1)),
                minimum=float(rewards.min()), maximum=float(rewards.max()),
                mean_training_seconds=float(np.mean([r["training"]["seconds"] for r in runs])))


def execute(config, seeds, test_episodes):
    runs = []
    for seed in seeds:
        q, training = train_candidate(config, seed)
        result = evaluate(q, seed + 100000, test_episodes)
        runs.append(dict(training=training, evaluation=result))
    return dict(runs=runs, summary=summaries(runs))


def main():
    configurations = [("q_baseline", dict(DEFAULT))]
    for gamma in (0.995, 1.0):
        configurations.append(("q_gamma_" + str(gamma), {**DEFAULT, "gamma": gamma}))
    configurations.extend([
        ("q_995_slow_rate", {**DEFAULT, "gamma": 0.995, "alpha": 0.25}),
        ("q_995_fast_rate", {**DEFAULT, "gamma": 0.995, "alpha_scale": 200}),
        ("q_995_exploration", {**DEFAULT, "gamma": 0.995, "epsilon_start": 0.5, "epsilon_end": 0.1}),
        ("sarsa_99", {**DEFAULT, "method": "sarsa", "epsilon_end": 0.0, "decay_fraction": 0.7}),
        ("sarsa_995", {**DEFAULT, "method": "sarsa", "gamma": 0.995, "epsilon_end": 0.0, "decay_fraction": 0.7}),
        ("sarsa_1", {**DEFAULT, "method": "sarsa", "gamma": 1.0, "epsilon_end": 0.0, "decay_fraction": 0.7}),
        ("mc_995_mean", {**DEFAULT, "method": "mc", "gamma": 0.995, "mc_rate": "mean", "epsilon_start": 0.3, "epsilon_end": 0.01}),
        ("mc_995_constant", {**DEFAULT, "method": "mc", "gamma": 0.995, "mc_rate": 0.02, "epsilon_start": 0.3, "epsilon_end": 0.01}),
        ("mc_1_constant", {**DEFAULT, "method": "mc", "gamma": 1.0, "mc_rate": 0.02, "epsilon_start": 0.3, "epsilon_end": 0.01})
    ])
    study = dict(screen_seeds=[3301, 3302, 3303], confirm_seeds=[4401, 4402, 4403, 4404, 4405],
                 protocol="Screen all 12 on 3 paired seeds and 1000 tests; confirm baseline, top two, and best gamma=.995 on five fresh paired seeds and 2000 tests. Select highest confirmation mean, retaining baseline on exact tie.", candidates=[])
    path = ROOT / "expanded_study.json"
    for name, config in configurations:
        result = dict(name=name, config=config, screen=execute(config, study["screen_seeds"], 1000))
        study["candidates"].append(result)
        path.write_text(json.dumps(study, indent=2))
        print(json.dumps(dict(stage="screen", name=name, **result["screen"]["summary"])), flush=True)
    ranked = sorted(study["candidates"], key=lambda r: r["screen"]["summary"]["mean"], reverse=True)
    finalist_names = {"q_baseline", ranked[0]["name"], ranked[1]["name"], next(r["name"] for r in ranked if r["config"]["gamma"] == 0.995)}
    for candidate in study["candidates"]:
        if candidate["name"] in finalist_names:
            candidate["confirmation"] = execute(candidate["config"], study["confirm_seeds"], 2000)
            path.write_text(json.dumps(study, indent=2))
            print(json.dumps(dict(stage="confirmation", name=candidate["name"], **candidate["confirmation"]["summary"])), flush=True)
    winner = max((r for r in study["candidates"] if "confirmation" in r), key=lambda r: r["confirmation"]["summary"]["mean"])
    study["selected"] = winner["name"]
    path.write_text(json.dumps(study, indent=2))
    (ROOT / "expanded_config.json").write_text(json.dumps(winner["config"], indent=2))
    print("SELECTED " + winner["name"], flush=True)


if __name__ == "__main__":
    main()
