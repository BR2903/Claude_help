# AI assistance: Codex drafted the training, tuning, evaluation, and validation code; executed results are recorded alongside it.
"""Sample-only tabular RL. No model access, planning, or hardcoded policy.

Protocol approved by user: ONE shared table, 30,000 total training steps,
standard 4x4 FrozenLake, 100-step episodes, 2,000 tests per official seed.
Run: python solution.py --seeds seeds_eval.csv --output results
Development tuning is a separate, explicitly requested experiment.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

DEFAULT = dict(method="qlearning", gamma=0.99, alpha=0.5,
               alpha_scale=50, alpha_power=0.6, epsilon_start=1.0,
               epsilon_end=0.05, decay_fraction=0.8, initial_q=0.0)


def environment():
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True,
                    max_episode_steps=100)


def action(q, state, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(4))
    row = q[state]
    choices = np.flatnonzero(row == row.max())
    return int(choices[int(rng.integers(len(choices)))])


def train(config, seed, steps=30000):
    start = time.perf_counter()
    env = environment()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    q = np.full((16, 4), config["initial_q"], dtype=np.float64)
    visits = np.zeros((16, 4), dtype=np.int64)
    state, _ = env.reset(seed=seed)
    episodes = successes = 0
    selected = None
    for step in range(steps):
        fraction = min(step / max(1, steps * config["decay_fraction"]), 1)
        epsilon = config["epsilon_start"] + fraction * (
            config["epsilon_end"] - config["epsilon_start"])
        a = action(q, state, rng, epsilon) if selected is None else selected
        nxt, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        selected = None
        # The approved task has a 100-step horizon; both endings stop returns.
        target = float(reward)
        if not done:
            if config["method"] == "sarsa":
                selected = action(q, nxt, rng, epsilon)
                target += config["gamma"] * q[nxt, selected]
            else:
                target += config["gamma"] * float(q[nxt].max())
        visits[state, a] += 1
        rate = config["alpha"] / (
            1 + visits[state, a] / config["alpha_scale"]
        ) ** config["alpha_power"]
        q[state, a] += rate * (target - q[state, a])
        state = nxt
        if done:
            episodes += 1
            successes += int(reward == 1)
            state, _ = env.reset()
            selected = None
    env.close()
    assert np.isfinite(q).all()
    return q, dict(seed=seed, transitions=steps, completed_episodes=episodes,
                   successes=successes, seconds=time.perf_counter() - start)


def evaluate(q, seed, episodes=2000):
    start = time.perf_counter()
    env = environment()
    # A separate environment is seeded once; subsequent resets advance its RNG.
    state, _ = env.reset(seed=seed)
    policy = np.argmax(q, axis=1)
    successes = holes = truncations = transitions = 0
    for episode in range(episodes):
        if episode:
            state, _ = env.reset()
        while True:
            state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            transitions += 1
            if terminated or truncated:
                successes += int(reward == 1)
                holes += int(terminated and reward == 0)
                truncations += int(truncated and not terminated)
                break
    env.close()
    assert successes + holes + truncations == episodes
    return dict(seed=int(seed), total_reward=successes / episodes,
                episodes=episodes, successes=successes, holes=holes,
                truncations=truncations, transitions=transitions,
                seconds=time.perf_counter() - start)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=Path(__file__).with_name("seeds_eval.csv"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    config = {**DEFAULT, **json.loads(args.config.read_text())}
    with args.seeds.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["seed"]
        seeds = [int(row["seed"]) for row in reader]
    assert len(seeds) == len(set(seeds)) == 10
    # Fixed in advance, not chosen from tuning or official evaluation outcomes.
    training_seed = 20261001
    q, training = train(config, training_seed)
    before = q.copy()
    rows = [evaluate(q, seed) for seed in seeds]
    assert np.array_equal(q, before)
    values = np.array([r["total_reward"] for r in rows])
    assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
    args.output.mkdir(parents=True, exist_ok=True)
    submission = args.output / "submission.csv"
    with submission.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "total_reward"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    np.save(args.output / "q_table.npy", q)
    elapsed = time.perf_counter() - started
    report = dict(config=config, training=training, runs=rows,
                  mean_reward=float(values.mean()), std_reward=float(values.std(ddof=1)),
                  best_reward=float(values.max()), worst_reward=float(values.min()),
                  optimal_reward=None, gap_to_optimal_reward=None, official_mae=None,
                  elapsed_seconds=elapsed, within_120_seconds=elapsed < 120,
                  python=platform.python_version(), numpy=np.__version__,
                  gymnasium=gym.__version__,
                  q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                  protocol="One 30000-transition training run; 10 evaluation seeds; 2000 episodes each; horizon 100; greedy argmax; reseed once per seed.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if elapsed >= 120:
        raise RuntimeError("Measured execution exceeded 120 seconds")


if __name__ == "__main__":
    main()
