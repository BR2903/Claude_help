# AI assistance: Codex drafted and executed this sampled Q-learning pipeline.
"""python agent.py: 30,000 training steps total, 2,000 tests per official seed.

User-requested protocol change: no 100-step TimeLimit. Episodes run to a
goal/hole. A wall-clock guard aborts without producing an incomplete CSV.
This protocol has not been verified against the organizer's evaluator.
"""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import gymnasium as gym

SEEDS = [13, 15, 45, 56, 21, 1, 7, 9, 50, 47]
TRAINING_SEED = 20261001
GAMMA = 1.0


def env_factory():
    # Public Gymnasium option: -1 disables the registered TimeLimit wrapper.
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True,
                    max_episode_steps=-1)


def guard(deadline):
    if time.perf_counter() >= deadline:
        raise RuntimeError("Runtime budget reached; no incomplete submission will be written.")


def train(deadline):
    env = env_factory()
    rng = np.random.default_rng(np.random.SeedSequence([TRAINING_SEED, 1]))
    q = np.zeros((16, 4), dtype=np.float64)
    visits = np.zeros((16, 4), dtype=np.int64)
    state, _ = env.reset(seed=TRAINING_SEED)
    episodes = successes = 0
    for step in range(30000):
        if step % 1024 == 0:
            guard(deadline)
        epsilon = 1.0 - 0.99 * min(step / 24000, 1.0)
        if rng.random() < epsilon:
            action = int(rng.integers(4))
        else:
            choices = np.flatnonzero(q[state] == q[state].max())
            action = int(choices[int(rng.integers(len(choices)))])
        nxt, reward, terminated, truncated, _ = env.step(action)
        assert not truncated, "Unexpected environment truncation"
        target = float(reward)
        if not terminated:
            target += GAMMA * float(q[nxt].max())
        visits[state, action] += 1
        alpha = 0.25 / (1 + visits[state, action] / 50) ** 0.6
        q[state, action] += alpha * (target - q[state, action])
        state = nxt
        if terminated:
            episodes += 1
            successes += int(reward == 1)
            state, _ = env.reset()
    env.close()
    return q, dict(seed=TRAINING_SEED, transitions=30000,
                   completed_episodes=episodes, successes=successes)


def evaluate(q, seed, deadline):
    env = env_factory()
    state, _ = env.reset(seed=seed)
    policy = np.argmax(q, axis=1)
    successes = transitions = longest = 0
    for episode in range(2000):
        guard(deadline)
        if episode:
            state, _ = env.reset()
        length = 0
        while True:
            if length % 1024 == 0:
                guard(deadline)
            state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            assert not truncated, "Unexpected environment truncation"
            length += 1
            if terminated:
                successes += int(reward == 1)
                break
        longest = max(longest, length)
        transitions += length
    env.close()
    return dict(seed=seed, total_reward=successes / 2000, episodes=2000,
                successes=successes, holes=2000-successes,
                transitions=transitions, longest_episode=longest, truncations=0)


def main():
    started = time.perf_counter()
    deadline = started + 115
    q, training = train(deadline)
    frozen = q.copy()
    rows = [evaluate(q, seed, deadline) for seed in SEEDS]
    assert np.array_equal(frozen, q) and np.isfinite(q).all()
    values = np.array([r["total_reward"] for r in rows])
    assert len(rows) == len(set(SEEDS)) == 10
    assert ((0 <= values) & (values <= 1)).all()
    guard(deadline)
    root = Path(__file__).resolve().parent
    report = dict(method="Q-learning", gamma=GAMMA, training=training, runs=rows,
                  mean_reward=float(values.mean()), std_reward=float(values.std(ddof=1)),
                  best_reward=float(values.max()), worst_reward=float(values.min()),
                  optimal_reward=None, gap_to_optimal_reward=None, official_mae=None,
                  episode_step_cap=None, evaluation_episodes_per_seed=2000,
                  protocol="One shared table; no TimeLimit; terminal goal/hole; seed each fresh evaluation environment once; frozen greedy policy.",
                  q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                  python=platform.python_version(), numpy=np.__version__, gymnasium=gym.__version__)
    np.save(root / "q_table.npy", q)
    with (root / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "total_reward"])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    report["elapsed_seconds"] = time.perf_counter()-started
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
