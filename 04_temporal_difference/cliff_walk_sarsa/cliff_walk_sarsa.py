"""
Tabular SARSA for the 4 x 12 Cliff Walking task.

Authors: Aispur Santiago and Amaya Bryan
Course: Intelligent Agents
Instructor: PhD Armas Rolando

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROWS = 4
COLS = 12
N_ACTIONS = 4
START = (3, 0)
GOAL = (3, 11)
CLIFF = {(3, col) for col in range(1, 11)}
ACTION_DELTAS = np.array(
    [
        [-1, 0],  # 0: up
        [0, 1],   # 1: right
        [1, 0],   # 2: down
        [0, -1],  # 3: left
    ],
    dtype=int,
)

ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.1
N_EPISODES = 500
N_RUNS = 50
MAX_STEPS = 1_000
MOVING_AVERAGE_WINDOW = 10


def step(state, action):
    """Apply one deterministic action and return (next_state, reward, done)."""
    candidate = np.asarray(state) + ACTION_DELTAS[action]
    row = int(np.clip(candidate[0], 0, ROWS - 1))
    col = int(np.clip(candidate[1], 0, COLS - 1))
    next_state = (row, col)

    if next_state in CLIFF:
        return START, -100, False
    if next_state == GOAL:
        return GOAL, -1, True
    return next_state, -1, False


def epsilon_greedy(q_values, state, epsilon, rng):
    """Sample from an epsilon-greedy policy with uniform random tie-breaking."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))

    row, col = state
    values = q_values[row, col]
    best_actions = np.flatnonzero(values == values.max())
    return int(rng.choice(best_actions))


def train_sarsa(seed, epsilon=EPSILON, episodes=N_EPISODES):
    """Train one independent SARSA run and return its Q-table and returns."""
    rng = np.random.default_rng(seed)
    q_values = np.zeros((ROWS, COLS, N_ACTIONS), dtype=float)
    episode_returns = np.zeros(episodes, dtype=float)

    for episode in range(episodes):
        state = START
        action = epsilon_greedy(q_values, state, epsilon, rng)
        total_return = 0.0

        for _ in range(MAX_STEPS):
            next_state, reward, done = step(state, action)
            total_return += reward
            row, col = state

            if done:
                target = reward
                q_values[row, col, action] += ALPHA * (
                    target - q_values[row, col, action]
                )
                break

            # This sampled next action is what makes the update on-policy.
            next_action = epsilon_greedy(q_values, next_state, epsilon, rng)
            next_row, next_col = next_state
            target = reward + GAMMA * q_values[next_row, next_col, next_action]
            q_values[row, col, action] += ALPHA * (
                target - q_values[row, col, action]
            )
            state, action = next_state, next_action

        episode_returns[episode] = total_return

    return q_values, episode_returns


def train_runs(epsilon):
    """Train seeds 0 through 49 and return every Q-table and return trace."""
    all_q_values = np.empty((N_RUNS, ROWS, COLS, N_ACTIONS), dtype=float)
    all_returns = np.empty((N_RUNS, N_EPISODES), dtype=float)
    for seed in range(N_RUNS):
        q_values, returns = train_sarsa(seed=seed, epsilon=epsilon)
        all_q_values[seed] = q_values
        all_returns[seed] = returns
    return all_q_values, all_returns


def moving_average(values, window=MOVING_AVERAGE_WINDOW):
    """Return a trailing average, using shorter windows for episodes 1--9."""
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    smoothed = np.empty_like(values)
    for index in range(len(values)):
        lower = max(0, index - window + 1)
        smoothed[index] = (
            cumulative[index + 1] - cumulative[lower]
        ) / (index + 1 - lower)
    return smoothed


def greedy_rollout(q_values, max_steps=100):
    """Roll out a greedy policy; return a partial path if it stalls or loops."""
    state = START
    path = [state]
    for _ in range(max_steps):
        row, col = state
        action = int(np.flatnonzero(q_values[row, col] == q_values[row, col].max())[0])
        next_state, _, done = step(state, action)
        path.append(next_state)
        if done or next_state == state or path.count(next_state) > 2:
            break
        state = next_state
    return path


def greedy_path(q_values):
    """Follow deterministic greedy actions from S to G.

    ``np.flatnonzero(...)[0]`` only resolves exact post-training ties for route
    display. Training-time ties are always broken uniformly at random.
    """
    path = greedy_rollout(q_values, max_steps=MAX_STEPS)
    if path[-1] != GOAL:
        raise RuntimeError("The extracted greedy policy did not reach the goal.")
    return path


def path_length(path):
    """Return the step count, or None if a greedy rollout did not reach G."""
    return len(path) - 1 if path[-1] == GOAL else None


def cliff_clearance(path):
    """Return the closest row to the cliff used in columns 1 through 10."""
    middle_rows = [row for row, col in path if 1 <= col <= 10]
    return max(middle_rows) if middle_rows else None


def plot_learning_curve(all_returns, output_path):
    mean_returns = all_returns.mean(axis=0)
    smoothed = moving_average(mean_returns)
    episodes = np.arange(1, N_EPISODES + 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(episodes, smoothed, color="#1769aa", linewidth=2)
    ax.axhline(-13, color="#d1495b", linestyle="--", linewidth=1.4,
               label="Optimal return (-13 points)")
    ax.set_title("SARSA Learning Curve on Cliff Walking")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean return (points per episode)\n10-episode moving average")
    ax.set_ylim(-100, 0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_greedy_path(path, output_path):
    cell_types = np.zeros((ROWS, COLS), dtype=int)
    for row, col in CLIFF:
        cell_types[row, col] = 1
    cell_types[START] = 2
    cell_types[GOAL] = 3

    cmap = ListedColormap(["#f7f7f2", "#3f4b63", "#dce9f9", "#dff2df"])
    fig, ax = plt.subplots(figsize=(12.0, 4.3))
    ax.imshow(cell_types, cmap=cmap, vmin=0, vmax=3, origin="upper")

    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="#b8b5aa", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"Greedy Route Learned by SARSA (seed 0, {len(path) - 1} steps)")

    path_array = np.asarray(path)
    ax.plot(path_array[:, 1], path_array[:, 0], color="#e4572e", linewidth=3,
            marker="o", markersize=5, label="Greedy route")
    ax.text(START[1], START[0], "S", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#1769aa")
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#17823b")
    ax.text(5.5, 3, "CLIFF: -100 and return to S", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    legend_items = [
        ax.lines[0],
        Patch(facecolor="#3f4b63", edgecolor="#b8b5aa", label="Cliff"),
    ]
    ax.legend(handles=legend_items, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_epsilon_sweep(results, output_path):
    colors = {0.01: "#2a9d8f", 0.1: "#1769aa", 0.2: "#e76f51"}
    episodes = np.arange(1, N_EPISODES + 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for epsilon, (_, all_returns, route_length) in results.items():
        smoothed = moving_average(all_returns.mean(axis=0))
        ax.plot(episodes, smoothed, linewidth=2, color=colors[epsilon],
                label=rf"$\epsilon={epsilon}$ ({route_length}-step route)")

    ax.axhline(-13, color="#555555", linestyle="--", linewidth=1.2,
               label="Optimal return (-13 points)")
    ax.set_title(r"Effect of Exploration Rate $\epsilon$ on SARSA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean return (points per episode)\n10-episode moving average")
    ax.set_ylim(-100, 0)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon_results = {}
    for epsilon in (0.01, 0.1, 0.2):
        all_q_values, all_returns = train_runs(epsilon)
        path = greedy_path(all_q_values[0])
        epsilon_results[epsilon] = (all_q_values, all_returns, len(path) - 1)

    default_q_values, default_returns, default_length = epsilon_results[EPSILON]
    default_path = greedy_path(default_q_values[0])

    plot_learning_curve(default_returns, output_dir / "learning_curve.png")
    plot_greedy_path(default_path, output_dir / "greedy_path.png")
    plot_epsilon_sweep(epsilon_results, output_dir / "epsilon_sweep.png")

    default_mean = default_returns.mean(axis=0)
    print("=== Experiment 1: learning curve (epsilon = 0.1) ===")
    print(f"mean return, episode 1          : {default_mean[0]:10.1f}")
    print(f"mean return, first 10 episodes  : {default_mean[:10].mean():10.1f}")
    print(f"mean return, last 100 episodes  : {default_mean[-100:].mean():10.2f}")
    print(f"episodes below the -100 y-limit : "
          f"{int((moving_average(default_mean) < -100).sum())}")

    print("\n=== Experiment 2: greedy route, seed 0 ===")
    print(f"path: {default_path}")
    print(f"summary: {default_length} steps, "
          f"rows used: {sorted({row for row, _ in default_path})}")

    print("\n=== Experiment 3: effect of epsilon ===")
    for epsilon, (all_q_values, all_returns, route_length) in epsilon_results.items():
        paths = [greedy_rollout(q_values) for q_values in all_q_values]
        reached_lengths = [
            length for path in paths if (length := path_length(path)) is not None
        ]
        clearances = [cliff_clearance(path) for path in paths]
        clearance_counts = {row: clearances.count(row) for row in (2, 1, 0)}
        last_100_return = all_returns.mean(axis=0)[-100:].mean()
        print(
            f"epsilon={epsilon:0.2f} | seed 0: {route_length} steps | "
            f"mean length {np.mean(reached_lengths):.2f} over "
            f"{len(reached_lengths)}/{N_RUNS} successful greedy rollouts | "
            f"crossing rows {clearance_counts} | "
            f"last-100 mean return {last_100_return:.2f}"
        )


if __name__ == "__main__":
    main()
