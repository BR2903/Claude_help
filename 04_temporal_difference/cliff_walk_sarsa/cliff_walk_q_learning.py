"""Q-learning from scratch on the 4 x 12 Cliff Walking grid world.

Authors: Aispur Santiago and Amaya Bryan
Course: Intelligent Agents
Instructor: PhD Armas Rolando

Dependencies: NumPy for the implementation and Matplotlib for figures only.
No Gym or reinforcement-learning package is used. The file also trains SARSA
under the same conditions for a fair comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# Environment ---------------------------------------------------------------

ROWS = 4
COLS = 12
N_ACTIONS = 4
START = (3, 0)
GOAL = (3, 11)
CLIFF = {(3, col) for col in range(1, 11)}

# 0 = up, 1 = right, 2 = down, 3 = left
ACTION_DELTAS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)

ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.1
N_EPISODES = 500
N_RUNS = 50
MAX_STEPS = 1_000


def step(state, action):
    """Apply an action and return (next_state, reward, terminal, fell)."""
    delta_row, delta_col = ACTION_DELTAS[action]
    next_row = state[0] + delta_row
    next_col = state[1] + delta_col

    # A wall leaves the agent in the same cell and still costs one point.
    if not (0 <= next_row < ROWS and 0 <= next_col < COLS):
        next_row, next_col = state

    next_state = (next_row, next_col)
    if next_state in CLIFF:
        return START, -100, False, True
    if next_state == GOAL:
        return GOAL, -1, True, False
    return next_state, -1, False, False


def epsilon_greedy(q_values, state, epsilon, rng):
    """Select an action, breaking greedy ties uniformly at random."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))

    row, col = state
    action_values = q_values[row, col]
    best_actions = np.flatnonzero(action_values == action_values.max())
    return int(rng.choice(best_actions))


# Learning algorithms -------------------------------------------------------

def train_q_learning(seed, epsilon=EPSILON, episodes=N_EPISODES):
    """Train one tabular Q-learning run.

    Q-learning is off-policy: behavior is epsilon-greedy, but the target uses
    the largest next-state action value, regardless of the next behavior
    action that will actually be sampled.
    """
    rng = np.random.default_rng(seed)
    q_values = np.zeros((ROWS, COLS, N_ACTIONS), dtype=float)
    returns = np.zeros(episodes, dtype=float)
    cliff_falls = np.zeros(episodes, dtype=int)

    for episode in range(episodes):
        state = START
        total_return = 0.0

        for _ in range(MAX_STEPS):
            action = epsilon_greedy(q_values, state, epsilon, rng)
            next_state, reward, terminal, fell = step(state, action)
            total_return += reward
            cliff_falls[episode] += int(fell)

            row, col = state
            if terminal:
                target = reward
            else:
                next_row, next_col = next_state
                # The max below is the defining off-policy Q-learning line.
                target = reward + GAMMA * np.max(q_values[next_row, next_col])

            q_values[row, col, action] += ALPHA * (
                target - q_values[row, col, action]
            )
            state = next_state

            if terminal:
                break

        returns[episode] = total_return

    return q_values, returns, cliff_falls


def train_sarsa(seed, epsilon=EPSILON, episodes=N_EPISODES):
    """Train SARSA under the same conditions as the Q-learning run."""
    rng = np.random.default_rng(seed)
    q_values = np.zeros((ROWS, COLS, N_ACTIONS), dtype=float)
    returns = np.zeros(episodes, dtype=float)
    cliff_falls = np.zeros(episodes, dtype=int)

    for episode in range(episodes):
        state = START
        action = epsilon_greedy(q_values, state, epsilon, rng)
        total_return = 0.0

        for _ in range(MAX_STEPS):
            next_state, reward, terminal, fell = step(state, action)
            total_return += reward
            cliff_falls[episode] += int(fell)
            row, col = state

            if terminal:
                target = reward
                q_values[row, col, action] += ALPHA * (
                    target - q_values[row, col, action]
                )
                break

            # SARSA uses the next action sampled from its behavior policy.
            next_action = epsilon_greedy(q_values, next_state, epsilon, rng)
            next_row, next_col = next_state
            target = reward + GAMMA * q_values[next_row, next_col, next_action]
            q_values[row, col, action] += ALPHA * (
                target - q_values[row, col, action]
            )
            state, action = next_state, next_action

        returns[episode] = total_return

    return q_values, returns, cliff_falls


def train_runs(training_function):
    """Train seeds 0 through 49 with one learning algorithm."""
    all_q_values = np.empty((N_RUNS, ROWS, COLS, N_ACTIONS), dtype=float)
    all_returns = np.empty((N_RUNS, N_EPISODES), dtype=float)
    all_falls = np.empty((N_RUNS, N_EPISODES), dtype=int)

    for seed in range(N_RUNS):
        q_values, returns, falls = training_function(seed)
        all_q_values[seed] = q_values
        all_returns[seed] = returns
        all_falls[seed] = falls

    return all_q_values, all_returns, all_falls


# Policy analysis -----------------------------------------------------------

def greedy_rollout(q_values, max_steps=100):
    """Return a deterministic greedy route, or a partial route if it loops."""
    state = START
    path = [state]
    visits = {state: 1}

    for _ in range(max_steps):
        row, col = state
        action_values = q_values[row, col]
        # Lowest-index tie resolution is used only to display one fixed route.
        action = int(np.flatnonzero(action_values == action_values.max())[0])
        next_state, _, terminal, _ = step(state, action)
        path.append(next_state)

        if terminal:
            return path

        visits[next_state] = visits.get(next_state, 0) + 1
        if visits[next_state] > 1:
            return path
        state = next_state

    return path


def path_length(path):
    """Return the route length, or None when the rollout did not reach G."""
    return len(path) - 1 if path[-1] == GOAL else None


def crossing_row(path):
    """Return the dominant row used over middle columns 1 through 10."""
    rows = [row for row, col in path if 1 <= col <= 10]
    if not rows:
        return None
    counts = np.bincount(rows, minlength=ROWS)
    return int(np.argmax(counts))


def policy_grid(q_values):
    """Return a printable greedy-policy grid."""
    symbols = np.array(["^", ">", "v", "<"])
    grid = np.full((ROWS, COLS), " ", dtype="<U1")

    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)
            if state == START:
                grid[row, col] = "S"
            elif state == GOAL:
                grid[row, col] = "G"
            elif state in CLIFF:
                grid[row, col] = "X"
            else:
                grid[row, col] = symbols[int(np.argmax(q_values[row, col]))]
    return grid


def print_grid(grid):
    for row in grid:
        print(" ".join(row))


# Figures -------------------------------------------------------------------

def draw_path(ax, path, algorithm, color):
    """Draw the Cliff Walking grid and one learned greedy route."""
    cell_types = np.zeros((ROWS, COLS), dtype=int)
    for row, col in CLIFF:
        cell_types[row, col] = 1
    cell_types[START] = 2
    cell_types[GOAL] = 3

    cmap = ListedColormap(["#f7f7f2", "#3f4b63", "#dce9f9", "#dff2df"])
    ax.imshow(cell_types, cmap=cmap, vmin=0, vmax=3, origin="upper")
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="#b8b5aa", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    route = np.asarray(path)
    ax.plot(route[:, 1], route[:, 0], color=color, linewidth=3,
            marker="o", markersize=5, label=f"{algorithm} greedy route")
    ax.text(START[1], START[0], "S", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1769aa")
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#17823b")
    ax.text(5.5, 3, "CLIFF: -100 and return to S", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")

    length = path_length(path)
    row = crossing_row(path)
    ax.set_title(f"{algorithm}: {length} steps, dominant row {row}")
    ax.legend(
        handles=[
            ax.lines[0],
            Patch(facecolor="#3f4b63", edgecolor="#b8b5aa", label="Cliff"),
        ],
        loc="upper center",
        ncol=2,
    )


def save_path_figures(q_path, sarsa_path, output_dir):
    """Save the Q-learning path and a direct Q-learning/SARSA comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.0, 4.3))
    draw_path(ax, q_path, "Q-learning", "#e4572e")
    fig.suptitle("Greedy Route Learned by Q-learning (seed 0)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "q_learning_greedy_path.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0))
    draw_path(axes[0], q_path, "Q-learning", "#e4572e")
    draw_path(axes[1], sarsa_path, "SARSA", "#1769aa")
    fig.suptitle("Cliff Walking: Q-learning versus SARSA (seed 0)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "q_learning_vs_sarsa_paths.png", dpi=180)
    plt.close(fig)


def summarize(name, all_q_values, all_returns, all_falls):
    """Print route and training statistics for one algorithm."""
    paths = [greedy_rollout(q_values) for q_values in all_q_values]
    lengths = np.array(
        [length for path in paths if (length := path_length(path)) is not None],
        dtype=float,
    )
    seed_zero_path = paths[0]
    seed_zero_length = path_length(seed_zero_path)

    print(f"\n=== {name} ===")
    print(f"Seed-0 greedy route: {seed_zero_path}")
    print(f"Seed-0 route length: {seed_zero_length}")
    print(f"Seed-0 dominant crossing row: {crossing_row(seed_zero_path)} "
          "(2 is beside the cliff; 0 is the top)")
    print(f"Successful greedy routes: {len(lengths)}/{N_RUNS}")
    print(f"Mean successful route length: {lengths.mean():.2f}")
    print(f"Last-100 mean training return: "
          f"{all_returns[:, -100:].mean():.2f}")
    print(f"Last-100 mean cliff falls per episode: "
          f"{all_falls[:, -100:].mean():.3f}")
    print("Seed-0 greedy policy:")
    print_grid(policy_grid(all_q_values[0]))

    return seed_zero_path


def main():
    q_tables, q_returns, q_falls = train_runs(train_q_learning)
    sarsa_tables, sarsa_returns, sarsa_falls = train_runs(train_sarsa)

    q_path = summarize("Q-learning (off-policy)", q_tables, q_returns, q_falls)
    sarsa_path = summarize("SARSA (on-policy)", sarsa_tables,
                           sarsa_returns, sarsa_falls)

    figure_dir = Path(__file__).resolve().parent / "figures"
    save_path_figures(q_path, sarsa_path, figure_dir)

    print("\n=== Direct comparison ===")
    print(f"Q-learning seed-0 path: {path_length(q_path)} steps, "
          f"dominant crossing row {crossing_row(q_path)}")
    print(f"SARSA seed-0 path     : {path_length(sarsa_path)} steps, "
          f"dominant crossing row {crossing_row(sarsa_path)}")
    print("Q-learning learns the greedy optimal route beside the cliff because its")
    print("target assumes the best next action. SARSA learns a longer, safer route")
    print("because its target includes the exploratory action it may actually take.")
    print(f"Figures saved in: {figure_dir}")


if __name__ == "__main__":
    main()
