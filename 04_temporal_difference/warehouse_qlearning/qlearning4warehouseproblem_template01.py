# -*- coding: utf-8 -*-
"""Warehouse route optimization using tabular Q-learning.

Original ASCII floor plan (gaps between cells are two-way passages):

+-----+-----+-----+-----+
|  A     B     C  |  D
+-----+     +     +     +
   E  |  F  |  G     H  |
+     +     +-----+     +
|  I     J     K     L  |
+-----+-----+     +-----+

Only A-L are states: gaps at the outside boundary are not extra actions.
Each move has equal cost, so the best route uses the fewest passages.
"""

import numpy as np


gamma = 0.75  # Discount factor
alpha = 0.9   # Learning rate
GOAL_REWARD = 10.0

# Part 1: Environment settings
location_to_state = {location: state for state, location in enumerate("ABCDEFGHIJKL")}
state_to_location = {state: location for location, state in location_to_state.items()}
actions = list(range(12))  # Action GoToX has the same index as state X.

# Columns: GoTo A, B, C, D, E, F, G, H, I, J, K, L.
# A zero means a forbidden move, not an available zero-reward move.
R = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # B
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # C
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # D
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # E
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # F
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # G
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # H
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # I
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # J
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # K
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # L
], dtype=float)


def _state(location):
    """Reject unknown labels with a helpful error."""
    if not isinstance(location, str) or location not in location_to_state:
        raise ValueError(f"Unknown location {location!r}; choose an uppercase letter A-L.")
    return location_to_state[location]


# Part 2: Q-learning
def train_q(ending_location, *, seed=42, max_updates=100_000):
    """Learn a fresh goal-specific table using random legal transitions.

    Reward is 1 per passage, replaced by 10 when entering the terminal goal.
    With gamma=.75, a route of d moves returns 4 + 6*(.75**(d-1));
    shorter routes therefore have strictly higher return. Wandering forever
    returns only 4. The terminal row stays zero: no fictitious self-loop.

    Stop when the maximum Bellman residual over ALL legal nonterminal
    transitions is below 1e-9, or raise if the update budget is exhausted.
    This is random-transition Q-learning, as specified by the template;
    complete episodes and epsilon-greedy exploration are not required here.
    """
    goal = _state(ending_location)
    if not isinstance(max_updates, int) or isinstance(max_updates, bool) or max_updates <= 0:
        raise ValueError("max_updates must be a positive integer.")
    rng = np.random.default_rng(seed)
    q = np.zeros_like(R)
    legal = [np.flatnonzero(row > 0) for row in R]
    nonterminal = np.array([s for s in actions if s != goal])
    sources, destinations = np.where(R > 0)
    active = sources != goal
    sources, destinations = sources[active], destinations[active]
    rewards = np.where(destinations == goal, GOAL_REWARD, R[sources, destinations])

    for update in range(1, max_updates + 1):
        current_state = int(rng.choice(nonterminal))
        current_action = int(rng.choice(legal[current_state]))
        next_state = current_action
        reward = GOAL_REWARD if next_state == goal else R[current_state, current_action]
        future = 0.0 if next_state == goal else q[next_state, legal[next_state]].max()
        td = reward + gamma * future - q[current_state, current_action]
        q[current_state, current_action] += alpha * td

        if update % 1000 == 0 or update == max_updates:
            values = np.array([q[s, legal[s]].max() for s in actions])
            values[goal] = 0.0
            targets = rewards + gamma * values[destinations]
            residual = np.max(np.abs(targets - q[sources, destinations]))
            if residual < 1e-9:
                return q

    raise RuntimeError(f"Q-learning did not converge within {max_updates} updates.")


# Part 3: Deployment
def _greedy_route(start, goal, q):
    """Extract a route from learned Q values, with legal moves and a loop guard."""
    path = [state_to_location[start]]
    visited = {start}
    current = start
    while current != goal:
        legal = np.flatnonzero(R[current] > 0)
        scores = q[current, legal]
        # Resolve numerical ties in alphabetical order for reproducible routes.
        best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-8))
        current = int(legal[best[0]])
        if current in visited:
            raise RuntimeError("The learned policy contains a cycle; check training convergence.")
        path.append(state_to_location[current])
        visited.add(current)
    return path


def route(starting_location, ending_location):
    """Return a shortest legal route, including both endpoints."""
    start, goal = _state(starting_location), _state(ending_location)
    if start == goal:
        return [starting_location]
    q = train_q(ending_location)
    return _greedy_route(start, goal, q)


def plot_route(path, *, save_path=None, show=True):
    """Draw the learned route on the ASCII map's 3-by-4 grid.

    Arrows show the deployed route, not the random training transitions.
    Use show=False and save_path='route.png' for a noninteractive export.
    Returns (figure, axes) so callers can further customize the plot.
    Requires matplotlib (python -m pip install matplotlib).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    path = list(path)
    if not path:
        raise ValueError("The route must contain at least one location.")
    states = [_state(location) for location in path]
    for left, right in zip(states, states[1:]):
        if R[left, right] <= 0:
            raise ValueError("The route contains a move through a wall.")

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"route": "#dbeafe", "start": "#bbf7d0", "goal": "#fed7aa"}
    for state in actions:
        row, col = divmod(state, 4)
        color = colors["route"] if state in states else "#f8fafc"
        if state == states[0]:
            color = colors["start"]
        if state == states[-1]:
            color = colors["goal"]
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=color,
                               edgecolor="#cbd5e1", linewidth=0.8))
        ax.text(col + 0.18, row + 0.22, state_to_location[state],
                ha="center", va="center", fontsize=17, weight="bold", color="#0f172a")

        # Internal walls come directly from the same legal-move matrix.
        if col < 3 and R[state, state + 1] == 0:
            ax.plot([col + 1, col + 1], [row, row + 1], color="#334155", lw=4)
        if row < 2 and R[state, state + 4] == 0:
            ax.plot([col, col + 1], [row + 1, row + 1], color="#334155", lw=4)

    # Preserve the original ASCII outer openings at D, E, and below K.
    ax.plot([0, 4], [0, 0], color="#334155", lw=4)
    for x, rows in ((0, (0, 2)), (4, (1, 2))):
        for row in rows:
            ax.plot([x, x], [row, row + 1], color="#334155", lw=4)
    for left, right in ((0, 2), (3, 4)):
        ax.plot([left, right], [3, 3], color="#334155", lw=4)

    centers = [(state % 4 + 0.5, state // 4 + 0.56) for state in states]
    for start, end in zip(centers, centers[1:]):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.5,
                                    mutation_scale=18, shrinkA=13, shrinkB=13))
    for step, (x, y) in enumerate(centers):
        ax.text(x, y, str(step), ha="center", va="center", fontsize=10,
                color="white", weight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", fc="#2563eb", ec="white"))

    if len(path) == 1:
        ax.text(centers[0][0], centers[0][1] + 0.26, "Start / Goal",
                ha="center", fontsize=10)
    else:
        for (x, y), label in ((centers[0], "Start"), (centers[-1], "Goal")):
            ax.text(x + 0.23, y + 0.26, label, ha="center", fontsize=10)

    ax.set(xlim=(-0.08, 4.08), ylim=(3.08, -0.08), aspect="equal")
    ax.axis("off")
    ax.set_title(f"Learned warehouse route: {path[0]} to {path[-1]}\n"
                 f"{len(path) - 1} moves | numbered steps start at 0", pad=18)
    ax.legend(handles=[Patch(facecolor=colors[key], label=label)
                       for key, label in (("start", "Start"), ("goal", "Goal"),
                                          ("route", "Visited location"))],
              loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)
    fig.text(0.5, 0.045, " → ".join(path), ha="center", fontsize=12)
    fig.text(0.5, 0.015, "Thick lines = walls; thin lines = passages. Outside openings are not states.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    Q = train_q("G")
    r = _greedy_route(_state("E"), _state("G"), Q)
    print("Q matrix for destination G (zeros are forbidden moves or terminal values):")
    print(np.round(Q, 2))
    print("Best route:", " -> ".join(r))
    print("Number of moves:", len(r) - 1)
    plot_route(r)
