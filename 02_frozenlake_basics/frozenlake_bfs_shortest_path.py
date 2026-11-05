"""Shortest safe path on a custom 4x4 FrozenLake map using Breadth-First Search.

Map:
    SFFF
    FHHH
    FFFH
    HFFG

Coordinates are (row, col). Actions follow the Gymnasium FrozenLake convention:
0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP.
"""

import json
from collections import deque

DESC = ["SFFF",
        "FHHH",
        "FFFH",
        "HFFG"]

ACTIONS = {0: ("LEFT", (0, -1)), 1: ("DOWN", (1, 0)), 2: ("RIGHT", (0, 1)), 3: ("UP", (-1, 0))}
N_ROWS, N_COLS = len(DESC), len(DESC[0])


def find_tile(tile):
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if DESC[r][c] == tile:
                return (r, c)


def neighbors(node):
    r, c = node
    for a, (name, (dr, dc)) in ACTIONS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < N_ROWS and 0 <= nc < N_COLS and DESC[nr][nc] != "H":
            yield a, name, (nr, nc)


def bfs(start, goal):
    frontier = deque([start])
    parent = {start: None}
    depth = {start: 0}
    trace = []
    while frontier:
        node = frontier.popleft()
        added = []
        for a, name, nxt in neighbors(node):
            if nxt not in parent:
                parent[nxt] = (node, a, name)
                depth[nxt] = depth[node] + 1
                frontier.append(nxt)
                added.append(nxt)
        trace.append((node, depth[node], added, list(frontier)))
        if node == goal:
            break
    path, actions = [goal], []
    while parent[path[-1]] is not None:
        prev, a, name = parent[path[-1]]
        actions.append((a, name))
        path.append(prev)
    return path[::-1], actions[::-1], trace, depth


def verify_with_gymnasium(actions):
    try:
        import gymnasium as gym
    except ImportError:
        return None
    env = gym.make("FrozenLake-v1", desc=DESC, is_slippery=False)
    state, _ = env.reset()
    reward, terminated = 0, False
    for a, _ in actions:
        state, reward, terminated, _, _ = env.step(a)
    env.close()
    return {"final_state": int(state), "reward": float(reward), "terminated": bool(terminated)}


def main():
    start, goal = find_tile("S"), find_tile("G")

    print("Node table")
    print(f"{'State':>5} | {'(r,c)':>6} | Tile | Type | Safe neighbors")
    for r in range(N_ROWS):
        for c in range(N_COLS):
            tile = DESC[r][c]
            kind = {"S": "Start", "F": "Frozen", "H": "Hole", "G": "Goal"}[tile]
            nbrs = [] if tile in "HG" else [f"{n}{p}" for _, n, p in neighbors((r, c))]
            print(f"{r * N_COLS + c:>5} | {str((r, c)):>6} | {tile:^4} | {kind:<6} | {', '.join(nbrs) or '-'}")

    path, actions, trace, depth = bfs(start, goal)

    print("\nBFS expansion trace")
    print(f"{'Step':>4} | {'Expanded':>8} | Depth | Newly discovered | Frontier after")
    for i, (node, d, added, frontier) in enumerate(trace, 1):
        print(f"{i:>4} | {str(node):>8} | {d:^5} | {str(added):<16} | {frontier}")

    print("\nShortest path")
    print(f"{'Step':>4} | {'From':>6} | Action    | {'To':>6} | Tile")
    for i, ((a, name), frm, to) in enumerate(zip(actions, path, path[1:]), 1):
        print(f"{i:>4} | {str(frm):>6} | {a} {name:<7} | {str(to):>6} | {DESC[to[0]][to[1]]}")

    result = {
        "start": list(start),
        "goal": list(goal),
        "path_length_steps": len(actions),
        "path": [list(p) for p in path],
        "path_states": [r * N_COLS + c for r, c in path],
        "actions": [a for a, _ in actions],
        "action_names": [n for _, n in actions],
        "gymnasium_check": verify_with_gymnasium(actions),
    }
    print("\nJSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
