import gymnasium as gym
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# Parameters
num_episodes = 5000
gamma = 0.9
seed = 40
random.seed(seed)
np.random.seed(seed)

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
env.action_space.seed(seed)
env.reset(seed=seed)

def random_policy(state):
    return env.action_space.sample()

def generate_episode(env, policy):
    episode = []
    state, _ = env.reset()
    done = False
    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
    return episode

def mc_first_visit(env, num_episodes, gamma=0.9):
    V = defaultdict(float)
    returns = defaultdict(list)

    for ep in range(num_episodes):
        episode = generate_episode(env, random_policy)
        visited_states = set()
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in visited_states:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited_states.add(state)
    return V

# Run MC Evaluation
V = mc_first_visit(env, num_episodes)
print("Estimated State Values:")
for s, v in sorted(V.items()):
    print(f"State {s}: {v:.3f}")

""" def plot_value_grid(V, grid_size=4):
    grid = np.array([V[s] for s in range(grid_size * grid_size)]).reshape(grid_size, grid_size)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap="viridis")
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white")
    plt.title("Estimated State Values")
    plt.colorbar()
    plt.show()

def plot_policy_grid(policy, grid_size=4):
    # FrozenLake action mapping: 0=Left, 1=Down, 2=Right, 3=Up
    arrows = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}
    grid = np.array([arrows[policy[s]] for s in range(grid_size * grid_size)]).reshape(grid_size, grid_size)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.zeros((grid_size, grid_size)), cmap="Greys", vmin=0, vmax=1)
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, grid[i, j], ha="center", va="center", fontsize=20)
    plt.title("Greedy Policy from V")
    plt.xticks([])
    plt.yticks([])
    plt.show()
 """
def greedy_policy_from_V(env, V, gamma=0.9):
    # Uses FrozenLake's known transition model for a one-step lookahead (V alone has no action info)
    P = env.unwrapped.P
    policy = {}
    Q = {}
    for state in P:
        action_values = []
        for action in P[state]:
            q = sum(prob * (reward + gamma * V[next_state] * (not terminated))
                    for prob, next_state, reward, terminated in P[state][action])
            action_values.append(q)
        Q[state] = action_values
        policy[state] = int(np.argmax(action_values))
    return policy, Q

def print_results(V, Q, policy):
    print("\nState Values (V):")
    for s in sorted(V):
        print(f"State {s}: {V[s]:.3f}")

    print("\nAction Values (Q):")
    for s in sorted(Q):
        print(f"State {s}: {[f'{q:.3f}' for q in Q[s]]}")

    print("\nLearned Policy:")
    for s in sorted(policy):
        print(f"State {s}: {policy[s]}")

def print_policy_matrix(env, policy, grid_size=4):
    # 'H' (hole) and 'G' (goal) are terminal states, they have no action
    desc = env.unwrapped.desc.astype(str)
    action_letters = {0: "L", 1: "D", 2: "R", 3: "U"}
    print("\nLearned Policy (matrix):")
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            cell = desc[i][j]
            if cell in ("H", "G"):
                row.append(cell)
            else:
                row.append(action_letters[policy[i * grid_size + j]])
        print(" ".join(row))

#plot_value_grid(V)
learned_policy, Q = greedy_policy_from_V(env, V, gamma)
print("\nLearned Policy (from V):")
print_results(V, Q, learned_policy)
print_policy_matrix(env, learned_policy)
#plot_policy_grid(learned_policy)

def run_policy(env, policy):
    total_steps = 0
    total_reward = 0
    state, _ = env.reset()
    done = False
    while not done:
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        total_steps += 1
        total_reward += reward
        done = terminated or truncated
    return total_steps, total_reward

# --- Test run: average steps and reward over multiple episodes ---
num_test_episodes = 1000
test_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
steps_list = []
rewards_list = []
for _ in range(num_test_episodes):
    steps, reward = run_policy(test_env, learned_policy)
    steps_list.append(steps)
    rewards_list.append(reward)
test_env.close()
print(f"\nAverage over {num_test_episodes} test episodes:")
print(f"Average steps: {np.mean(steps_list):.2f}")
print(f"Average reward: {np.mean(rewards_list):.2f}")

# --- Visualization: human render ---
#print("\nExample Run (greedy policy derived from V, human render):")
#env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
#total_steps, total_reward = run_policy(env, learned_policy)
#env.close()
#print(f"Total steps: {total_steps}, Total reward: {total_reward}")