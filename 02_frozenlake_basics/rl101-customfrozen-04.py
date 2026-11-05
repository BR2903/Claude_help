import gymnasium as gym
import numpy as np

# Value Iteration only.
# Keep it simple and focused on the Bellman update.

env = gym.make("FrozenLake-v1", is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

gamma = 0.95
theta = 1e-8

# Start with zero value for every state.
V = np.zeros(num_states)

# Optional: identify terminal states, but not required for the algorithm itself.
# In FrozenLake, terminal states are the goal and the holes.
terminal_states = {5, 7, 11, 12, 15}

# Value Iteration loop
for iteration in range(1, 1000):
    new_V = V.copy()
    delta = 0.0

    for state in range(num_states):
        if state in terminal_states:
            continue

        best_value = -float("inf")

        for action in range(num_actions):
            q = 0.0
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                q += prob * (reward + gamma * V[next_state])
            best_value = max(best_value, q)

        new_V[state] = best_value
        delta = max(delta, abs(new_V[state] - V[state]))

    V = new_V

    if delta < theta:
        print(f"Converged after {iteration} iterations")
        break

# Print the value table in a 4x4 grid.
print("\nValue table:")
print(np.round(V.reshape(4, 4), 3))

print("\nSimple idea:")
print("For each state, we compute:")
print("V(s) = max_a sum_s' P(s'|s,a) * (R + gamma * V(s'))")
print("This is the Bellman optimality equation.")
print("We keep updating V until the change is tiny.")
