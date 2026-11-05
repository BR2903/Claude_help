import gymnasium as gym
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# Parameters
num_steps = 100000
gamma = 0.9
alpha = 0.1
seed = 40
random.seed(seed)
np.random.seed(seed)

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
env.action_space.seed(seed)
env.reset(seed=seed)
num_actions = env.action_space.n

def random_policy(state):
    return env.action_space.sample()

def sarsa_policy_evaluation(env, policy, num_steps, gamma=0.9, alpha=0.1):
    Q = defaultdict(float)
    state, _ = env.reset()
    action = policy(state)
    i = 0
    while i < num_steps:
        next_state, reward, terminated, truncated, _ = env.step(action)
        i += 1
        next_action = policy(next_state)
        delta = reward if (terminated or truncated) else reward + gamma * Q[(next_state, next_action)]
        Q[(state, action)] += alpha * (delta - Q[(state, action)])
        state, action = next_state, next_action
        if terminated or truncated:
            state, _ = env.reset()
            action = policy(state)
    return Q

# Run SARSA Evaluation
Q = sarsa_policy_evaluation(env, random_policy, num_steps, gamma, alpha)
print("Estimated Action Values:")
for (s, a), q in sorted(Q.items()):
    print(f"State {s}, Action {a}: {q:.3f}")

def greedy_policy_from_Q(Q, num_states, num_actions):
    policy = {}
    for s in range(num_states):
        action_values = [Q[(s, a)] for a in range(num_actions)]
        policy[s] = int(np.argmax(action_values))
    return policy

def print_results(Q, policy, num_states, num_actions):
    print("\nAction Values (Q):")
    for s in range(num_states):
        print(f"State {s}: {[f'{Q[(s, a)]:.3f}' for a in range(num_actions)]}")

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

num_states = env.observation_space.n
learned_policy = greedy_policy_from_Q(Q, num_states, num_actions)
print("\nLearned Policy (from Q):")
print_results(Q, learned_policy, num_states, num_actions)
print_policy_matrix(env, learned_policy)

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
#print("\nExample Run (greedy policy derived from Q, human render):")
#env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
#total_steps, total_reward = run_policy(env, learned_policy)
#env.close()
#print(f"Total steps: {total_steps}, Total reward: {total_reward}")
