import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

# Parameters
num_episodes = 5000
gamma = 0.9
seed = 40
random.seed(seed)
np.random.seed(seed)

# Environment
env = gym.make("FrozenLake8x8-v1", is_slippery=True, render_mode="ansi")
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

# def mc_first_visit(env, num_episodes, gamma=0.9):
#     V = defaultdict(float)
#     returns = defaultdict(list)

#     for ep in range(num_episodes):
#         episode = generate_episode(env, random_policy)
#         visited_states = set()
#         G = 0
#         for t in reversed(range(len(episode))):
#             state, action, reward = episode[t]
#             G = gamma * G + reward
#             if state not in visited_states:
#                 returns[state].append(G)
#                 V[state] = np.mean(returns[state])
#                 visited_states.add(state)
#     return V

def mc_first_visit_Q_policy(env, num_episodes, gamma=0.9):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    policy = np.random.randint(env.action_space.n, size=env.observation_space.n)

    for ep in range(num_episodes):
        print(f"episode:{ep}")
        episode = generate_episode(env, random_policy)
        visited_states = set()
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in visited_states:
                returns[state,action].append(G)
                #Compute Q Values
                Q[state][action] = np.mean(returns[(state,action)])
                policy[state] = np.argmax(Q[state])  # policy improvement

    return policy, Q

# Run MC Evaluation
policy, Q = mc_first_visit_Q_policy(env, num_episodes)

print("Final Policy:")
print(policy.reshape(8,8))

print("Final Q-values:")
for s in range(env.observation_space.n):
    print(f"State {s}: {Q[s]}")

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
test_env = gym.make("FrozenLake8x8-v1", is_slippery=True, render_mode="ansi")
steps_list = []
rewards_list = []
for _ in range(num_test_episodes):
    steps, reward = run_policy(test_env, policy)
    steps_list.append(steps)
    rewards_list.append(reward)
test_env.close()
print(f"\nAverage over {num_test_episodes} test episodes:")
print(f"Average steps: {np.mean(steps_list):.2f}")
print(f"Average reward: {np.mean(rewards_list):.2f}")

# --- Visualization: human render ---
print("\nExample Run (learned policy, human render):")
env = gym.make("FrozenLake8x8-v1", is_slippery=True, render_mode="human")
total_steps, total_reward = run_policy(env, policy)
env.close()
print(f"Total steps: {total_steps}, Total reward: {total_reward}")