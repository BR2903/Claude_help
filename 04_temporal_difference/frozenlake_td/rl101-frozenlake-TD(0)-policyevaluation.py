import gymnasium as gym
import numpy as np
from collections import defaultdict
import random


# Parameters
num_steps = 20000
gamma = 0.9
alpha = 0.1
seed = 40
random.seed(seed)
np.random.seed(seed)

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
env.action_space.seed(seed)
env.reset(seed=seed)

def random_policy(state):
    return env.action_space.sample()

def td0_policy_evaluation(env, policy, num_steps, gamma=0.9, alpha=0.1):
    V = defaultdict(float)
    state, _ = env.reset()
    i = 0
    while i < num_steps:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        i += 1
        target = reward if (terminated or truncated) else reward + gamma * V[next_state]
        V[state] += alpha * (target - V[state])
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
    return V

def td0_policy_evaluation_q(env, policy, num_steps, gamma=0.9, alpha=0.1):
    Q = defaultdict(float)
    state, _ = env.reset()
    action = policy(state)
    i = 0
    while i < num_steps:
        next_state, reward, terminated, truncated, _ = env.step(action)
        i += 1
        next_action = policy(next_state)
        # on-policy TD target uses the action the fixed policy would take next
        target = reward if (terminated or truncated) else reward + gamma * Q[(next_state, next_action)]
        Q[(state, action)] += alpha * (target - Q[(state, action)])
        state, action = next_state, next_action
        if terminated or truncated:
            state, _ = env.reset()
            action = policy(state)
    return Q

def print_values_as_matrix(V, env):
    nrow, ncol = env.unwrapped.desc.shape
    for r in range(nrow):
        row_values = [V[r * ncol + c] for c in range(ncol)]
        print(" ".join(f"{v:6.3f}" for v in row_values))

def print_q_as_matrix(Q, env):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    header = "State  " + " ".join(f"A{a:<5d}" for a in range(num_actions))
    print(header)
    for s in range(num_states):
        row_values = [Q[(s, a)] for a in range(num_actions)]
        print(f"{s:<6d} " + " ".join(f"{q:6.3f}" for q in row_values))

# Run TD(0) Evaluation
V = td0_policy_evaluation(env, random_policy, num_steps, gamma, alpha)

print("\nEstimated State Values (matrix format):")
print_values_as_matrix(V, env)

# Run TD(0) Evaluation for Q
Q = td0_policy_evaluation_q(env, random_policy, num_steps, gamma, alpha)
print("\nEstimated Action Values (matrix format, rows=states, columns=actions):")
print_q_as_matrix(Q, env)