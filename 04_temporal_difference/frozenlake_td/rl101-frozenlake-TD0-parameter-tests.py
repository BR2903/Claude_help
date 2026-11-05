import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

# Base parameters (kept fixed except for the one under test in each experiment)
BASE_NUM_STEPS = 20000
BASE_GAMMA = 0.9
BASE_ALPHA = 0.1
SEED = 40


def make_env(seed=SEED):
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env


def random_policy(env):
    return lambda state: env.action_space.sample()


def td0_policy_evaluation(env, policy, num_steps, gamma, alpha):
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


def values_as_matrix(V, env):
    nrow, ncol = env.unwrapped.desc.shape
    return [[V[r * ncol + c] for c in range(ncol)] for r in range(nrow)]


def print_matrix(matrix, label):
    print(f"  {label}")
    for row in matrix:
        print("    " + " ".join(f"{v:6.3f}" for v in row))


def run_experiment(title, param_name, values):
    print(f"\n=== {title} (varying {param_name}) ===")
    for value in values:
        random.seed(SEED)
        np.random.seed(SEED)
        env = make_env()
        kwargs = dict(num_steps=BASE_NUM_STEPS, gamma=BASE_GAMMA, alpha=BASE_ALPHA)
        kwargs[param_name] = value
        V = td0_policy_evaluation(env, random_policy(env), **kwargs)
        print_matrix(values_as_matrix(V, env), f"{param_name} = {value}")
        env.close()


# Test 1: learning rate (alpha) - controls how fast new estimates overwrite old ones.
# Low alpha -> slow, smoother convergence; high alpha -> faster but noisier/less stable estimates.
run_experiment("Test 1: Effect of learning rate alpha", "alpha", [0.01, 0.1, 0.5])

# Test 2: discount factor (gamma) - controls how much future rewards are valued.
# Low gamma -> myopic (values near zero, only immediate reward matters);
# high gamma -> far-sighted (values propagate further from the goal state).
run_experiment("Test 2: Effect of discount factor gamma", "gamma", [0.5, 0.9, 0.99])

# Test 3: number of training steps (num_steps) - controls how much experience is collected.
# Fewer steps -> under-trained, noisy/incomplete value estimates;
# more steps -> better convergence toward the true value function.
run_experiment("Test 3: Effect of number of training steps", "num_steps", [2000, 20000, 200000])
