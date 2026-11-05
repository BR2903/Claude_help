"""
Policy Improvement on FrozenLake-v1 (Gymnasium)

Implements the Policy Improvement algorithm:

    Require: state value function V_pi for pi, discount rate gamma
    Initialize: Q_pi(s, a) = 0, for all s in S, a in A(s)

    # Compute the state-action value function using the estimated state value function
    for s in S:
        for a in A(s):
            Q_pi(s, a) <- R(s, a) + gamma * sum_{s'} P(s'|s, a) * V_pi(s')

    # Compute an improved deterministic policy
    for s in S:
        A* in argmax_a Q_pi(s, a), breaking ties deterministically
        for a in A(s):
            pi'(a|s) = 1 if a == A* else 0
"""

import time

import numpy as np
import gymnasium as gym


def build_uniform_policy(nS, nA):
    return np.ones((nS, nA)) / nA


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_improvement(env, V, gamma=0.9):
    """Compute Q_pi from V_pi, then return the greedy deterministic policy pi'."""
    nS = env.observation_space.n
    nA = env.action_space.n
    P = env.unwrapped.P

    # Compute Q_pi(s, a) = R(s, a) + gamma * sum_s' P(s'|s,a) * V_pi(s')
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            q_sa = 0
            for prob, next_state, reward, done in P[s][a]:
                q_sa += prob * (reward + gamma * V[next_state])
            Q[s, a] = q_sa

    # Build improved deterministic policy: pi'(a|s) = 1 for A* = argmax_a Q(s,a), else 0
    policy_prime = np.zeros((nS, nA))
    for s in range(nS):
        best_action = np.argmax(Q[s])  # ties broken deterministically (first max)
        policy_prime[s, best_action] = 1.0

    return policy_prime, Q


def print_value_grid(V, env):
    desc = env.unwrapped.desc
    rows, cols = desc.shape
    V_grid = V.reshape(rows, cols)
    for r in range(rows):
        rowvals = ["{:+.3f}".format(v) for v in V_grid[r]]
        print("  ".join(rowvals))


ACTION_ARROWS = {0: "<", 1: "v", 2: ">", 3: "^"}


def print_policy_grid(policy, env):
    """Print the deterministic greedy action for each state as an arrow grid."""
    desc = env.unwrapped.desc
    rows, cols = desc.shape
    best_actions = np.argmax(policy, axis=1).reshape(rows, cols)
    for r in range(rows):
        row_desc = desc[r]
        row_str = []
        for c in range(cols):
            cell = row_desc[c].decode("utf-8") if isinstance(row_desc[c], bytes) else row_desc[c]
            if cell in ("H", "G"):
                row_str.append(cell)
            else:
                row_str.append(ACTION_ARROWS[best_actions[r, c]])
        print("  ".join(row_str))


def demonstrate_policy(policy, gamma, n_episodes=3, max_steps=100, sleep_time=0.5):
    """Render episodes where the agent follows the greedy deterministic policy."""
    demo_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    for episode in range(1, n_episodes + 1):
        obs, info = demo_env.reset()
        print(f"\n--- Episode {episode} ---")
        for step in range(max_steps):
            action = np.argmax(policy[obs])
            obs, reward, terminated, truncated, info = demo_env.step(action)
            time.sleep(sleep_time)
            if terminated or truncated:
                outcome = "reached the goal!" if reward == 1 else "fell in a hole or ran out of moves."
                print(f"Episode {episode} finished after {step + 1} steps, {outcome}")
                break
    demo_env.close()


if __name__ == "__main__":
    gamma = 0.9
    theta = 1e-6

    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
    nS = env.observation_space.n
    nA = env.action_space.n

    # Start from a uniform random policy and evaluate it
    policy = build_uniform_policy(nS, nA)
    V = policy_evaluation(env, policy, gamma, theta)

    print("State values V_pi for the uniform random policy:")
    print_value_grid(V, env)

    print("\nUniform policy (greedy-arrow view of the argmax action, for reference):")
    print_policy_grid(policy, env)

    # Improve the policy using V_pi
    policy_prime, Q = policy_improvement(env, V, gamma)

    print("\nState-action values Q_pi(s, a):")
    print(Q)

    print("\nImproved deterministic policy pi':")
    print(policy_prime)

    print("\nImproved policy (arrow view):")
    print_policy_grid(policy_prime, env)

    env.close()

    # Demonstrate how the agent moves under the improved policy
    demonstrate_policy(policy_prime, gamma)