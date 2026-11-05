import gymnasium as gym
import numpy as np
import time



#policy = np.ones([nS, nA]) / nA  # uniform random policy

def build_uniform_policy(nS, nA):
    return np.ones((nS, nA)) / nA

def build_greedy_policy_towards_action(nS, nA, preferred_action):
    """Deterministic policy always take preferred_action where possible."""
    policy = np.zeros((nS, nA))
    policy[:, preferred_action] = 1.0
    return policy


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    """
    Iterative policy evaluation.

    Args:
      env: gymnasium env (FrozenLake-v1)
      policy: array shape (nS, nA) with π(a|s)
      gamma: discount factor
      theta: convergence threshold

    Returns:
      V: numpy array shape (nS,) with state values under policy
    
    
    Follow the following pseudocode:
    Input: Policy π(s,a), Transition model P(s’|s,a), Rewards R(s,a,s’), Discount γ, Threshold θ
    Initialize V(s) = 0 for all states s
    Repeat:
        Δ = 0
        For each state s:
            v = V(s)
            V(s) = sum_a π(a|s) * sum_{s’,r} P(s’,r|s,a) * [ r + γ * V(s’) ]
            Δ = max(Δ, |v - V(s)|)
    Until Δ < θ
    Return V
    """
    

    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0

        for state in range(env.observation_space.n):
            old_value = V[state]
            new_value = 0

            for action in range(env.action_space.n):
                action_value = 0

                for transition in env.unwrapped.P[state][action]:
                    probability, next_state, reward, done = transition[:4]
                    future_value = 0 if done else gamma * V[next_state]
                    action_value += probability * (reward + future_value)

                new_value += policy[state, action] * action_value

            V[state] = new_value
            delta = max(delta, abs(old_value - new_value))

        if delta < theta:
            return V


# Run evaluation

# Create FrozenLake with human render
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Parameters
gamma = 0.9
theta = 1e-6
nS = env.observation_space.n
nA = env.action_space.n


policy = build_uniform_policy(nS, nA)

#prefer_right = 2
#policy = build_greedy_policy_towards_action(nS, nA, prefer_right)

V = policy_evaluation(env, policy, gamma, theta)
print("Policy")
print(policy)
print("\nFinal State Values:")
print(V.reshape((int(np.sqrt(nS)), -1)))  # show in grid form if square


# # Demonstration episode (to visualize agent moves under policy)
# obs, info = env.reset()
# done = False
# while not done:
#     action = np.random.choice(env.action_space.n, p=policy[obs])
#     obs, reward, terminated, truncated, info = env.step(action)
#     time.sleep(0.9)  # Slow down so you can watch
#     done = terminated or truncated

# env.close()