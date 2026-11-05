import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

state = 6
action = 0
print(env.unwrapped.P[state][action])