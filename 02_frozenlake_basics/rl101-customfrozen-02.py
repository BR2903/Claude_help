import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi")

print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)

# Try random exploration
state, _ = env.reset()
for t in range(10):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}: state={state}, action={action}, reward={reward}, next={next_state}")
    state = next_state
    if terminated or truncated:
        state, _ = env.reset()