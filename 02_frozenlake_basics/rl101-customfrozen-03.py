
import gymnasium as gym 

# Create the Frozen Lake environment 
env = gym.make('FrozenLake-v1') 

# Compute the size of the action space 
num_actions = env.action_space.n 

# Choose the initial state
state = 3

# Extract transitions for each state-action pair
for action in range(num_actions):
    transitions = env.unwrapped.P[state][action]
    # Print details of each transition
    for transition in transitions:
        probability, next_state, reward, done = transition
        print(f"Probability: {probability}, Next State: {next_state}, Reward: {reward}, Done: {done}")