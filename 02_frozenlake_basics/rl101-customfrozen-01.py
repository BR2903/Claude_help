import gymnasium as gym 

#Create the Frozen Lake environment 
env = gym.make('FrozenLake-v1') 

#Compute the size of the action space 
num_actions = env.action_space.n 

#Compute the size of the state space 
num_states = env.observation_space.n 

print("Number of actions:", num_actions)  
print("Number of states:", num_states) 