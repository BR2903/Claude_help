# -*- coding: utf-8 -*-
"""qlearning4WarehouseProblem.py

#Warehouse Process Optimization Problem by Qlearning
"""

import numpy as np


#qlearning gamma and alpha settings

gamma = 0.75 #Discount Factor
alpha = 0.9  #Learning Rate

"""# Part 1 Environment Settings
+-----+-----+-----+-----+
|  A     B     C  |  D   
+-----+     +     +     +
   E  |  F  |  G     H  |
+     +     +-----+     +
|  I     J     K     L  |
+-----+-----+     +-----+

"""

#Part 1 Env Settings

# States

location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}


# Actions: GoToA, GoToB, ... , GoToL
actions = [0,1,2,3,4,5,6,7,8,9,10,11]


# Rewards:
                #GoToA,GoToB,...,GoToL
         #StateA
         #StateB
         #...
         #StateL

              #GoTo:
              #A,B,C,D,E,F,G,H,I,J,K,L.  #States
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0], #A
              [1,0,1,0,0,1,0,0,0,0,0,0], #B
              [0,1,0,0,0,0,1,0,0,0,0,0], #C
              [0,0,0,0,0,0,0,1,0,0,0,0], #D
              [0,0,0,0,0,0,0,0,1,0,0,0], #E
              [0,1,0,0,0,0,0,0,0,1,0,0], #F
              [0,0,1,0,0,0,0,1,0,0,0,0], #G
              [0,0,0,1,0,0,1,0,0,0,0,1], #H
              [0,0,0,0,1,0,0,0,0,1,0,0], #I
              [0,0,0,0,0,1,0,0,1,0,1,0], #J
              [0,0,0,0,0,0,0,0,0,1,0,1], #K
              [0,0,0,0,0,0,0,1,0,0,1,0]]) #L

"""# Part 2 Q Matrix

#Q Learning Algorithm (Pseudocode)

**Initialization**

For all pairs of states s and actions a, the Q values are initialized to 0:

∀s∈S, a∈A, Q0(s,a)=0

We start in the initial state s0. We perform a possible random action and arrive at the first state s1.

**For each instant t≥1**, we will repeat the following a certain number of times:

1. We select a random state st from our 12 possible states:

    St = random(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

2. We perform a random action at that can lead to the next possible state, i.e., such that R(st,at) > 0:

     at = random(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) t.q. R(st,at) > 0:

3. We reach the next state st+1 and get the reward R(st,at)

4. Compute Temporal Difference TDt(st,at) :

    TDt(st,at) = R(st,at) + γ maxa(Q(st+1,a)) - Q(st,at)

5. We update the Q value by applying the Bellman equation:

    Qt(st,at) = Qt-1(st,at) + α TDt(st,at)
"""

#Q Initialization
Q = np.array(np.zeros([12, 12]))

# for i in range(1000):
#     current_state = np.random.randint(0, 12)
#     playable_actions = []
#     for j in range(12):
#         if R[current_state, j] > 0:
#             playable_actions.append(j)

#     current_action = np.random.choice(playable_actions)
#     next_state = current_action #only for this specific case: GoToX = StateX
#     TD = R[current_state, current_action] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, current_action]
#     #Q[current_state, current_action] += alpha * TD
#     Q[current_state, current_action] = Q[current_state, current_action] + alpha * TD

#print(Q.astype(int))

"""# Part 3 Deployment"""

#Inverse transform from states to locations 
state_to_location = {state: location for location, state in location_to_state.items()}

#Training Q inside route to consider different end locations 
def route(starting_location, ending_location):
  R_new = np.copy(R)
  ending_state = location_to_state[ending_location]
  # Assign high reward at ending_state
  R_new[ending_state, ending_state] = 10


  
  #Q Initialization
  Q = np.array(np.zeros([12, 12]))
  ##ToDo 
  ##Training phase


  #Define Route
  route = [starting_location]
  next_location = starting_location
  while (next_location != ending_location):
    starting_state = location_to_state[starting_location]
    next_state = np.argmax(Q[starting_state,])
    next_location = state_to_location[next_state]
    route.append(next_location)
    starting_location = next_location
  return route


r = route ('D','A')
print (r)