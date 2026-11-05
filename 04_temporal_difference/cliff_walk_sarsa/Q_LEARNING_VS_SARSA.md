# Q-learning versus SARSA on Cliff Walking

## Files found and implementation choice

The workspace already contained `rl101-frozenlake-TD-Qlearning-Epsilon.py`.
That program is for FrozenLake and depends on Gymnasium, so it does not meet
the requirement for a from-scratch Cliff Walking solution using only Python
and NumPy. The new implementation is `cliff_walk_q_learning.py`. It defines
the grid dynamics directly and includes both Q-learning and SARSA so they can
be trained with identical hyperparameters and seeds. NumPy implements the
environment and learning; Matplotlib is used only to display the paths.

## Shared setup

- Grid: 4 rows by 12 columns.
- Start: `(3, 0)`; goal: `(3, 11)`.
- Cliff: `(3, 1)` through `(3, 10)`.
- Rewards: `-1` per step and `-100` for a fall.
- A fall returns the agent to the start without ending the episode.
- Q-table shape: `(4, 12, 4)`.
- Parameters: alpha `0.5`, gamma `1.0`, epsilon `0.1`.
- Training: 500 episodes for each of 50 seeds.
- Greedy ties during training are broken uniformly at random.

## The essential code difference

Q-learning uses the largest next-state value in its target:

```python
target = reward + GAMMA * np.max(q_values[next_row, next_col])
```

This is **off-policy**. The agent behaves epsilon-greedily during training,
but the update evaluates an ideal greedy action at the next state.

SARSA first samples the action it will actually take and uses its value:

```python
next_action = epsilon_greedy(q_values, next_state, epsilon, rng)
target = reward + GAMMA * q_values[next_row, next_col, next_action]
```

This is **on-policy**. Its learned values include the risk created by continued
exploration.

## Measured comparison

| Measurement | Q-learning | SARSA |
|---|---:|---:|
| Seed-0 greedy route | 13 steps | 17 steps |
| Dominant crossing row | 2 (beside cliff) | 0 (top row) |
| Successful greedy routes | 50/50 | 44/50 |
| Mean successful route length | 13.00 | 17.00 |
| Last-100 mean training return | -48.92 | -25.16 |
| Last-100 cliff falls per episode | 0.324 | 0.041 |

Q-learning learns the shortest greedy route: up once, right across row 2, and
down into the goal. That route is optimal when exploration is disabled, but
the epsilon-greedy training behavior occasionally moves down into the cliff.
Consequently, Q-learning has a worse online training return and more falls.

SARSA anticipates its own exploratory actions. It moves farther from the cliff
and accepts four extra steps in seed 0, but falls much less often and therefore
earns a better return while training. Thus “better” depends on the evaluation:
Q-learning produces the shortest final greedy route, while SARSA is safer when
epsilon remains positive.

## Running the comparison

From the project folder:

```powershell
python .\cliff_walk_sarsa\cliff_walk_q_learning.py
```

The script prints both greedy paths, policy grids, route statistics, returns,
and cliff-fall rates. It also creates:

- `figures/q_learning_greedy_path.png`
- `figures/q_learning_vs_sarsa_paths.png`

The first figure displays the Q-learning path by itself. The second places the
Q-learning and SARSA paths in vertically aligned grids for direct comparison.
