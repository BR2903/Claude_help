# AI Project

Coursework for **Intelligent Agents** (reinforcement learning), Aispur Santiago and Amaya Bryan.
Folders follow the order of the course.

| Folder | Contents |
|---|---|
| `01_dynamic_programming/` | Bellman equations on FrozenLake: stochastic Bellman / value iteration (`AispurS_Stochastic_Bellman.py`, outputs in `rl101_results/`), policy evaluation lab 1.4, DP policy evaluation and improvement, value-iteration defense guide (PDF) |
| `02_frozenlake_basics/` | Custom FrozenLake maps (`rl101-customfrozen-01..04.py`), transition-model inspection (`frozenlake_inspect_transitions.py`), BFS shortest safe path |
| `03_monte_carlo/` | First-visit and every-visit Monte Carlo (`rl101-EveryVisit.py` writes `rewards.csv` to the current directory) |
| `04_temporal_difference/` | `frozenlake_td/`: TD(0), SARSA, Q-learning scripts · `cliff_walk_sarsa/`: Cliff Walking SARSA vs Q-learning report (LaTeX + PDF) · `warehouse_qlearning/`: warehouse Q-learning templates and tests · `frozenlake_competition/`: midterm competition solution and studies |
| `05_dqn_cartpole/` | `notebooks/`: NN fundamentals lab 2.2, CartPole activity, DQN replay/target-net labs, Double DQN lab, NN/cores comparisons · `defense/`: CartPole defense slides (`.tex`/`.pptx`, figures in `slides_figures/`) and project analyses |
| `rl_dqn_project/` | Final project, tabular RL vs DQN (see `Project Guide_ RL and DQN.pdf` and `rl_dqn_project/README.md`) |
| `misc/` | Files from other classes (math guides, DB diagram, `multiply.py`) |

## Environment

One shared virtual environment at `.venv/` (Python 3.13, gymnasium 1.3 with Box2D, PyTorch CPU, matplotlib, pandas, scipy, pytest).

```fish
cd ~/Documents/Intelligent\ Agents
source .venv/bin/activate.fish
cd 04_temporal_difference/warehouse_qlearning; python -m unittest test_qlearning_warehouse
```

Scripts write their outputs next to themselves or into the current directory, so run each from its own folder.
