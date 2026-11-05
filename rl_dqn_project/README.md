# Tabular RL vs DQN — Tier 1 & Tier 2 project

**Team:** Aispur Santiago · Amaya Bryan — *Intelligent Agents* (PhD Armas Rolando)

From-scratch tabular Q-learning and DQN (PyTorch), compared on **cumulative environment steps**.
Every configuration runs with **10 fixed seeds** and gets periodic **frozen-weight greedy evaluation**.
Plots show 95 % confidence bands.

**Start here:** [`ANALYSIS.md`](ANALYSIS.md) (what we learned) · [`report/main.pdf`](report/main.pdf) (8-page report)
· [`DEFENSE_GUIDE.md`](DEFENSE_GUIDE.md) (code walkthrough and expected questions)

| Study | Environment | Script | What it shows |
|---|---|---|---|
| Tier 1 | CartPole-v1 | `experiments/tier1_cartpole.py` | tabular (6/10/20 bins) vs DQN; pole-angle bin boundaries; DQN ε-decay rate |
| Tier 1 (sparse reward) | MountainCar-v0 | `experiments/tier1_mountaincar.py` | negative result: DQN never sees the flag; optimistic tabular does |
| Tier 2 | LunarLander-v3 | `experiments/tier2_ablation.py` | ablations (no target net, no replay) + Double / Dueling DQN |
| Tier 2 (hyper-parameters) | LunarLander-v3 | `experiments/tier2_hparams.py` | learning rate {1e-4, 5e-4, 2e-3} and target-update period {250, 1000, 4000}, one at a time around the default |
| Final evaluation | all three | `experiments/eval100.py` | final weights of every run, 100 greedy episodes, against the official 100-episode thresholds |
| Agent videos | all three | `experiments/make_gifs.py` | one GIF per main agent and per instructive failure → `results/videos/` |
| Support | CartPole-v1 | `experiments/obs_ranges.py` | empirical velocity ranges that justify the clip bounds |

## Reproduce everything (one command)

```bash
cd "~/Documents/Intelligent Agents/rl_dqn_project"
source ../.venv/bin/activate.fish        # or: source ../.venv/bin/activate (bash)
python experiments/run_all.py             # 190 runs, ~25 CPU-hours; cached runs are skipped; then eval100 + GIFs
python analysis.py                        # every number in ANALYSIS.md / the report -> results/analysis.md
```

Fresh machine setup:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Useful flags (all experiment scripts and `run_all.py` accept them):

| Flag | Meaning |
|---|---|
| `--quick` | 2-seed smoke test with tiny budgets → `results/quick/` (never overwrites real results) |
| `--plot-only` | rebuild figures/tables from cached runs (seconds) |
| `--force` | recompute even if a run is cached |
| `--seeds N --seed-start K` | run seeds K … K+N−1, so several machines can split the seeds; then copy the `results/` folders together |
| `--workers N` | parallel processes (default: cores − 1; each run uses 1 thread) |

Other commands:

```bash
python -m pytest -q tests                              # 10 sanity tests (update rules, gather, targets, reproducibility)
python experiments/watch.py --env LunarLander-v3       # watch the trained agent (pygame window)
python experiments/watch.py --env CartPole-v1 --tabular --bins 10
cd report && latexmk -pdf main.tex                     # build the report PDF
```

## Repository layout

```
agents/
  tabular.py        TabularAgent: ε-greedy Q-learning on a discretized state
  dqn.py            QNetwork (MLP, optional dueling head), ReplayBuffer, DQNAgent (ablation switches)
envs/
  discretizers.py   state binning (uniform or hand-made edges), CartPole/MountainCar bounds + justification
common/
  loop.py           training loop (step-indexed logs) + frozen greedy evaluation every N steps
  runner.py         parallel (config × seed) runner, JSON cache, saves weights
  stats.py          step-aligned curves, Student-t 95% CI, summary metrics
  plotting.py       mean ± CI shading, figure style
  schedules.py      linear ε schedule
  seeding.py        seeds python / numpy / torch (env seeded via reset(seed=...))
  policy.py         rebuilds the greedy policy of any finished run from its JSON spec + weights
experiments/        one script per study, run_all.py, obs_ranges.py, eval100.py, make_gifs.py, watch.py
results/
  <experiment>/<config>/seed_<k>.json   full logs + exact spec of every run
  <experiment>/<config>/seed_<k>.pt     final DQN weights   (.npz = tabular Q-table)
  figures/          PDF + PNG figures used by the report
  tables/           CSV + LaTeX tables used by the report
  hosts.json        which machine ran which seeds (CPU times use the laptop runs only)
  final_eval100.json  100-episode greedy returns of the final weights of every run (eval100.py)
  videos/           GIFs of trained agents (make_gifs.py; median seed by default)
  analysis.md/.json full quantitative analysis written by analysis.py
analysis.py         statistics behind every claim (Welch tests, efficiency ratios, forgetting, divergence)
ANALYSIS.md         what we learned, with all tables
report/             LaTeX source + compiled main.pdf (8 pages: main.tex, body.tex, references.bib)
tests/              pytest sanity checks
DEFENSE_GUIDE.md    code walkthrough + answers to expected oral-defense questions/live modifications
```

## Reproducibility contract

- **Seeds:** for seed `k`, the scripts seed `random`, `numpy`, `torch.manual_seed(k)`, the agent's own
  `np.random.default_rng(k)`, `env.reset(seed=k)` (first reset) and `env.action_space.seed(k)`.
  Evaluation episodes use the disjoint seeds `1_000_000 + 1000k + i`, the same ones at every checkpoint.
- Every result JSON stores the **full spec** (hyper-parameters, budget, seed) and library versions.
  `tests/test_core.py::test_runs_are_reproducible` checks that two runs with the same seed match bit for bit.
- The x-axis is always **cumulative environment steps**. Wall-clock and CPU seconds are logged per run
  (see the `CPU s/run` table columns).
- Versions are pinned in `requirements.txt` (Python 3.13.5, CPU PyTorch).
  Other hardware can differ in the last float digits. Seeds 0–4 (and 8–9 for LunarLander) ran on the laptop and
  the rest on GitHub Codespaces (`results/hosts.json`); `--seed-start` makes that split possible.
  The Tier 2 hyper-parameter runs (`lr_*`, `target_update_freq_*`, seeds 0–4) ran on a third machine
  (Ryzen 7 5800U, Windows 11, same pinned packages), so their table has no CPU-time column.

## VS Code

Open this folder (`code .`). The interpreter is preset to `../.venv`, and pytest is configured.
**Run and Debug** has the quick experiments and the watcher. **Terminal → Run Task** has the full run,
the smoke test, the unit tests and the report build.
