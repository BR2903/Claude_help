# Oral defense guide

Both of us must be able to explain **every line**. Read this next to the code: each section gives
the file, the lines that matter, and what to say. The last sections hold the expected questions and
the step-by-step recipes for live modifications.

---

## 1. The 60-second story (numbers: 10 seeds, `python analysis.py`)

1. **Tier 1, CartPole.**
   - DQN reaches 475 in **56k** steps against **382k** for the best table (10 bins), so it is ~7× more
     sample-efficient, because it generalizes between nearby states.
   - Each DQN step costs **23×** more CPU, so the table gets there with **3.4× less compute**.
   - 10 bins beat 6 (too coarse to express a precise policy) and 20 (95 % of the cells never visited).
   - Every fast-decay DQN seed solved the task and then **forgot** it (fell below 200), which is why we
     evaluate during training.
2. **MountainCar (negative result).**
   - The reward is −1 everywhere until the flag, so DQN's targets make Q converge to −1/(1−γ) = −100:
     "all states are equally bad". 14/20 DQN runs never found the flag.
   - The **same** tabular agent finds the flag 10/10 times with Q₀ = 0 (optimistic) and 0/10 times with
     Q₀ = −100. Optimism, not "tabular", is what explores.
   - When DQN *did* find the flag (6/20), its Q-values **diverged** to as much as +95,000, although every
     true value is ≤ 0. That is the deadly triad. Its greedy policy still improved, because argmax only
     depends on how the values rank.
3. **Tier 2, LunarLander.**
   - **No target network:** learns just as fast (same peak, 275), but 10/10 seeds later crash below 0.
     Its final score is 176 vs 234 (p = 0.014). The moving target costs stability.
   - **No replay buffer:** peaks at 149 and ends at −376, with Q 5.9× too high and gradients 15× larger.
     Correlated, batch-1 updates break SGD's i.i.d. assumption.
   - **Double / Dueling:** no measurable difference at this scale (p = 0.98 and 0.82). We say so honestly.
4. **Methodology.**
   - 10 seeds, 95 % Student-t CIs (t₉ = 2.26), and Welch tests for the key claims.
   - The x-axis is always environment steps.
   - Greedy (ε = 0) evaluation with frozen weights every 5k/10k steps on fixed evaluation seeds, plotted
     separately from training returns.
   - CPU time comes from one machine.
   - Seeds 5–9 ran on GitHub Codespaces, and each run is an independent JSON file
     (`results/hosts.json`).

---

## 2. Code walkthrough

### `envs/discretizers.py`
- `Discretizer.__init__`: for `k` bins on `[low, high]` we keep the `k−1` **interior** edges,
  `np.linspace(low, high, k+1)[1:-1]`.
- `bin_ids`: `np.searchsorted(edges, x, side="right")` counts the edges ≤ x, which gives a bin id in `0…k−1`.
  Values beyond the bounds fall into the first/last bin, so clipping is implicit.
- `__call__`: `np.ravel_multi_index` turns the per-dimension ids into **one integer**
  (a mixed-radix number), so the Q-table is a 2-D array `Q[n_states, n_actions]`.
- Bounds: position ±2.4 and angle ±0.2095 rad (12°) are the **termination** limits. The velocities are
  unbounded, so we clip at 3.0 / 3.5. `experiments/obs_ranges.py` shows that 100 % of visited values
  (random and trained policy) fall inside these bounds: max |ẋ| = 2.83, max |θ̇| = 2.92 (table `t1_obs_ranges`).
- `edges={dim: [...]}` gives hand-made, non-uniform boundaries (used for the angle study).

### `agents/tabular.py`
- `act`: ε-greedy. `self.eps(t)` is a linear schedule over environment steps.
- `_argmax_random_tie`: with a zero-initialized table every action ties at first. Always picking
  action 0 would bias exploration, so ties are broken at random.
- `observe`: the update
  `Q[s,a] += α (r + γ·max_a' Q[s',a'] − Q[s,a])`.
  **`terminated` → no bootstrap** (a terminal state has value 0).
  **`truncated` (time limit) → still bootstrap**: the pole did not fall, the clock just ran out.
  Mixing the two up is a classic bug. CartPole truncates at 500 steps and MountainCar at 200.
- It is **off-policy**: the behavior policy is ε-greedy, but the target uses `max` (greedy policy).
  SARSA would use `Q[s', a_next]` instead.
- `init_value`: optimistic initialization, used for MountainCar.

### `agents/dqn.py`
- `QNetwork`: MLP `obs → hidden → hidden → n_actions` (one output per action, so a single forward pass
  gives every Q(s,·)). With `dueling`: `Q = V + A − mean(A)`. Subtracting the mean makes V and A
  identifiable; otherwise any constant could move between them.
- `ReplayBuffer`: pre-allocated circular arrays. `ptr` wraps around and overwrites the oldest data.
  `dones` stores **only `terminated`**, never `truncated`.
- `DQNAgent.observe`:
  - With replay: store the transition, then every `train_freq` steps (after `learning_starts` warm-up)
    sample a uniform random batch.
  - Without replay: update on the single current transition (online Q-learning with a network).
- `_update`, line by line:
  - `q_all = self.q(s)` has shape `[B, n_actions]`.
  - `q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)`: **gather** takes, in each row `i`, the column
    `a[i]`, i.e. the Q-value of the action **actually taken**. Those are the values whose error we
    regress. The `max` is only used for the next-state target.
  - `with torch.no_grad()`: the target is a **constant** (semi-gradient TD). Gradients flow only through `q_sa`.
  - `target_net = self.q_target if self.use_target else self.q`: the ablation switch.
  - Double DQN: `a2 = argmax self.q(s2)` (the online network selects), then `target_net(s2).gather(a2)`
    (the target network evaluates).
  - `y = r + γ (1 − done) q_next`.
  - `smooth_l1_loss` (Huber): it behaves like MSE for small errors and is linear for large ones, so a
    single huge TD error can't blow up the gradient. `clip_grad_norm_(…, 10)` adds a second safety net.
- `_maybe_update_target`: a **hard** copy, `load_state_dict`, every `target_update_freq` environment
  steps (our default), or a **soft/Polyak** update `θ_t ← τθ + (1−τ)θ_t` when `tau < 1`.
- `greedy` runs under `@torch.no_grad()` and never calls the optimizer, so the weights are frozen
  during evaluation.

### `common/loop.py`
- One loop serves both agents, so the comparison is fair by construction.
- `t` is the global environment-step counter, used for ε, learning starts, the target-update clock,
  and evaluation every `eval_every` steps.
- `evaluate`: ε = 0, a separate environment, and the same `seed_base + k` start states at every
  checkpoint, so the curve tracks the policy and not luck in the start states.
- Training-episode returns are logged with the **step at which each episode ended** (for step-based x-axes).

### `common/runner.py` / `common/stats.py` / `common/plotting.py`
- Each run executes in its own process with `torch.set_num_threads(1)`. Results are written atomically
  (`.tmp` + `replace`) and runs already on disk are skipped.
- `mean_ci`: `t_{n−1,0.975} · s/√n`. With 5 seeds t = 2.776, which is wider (and more honest) than 1.96.
- `train_matrix`: each run's 20-episode moving average is read off a common step grid, so we can average seeds.
- `summarize`: **final** (last 3 checkpoints), **best**, **AUC** (mean over all checkpoints),
  **drop = best − final** (catastrophic forgetting), and steps to threshold.

---

## 3. Expected questions and answers

**"Change the pole-angle bin boundaries. What happens?"**
Bins that are too coarse around the vertical stop the agent from telling a slight tilt from upright,
so it reacts late and falls. Our `tabular_angle_coarse` (2 bins, sign only) shows this. Bins that are
too fine need more data per cell: every extra angle bin multiplies the table by 10·10·10 cells for the
other dimensions. `tabular_angle_fine_center` puts resolution where the pole spends its time. See
Fig. `t1_angle_boundaries` and the coverage column.

**"Explain `gather()`."** See `_update` above: it selects Q(sᵢ, aᵢ) for the actions taken, not the maximum.

**"How is the target network updated?"** A hard copy with `load_state_dict` every 500 (CartPole) /
1000 (LunarLander) environment steps. `tau < 1` switches to Polyak averaging.

**"Why does no-target fail / oscillate?"** Each gradient step on θ also moves the target
`r + γ max Q_θ(s′)`. Because similar states share weights, raising Q(s,a) also raises Q(s′,·). That
feedback loop shows up as a larger TD loss, gradient norms hitting the clip, and Q-value drift
(Fig. `t2_diagnostics`).

**"Why does no-replay fail?"** Consecutive transitions are nearly identical, so SGD sees a
non-stationary, correlated stream and overfits the current part of the trajectory. It forgets other
regions (catastrophic interference), and each transition is used once and thrown away, which is
sample-inefficient.

**"Why steps and not episodes?"** Early tabular episodes last ~15–20 steps while a good DQN
episode lasts 500. Per episode, the agents consume very different amounts of data and compute.

**"Why is DQN slower in wall-clock if it learns in fewer steps?"** Every step runs a forward pass
for acting plus a backward pass on a batch of 64–128, against a single table lookup for tabular.
Look at the `CPU s/run` columns.

**"Why does DQN fail on MountainCar?"** Covered in Section 1. Show Fig. `t1_mountaincar` (c): every seed sits at
−100 until its first flag. Then show (b): 14/20 DQN runs never reach it.

**"Your Q-values are +90,000 in a task where rewards are −1. Is that a bug?"** No. The test
`test_dqn_targets_terminal_and_bootstrap` proves y = r on terminal transitions, and divergence starts only
*after* the first flag (Q ≈ −100 before it). This is the deadly triad (Sutton & Barto §11.3): function
approximation + bootstrapping + off-policy data. The value cliff at the flag can't be fit locally by the MLP,
the error leaks to neighbouring states, and `max` in the target amplifies it. A table can't diverge, because
each update only moves its own cell. Fixes we'd try: normalize observations (velocity is ~0.07 and
position ~1), a lower learning rate, Double DQN, or a soft target update (τ = 0.005).

**"Double DQN didn't help. Did you implement it wrong?"** `td_targets` selects with `self.q` and evaluates with
`self.q_target` (tested). With 4 actions, a 1,000-step target refresh and a Huber loss, the overestimation
is small to begin with: full DQN's mean Q is 43.4 and Double's is 43.2. The original gains were on Atari.

**"Why is no-replay so much more expensive?"** It updates on *every* step instead of every 4th, so it makes
4× more gradient steps, each on a batch of 1. It's slower and still worse.

**"Why the Huber loss?"** Robustness to large TD errors early in training (outliers); it is the
same choice as Mnih et al. (2015).

**"Terminated vs truncated?"** See the tabular section. For a time-limit truncation we still bootstrap.

**"Is your result statistically meaningful?"** 10 seeds, 95 % t-CIs, and Welch t-tests (p-values in
`results/analysis.md`). Where CIs overlap (6 vs 20 bins, p = 0.07; Double/Dueling vs full), we call it a
trend or a null result. Going from 5 to 10 seeds actually reversed our 6-vs-10-bins ranking, which is
exactly why seeds matter.

**"What would you use for HalfCheetah / BipedalWalker / Atari?"** See the report's theory section:
5⁶ = 15,625 discrete actions, 10¹⁷ table cells, 256⁷⁰⁵⁶ pixel states. Continuous actions call for
PPO, SAC or TD3, and pixels call for a CNN function approximator.

---

## 4. Live-modification recipes

Each change takes one edit plus a `--quick` run (~1–3 min, output goes to `results/quick/`).
Revert with `git checkout -- <file>` afterwards, if the folder is under git.

| Request | Where | Edit | Run |
|---|---|---|---|
| Change the ε-decay rate | `experiments/tier1_cartpole.py`, `dqn(...)` default or `dqn_eps*` entries | `eps_decay_steps=...` | `python experiments/tier1_cartpole.py --quick` |
| Change the number of bins | `tabular_bins*` entries | `tab(14)` | same |
| Change the pole-angle boundaries | `tabular_angle_fine_center` | `angle_edges=[...]` (sorted, radians) | same |
| Change the velocity clip range | `envs/discretizers.py`, `CARTPOLE_LOW/HIGH` | e.g. `±1.5` | same |
| Soft target update | `dqn(...)` / `BASE` | `tau=0.005` | `python experiments/tier2_ablation.py --quick` |
| Target-update frequency | `target_update_freq` | e.g. `100` or `10_000` | same |
| Remove the target network or replay buffer | already configs | `use_target_network=False` / `use_replay_buffer=False` | same |
| MSE instead of Huber | `agents/dqn.py`, `_update` | `F.mse_loss(q_sa, y)` | any `--quick` |
| Different γ | config `gamma` | `0.9` | any `--quick` |

To try a one-off agent interactively without the runner:

```python
import sys; sys.path.insert(0, ".")
from common.runner import run_single
log = run_single({"env_id": "CartPole-v1", "agent": "dqn", "seed": 0, "total_steps": 20_000,
                  "eval_every": 5_000, "eval_episodes": 5, "weights_path": "/tmp/w.pt",
                  "agent_kwargs": {"eps_decay_steps": 5_000}})
print(log["eval_return_mean"])
```
