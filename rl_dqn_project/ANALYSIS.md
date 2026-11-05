# Analysis: what we learned

Every number here is produced by `python analysis.py` from the raw run logs in `results/`. The script also
writes `results/analysis.md` (every table) and `results/analysis.json` (per-seed values). Results are the mean
± 95 % Student-t CI over **10 seeds**. "Greedy" means ε = 0 with frozen weights, evaluated every 5k
(CartPole/MountainCar) or 10k (LunarLander) steps on fixed evaluation seeds.

---

## 0. The ten lessons

1. **DQN wins on samples; tabular wins on compute.** On CartPole, DQN solved the task in **~7× fewer
   environment steps** (56k vs 382k), but each step cost **23× more CPU**. Reaching 475 took tabular
   **3.4× less CPU time** (37 s vs 124 s). "Which is faster" depends on whether environment interaction or
   computation is the scarce resource.
2. **Generalization is the whole story.** DQN's advantage comes from one gradient step moving Q for *nearby*
   states too. A table only learns about cells it has visited, and with 20 bins 95 % of the table was never
   updated after 1M steps.
3. **Resolution has to be paid for with data.** 10 bins beat both 6 bins (too coarse to express a precise
   policy) and 20 bins (too many cells to fill). Fine pole-angle bins near vertical only helped once enough
   data had arrived. A 2-bin angle (sign only) was clearly worse early (p < 10⁻⁶).
4. **Evaluate during training, not just at the end.** Every fast- and medium-decay DQN seed on CartPole
   (20/20) hit ≥475 and later collapsed below 200 (catastrophic forgetting). Judged only by its last
   checkpoint, a method that had solved the task looks mediocre.
5. **Exploration needs depend on the task.** CartPole rewards *fast* ε-decay, because random actions already
   cover the states that matter. MountainCar needs *directed* exploration, which ε-greedy doesn't provide.
6. **Optimism is exploration.** On MountainCar, the same tabular agent finds the flag on every seed with
   Q₀ = 0 (optimistic) and never with Q₀ = −100. The cause is the initialization, not "tabular vs deep".
7. **Sparse rewards starve DQN.** Until the flag is seen, every DQN target is −1 + γ·Q, so Q converges to
   −1/(1−γ) = −100 everywhere ("all states are equally bad"). That leaves no gradient pointing toward the goal.
8. **Function approximation + bootstrapping + off-policy data can diverge** (the deadly triad). In MountainCar
   every true Q is ≤ 0. Yet after DQN first reached the flag, 4 of the 6 runs that found it pushed Q above
   +4,900 (one reached +79, one stayed ≤ 0). The greedy policy still improved, because argmax only depends on
   how the values rank. A unit test rules out a terminal-handling bug.
9. **Both DQN stabilizers matter, in different ways.** Without a target network, LunarLander learns just as
   fast but cannot *stay* good. Without replay, it never really learns, and its Q-values inflate far beyond
   what its policy earns.
10. **Rigor changes conclusions.** Going from 5 to 10 seeds reversed our 6-vs-10-bins ranking and turned
    several "differences" into overlapping intervals. We report p-values and say "trend" when p > 0.05.

---

## 1. Methodology choices and why

| Choice | Why |
|---|---|
| x-axis = cumulative env steps | Tabular episodes last ~20 steps early on, while good DQN episodes last 500. An "episode" axis would be an unfair clock. |
| greedy eval every N steps, fixed eval seeds, frozen weights | separates *what the policy knows* from exploration noise; fixed start states make checkpoints comparable |
| 10 seeds, Student-t 95 % CI | t₉ = 2.26 > 1.96: honest bands for small n; plus Welch t-tests for the key claims |
| per-run JSON with full spec, atomic writes, cache | any figure can be rebuilt (`--plot-only`); crashed runs resume; seeds can be split across machines |
| CPU time from one machine only | seeds 5–9 ran on GitHub Codespaces (`results/hosts.json`), and timings from different CPUs are not averaged |
| `terminated` vs `truncated` | time-limit cuts are not terminal, so we still bootstrap. Mixing these up silently biases values. |
| Huber loss + grad-norm clip 10 | bounded gradients for large early TD errors (Mnih et al., 2015) |

---

## 2. Tier 1, CartPole-v1

| Config | Final | Best | AUC | Steps to 475 | Solved | Drop |
|---|---|---|---|---|---|---|
| Tabular, 6 bins | 356 ± 51 | 494 ± 6 | 303 ± 11 | 518 ± 59k | 10/10 | 138 ± 49 |
| Tabular, 10 bins | **392 ± 81** | 500 ± 0 | 294 ± 30 | 382 ± 58k | 10/10 | 108 ± 81 |
| Tabular, 20 bins | 263 ± 96 | 475 ± 31 | 269 ± 47 | 505 ± 70k | 7/10 | 212 ± 88 |
| Angle: 2 bins (sign) | 305 ± 73 | 486 ± 24 | 226 ± 23 | 608 ± 159k | 8/10 | 181 ± 78 |
| Angle: fine near 0 | 362 ± 74 | 477 ± 25 | 274 ± 31 | 474 ± 114k | 6/10 | 114 ± 62 |
| DQN, ε decay 5k | 340 ± 134 | 500 ± 0 | **324 ± 36** | **56 ± 16k** | 10/10 | 160 ± 134 |
| DQN, ε decay 25k | 297 ± 115 | 500 ± 0 | 223 ± 34 | 98 ± 21k | 10/10 | 203 ± 115 |
| DQN, ε decay 100k | 140 ± 66 | 309 ± 91 | 103 ± 9 | 180k | 2/10 | 170 ± 62 |

(Tabular: 1M steps. DQN: 200k steps. "Steps to 475" averages only the seeds that got there.)

**Sample efficiency vs compute.**
- At an *equal* 200k-step budget, DQN (ε 5k) scores 356 ± 95 against 175–215 for every table
  (Welch t = 4.2, p = 0.002 vs 10 bins).
- Per step, tabular costs 0.097 ms and DQN 2.24 ms (23×). DQN does a forward pass to act and a backward
  pass on 64 samples; tabular does one array lookup and one scalar update.

**Bins.**

| Bins | States | Visited after 1M steps |
|---|---|---|
| 6 | 1,296 | 38.7 % (~500) |
| 10 | 10,000 | 18.0 % (~1,800) |
| 20 | 160,000 | 4.6 % (~7,400) |

With 20 bins the agent sees more *distinct* states, yet each is visited far less often, so the Q-values stay
noisy. The seed-to-seed spread of the final score grows with the bin count (std 71 → 114 → 134). The final
difference between 6 and 20 bins is a trend (p = 0.07), not a significant result.

**Velocity clipping is justified by data** (`experiments/obs_ranges.py`, table `t1_obs_ranges`). Over 50k steps
under both a random and a trained policy, |ẋ| ≤ 2.83 and |θ̇| ≤ 2.92, inside the clip bounds of 3.0 and
3.5. So no bin is wasted on velocities that never occur, and no visited value is clipped.

**ε-decay.**
- Fast decay is best early (190 ± 52 over the first 50k steps) *and* late (379 ± 48 after 150k).
- Slow decay (100k of a 200k budget) spends half the run acting mostly at random, and only 2/10 seeds
  reach 475.
- Forgetting is the price of DQN's speed: once the policy is good, the buffer holds only near-upright
  states, and the network overwrites what it knew about recovering from large tilts.

---

## 3. Tier 1, MountainCar-v0 (sparse reward, planned negative result)

All four agents use the same ε schedule (1 → 0.05 over 30k steps; 150k for `dqn_long_eps`) and 500k steps.

| Config | Final greedy | Best greedy | Seeds that found the flag | Median first flag | Final mean Q |
|---|---|---|---|---|---|
| Tabular, Q₀ = 0 (optimistic) | −176 ± 10 | −148 ± 6 | **10/10** | 113k | — |
| Tabular, Q₀ = −100 | −200 ± 0 | −200 ± 0 | **0/10** | never | — |
| DQN, ε decay 30k | −189 ± 17 | −168 ± 34 | 3/10 | 94k | diverged in 3/3 that found it |
| DQN, ε decay 150k | −174 ± 30 | −164 ± 32 | 3/10 | 118k | diverged in 1/3, mild in 1, none in 1 |

**Why the optimistic table explores.**
- Every true value is negative, so Q₀ = 0 is an over-estimate.
- Each visit lowers the tried action's value. Untried actions then look best, so the agent systematically
  tries what it hasn't tried yet. This is directed exploration, not random dithering.
- Set Q₀ to the "realistic" −1/(1−γ) = −100 and the identical algorithm never finds the flag.

**Why DQN usually fails (14/20 runs).**
- Until the flag is reached, every target is −1 + γ·max Q, which converges to −100 everywhere.
- Those 14 runs end with mean Q = −99.4 ± 0.1 and a greedy return of exactly −200: a flat value
  landscape with no gradient toward the goal.
- A *longer* ε-decay did not help (3/10 in both cases). More random noise is not more exploration when the
  goal needs ~100 coordinated pushes to build momentum.

**What happens when DQN does find the flag (6/20 runs).**
- Their greedy policy ends *better* than the table's (−138 ± 38 vs −176 ± 10). Generalization pays off once
  there is a signal to generalize.
- Their Q-values diverge. In a task where no state is worth more than 0, mean Q reached +4,923, +24,799,
  +87,233 and +95,715 in four runs, and the pre-clip gradient norm went from ~0.4 to as much as 7,861.
- The flag introduces a sharp value cliff (−1 next to the goal, −100 elsewhere) that a smooth MLP
  cannot fit locally. The error spreads to neighbours, and the max in every bootstrap target amplifies
  it: the deadly triad (Sutton & Barto §11.3). A table cannot do this, because each update only moves
  its own cell toward its target.
- `tests/test_core.py::test_dqn_targets_terminal_and_bootstrap` verifies y = r on terminal transitions,
  so this is not a done-flag bug.

**Takeaway.** Exploration in sparse-reward tasks must be *directed* (optimism, count bonuses, intrinsic
motivation), and DQN on such tasks needs safeguards against divergence, such as observation normalization
and smaller learning rates, or Double DQN with a slower target.

---

## 4. Tier 2, LunarLander-v3 (ablations)

| Config | Final | Best | Drop | Solved | Welch p vs full | Final mean Q | Max grad-norm | CPU min/run |
|---|---|---|---|---|---|---|---|---|
| Full DQN | 234 ± 29 | 273 ± 6 | 39 ± 29 | 10/10 | — | 43.4 | 19 | 10.4 |
| No target network | 176 ± 37 | 275 ± 7 | 99 ± 35 | 10/10 | **0.014** | 48.4 | 20 | 10.0 |
| No replay buffer | −376 ± 188 | 149 ± 28 | 525 ± 191 | 2/10 | **4e-05** | **256.9** | **296** | 24.0 |
| Double DQN | 234 ± 29 | 276 ± 5 | 42 ± 29 | 10/10 | 0.98 | 43.2 | 21 | 10.2 |
| Dueling DQN | 238 ± 33 | 276 ± 6 | 38 ± 29 | 10/10 | 0.82 | 46.9 | 22 | 9.8 |

**Target network = stability, not speed.**
- Without it, learning is as fast and the peak as high (275 vs 273).
- But **10/10 seeds** that reached ≥ 200 later fell below 0, against 2/10 with a target network.
- Its TD loss stays higher for the whole run: the network chases a target that its own updates keep moving.

**Replay buffer = learning at all.**
- Online DQN on consecutive, nearly identical transitions (batch size 1) peaks at 149 and then degrades to −376.
- Its mean Q is 5.9× the full agent's while its policy is far worse, which is clear overestimation.
- Its gradients hit ~300 before clipping (vs ~20), and its loss never falls.
- The reasons:
  - correlated data breaks the i.i.d. assumption of SGD;
  - each transition is seen once and discarded;
  - every update overwrites what the network knew about other regions (catastrophic interference).
- It is also the most expensive variant (24 vs 10 CPU-minutes per run), because it updates on every step
  instead of every 4th.

**Double / Dueling: a null result, reported as such.**
- No measurable difference at this scale (p = 0.98 and 0.82).
- Double DQN doesn't even lower the Q level (43.2 vs 43.4). With 4 actions, a 1,000-step target refresh and a
  Huber loss, the max-overestimation it fixes is already small.
- The original gains were on Atari (18 actions, much noisier estimates).

---

## 5. Limits of both families in complex environments (theory)

| Problem | Number | Consequence |
|---|---|---|
| Discretizing HalfCheetah's 6-D action, 5 levels/joint | 5⁶ = **15,625** actions | output layer and max over 15k actions; ε-greedy over 15k arms; still too coarse for smooth torques |
| Ant's 8-D action, 5 levels | 5⁸ ≈ 390,625 | worse |
| Table for a 17-D state, 10 bins/dim | 10¹⁷ cells per action | ~800 PB of float64 per action; cells can never all be visited |
| Our own 4-D CartPole, 6 → 20 bins | 1,296 → 160,000 cells | visited fraction falls 39 % → 4.6 % (measured) |
| 84×84 grayscale Atari frame, 256 levels | 256⁷⁰⁵⁶ ≈ 10¹⁶⁹⁹² states | exact revisits essentially never happen, so a table never reuses an update |

**What works instead.**
- **Continuous control:** actor–critic methods that output actions directly.
  - PPO: on-policy, Gaussian policy, clipped updates.
  - SAC: off-policy, entropy-regularized, strong exploration.
  - TD3: deterministic actor, twin critics with a min, against overestimation.
- **Pixels:** convolutional function approximation, i.e. generalization across visually similar frames.
  That is the same mechanism behind DQN's 7× sample-efficiency win on CartPole, scaled up.

---

## 6. Threats to validity (what a reviewer could challenge)

- **Hyper-parameters were not tuned per method.** We used one sensible configuration per agent (CleanRL /
  SB3-zoo-like values) and changed only the ablated switch. A tuned tabular agent (decaying α, adaptive
  bins) or a tuned DQN could shift absolute numbers. The *mechanisms* we report (generalization, forgetting,
  flat values, divergence, instability) are what we claim, not exact scores.
- **Budgets differ by design.** CartPole tables got 1M steps and DQN 200k, while MountainCar and LunarLander
  gave every agent 500k. Equal-budget comparisons are made explicitly at 200k steps.
- **Two machine types.** Seeds 0–4 (and 8–9 on LunarLander) ran on the laptop, the rest on 4-vCPU
  Codespaces with the same pinned packages. Learning metrics pool all seeds; CPU-time figures use laptop
  runs only (`results/hosts.json`).
- **Display smoothing.** Greedy curves in figures use a 3–5-checkpoint trailing mean per seed. All tables
  and tests use raw values.
- **Multiple comparisons.** We ran several Welch tests without correction. The headline results
  (p ≤ 0.002 to < 10⁻⁶) survive any reasonable correction; p = 0.014 (no-target final) and p = 0.07 (bins)
  should be read as moderate evidence and a trend, respectively.

---

## 7. Compute

**170 runs, 22.3 CPU-hours in total** (one thread per run).

| Run (reference laptop) | CPU time per run |
|---|---|
| 1M-step CartPole table | ~100 s |
| 200k-step CartPole DQN | ~450 s |
| 500k-step LunarLander DQN | ~10 min (24 min without replay) |

To make the deadline, seeds were split across the laptop and two GitHub Codespaces. Each run is an
independent JSON file, so the results merged by copying files.

---

## 8. Reproduce

```bash
python experiments/run_all.py          # all experiments (cached runs are skipped)
python analysis.py                     # every number in this file
python -m pytest -q tests              # 10 sanity tests
cd report && latexmk -pdf main.tex     # the 8-page report
```

