"""Tier 2: DQN component ablations (and Tier-3B extensions) on LunarLander-v3.

Configs (identical hyper-parameters except the switch named):
  full         replay buffer + target network (hard copy every 1,000 steps)
  no_target    targets from the online network itself  -> moving target
  no_replay    online DQN: 1 update per step on the latest transition (batch = 1)  -> correlated data
  double       Double DQN target (online selects, target evaluates)
  dueling      dueling V/A head (Wang et al., 2016)

Usage:  python experiments/tier2_ablation.py [--quick] [--plot-only]
"""
from __future__ import annotations

import _common  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from _common import figpath, fmt, parse_args, seed_list, use_quick_dirs, write_table
from common import plotting as P
from common.runner import load_runs, run_experiment
from common.stats import eval_matrix, stat_matrix, summarize, train_matrix

ENV = "LunarLander-v3"
SOLVED = 200.0

BASE = {"hidden": [256, 256], "lr": 5e-4, "gamma": 0.99, "buffer_size": 100_000, "batch_size": 128,
        "learning_starts": 10_000, "train_freq": 4, "target_update_freq": 1_000,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 100_000, "grad_clip": 10.0}


def build_configs(quick: bool) -> dict[str, dict]:
    steps = 40_000 if quick else 500_000

    def cfg(**kw):
        return {"env_id": ENV, "agent": "dqn", "total_steps": steps, "eval_every": 10_000, "eval_episodes": 10,
                "agent_kwargs": {**BASE, **kw}}

    return {
        "full": cfg(),
        "no_target": cfg(use_target_network=False),
        "no_replay": cfg(use_replay_buffer=False),
        "double": cfg(double=True),
        "dueling": cfg(dueling=True),
    }


LABELS = {"full": "Full DQN", "no_target": "No target network", "no_replay": "No replay buffer",
          "double": "Double DQN", "dueling": "Dueling DQN"}
COLORS = {"full": P.PALETTE[0], "no_target": P.PALETTE[1], "no_replay": P.PALETTE[2],
          "double": P.PALETTE[6], "dueling": P.PALETTE[3]}


def figures(exp: str) -> None:
    R = {k: load_runs(exp, k) for k in LABELS}
    R = {k: v for k, v in R.items() if v}

    groups = [("t2_ablation", ["full", "no_target", "no_replay"], "Ablation"),
              ("t2_extensions", ["full", "double", "dueling"], "Extensions")]
    for fname, names, _ in groups:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
        for name in names:
            if name not in R:
                continue
            steps, E = eval_matrix(R[name])
            P.shaded(axes[0], steps, E, LABELS[name], COLORS[name], smooth=3)
            grid = np.linspace(0, R[name][0]["total_steps"], 201)
            P.shaded(axes[1], grid, train_matrix(R[name], grid), LABELS[name], COLORS[name])
        for ax, t in zip(axes, ["(a) Greedy evaluation (ε = 0)", "(b) Training returns (20-ep. avg.)"]):
            P.style_steps_axis(ax)
            P.hline(ax, SOLVED, "solved = 200")
            ax.set_title(t, loc="left")
        axes[0].set_ylabel("Episode return")
        fig.tight_layout()
        P.legend_below(fig, axes[0], ncol=3)
        P.save(fig, figpath(fname))

    # diagnostics: TD loss (log scale), mean Q of taken actions, gradient norm
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for name in ["full", "no_target", "no_replay", "double"]:
        if name not in R:
            continue
        for ax, key in zip(axes, ["loss", "q_mean", "grad_norm"]):
            s, M = stat_matrix(R[name], key)
            # 10k-step block averages to keep the lines readable
            k = 10
            n = (M.shape[1] // k) * k
            P.shaded(ax, s[:n].reshape(-1, k).mean(1), M[:, :n].reshape(M.shape[0], -1, k).mean(2),
                     LABELS[name], COLORS[name])
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Huber TD loss", loc="left")
    axes[1].set_title("(b) Mean Q(s,a) on batch", loc="left")
    axes[2].set_title("(c) Grad. norm (pre-clip)", loc="left")
    axes[2].set_yscale("log")
    for ax in axes:
        P.style_steps_axis(ax, "Env. steps")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    P.save(fig, figpath("t2_diagnostics"))

    rows = []
    for name, runs in R.items():
        s = summarize(runs, threshold=SOLVED)
        stt = s["steps_to_threshold"]
        _, Q = stat_matrix(runs, "q_mean")
        rows.append([LABELS[name], fmt(s["final"]), fmt(s["best"]), fmt(s["auc"]), fmt(s["drop"]),
                     ("--" if stt[2] == 0 else f"{stt[0] / 1e3:.0f}k") + f" ({stt[2]}/{s['n_seeds']})",
                     f"{np.nanmean(Q[:, -50:]):.1f}",
                     f"{s['cpu_seconds'][0] / 60:.1f}"])
    write_table("t2_lunarlander",
                ["Config", "Final", "Best", "AUC", "Drop", "Mean steps to 200 (seeds)", "Final mean Q", "CPU min/run"],
                rows,
                "Tier 2, LunarLander-v3, 500k steps, greedy evaluation every 10k steps (10 episodes). "
                "Mean $\\pm$ 95\\% t-CI over seeds. Final mean Q = average Q(s,a) on training batches over the last 50k steps.")


def main():
    args = parse_args(__doc__)
    exp = "tier2_lunarlander_quick" if args.quick else "tier2_lunarlander"
    seeds = seed_list(args)
    if args.quick:
        use_quick_dirs()
    if not args.plot_only:
        run_experiment(exp, build_configs(args.quick), seeds, args.workers, args.force)
    figures(exp)


if __name__ == "__main__":
    main()
