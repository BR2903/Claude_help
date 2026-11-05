"""Tier 1 (sparse-reward variant): tabular Q-learning vs DQN on MountainCar-v0.

This is the planned NEGATIVE-RESULT study. MountainCar gives -1 every step until the car reaches
the flag (episodes are cut at 200 steps). All agents use the SAME epsilon schedule
(1.0 -> 0.05 over 30k steps) so the only differences are:
  * tabular, init Q = 0    -> optimistic (every true value is < 0): untried actions look best
  * tabular, init Q = -100 -> pessimistic/realistic, = -1/(1-gamma): no optimism bonus
  * DQN (short and long epsilon decay): neural net initialised near 0, but trained on a buffer
    that, until the flag is first reached, contains ONLY reward -1 transitions.

Diagnostics: greedy evaluation return, cumulative number of training episodes that reached the
flag, and the DQN's mean Q(s,a) (it should approach -1/(1-gamma) = -100 everywhere if the flag
is never seen, i.e. "all states are equally bad").

Usage:  python experiments/tier1_mountaincar.py [--quick] [--plot-only]
"""
from __future__ import annotations

import _common  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from _common import figpath, fmt, parse_args, seed_list, use_quick_dirs, write_table
from common import plotting as P
from common.runner import load_runs, run_experiment
from common.stats import eval_matrix, stat_matrix, summarize

ENV = "MountainCar-v0"
GAMMA = 0.99
EPS = {"eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 30_000}


def build_configs(quick: bool) -> dict[str, dict]:
    steps = 30_000 if quick else 500_000

    def tab(init):
        return {"env_id": ENV, "agent": "tabular", "total_steps": steps, "eval_every": 5_000, "eval_episodes": 10,
                "discretizer": {"kind": "mountaincar", "bins": 20},
                "agent_kwargs": {"alpha": 0.1, "gamma": GAMMA, "init_value": init, **EPS}}

    def dqn(**kw):
        base = {"hidden": [128, 128], "lr": 5e-4, "gamma": GAMMA, "buffer_size": 50_000, "batch_size": 64,
                "learning_starts": 1_000, "train_freq": 1, "target_update_freq": 500, **EPS}
        return {"env_id": ENV, "agent": "dqn", "total_steps": steps, "eval_every": 5_000, "eval_episodes": 10,
                "agent_kwargs": {**base, **kw}}

    return {
        "tabular_optimistic": tab(0.0),
        "tabular_pessimistic": tab(-1.0 / (1.0 - GAMMA)),
        "dqn": dqn(),
        "dqn_long_eps": dqn(eps_decay_steps=150_000),
    }


LABELS = {"tabular_optimistic": "Tabular, Q₀ = 0 (optimistic)", "tabular_pessimistic": "Tabular, Q₀ = −100",
          "dqn": "DQN, ε decay 30k", "dqn_long_eps": "DQN, ε decay 150k"}
COLORS = {"tabular_optimistic": P.PALETTE[0], "tabular_pessimistic": P.PALETTE[1],
          "dqn": P.PALETTE[6], "dqn_long_eps": P.PALETTE[2]}


def cumulative_goals(runs, grid):
    out = np.zeros((len(runs), len(grid)))
    for i, r in enumerate(runs):
        ends = np.asarray(r["train_episode_end_step"])
        succ = np.cumsum(np.asarray(r["train_terminated"], dtype=float))
        pos = np.searchsorted(ends, grid, side="right") - 1
        out[i] = np.where(pos >= 0, succ[np.maximum(pos, 0)], 0.0)
    return out


def first_goal_step(run):
    for t, term in zip(run["train_episode_end_step"], run["train_terminated"]):
        if term:
            return t
    return np.nan


def figures(exp: str) -> None:
    R = {k: load_runs(exp, k) for k in LABELS}
    R = {k: v for k, v in R.items() if v}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for name, runs in R.items():
        steps, E = eval_matrix(runs)
        P.shaded(axes[0], steps, E, LABELS[name], COLORS[name], smooth=3)
        # (b) and (c) are bimodal across seeds (flag found or never found), so a mean +- CI would be
        # misleading (it even goes below 0 for counts). Show every seed as a thin line + the median.
        grid = np.linspace(0, runs[0]["total_steps"], 201)
        G = cumulative_goals(runs, grid)
        for row in G:
            axes[1].plot(grid, row, color=COLORS[name], lw=0.6, alpha=0.35)
        axes[1].plot(grid, np.median(G, axis=0), color=COLORS[name], lw=1.8)
        if runs[0]["spec"]["agent"] == "dqn":
            s, Q = stat_matrix(runs, "q_mean")
            for row in Q:
                axes[2].plot(s, row, color=COLORS[name], lw=0.7, alpha=0.6)
    axes[0].set_title("(a) Greedy evaluation", loc="left")
    axes[0].set_ylabel("Return")
    axes[1].set_title("(b) Flag reached (each seed)", loc="left")
    axes[1].set_ylabel("Cumulative successful episodes")
    axes[2].set_title("(c) DQN mean Q(s,a), each seed", loc="left")
    axes[2].set_yscale("symlog", linthresh=100)
    axes[2].axhline(-1 / (1 - GAMMA), color=P.TEXT2, lw=0.8, ls=":")
    axes[2].axhline(0, color=P.TEXT2, lw=0.6)
    axes[2].annotate("−1/(1−γ) = −100", (0.02, -1 / (1 - GAMMA)), xycoords=("axes fraction", "data"),
                     va="top", fontsize=7, color=P.TEXT2)
    axes[2].annotate("true Q ≤ 0", (0.02, 0), xycoords=("axes fraction", "data"),
                     va="bottom", fontsize=7, color=P.TEXT2)
    for ax in axes:
        P.style_steps_axis(ax, "Env. steps")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    P.save(fig, figpath("t1_mountaincar"))

    rows = []
    for name, runs in R.items():
        s = summarize(runs, threshold=-110.0)  # Gymnasium's threshold for MountainCar-v0
        fg = np.array([first_goal_step(r) for r in runs], dtype=float)
        n_succ = [int(np.sum(r["train_terminated"])) for r in runs]
        rows.append([LABELS[name].replace("₀", "0"), fmt(s["final"]), fmt(s["best"]),
                     f"{np.sum(~np.isnan(fg))}/{len(runs)}",
                     "--" if np.all(np.isnan(fg)) else f"{np.nanmedian(fg) / 1e3:.1f}k",
                     f"{np.mean(n_succ):.0f}", f"{s['cpu_seconds'][0]:.0f}"])
    write_table("t1_mountaincar",
                ["Config", "Final greedy", "Best greedy", "Seeds reaching flag", "Median first flag", "Successful train eps", "CPU s/run"],
                rows,
                "MountainCar-v0 (sparse reward, 500k steps, same $\\varepsilon$ schedule for all). "
                "Greedy returns are mean $\\pm$ 95\\% CI; -200 means the flag was never reached.")


def main():
    args = parse_args(__doc__)
    exp = "tier1_mountaincar_quick" if args.quick else "tier1_mountaincar"
    seeds = seed_list(args)
    if args.quick:
        use_quick_dirs()
    if not args.plot_only:
        run_experiment(exp, build_configs(args.quick), seeds, args.workers, args.force)
    figures(exp)


if __name__ == "__main__":
    main()
