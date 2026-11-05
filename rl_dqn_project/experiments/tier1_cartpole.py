"""Tier 1: tabular Q-learning vs DQN on CartPole-v1.

Studies
  A. Tabular vs DQN on a common x-axis (cumulative environment steps).
  B. Tabular bin-count ablation: 6, 10, 20 uniform bins per dimension.
  C. Pole-angle boundary study (oral-defense question 1): 2 coarse angle bins vs bins
     concentrated around the vertical, all other dimensions at 10 bins.
  D. DQN epsilon-decay ablation: fast (5k), medium (25k), slow (100k steps) linear decay.

Usage:  python experiments/tier1_cartpole.py            # full run (5 seeds) + figures + tables
        python experiments/tier1_cartpole.py --quick    # 2-minute smoke test
        python experiments/tier1_cartpole.py --plot-only
"""
from __future__ import annotations

import _common  # noqa: F401  (sets sys.path)
import numpy as np
import matplotlib.pyplot as plt

from _common import figpath, fmt, parse_args, seed_list, use_quick_dirs, write_table
from common import plotting as P
from common.runner import load_runs, run_experiment
from common.stats import eval_matrix, summarize, train_matrix

ENV = "CartPole-v1"
SOLVED = 475.0  # Gymnasium's reward threshold for CartPole-v1


def build_configs(quick: bool) -> dict[str, dict]:
    tab_steps = 60_000 if quick else 1_000_000
    dqn_steps = 20_000 if quick else 200_000
    ev = 5_000

    def tab(bins=10, angle_edges=None, **kw):
        disc = {"kind": "cartpole", "bins": bins}
        if angle_edges is not None:
            disc["angle_edges"] = angle_edges
        return {"env_id": ENV, "agent": "tabular", "total_steps": tab_steps, "eval_every": ev, "eval_episodes": 10,
                "discretizer": disc,
                "agent_kwargs": {"alpha": 0.1, "gamma": 0.99, "eps_start": 1.0, "eps_end": 0.01,
                                 "eps_decay_steps": tab_steps // 2, "init_value": 0.0, **kw}}

    def dqn(**kw):
        base = {"hidden": [128, 128], "lr": 5e-4, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64,
                "learning_starts": 1_000, "train_freq": 1, "target_update_freq": 500,
                "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 25_000}
        return {"env_id": ENV, "agent": "dqn", "total_steps": dqn_steps, "eval_every": ev, "eval_episodes": 10,
                "agent_kwargs": {**base, **kw}}

    return {
        "tabular_bins6": tab(6),
        "tabular_bins10": tab(10),
        "tabular_bins20": tab(20),
        # angle boundaries: only 2 bins (left/right of vertical) vs 10 bins concentrated near 0
        "tabular_angle_coarse": tab(10, angle_edges=[0.0]),
        "tabular_angle_fine_center": tab(10, angle_edges=[-0.1, -0.05, -0.025, -0.01, 0.0, 0.01, 0.025, 0.05, 0.1]),
        "dqn_eps5k": dqn(eps_decay_steps=5_000),
        "dqn_eps25k": dqn(eps_decay_steps=25_000),   # = default DQN
        "dqn_eps100k": dqn(eps_decay_steps=100_000),
    }


LABELS = {
    "tabular_bins6": "Tabular, 6 bins", "tabular_bins10": "Tabular, 10 bins", "tabular_bins20": "Tabular, 20 bins",
    "tabular_angle_coarse": "Angle: 2 bins (sign only)", "tabular_angle_fine_center": "Angle: fine near 0",
    "dqn_eps5k": "DQN, ε decay 5k", "dqn_eps25k": "DQN, ε decay 25k", "dqn_eps100k": "DQN, ε decay 100k",
}


def figures(exp: str) -> None:
    R = {name: load_runs(exp, name) for name in LABELS}
    R = {k: v for k, v in R.items() if v}

    # ---- A: tabular vs DQN, greedy eval (left) and training returns (right) --------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    show = ["tabular_bins6", "tabular_bins10", "tabular_bins20", "dqn_eps25k"]
    colors = {"tabular_bins6": P.PALETTE[0], "tabular_bins10": P.PALETTE[1],
              "tabular_bins20": P.PALETTE[2], "dqn_eps25k": P.PALETTE[6]}
    for name in show:
        if name not in R:
            continue
        steps, E = eval_matrix(R[name])
        P.shaded(axes[0], steps, E, LABELS[name], colors[name], smooth=5)
        grid = np.linspace(0, R[name][0]["total_steps"], 201)
        P.shaded(axes[1], grid, train_matrix(R[name], grid), LABELS[name], colors[name])
    for ax, title in zip(axes, ["(a) Greedy eval. (ε = 0, frozen)", "(b) Training returns (ε-greedy)"]):
        P.style_steps_axis(ax)
        P.hline(ax, SOLVED, "solved = 475")
        ax.set_title(title, loc="left")
        ax.set_xlim(0, None)
        ax.set_ylim(0, 520)
    axes[0].set_ylabel("Episode return")
    fig.tight_layout()
    P.legend_below(fig, axes[0])
    P.save(fig, figpath("t1_tabular_vs_dqn"))

    # ---- C: pole-angle boundaries ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, name in enumerate(["tabular_bins10", "tabular_angle_coarse", "tabular_angle_fine_center"]):
        if name in R:
            steps, E = eval_matrix(R[name])
            label = "Angle: 10 uniform" if name == "tabular_bins10" else LABELS[name]
            P.shaded(ax, steps, E, label, P.PALETTE[i], smooth=5)
    P.style_steps_axis(ax)
    P.hline(ax, SOLVED, "475")
    ax.set_ylabel("Greedy return")
    ax.set_ylim(0, 520)
    ax.set_title("Tabular: pole-angle bin boundaries", loc="left")
    ax.legend(loc="lower right")
    P.save(fig, figpath("t1_angle_boundaries"))

    # ---- D: epsilon decay, early vs late -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for i, name in enumerate(["dqn_eps5k", "dqn_eps25k", "dqn_eps100k"]):
        if name not in R:
            continue
        steps, E = eval_matrix(R[name])
        P.shaded(axes[0], steps, E, LABELS[name], P.PALETTE[i], smooth=3)
        grid = np.linspace(0, R[name][0]["total_steps"], 201)
        P.shaded(axes[1], grid, train_matrix(R[name], grid), LABELS[name], P.PALETTE[i])
        d = R[name][0]["spec"]["agent_kwargs"]["eps_decay_steps"]
        for ax in axes:
            ax.axvline(d, color=P.PALETTE[i], lw=0.7, ls="--", alpha=0.7)
    for ax, title in zip(axes, ["(a) Greedy evaluation", "(b) Training returns"]):
        P.style_steps_axis(ax)
        ax.set_ylim(0, 520)
        P.hline(ax, SOLVED, "475")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Episode return")
    fig.tight_layout()
    P.legend_below(fig, axes[0], ncol=3)
    P.save(fig, figpath("t1_dqn_eps_decay"))

    # ---- tables -----------------------------------------------------------------------------------
    rows = []
    for name, runs in R.items():
        s = summarize(runs, threshold=SOLVED)
        steps, E = eval_matrix(runs)
        early = E[:, (steps > 0) & (steps <= 50_000)].mean(axis=1)  # early-learning window
        cov = np.mean([r.get("state_coverage", np.nan) for r in runs])
        rows.append([LABELS[name], f"{runs[0]['total_steps'] / 1e3:.0f}k",
                     fmt(s["final"]), fmt(s["best"]), fmt(s["auc"]), f"{np.mean(early):.1f}",
                     fmt(s["drop"]),
                     fmt((s["steps_to_threshold"][0] / 1e3, s["steps_to_threshold"][1] / 1e3,
                          s["steps_to_threshold"][2]), 0) + f" ({int(s['solved_frac'][0] * s['n_seeds'] + 0.5)}/{s['n_seeds']})",
                     "--" if np.isnan(cov) else f"{100 * cov:.1f}%",
                     f"{s['cpu_seconds'][0]:.0f}"])
    write_table(
        "t1_cartpole",
        ["Config", "Steps", "Final", "Best", "AUC", "Early (≤50k)", "Drop", "k-steps to 475 (solved)", "States seen", "CPU s/run†"],
        rows,
        "Tier 1, CartPole-v1. Greedy-evaluation returns (10 episodes per checkpoint, every 5k steps), mean "
        "$\\pm$ 95\\% t-CI over seeds. Final = mean of the last 3 checkpoints; AUC = mean over all checkpoints; "
        "Drop = best $-$ final (forgetting); Early = mean greedy return over the first 50k steps. $^\\dagger$CPU time from the runs on the reference laptop only (\\texttt{results/hosts.json}).")


def main():
    args = parse_args(__doc__)
    exp = "tier1_cartpole_quick" if args.quick else "tier1_cartpole"
    seeds = seed_list(args)
    if args.quick:
        use_quick_dirs()
    if not args.plot_only:
        run_experiment(exp, build_configs(args.quick), seeds, args.workers, args.force)
    figures(exp)


if __name__ == "__main__":
    main()
