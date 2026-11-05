"""Tier 2: one-at-a-time hyper-parameter study of DQN on LunarLander-v3.

Two parameters, three values each, varied one at a time around the documented default
(BASE in tier2_ablation.py; the default run is the cached `full` config of the ablation study):

  learning rate         lr                 in {1e-4, 5e-4 (default), 2e-3}
  target-update period  target_update_freq in {250, 1000 (default), 4000} env steps

Every other hyper-parameter is identical to `full`. Only the seeds of this run are compared
(the default config also has seeds 5-9 from the ablation study; they are left out here so all
rows have the same seeds). No CPU-time column: the new runs were made on a different machine
(Ryzen 7 5800U, Windows 11) than the reference laptop, so their timings are not comparable.

Usage:  python experiments/tier2_hparams.py [--quick] [--plot-only] [--seeds N] [--workers N]
"""
from __future__ import annotations

import _common  # noqa: F401
import matplotlib.pyplot as plt

from _common import figpath, fmt, parse_args, seed_list, use_quick_dirs, write_table
from common import plotting as P
from common.runner import load_runs, run_experiment
from common.stats import eval_matrix, summarize
from tier2_ablation import BASE, ENV, SOLVED

# (parameter, values, label formatter); the BASE value of each parameter is the shared default
STUDIES = [
    ("lr", [1e-4, 5e-4, 2e-3], lambda v: f"lr = {v:g}"),
    ("target_update_freq", [250, 1_000, 4_000], lambda v: f"target update every {v:,}"),
]
TITLES = {"lr": "(a) Learning rate", "target_update_freq": "(b) Target-network update period"}


def config_name(param: str, value) -> str:
    """The default value maps to the ablation study's `full` run, so it is trained only once."""
    return "full" if value == BASE[param] else f"{param}_{value:g}"


def build_configs(quick: bool) -> dict[str, dict]:
    steps = 40_000 if quick else 500_000
    configs = {}
    for param, values, _ in STUDIES:
        for v in values:
            configs[config_name(param, v)] = {
                "env_id": ENV, "agent": "dqn", "total_steps": steps, "eval_every": 10_000, "eval_episodes": 10,
                "agent_kwargs": {**BASE, param: v}}
    return configs


def figures(exp: str, seeds: list[int]) -> None:
    def runs_for(name):
        return [r for r in load_runs(exp, name) if r["spec"]["seed"] in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    rows = []
    for ax, (param, values, label) in zip(axes, STUDIES):
        for i, v in enumerate(values):
            runs = runs_for(config_name(param, v))
            if not runs:
                continue
            tag = label(v) + (" (default)" if v == BASE[param] else "")
            steps, E = eval_matrix(runs)
            P.shaded(ax, steps, E, tag, P.PALETTE[i], smooth=3)

            s = summarize(runs, threshold=SOLVED)
            stt = s["steps_to_threshold"]
            rows.append([param, f"{v:g}" + (" (default)" if v == BASE[param] else ""),
                         fmt(s["final"]), fmt(s["best"]), fmt(s["auc"]), fmt(s["drop"]),
                         ("--" if stt[2] == 0 else f"{stt[0] / 1e3:.0f}k") + f" ({stt[2]}/{s['n_seeds']})"])
        P.style_steps_axis(ax)
        P.hline(ax, SOLVED, "solved = 200")
        ax.set_title(TITLES[param], loc="left")
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("Greedy return (ε = 0)")
    fig.tight_layout()
    P.save(fig, figpath("t2_hparams"))

    write_table("t2_hparams",
                ["Parameter", "Value", "Final", "Best", "AUC", "Drop", "Mean steps to 200 (seeds)"],
                rows,
                "Tier 2 hyper-parameter study, LunarLander-v3, 500k steps, one parameter varied at a time "
                "around the default (all other settings as Full DQN). Greedy evaluation every 10k steps "
                "(10 episodes). Mean $\\pm$ 95\\% t-CI over seeds.")


def main():
    args = parse_args(__doc__)
    exp = "tier2_lunarlander_quick" if args.quick else "tier2_lunarlander"
    seeds = seed_list(args)
    if args.quick:
        use_quick_dirs()
    if not args.plot_only:
        run_experiment(exp, build_configs(args.quick), seeds, args.workers, args.force)
    figures(exp, seeds)


if __name__ == "__main__":
    main()
