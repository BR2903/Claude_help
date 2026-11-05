"""Full quantitative analysis of every experiment, computed from the raw run logs in results/.

    python analysis.py            # prints the analysis and writes results/analysis.json + results/analysis.md

Every number quoted in ANALYSIS.md and in the report comes from this script. It never retrains
anything; it only reads results/<experiment>/<config>/seed_*.json.

Sections
  1. Tier 1 CartPole: sample efficiency (steps), compute (CPU), bins, angle boundaries, epsilon decay,
     forgetting.
  2. Tier 1 MountainCar: flag discovery, optimistic vs pessimistic init, DQN Q collapse to -1/(1-gamma).
  3. Tier 2 LunarLander: ablations and extensions, Welch t-tests vs full DQN, Q overestimation.
  4. Compute budget of the whole study.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.runner import RESULTS, load_runs  # noqa: E402
from common.stats import eval_matrix, mean_ci, stat_matrix, timing_runs  # noqa: E402

OUT: dict = {}
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def ci(x) -> str:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n/a"
    m, h = mean_ci(x)
    return f"{m:.1f} ± {h:.1f} (n={len(x)})"


def per_seed(runs: list[dict], threshold: float | None = None) -> dict[str, np.ndarray]:
    steps, E = eval_matrix(runs)
    d = {"final": E[:, -3:].mean(1), "best": E.max(1), "auc": E.mean(1)}
    d["drop"] = d["best"] - d["final"]
    if threshold is not None:
        d["steps_to_threshold"] = np.array(
            [steps[np.argmax(r >= threshold)] if (r >= threshold).any() else np.nan for r in E], dtype=float)
    # timings only from the reference machine (results/hosts.json); learning metrics use every seed
    d["cpu_s"] = np.array([r["cpu_seconds"] for r in timing_runs(runs)])
    d["steps"] = np.array([r["total_steps"] for r in timing_runs(runs)], dtype=float)
    return d


def welch(a, b) -> tuple[float, float]:
    """Welch's t-test (unequal variances); returns (t, two-sided p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = st.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def collapses(E: np.ndarray, peak: float, fall_to: float) -> np.ndarray:
    """Per seed: did the greedy return reach `peak` and LATER fall below `fall_to`? (forgetting events)"""
    out = []
    for row in E:
        hit = np.flatnonzero(row >= peak)
        out.append(bool(len(hit)) and bool((row[hit[0]:] < fall_to).any()))
    return np.array(out)


# =============================================================================================
def cartpole() -> None:
    exp = "tier1_cartpole"
    say("## 1. Tier 1 - CartPole-v1")
    names = ["tabular_bins6", "tabular_bins10", "tabular_bins20", "tabular_angle_coarse",
             "tabular_angle_fine_center", "dqn_eps5k", "dqn_eps25k", "dqn_eps100k"]
    R = {n: load_runs(exp, n) for n in names}
    D = {n: per_seed(r, 475.0) for n, r in R.items() if r}
    res = {}

    say("\n### 1.1 Summary (greedy evaluation)")
    say("| config | final | best | AUC | drop (forgetting) | steps to 475 | seeds solved | ms CPU / step |")
    say("|---|---|---|---|---|---|---|---|")
    for n, d in D.items():
        solved = int(np.sum(~np.isnan(d["steps_to_threshold"])))
        ms = 1000 * np.mean(d["cpu_s"] / d["steps"])
        say(f"| {n} | {ci(d['final'])} | {ci(d['best'])} | {ci(d['auc'])} | {ci(d['drop'])} | "
            f"{ci(d['steps_to_threshold'] / 1e3)}k | {solved}/{len(d['final'])} | {ms:.3f} |")
        res[n] = {k: v.tolist() for k, v in d.items()}

    say("\n### 1.2 Sample efficiency vs compute (tabular 10 bins vs best DQN)")
    tab, dqn = D["tabular_bins10"], D["dqn_eps5k"]
    s_tab, s_dqn = np.nanmean(tab["steps_to_threshold"]), np.nanmean(dqn["steps_to_threshold"])
    ms_tab = np.mean(tab["cpu_s"] / tab["steps"])
    ms_dqn = np.mean(dqn["cpu_s"] / dqn["steps"])
    cpu_tab, cpu_dqn = s_tab * ms_tab, s_dqn * ms_dqn
    say(f"- steps to first greedy >= 475: tabular {s_tab / 1e3:.0f}k vs DQN {s_dqn / 1e3:.0f}k "
        f"-> DQN is {s_tab / s_dqn:.1f}x more SAMPLE-efficient")
    say(f"- CPU per env step: tabular {1e3 * ms_tab:.3f} ms vs DQN {1e3 * ms_dqn:.3f} ms -> {ms_dqn / ms_tab:.0f}x")
    say(f"- CPU-seconds to reach 475: tabular {cpu_tab:.0f} s vs DQN {cpu_dqn:.0f} s "
        f"-> tabular is {cpu_dqn / cpu_tab:.1f}x more COMPUTE-efficient")
    # equal-budget comparison at 200k steps
    at = {}
    for n in ["tabular_bins6", "tabular_bins10", "tabular_bins20", "dqn_eps5k", "dqn_eps25k"]:
        s, E = eval_matrix(R[n])
        at[n] = E[:, (s >= 180_000) & (s <= 200_000)].mean(1)
    say("- greedy return at an EQUAL budget of 200k steps (mean of checkpoints 180k-200k): "
        + "; ".join(f"{n} {ci(v)}" for n, v in at.items()))
    t, p = welch(at["dqn_eps5k"], at["tabular_bins10"])
    say(f"- Welch t-test DQN(eps5k) vs tabular(10 bins) at 200k: t = {t:.2f}, p = {p:.2g}")
    res["efficiency"] = {"steps_tab": s_tab, "steps_dqn": s_dqn, "ms_tab": 1e3 * ms_tab, "ms_dqn": 1e3 * ms_dqn,
                         "cpu_to_475_tab": cpu_tab, "cpu_to_475_dqn": cpu_dqn,
                         "at200k": {k: v.tolist() for k, v in at.items()}, "welch_200k": [t, p]}

    say("\n### 1.3 Number of bins: precision vs data")
    for n in ["tabular_bins6", "tabular_bins10", "tabular_bins20"]:
        cov = np.array([r["state_coverage"] for r in R[n]])
        ns = R[n][0]["n_states"]
        say(f"- {n}: {ns:,} states, visited {100 * cov.mean():.1f}% (~{cov.mean() * ns:,.0f} states); "
            f"final {ci(D[n]['final'])}; seed-to-seed std of final = {np.std(D[n]['final'], ddof=1):.0f}")
    t, p = welch(D["tabular_bins6"]["final"], D["tabular_bins20"]["final"])
    say(f"- Welch t-test final(6 bins) vs final(20 bins): t = {t:.2f}, p = {p:.2g}")

    say("\n### 1.4 Pole-angle boundaries (other dims 10 bins)")
    early = {}
    for n in ["tabular_bins10", "tabular_angle_coarse", "tabular_angle_fine_center"]:
        s, E = eval_matrix(R[n])
        early[n] = E[:, (s > 0) & (s <= 200_000)].mean(1)
        say(f"- {n}: mean greedy return over first 200k steps {ci(early[n])}; AUC {ci(D[n]['auc'])}; "
            f"final {ci(D[n]['final'])}")
    t, p = welch(early["tabular_angle_coarse"], early["tabular_bins10"])
    say(f"- Welch t-test early(coarse angle) vs early(uniform): t = {t:.2f}, p = {p:.2g}")

    say("\n### 1.5 Epsilon decay and catastrophic forgetting (DQN)")
    for n in ["dqn_eps5k", "dqn_eps25k", "dqn_eps100k"]:
        s, E = eval_matrix(R[n])
        e50 = E[:, (s > 0) & (s <= 50_000)].mean(1)
        late = E[:, s >= 150_000].mean(1)
        col = collapses(E, peak=475, fall_to=200)
        say(f"- {n}: early (<=50k) {ci(e50)}; late (>=150k) {ci(late)}; drop {ci(D[n]['drop'])}; "
            f"seeds that reached >=475 and later fell below 200: {int(col.sum())}/{len(col)}")
    OUT["cartpole"] = res


# =============================================================================================
def mountaincar() -> None:
    exp = "tier1_mountaincar"
    say("\n## 2. Tier 1 - MountainCar-v0 (sparse reward)")
    names = ["tabular_optimistic", "tabular_pessimistic", "dqn", "dqn_long_eps"]
    R = {n: load_runs(exp, n) for n in names}
    res = {}
    say("| config | final greedy | best greedy | seeds that ever reached the flag (training) | "
        "median first flag | flag episodes / run | final mean Q (last 50k) |")
    say("|---|---|---|---|---|---|---|")
    for n, runs in R.items():
        if not runs:
            continue
        d = per_seed(runs)
        first = np.array([next((t for t, x in zip(r["train_episode_end_step"], r["train_terminated"]) if x), np.nan)
                          for r in runs], dtype=float)
        succ = np.array([np.sum(r["train_terminated"]) for r in runs])
        qfin = "--"
        if runs[0]["spec"]["agent"] == "dqn":
            _, Q = stat_matrix(runs, "q_mean")
            qfin = ci(np.nanmean(Q[:, -50:], axis=1))
        med = f"{np.nanmedian(first) / 1e3:.0f}k" if (~np.isnan(first)).any() else "never"
        say(f"| {n} | {ci(d['final'])} | {ci(d['best'])} | {int(np.sum(~np.isnan(first)))}/{len(runs)} | {med} | "
            f"{succ.mean():.0f} | {qfin} |")
        res[n] = {"final": d["final"].tolist(), "first_flag": first.tolist(), "flag_episodes": succ.tolist()}
    gamma = R["dqn"][0]["spec"]["agent_kwargs"]["gamma"] if R["dqn"] else 0.99
    say(f"\n- reference: if the flag is never seen, every Q converges to -1/(1-gamma) = {-1 / (1 - gamma):.0f}")
    if R["dqn"]:
        never = [r for r in R["dqn"] + R["dqn_long_eps"] if not any(r["train_terminated"])]
        seen = [r for r in R["dqn"] + R["dqn_long_eps"] if any(r["train_terminated"])]
        for label, group in [("DQN runs that NEVER reached the flag", never), ("DQN runs that DID reach it", seen)]:
            if group:
                _, Q = stat_matrix(group, "q_mean")
                say(f"- {label}: n={len(group)}, final mean Q {ci(np.nanmean(Q[:, -50:], axis=1))}, "
                    f"final greedy {ci([np.mean(r['eval_return_mean'][-3:]) for r in group])}")
        # divergence after the first flag: every true Q is <= 0 here (reward -1 per step, 0 never paid)
        say("- Q divergence check (all true values are <= 0 in MountainCar):")
        for r in seen:
            q = np.array([s["q_mean"] for s in r["stats"]])
            g = np.array([s["grad_norm"] for s in r["stats"]])
            steps = np.asarray(r["stats_step"])
            first = next(t for t, x in zip(r["train_episode_end_step"], r["train_terminated"]) if x)
            pre, post = steps < first, steps >= first
            say(f"  - {r['spec']['config']} seed {r['spec']['seed']}: first flag at {first / 1e3:.0f}k; "
                f"mean Q before {np.nanmean(q[pre]):.0f}, max after {np.nanmax(q[post]):.0f}; "
                f"grad-norm before {np.nanmedian(g[pre]):.1f}, max after {np.nanmax(g[post]):.0f}")
    OUT["mountaincar"] = res


# =============================================================================================
def lunarlander() -> None:
    exp = "tier2_lunarlander"
    say("\n## 3. Tier 2 - LunarLander-v3 (ablations)")
    names = ["full", "no_target", "no_replay", "double", "dueling"]
    R = {n: load_runs(exp, n) for n in names}
    D = {n: per_seed(r, 200.0) for n, r in R.items() if r}
    res = {}
    say("| config | final | best | AUC | drop | seeds solved (>=200) | Welch p vs full (final) | final mean Q | "
        "max grad-norm (median over seeds) | CPU min/run |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for n, d in D.items():
        solved = int(np.sum(~np.isnan(d["steps_to_threshold"])))
        p = "--" if n == "full" else f"{welch(d['final'], D['full']['final'])[1]:.2g}"
        _, Q = stat_matrix(R[n], "q_mean")
        _, G = stat_matrix(R[n], "grad_norm")
        say(f"| {n} | {ci(d['final'])} | {ci(d['best'])} | {ci(d['auc'])} | {ci(d['drop'])} | "
            f"{solved}/{len(d['final'])} | {p} | {np.nanmean(Q[:, -50:]):.1f} | "
            f"{np.median(np.nanmax(G, axis=1)):.1f} | {np.mean(d['cpu_s']) / 60:.1f} |")
        res[n] = {k: v.tolist() for k, v in d.items()}

    say("\n- Q-value level vs policy quality. Mean Q on training batches (discounted value of arbitrary "
        "visited states) is not directly comparable to an episode return, so we compare it ACROSS configs: "
        "a config whose Q is far above full DQN's while its policy is far worse is overestimating.")
    qf = np.nanmean(stat_matrix(R["full"], "q_mean")[1][:, -50:]) if "full" in R else np.nan
    for n in names:
        if n in D:
            _, Q = stat_matrix(R[n], "q_mean")
            q = np.nanmean(Q[:, -50:])
            say(f"  - {n}: final mean Q {q:.1f} ({q / qf:.2f}x full DQN) with final greedy return "
                f"{np.mean(D[n]['final']):.1f}")
    for n in ["no_target", "no_replay", "full", "double", "dueling"]:
        if n in R:
            _, E = eval_matrix(R[n])
            col = collapses(E, peak=200, fall_to=0)
            say(f"- {n}: seeds that reached >=200 and later fell below 0: {int(col.sum())}/{len(col)}")
    OUT["lunarlander"] = res


# =============================================================================================
def lunarlander_hparams() -> None:
    """One-at-a-time study (experiments/tier2_hparams.py). Compared on the seeds the study ran,
    so the default (`full`) row uses the same seeds as the varied rows."""
    exp = "tier2_lunarlander"
    groups = {"learning rate": ["lr_0.0001", "full", "lr_0.002"],
              "target update period": ["target_update_freq_250", "full", "target_update_freq_4000"]}
    study_seeds = sorted({r["spec"]["seed"] for n in ["lr_0.0001", "lr_0.002"] for r in load_runs(exp, n)})
    if not study_seeds:
        return
    say("\n## 3b. Tier 2 - LunarLander-v3 hyper-parameter study (one at a time around the default)")
    say(f"- seeds compared: {study_seeds}")
    say("| parameter | config | final | best | AUC | drop | seeds solved (>=200) | Welch p vs default (final) |")
    say("|---|---|---|---|---|---|---|---|")
    res = {}
    for g, names in groups.items():
        D = {}
        for n in names:
            runs = [r for r in load_runs(exp, n) if r["spec"]["seed"] in study_seeds]
            if runs:
                D[n] = per_seed(runs, 200.0)
        for n, d in D.items():
            solved = int(np.sum(~np.isnan(d["steps_to_threshold"])))
            p = "--" if n == "full" else f"{welch(d['final'], D['full']['final'])[1]:.2g}"
            say(f"| {g} | {n} | {ci(d['final'])} | {ci(d['best'])} | {ci(d['auc'])} | {ci(d['drop'])} | "
                f"{solved}/{len(d['final'])} | {p} |")
            res[n] = {k: v.tolist() for k, v in d.items()}
    OUT["lunarlander_hparams"] = res


# =============================================================================================
def compute_budget() -> None:
    say("\n## 4. Compute budget")
    total_cpu, total_runs = 0.0, 0
    for p in sorted(RESULTS.glob("tier*/*/seed_*.json")):
        if "_quick" in str(p):
            continue
        r = json.loads(p.read_text())
        total_cpu += r["cpu_seconds"]
        total_runs += 1
    say(f"- {total_runs} runs, {total_cpu / 3600:.1f} CPU-hours in total (one thread per run, all machines; "
        f"see results/hosts.json for which machine ran which seeds)")
    OUT["compute"] = {"runs": total_runs, "cpu_hours": total_cpu / 3600}


if __name__ == "__main__":
    say("# Quantitative analysis (generated by analysis.py)\n")
    say("Mean ± 95% Student-t CI over seeds. Greedy = epsilon 0, frozen weights. "
        "Final = mean of the last 3 checkpoints.\n")
    cartpole()
    mountaincar()
    lunarlander()
    lunarlander_hparams()
    compute_budget()
    (RESULTS / "analysis.json").write_text(json.dumps(OUT, indent=1, default=float))
    (RESULTS / "analysis.md").write_text("\n".join(LINES) + "\n")
    print("\nwrote results/analysis.json and results/analysis.md")
