"""Aggregation across seeds: step-aligned curves, 95% confidence intervals, summary metrics."""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def mean_ci(values: np.ndarray, axis: int = 0, level: float = 0.95):
    """Mean and half-width of a Student-t confidence interval across seeds (axis 0).

    With n seeds: half = t_{n-1, 0.975} * std(ddof=1) / sqrt(n). With few seeds (3-5) the t
    multiplier (2.78 for n=5) is noticeably larger than the normal 1.96, which is the honest choice.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    if n < 2:
        return mean, np.zeros_like(mean)
    sem = values.std(axis=axis, ddof=1) / np.sqrt(n)
    return mean, sem * st.t.ppf(0.5 + level / 2, n - 1)


def eval_matrix(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """[n_seeds, n_checkpoints] greedy-evaluation returns (all runs share the same checkpoints)."""
    steps = np.asarray(runs[0]["eval_step"])
    return steps, np.array([r["eval_return_mean"] for r in runs])


def train_matrix(runs: list[dict], grid: np.ndarray, window: int = 20) -> np.ndarray:
    """Training returns put on a common step grid.

    Training episodes end at different steps in every run, so for each run we take a moving
    average over the last `window` episodes, indexed by the step at which each episode ended,
    and read it off at every grid point (value of the latest finished episode, i.e. a step
    function). Grid points before the first finished episode get NaN.
    """
    out = np.full((len(runs), len(grid)), np.nan)
    for i, r in enumerate(runs):
        ends = np.asarray(r["train_episode_end_step"])
        rets = np.asarray(r["train_return"], dtype=float)
        if len(rets) == 0:
            continue
        csum = np.cumsum(np.insert(rets, 0, 0.0))
        idx = np.arange(1, len(rets) + 1)
        lo = np.maximum(idx - window, 0)
        smooth = (csum[idx] - csum[lo]) / (idx - lo)
        pos = np.searchsorted(ends, grid, side="right") - 1
        ok = pos >= 0
        out[i, ok] = smooth[pos[ok]]
    return out


def stat_matrix(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    steps = np.asarray(runs[0]["stats_step"])
    return steps, np.array([[s.get(key, np.nan) for s in r["stats"]] for r in runs], dtype=float)


def timing_runs(runs: list[dict]) -> list[dict]:
    """Runs whose CPU/wall-clock times are comparable: those run on the reference machine.

    results/hosts.json lists which seeds ran on which machine. Learning curves always use every seed,
    but timings from different CPUs must not be averaged, so we keep only the 'laptop' seeds.
    Without hosts.json (e.g. a fresh single-machine reproduction) every run is used.
    """
    from pathlib import Path
    import json

    hosts = Path(__file__).resolve().parents[1] / "results" / "hosts.json"
    if not runs or not hosts.exists():
        return runs
    exp = runs[0]["spec"].get("experiment", "")
    seeds = json.loads(hosts.read_text()).get(exp, {}).get("laptop")
    if seeds is None:
        return runs
    keep = [r for r in runs if r["spec"]["seed"] in seeds]
    return keep or runs


def summarize(runs: list[dict], threshold: float | None = None, last_k: int = 3) -> dict:
    """Per-config metrics, each computed per seed and then reported as mean +- 95% CI.

    final  = mean greedy return over the last `last_k` checkpoints
    best   = best checkpoint
    auc    = mean over ALL checkpoints (rewards learning speed AND stability)
    drop   = best - final: how much performance was lost after the peak (catastrophic forgetting)
    steps_to_threshold = first checkpoint whose greedy return >= threshold (NaN if never)
    """
    steps, E = eval_matrix(runs)
    per_seed = {
        "final": E[:, -last_k:].mean(axis=1),
        "best": E.max(axis=1),
        "auc": E.mean(axis=1),
    }
    per_seed["drop"] = per_seed["best"] - per_seed["final"]
    if threshold is not None:
        hit = E >= threshold
        per_seed["steps_to_threshold"] = np.array(
            [steps[np.argmax(h)] if h.any() else np.nan for h in hit], dtype=float)
        per_seed["solved_frac"] = hit.any(axis=1).astype(float)
    per_seed["cpu_seconds"] = np.array([r["cpu_seconds"] for r in timing_runs(runs)])
    per_seed["wall_seconds"] = np.array([r["wall_seconds"] for r in timing_runs(runs)])
    per_seed["episodes"] = np.array([r["n_episodes"] for r in runs], dtype=float)

    out = {"n_seeds": len(runs)}
    for k, v in per_seed.items():
        good = v[~np.isnan(v)]
        if len(good):
            m, h = mean_ci(good)
            out[k] = (float(m), float(h), int(len(good)))
        else:
            out[k] = (float("nan"), float("nan"), 0)
    return out
