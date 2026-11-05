"""Record one greedy episode of trained agents as GIFs (qualitative evidence for the report/defense).

Which seed: by default the MEDIAN seed of the config, ranked by its 100-episode final evaluation
(results/final_eval100.json from eval100.py), so the clip shows a typical agent, not the luckiest one.
`--pick best` uses the best seed instead. Without final_eval100.json, seeds are ranked by the mean of
their last 3 greedy checkpoints.

Output: results/videos/<env>__<config>__seed<k>.gif
Usage:  python experiments/make_gifs.py [--pick median|best]
"""
from __future__ import annotations

import argparse
import json

import _common  # noqa: F401
import numpy as np

from _common import ROOT

RESULTS = ROOT / "results"
OUT = RESULTS / "videos"
# (experiment, config): the main agent of each study plus the instructive failures
CLIPS = [
    ("tier1_cartpole", "tabular_bins10"),
    ("tier1_cartpole", "dqn_eps5k"),
    ("tier1_mountaincar", "tabular_optimistic"),
    ("tier1_mountaincar", "dqn"),
    ("tier2_lunarlander", "full"),
    ("tier2_lunarlander", "no_replay"),
]
WIDTH = 360        # px; keeps every GIF well under a few MB
MAX_FRAMES = 400   # long episodes are sub-sampled evenly to at most this many frames


def seed_scores(exp: str, config: str) -> dict[int, float]:
    cache_file = RESULTS / "final_eval100.json"
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        scores = {int(k.split("seed_")[1].split(".")[0]): float(np.mean(v))
                  for k, v in cache.items() if k.startswith(f"{exp}/{config}/")}
        if scores:
            return scores
    scores = {}
    for p in (RESULTS / exp / config).glob("seed_*.json"):
        r = json.loads(p.read_text())
        scores[r["spec"]["seed"]] = float(np.mean(r["eval_return_mean"][-3:]))
    return scores


def pick_seed(scores: dict[int, float], how: str) -> int:
    ranked = sorted(scores, key=lambda s: (scores[s], -s))
    return ranked[-1] if how == "best" else ranked[(len(ranked) - 1) // 2]


def record(exp: str, config: str, seed: int) -> None:
    import gymnasium as gym
    from PIL import Image, ImageDraw

    from common.policy import load_policy

    json_path = RESULTS / exp / config / f"seed_{seed}.json"
    spec = json.loads(json_path.read_text())["spec"]
    env = gym.make(spec["env_id"], render_mode="rgb_array", **spec.get("env_kwargs", {}))
    policy = load_policy(json_path, env.observation_space.shape[0], env.action_space.n)

    obs, _ = env.reset(seed=3_000_000 + seed)  # a start state never used in training or evaluation
    frames, ret, done = [env.render()], 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy(obs))
        ret += r
        done = term or trunc
        frames.append(env.render())
    env.close()

    n = len(frames)
    keep = np.unique(np.linspace(0, n - 1, min(n, MAX_FRAMES)).round().astype(int))
    fps = env.metadata.get("render_fps", 50) * len(keep) / n  # real-time playback after sub-sampling
    images = []
    for i in keep:
        im = Image.fromarray(frames[i])
        im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
        # white text with a black outline: readable on CartPole's white and LunarLander's black backgrounds
        ImageDraw.Draw(im).text((6, 4), f"{config}  seed {seed}  return {ret:.0f}", fill=(255, 255, 255),
                                stroke_width=2, stroke_fill=(0, 0, 0))
        images.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{spec['env_id']}__{config}__seed{seed}.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0,
                   optimize=True)
    print(f"  wrote videos/{out.name}  ({n} steps, return {ret:.1f}, {out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick", choices=["median", "best"], default="median")
    a = ap.parse_args()
    for exp, config in CLIPS:
        scores = seed_scores(exp, config)
        if not scores:
            print(f"  skip {exp}/{config}: no runs")
            continue
        record(exp, config, pick_seed(scores, a.pick))


if __name__ == "__main__":
    main()
