# AI assistance: Codex drafted and executed this competition pipeline.
"""Run the configuration locked by expanded_study.py, then reproduce its results.

Usage: python run_selected.py [--verify]
All inputs are relative to this file, not the current working directory.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from expanded_study import train_candidate
from solution import evaluate


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "expanded_results"
    config = json.loads((root / "expanded_config.json").read_text())
    with (root / "seeds_eval.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["seed"]
        seeds = [int(r["seed"]) for r in reader]
    assert len(seeds) == len(set(seeds)) == 10
    q, training = train_candidate(config, 20261001)
    if config["method"] == "mc":
        assert training["completed_episodes"] == 30000
    else:
        assert training["transitions"] == 30000
    frozen = q.copy()
    rows = [evaluate(q, seed, 2000) for seed in seeds]
    assert np.array_equal(frozen, q)
    values = np.array([r["total_reward"] for r in rows])
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    report = dict(config=config, training=training, runs=rows,
                  mean_reward=float(values.mean()), std_reward=float(values.std(ddof=1)),
                  best_reward=float(values.max()), worst_reward=float(values.min()),
                  optimal_reward=None, gap_to_optimal_reward=None, official_mae=None,
                  elapsed_seconds=time.perf_counter()-start,
                  q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                  python=platform.python_version(), numpy=np.__version__, gymnasium=gym.__version__,
                  protocol="User-approved shared table; 100-step horizon; 2000 episodes per seed; each fresh evaluation environment seeded once; frozen greedy Q policy. Professor clarified MC=30000 episodes, TD=30000 total transitions.")
    assert report["elapsed_seconds"] < 120, "Runtime exceeded cap; output not approved"
    if args.verify:
        original = json.loads((output / "report.json").read_text())
        assert original["q_sha256"] == report["q_sha256"]
        for actual, expected in zip(rows, original["runs"]):
            for key in actual:
                if key != "seconds":
                    assert actual[key] == expected[key]
        with (output / "submission.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["seed", "total_reward"]
            submitted = list(reader)
        assert len(submitted) == 10
        assert [int(r["seed"]) for r in submitted] == seeds
        assert [float(r["total_reward"]) for r in submitted] == list(values)
        validation = dict(ten_rows=True, correct_columns=True, exact_training_replay=True,
                          exact_evaluation_replay=True, frozen_q=True,
                          budget_respected=True, within_120_seconds=True)
        (output / "validation.json").write_text(json.dumps(validation, indent=2))
        print(json.dumps(validation, indent=2))
    else:
        output.mkdir(exist_ok=True)
        with (output / "submission.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "total_reward"])
            writer.writeheader()
            writer.writerows({key: r[key] for key in writer.fieldnames} for r in rows)
        np.save(output / "q_table.npy", q)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
