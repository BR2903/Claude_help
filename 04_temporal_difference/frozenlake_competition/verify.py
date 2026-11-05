"""Verify submitted measurements by repeating the full frozen pipeline."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from solution import train, evaluate

root = Path(__file__).resolve().parent
start = time.perf_counter()
report = json.loads((root / "results/report.json").read_text())
with (root / "results/submission.csv").open(newline="") as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == ["seed", "total_reward"]
    submitted = list(reader)
with (root / "seeds_eval.csv").open(newline="") as f:
    expected = [int(row["seed"]) for row in csv.DictReader(f)]
assert [int(row["seed"]) for row in submitted] == expected
assert len(submitted) == len(set(expected)) == 10
q, training = train(report["config"], report["training"]["seed"])
assert training["transitions"] == 30000
assert hashlib.sha256(q.tobytes()).hexdigest() == report["q_sha256"]
frozen = q.copy()
for row, original in zip(submitted, report["runs"]):
    replay = evaluate(q, int(row["seed"]))
    for key in ("seed", "episodes", "successes", "holes", "truncations", "transitions", "total_reward"):
        assert replay[key] == original[key], (key, replay, original)
    assert replay["total_reward"] == float(row["total_reward"])
assert np.array_equal(q, frozen)
validation = dict(ten_rows=True, correct_columns=True, distinct_seeds=True,
                  training_transitions=30000, evaluation_episodes=20000,
                  frozen_q=True, exact_training_replay=True,
                  exact_evaluation_replay=True, seconds=time.perf_counter()-start)
(root / "results/validation.json").write_text(json.dumps(validation, indent=2))
print(json.dumps(validation, indent=2))
