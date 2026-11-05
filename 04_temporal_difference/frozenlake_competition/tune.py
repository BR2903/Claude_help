"""Independent development experiments. Never uses the official seed file."""
import json
from pathlib import Path
from solution import DEFAULT, train, evaluate

ROOT = Path(__file__).resolve().parent
configs = []
for gamma in (0.95, 0.99, 0.995):
    for initial in (0.0, 0.5):
        configs.append({**DEFAULT, "gamma": gamma, "initial_q": initial})
configs.append({**DEFAULT, "method": "sarsa"})
configs.append({**DEFAULT, "epsilon_start": 0.5, "epsilon_end": 0.1})
records = []
for index, config in enumerate(configs):
    runs = []
    for seed in (1101, 1102, 1103, 1104, 1105):
        q, training = train(config, seed)
        result = evaluate(q, seed + 100000, 300)
        runs.append(dict(training=training, evaluation=result))
    mean = sum(r["evaluation"]["total_reward"] for r in runs) / len(runs)
    record = dict(index=index, config=config, mean=mean, runs=runs)
    records.append(record)
    (ROOT / "tuning.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(index=index, mean=mean, config=config)), flush=True)
finalists = sorted(records, key=lambda r: r["mean"], reverse=True)[:2]
for record in finalists:
    holdout = []
    for seed in (2201, 2202, 2203, 2204, 2205):
        q, training = train(record["config"], seed)
        holdout.append(dict(training=training, evaluation=evaluate(q, seed + 100000, 1000)))
    record["holdout"] = holdout
    record["holdout_mean"] = sum(r["evaluation"]["total_reward"] for r in holdout) / len(holdout)
    print(json.dumps(dict(index=record["index"], holdout_mean=record["holdout_mean"])), flush=True)
winner = max(finalists, key=lambda r: r["holdout_mean"])
(ROOT / "tuning.json").write_text(json.dumps(records, indent=2))
(ROOT / "config.json").write_text(json.dumps(winner["config"], indent=2))
print("Selected configuration " + str(winner["index"]), flush=True)
