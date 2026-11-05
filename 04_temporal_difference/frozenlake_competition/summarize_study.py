"""Build an auditable JSON summary from measured experiment files."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
study = json.loads((root / "robustness_study.json").read_text())
expanded = json.loads((root / "expanded_study.json").read_text())
final = json.loads((root / "expanded_results/report.json").read_text())
original = json.loads((root / "results/report.json").read_text())
validation = json.loads((root / "expanded_results/validation.json").read_text())
winner = next(c for c in study["candidates"] if c["name"] == study["selected"])
incumbent = next(c for c in study["candidates"] if c["name"] == "incumbent")
differences = np.array([a["evaluation"]["total_reward"] - b["evaluation"]["total_reward"] for a,b in zip(winner["confirmation"]["runs"], incumbent["confirmation"]["runs"])])
halfwidth = 2.776445105 * differences.std(ddof=1) / np.sqrt(len(differences))
summary = dict(
    budget_clarification=dict(source="Professor clarification reported by user", mc_training_episodes=30000, td_total_training_steps=30000, runtime_seconds=120),
    selected_method=final["config"]["method"], selected_configuration=study["selected"], hyperparameters=final["config"],
    comparison=[dict(name=c["name"], method=c["config"]["method"], **c["confirmation"]["summary"]) for c in study["candidates"] if "confirmation" in c],
    expanded_screen=[dict(name=c["name"], **c["screen"]["summary"]) for c in expanded["candidates"]],
    paired_comparison=dict(training_seeds=study["confirm_seeds"], mean_difference=float(differences.mean()), differences=list(differences), descriptive_t_interval_95=[float(differences.mean()-halfwidth),float(differences.mean()+halfwidth)], limitation="Five pairs; selection used these data. Interval is descriptive, not adjusted for model selection."),
    official_seed_evaluation=dict(mean_reward=final["mean_reward"], std_reward=final["std_reward"], best_reward=final["best_reward"], worst_reward=final["worst_reward"], seconds=final["elapsed_seconds"], rows=[dict(seed=r["seed"],total_reward=r["total_reward"]) for r in final["runs"]]),
    previous_submission_comparison=dict(mean_change=final["mean_reward"]-original["mean_reward"], identical_greedy_policy=bool(np.array_equal(np.load(root / "results/q_table.npy").argmax(axis=1), np.load(root / "expanded_results/q_table.npy").argmax(axis=1))), identical_submission=(root / "results/submission.csv").read_bytes()==(root / "expanded_results/submission.csv").read_bytes()),
    reference=dict(optimal_reward=None,gap_to_optimal_reward=None,official_mae=None,can_achieve_one=None),
    interpretation=dict(total_reward="Number of successful episodes divided by 2000; each episode contributes zero or one.", score="Mean absolute per-seed error against unknown empirical reference values. Higher success need not monotonically lower this MAE.", scope="One shared trained table; official-seed spread is evaluation variation, not independent training variability."),
    unresolved=["Official environment version, map, horizon, RNG coupling, and training-per-seed protocol are not verified.","The previously user-approved 100-step horizon and shared-table protocol remain assumptions, not newly confirmed organizer rules.","No reference score or leaderboard threshold is available; winning and closeness to optimal cannot be established."],
    validation=validation)
(root / "updated_analysis.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
