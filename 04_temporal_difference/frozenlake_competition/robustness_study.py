# AI assistance: Codex drafted and executed this bounded follow-up comparison.
"""Fresh development seeds only; all TD runs use exactly 30,000 steps."""
import json
from pathlib import Path
from expanded_study import execute

ROOT = Path(__file__).resolve().parent


def main():
    previous = json.loads((ROOT / "expanded_study.json").read_text())
    incumbent = next(c for c in previous["candidates"] if c["name"] == previous["selected"])
    base = incumbent["config"]
    configs = [
        ("incumbent", dict(base)),
        ("alpha_01", {**base, "alpha": 0.1}),
        ("alpha_015", {**base, "alpha": 0.15}),
        ("constant_005", {**base, "alpha": 0.05, "alpha_power": 0.0}),
        ("optimistic_01", {**base, "initial_q": 0.1}),
        ("optimistic_03", {**base, "initial_q": 0.3}),
        ("epsilon_end_001", {**base, "epsilon_end": 0.01}),
        ("epsilon_decay_05", {**base, "decay_fraction": 0.5}),
        ("gamma_1_slow", {**base, "gamma": 1.0})
    ]
    study = dict(screen_seeds=[5501,5502,5503], confirm_seeds=[6601,6602,6603,6604,6605],
                 design="Nine Q settings on 3 fresh paired seeds with 1000 tests; compare top two plus incumbent and best earlier SARSA/MC on 5 fresh paired seeds with 2000 tests. Pick highest confirmation mean, incumbent on exact tie. No official seeds.", candidates=[])
    path = ROOT / "robustness_study.json"
    for name, config in configs:
        candidate = dict(name=name, config=config, screen=execute(config, study["screen_seeds"], 1000))
        study["candidates"].append(candidate)
        path.write_text(json.dumps(study, indent=2))
        print(json.dumps(dict(stage="screen", name=name, **candidate["screen"]["summary"])), flush=True)
    top = sorted(study["candidates"], key=lambda r:r["screen"]["summary"]["mean"], reverse=True)[:2]
    finalists = {"incumbent", *(r["name"] for r in top)}
    for family in ("sarsa", "mc"):
        winner = max((c for c in previous["candidates"] if c["config"]["method"] == family), key=lambda r:r["screen"]["summary"]["mean"])
        study["candidates"].append(dict(name=winner["name"], config=winner["config"], prior_screen=winner["screen"]))
        finalists.add(winner["name"])
    for candidate in study["candidates"]:
        if candidate["name"] in finalists:
            candidate["confirmation"] = execute(candidate["config"], study["confirm_seeds"], 2000)
            path.write_text(json.dumps(study, indent=2))
            print(json.dumps(dict(stage="confirmation", name=candidate["name"], **candidate["confirmation"]["summary"])), flush=True)
    winner = max((r for r in study["candidates"] if "confirmation" in r), key=lambda r:r["confirmation"]["summary"]["mean"])
    study["selected"] = winner["name"]
    path.write_text(json.dumps(study, indent=2))
    (ROOT / "expanded_config.json").write_text(json.dumps(winner["config"], indent=2))
    print("SELECTED " + winner["name"], flush=True)


if __name__ == "__main__":
    main()
