# FrozenLake competition deliverables

AI assistance: Codex drafted and executed the implementation and development experiments. The student should review the code and disclose assistance accurately under the course policy.

Run `python solution.py` after installing `requirements.txt`. The script reads `config.json` and `seeds_eval.csv` beside itself and writes `results/submission.csv`, `results/report.json`, and `results/q_table.npy`. Only upload `submission.csv` to the numerical submission field. No upload was performed.

## Evaluation contract

The user approved these assumptions in chat: one shared table trained for at most 30,000 total transitions, followed by 2,000 frozen-policy episodes for each official seed, using the standard 4x4 map and a 100-step horizon. This is conservative about the final script's training budget but is not an independently verified organizer evaluator. Each evaluation seed initializes a fresh environment once, with later resets continuing its random stream. Training uses the predeclared independent seed 20261001. Evaluation actions use deterministic first-maximum Q-table tie-breaking. Evaluation never updates the table. Termination and horizon truncation both end returns under the approved finite-horizon interpretation; a stationary table cannot represent time-dependent action values exactly.

Only sampled state, action, reward, next-state, and termination data enter updates. The code never reads a transition model, environment internals, transition probabilities, a reference policy, or optimal targets. Returned info dictionaries are discarded. Rewards remain the environment's original 0 and 1. No state-specific or seed-specific policy rules exist.

## Selection evidence

`tune.py` evaluates eight predeclared configurations on five development training seeds (1101–1105), with separate evaluation streams and 300 episodes per configuration/run. The two leading configurations receive five fresh training seeds (2201–2205) and 1,000 evaluation episodes each. `tuning.json` retains every outcome, including worse runs. Official seeds are never used in configuration selection. This bounded search compares seven Q-learning settings and one SARSA setting; it does not prove global superiority over all allowed algorithms.

Selected: Q-learning, gamma 0.99, initial Q=0, learning rate 0.5/(1+N(s,a)/50)^0.6, and linear epsilon decay from 1.0 to 0.05 over the first 24,000 transitions. The rate uses the visit count after incrementing it. Epsilon stays at 0.05 for the remaining 6,000 transitions. Random training tie-breaking uses a separate learner RNG. Development experiments are separate from the final end-to-end submission run and consumed additional training interactions; if the organizer intended a cumulative cap across all development, clarification is necessary.

The selected configuration achieved 0.731333 mean development success and 0.745 on the fresh development comparison. The runner-up achieved 0.736 and 0.7068 respectively. These results motivated selection without using official-seed feedback. `tune.py` is an audit/reproduction tool, not part of the timed final submission command.

## Interpretation

`total_reward` is successes/2000, not the sum of all rewards. The official-seed standard deviation uses the sample definition (ddof=1). Because all ten rows use the same trained table, that spread measures evaluation variation conditional on this table; it does not measure variation across independently trained agents. Training-seed variation is recorded separately in `tuning.json`.

The organizer's optimal reference values and precise seed synchronization are unavailable. Optimal reward, gap to optimal, and official MAE remain null. A hypothetical reward of 1 must not be presented as the optimal reachable success rate. The script checks its own measured runtime against 120 seconds, but timing on the organizer's hardware is not guaranteed.
