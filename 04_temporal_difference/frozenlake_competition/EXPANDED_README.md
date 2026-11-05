# Updated comparison after the professor's budget clarification

AI assistance: Codex drafted and executed the implementation, development comparisons, and verification code. Review and disclose this assistance under the course policy.

## Confirmed budget correction

The professor clarified, as reported by the user: Monte Carlo receives at most 30,000 training episodes; SARSA and Q-learning receive at most 30,000 total training steps. A TD run here makes exactly 30,000 calls to step(). A Monte Carlo run completes exactly 30,000 episodes. It is legitimate for those MC episodes to contain more than 30,000 transitions under this method-specific allowance. Each final pipeline must also meet the end-to-end 120-second cap.

The previous user-approved evaluation choices remain in force: one shared trained Q-table, the standard 4x4 map, a 100-step episode horizon, and 2,000 frozen-policy episodes per official seed. These choices are not independently established organizer rules. The supplied material still lacks the exact evaluator, reference values, RNG synchronization, and explicit evaluation horizon. Development experiments use additional training runs outside the final pipeline.

## Reproduce

Install `requirements.txt`. Run `python run_selected.py` for the selected pipeline. Upload only `expanded_results/submission.csv` as the numerical submission. Run `python run_selected.py --verify` separately to replay the complete experiment and compare the Q-table hash, every result, and CSV entries. A verification replay is a separate audit run, not part of the timed final pipeline.

`run_selected.py` uses `expanded_config.json`, `seeds_eval.csv`, `solution.py`, and `expanded_study.py` beside it. The `expanded_study.py` main function does not run on import. Its MC routine collects actual sampled trajectories and performs first-visit incremental updates from their discounted returns. TD routines use one sampled transition per update. No routine accesses environment models or transition probabilities. All info dictionaries are ignored. Evaluation actions depend on the Q-table and observed state only.

The original `solution.py` command and `results` directory reproduce the previous baseline. They are retained for audit, not overwritten with the new results.

## Search design

Twelve configurations were declared before screening. They cover Q-learning, SARSA, and first-visit incremental Monte Carlo. Q-learning tests discount factors 0.99, 0.995, and 1, smaller and more persistent learning rates, and a changed exploration schedule. SARSA tests the same three discounts with exploration decaying to zero. MC tests gamma 0.995 with sample-average and constant-rate updates, and gamma 1 with constant-rate updates. The bounded search is not an exhaustive search of all possible algorithms or hyperparameters.

Every configuration uses screening training seeds 3301, 3302, 3303, paired across configurations, with 1,000 evaluation episodes per run. The old baseline, the two screening leaders, and the best gamma=0.995 configuration advance without duplication. Confirmation uses five fresh training seeds 4401–4405 and 2,000 episodes per run. Evaluation streams use the corresponding training seed plus 100000. The final selection uses mean confirmation success; exact ties retain the baseline by ordering. Official seed outcomes are not used to select the configuration or training seed. The final training seed remains 20261001.

`expanded_study.json` records every individual run, training cost, configuration, sample standard deviation, and selection. Selection among several configurations introduces optimism into confirmation results; five training repetitions cannot prove universal superiority. Because the hidden reference is unavailable, all comparisons optimize a success-rate proxy, not observed Kaggle MAE.

### Additional robustness search

The user's follow-up requested a more aggressive search. `robustness_study.py` starts with the expanded-study winner and screens nine Q-learning configurations on fresh training seeds 5501–5503 with 1,000 test episodes. The changes cover smaller learning rates, constant learning rates, optimistic initialization, exploration decay speed, final exploration, and gamma=1. The top two, the incumbent, and the strongest screened SARSA and MC settings are compared on fresh seeds 6601–6605 with 2,000 test episodes each. This stage writes its selected configuration to `expanded_config.json`; `robustness_study.json` is the final selection record. The final evaluation seed and test protocol remain unchanged. The confirmation data are used for selection, not represented as an untouched final statistical test.

Optimistic initialization is uniform across states/actions and contains no map, policy, or seed-specific information. It was treated as a standard initialization of a sampled Q-learning method. Visit-count bonuses in action selection and algorithm ensembles were not introduced because their compatibility with the Q-table-only and named-method rules is less explicit. Counts are used solely for the existing learning-rate schedule.

## Optimization boundaries

Safe candidates for further development include exploration and learning-rate schedules, discounting, sample-based Q initialization, first/every-visit MC variants, and efficient non-rendered execution. They require controlled comparisons and must respect the named method family and sample allowance. Increasing the evaluation horizon to inflate returns, conditioning policies on official seeds, inventing submission values, or reading an environment model is not an authorized optimization. No extra replay, model-based planning, or external RL library is used here.

Gamma=1 aligns discounted-return weighting with the stated undiscounted success fraction under a finite horizon, but does not guarantee better finite-sample learning. Gamma=0.995 still discounts delayed success and is an empirical candidate, not a guaranteed improvement. A stationary Q-table also lacks a separate remaining-time dimension. These facts limit what a change in gamma alone can achieve.

`test_updates.py` checks first-visit discounted returns on a repeated-state sampled trajectory and confirms that true termination and the adopted finite horizon do not bootstrap. `run_selected.py --verify` checks the full pipeline separately. All reported optimal reward, optimality gap, and official MAE fields remain null without reference data.

## Actual final outcome

The final selection is Q-learning with gamma=0.995, learning-rate numerator 0.25, and epsilon declining from 1 to 0.01 over 24,000 of its 30,000 transitions. It scored 0.742 mean success in the five-run fresh development comparison; the incumbent scored 0.736, Monte Carlo 0.7303, and SARSA 0.7137. The selected configuration's 0.006 paired advantage over the incumbent is not statistically decisive with these five pairs, and selection used this comparison data.

Final official-seed mean success is 0.73345, exactly unchanged from the original submission. The final learned Q-values differ, but the entire greedy action table is identical, which explains the identical evaluation results. Do not present this as a leaderboard improvement or proof of optimality. The new pipeline completed in approximately 7.26 seconds locally and reproduced every evaluation result and its Q-table hash in a complete replay.
