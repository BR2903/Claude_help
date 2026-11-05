# Uncapped-episode submission

AI assistance: Codex drafted and executed this implementation and the recorded development comparison.

Install `numpy==2.3.5` and `gymnasium==1.3.0`, then run `python agent.py`. The script produces `submission.csv`, `report.json`, and `q_table.npy` beside itself. No external data or companion Python modules are required.

The user explicitly requested removing the imposed 100-step horizon. Public `gym.make(..., max_episode_steps=-1)` disables that wrapper. All 2,000 episodes for each of 10 official seeds now run to a goal or hole. A wall-clock guard aborts instead of counting unfinished episodes or producing a partial submission. No environment model, transition probabilities, state-specific policy, or official-seed-dependent actions are used. Returned info dictionaries are ignored.

The final pipeline uses one shared Q-table and exactly 30,000 total training transitions, preserving the previously approved shared-table design and the professor's TD step budget. The learner is trained from scratch with seed 20261001. Evaluation uses a separate environment for each official seed, seeded once and then reset without reseeding. Evaluation freezes Q-values and uses deterministic greedy actions.

A short development-only comparison tested gamma 0.995, 0.999, and 1.0 on training seeds 8801–8803, each with 2,000 evaluation episodes on separate seeds. The respective mean success fractions were 0.8105, 0.8098333333, and 0.8108333333. Gamma 1.0 was selected by the pre-specified highest-mean criterion; the differences are very small and do not establish statistical superiority. All runs are retained in `gamma_study.json`. Development repetitions are separate from the final timed pipeline.

This is a changed evaluation protocol, not proof that the official competition evaluator uses an unlimited horizon. The organizer's exact horizon, reset protocol, reference results, and public scoring subset remain unavailable. The shared-table protocol also remains a user-approved assumption. Local success rate is not the official MAE. Do not report an optimality gap or guaranteed Kaggle improvement without reference data.
