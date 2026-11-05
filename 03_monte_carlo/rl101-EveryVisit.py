import csv
import gymnasium as gym
import numpy as np


# ============================================================
# Competition configuration
# ============================================================

ENV_NAME = "FrozenLake8x8-v1"
MAX_STEPS = 200
TEST_EPISODES = 1000

# Fixed local evaluation seeds.
#
# The competition description says "Fixed seeds" but does not
# provide their actual values. Therefore we use a deterministic
# fixed list for reproducible local evaluation.
#
# If your professor later provides the official 1000 seeds,
# replace this tuple with those exact seeds.
EVAL_SEEDS = tuple(range(TEST_EPISODES))

# Small, reasoned VI comparison. This is NOT a brute-force search.
GAMMAS = (
    0.99,
    0.995,
    0.999,
    0.9999,
    1.0,
)

VI_TOL = 1e-12
VI_MAX_ITERATIONS = 100_000

PROBABILITY_TOL = 1e-12

SUBMISSION_FILE = "rewards.csv"

ACTION_NAMES = {
    0: "L",
    1: "D",
    2: "R",
    3: "U",
}

# Used only when two actions are exactly equal numerically.
ACTION_PRIORITY = (0, 1, 2, 3)


# ============================================================
# Environment construction and validation
# ============================================================

def build_environment():
    """
    Build the exact competition environment.
    """
    return gym.make(
        ENV_NAME,
        is_slippery=True,
    )


def get_map(env):
    """
    Return the map as an 8x8 Unicode array.
    """
    return np.asarray(env.unwrapped.desc).astype("U1")


def validate_environment(env):
    """
    Validate the important competition assumptions.
    """
    if env.observation_space.n != 64:
        raise AssertionError(
            f"Expected 64 states, got {env.observation_space.n}"
        )

    if env.action_space.n != 4:
        raise AssertionError(
            f"Expected 4 actions, got {env.action_space.n}"
        )

    desc = get_map(env)

    if desc.shape != (8, 8):
        raise AssertionError(
            f"Expected an 8x8 map, got {desc.shape}"
        )

    if np.sum(desc == "S") != 1:
        raise AssertionError(
            "Expected exactly one start tile S."
        )

    if np.sum(desc == "G") != 1:
        raise AssertionError(
            "Expected exactly one goal tile G."
        )

    if env.spec is None:
        raise AssertionError(
            "Environment specification is missing."
        )

    if env.spec.max_episode_steps != MAX_STEPS:
        raise AssertionError(
            f"Expected max_episode_steps={MAX_STEPS}, "
            f"got {env.spec.max_episode_steps}"
        )

    start_state = int(
        np.flatnonzero(desc.ravel() == "S")[0]
    )

    state, _ = env.reset(seed=0)

    if int(state) != start_state:
        raise AssertionError(
            f"Expected start state {start_state}, got {state}"
        )

    return desc, start_state


def validate_transition_model(P, desc, n_states, n_actions):
    """
    Verify:
      - probabilities sum to 1
      - slippery stochastic dynamics exist
      - goal gives reward 1 and terminates
      - holes terminate with reward 0
    """
    flat_map = desc.ravel()

    stochastic_transition_found = False
    goal_transition_found = False
    hole_transition_found = False

    for state in range(n_states):
        if state not in P:
            raise AssertionError(
                f"State {state} missing from P."
            )

        for action in range(n_actions):
            if action not in P[state]:
                raise AssertionError(
                    f"Action {action} missing for state {state}."
                )

            transitions = P[state][action]

            if len(transitions) == 0:
                raise AssertionError(
                    f"No transitions for state={state}, action={action}."
                )

            probability_sum = 0.0
            possible_next_states = set()

            for transition in transitions:
                if len(transition) != 4:
                    raise AssertionError(
                        "Expected transition tuple "
                        "(probability, next_state, reward, terminated)."
                    )

                probability, next_state, reward, terminated = transition

                probability = float(probability)
                next_state = int(next_state)
                reward = float(reward)
                terminated = bool(terminated)

                if not 0.0 <= probability <= 1.0:
                    raise AssertionError(
                        f"Invalid probability {probability}."
                    )

                if not 0 <= next_state < n_states:
                    raise AssertionError(
                        f"Invalid next state {next_state}."
                    )

                if reward not in (0.0, 1.0):
                    raise AssertionError(
                        f"Unexpected reward {reward}."
                    )

                probability_sum += probability

                if probability > 0.0:
                    possible_next_states.add(next_state)

                current_tile = flat_map[state]
                next_tile = flat_map[next_state]

                if current_tile not in ("H", "G"):

                    if next_tile == "H":
                        hole_transition_found = True

                        if not terminated:
                            raise AssertionError(
                                "Entering a hole must terminate."
                            )

                        if reward != 0.0:
                            raise AssertionError(
                                "Entering a hole must give reward 0."
                            )

                    elif next_tile == "G":
                        goal_transition_found = True

                        if not terminated:
                            raise AssertionError(
                                "Entering the goal must terminate."
                            )

                        if reward != 1.0:
                            raise AssertionError(
                                "Entering the goal must give reward 1."
                            )

                    elif terminated:
                        raise AssertionError(
                            "Unexpected termination on a safe tile."
                        )

            if not np.isclose(
                probability_sum,
                1.0,
                atol=PROBABILITY_TOL,
                rtol=0.0,
            ):
                raise AssertionError(
                    f"Transition probabilities for "
                    f"state={state}, action={action} "
                    f"sum to {probability_sum}, not 1."
                )

            if (
                flat_map[state] not in ("H", "G")
                and len(possible_next_states) > 1
            ):
                stochastic_transition_found = True

    if not stochastic_transition_found:
        raise AssertionError(
            "No stochastic transitions detected. "
            "Environment may not be slippery."
        )

    if not goal_transition_found:
        raise AssertionError(
            "No transition into the goal was found."
        )

    if not hole_transition_found:
        raise AssertionError(
            "No transition into a hole was found."
        )


# ============================================================
# Exact model representation
# ============================================================

def build_model_arrays(P, n_states, n_actions):
    """
    Build convenient NumPy arrays.

    immediate_reward[s, a]
        Expected immediate reward.

    continuation[s, a, s']
        Probability of reaching s' WITHOUT terminating.

    Terminal transitions are deliberately excluded from the
    continuation matrix so terminal states are never bootstrapped.
    """
    immediate_reward = np.zeros(
        (n_states, n_actions),
        dtype=np.float64,
    )

    continuation = np.zeros(
        (n_states, n_actions, n_states),
        dtype=np.float64,
    )

    for state in range(n_states):
        for action in range(n_actions):

            for probability, next_state, reward, terminated in P[state][action]:

                probability = float(probability)
                next_state = int(next_state)
                reward = float(reward)
                terminated = bool(terminated)

                immediate_reward[state, action] += (
                    probability * reward
                )

                if not terminated:
                    continuation[
                        state,
                        action,
                        next_state,
                    ] += probability

    if not np.all(np.isfinite(immediate_reward)):
        raise AssertionError(
            "Reward model contains NaN or infinity."
        )

    if not np.all(np.isfinite(continuation)):
        raise AssertionError(
            "Transition model contains NaN or infinity."
        )

    return continuation, immediate_reward


# ============================================================
# Deterministic policy extraction
# ============================================================

def deterministic_argmax(values):
    """
    Deterministic argmax without blindly depending on np.argmax's
    undocumented-for-us tie preference.

    Exact numerical ties use ACTION_PRIORITY.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.shape != (4,):
        raise AssertionError(
            f"Expected 4 Q-values, got shape {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise AssertionError(
            "Q-values contain NaN or infinity."
        )

    best_value = float(np.max(values))

    tied_actions = [
        action
        for action in range(4)
        if float(values[action]) == best_value
    ]

    priority = {
        action: rank
        for rank, action in enumerate(ACTION_PRIORITY)
    }

    return min(
        tied_actions,
        key=lambda action: priority[action],
    )


# ============================================================
# Value Iteration
# ============================================================

def value_iteration(
    continuation,
    immediate_reward,
    gamma,
    tolerance=VI_TOL,
):
    """
    Standard model-based Value Iteration.

    V(s) = max_a [
        R(s,a) + gamma * sum_s' P_nonterminal(s'|s,a) V(s')
    ]

    gamma=1.0 is allowed to run only if it actually converges.
    Failure to converge raises an error.
    """
    n_states, n_actions = immediate_reward.shape

    values = np.zeros(
        n_states,
        dtype=np.float64,
    )

    final_residual = None

    for iteration in range(
        1,
        VI_MAX_ITERATIONS + 1,
    ):

        q_values = (
            immediate_reward
            + gamma
            * np.einsum(
                "san,n->sa",
                continuation,
                values,
                optimize=True,
            )
        )

        new_values = np.max(
            q_values,
            axis=1,
        )

        if not np.all(np.isfinite(new_values)):
            raise RuntimeError(
                f"Value Iteration generated NaN/inf "
                f"for gamma={gamma}."
            )

        final_residual = float(
            np.max(
                np.abs(
                    new_values - values
                )
            )
        )

        values = new_values

        if final_residual < tolerance:
            break

    else:
        raise RuntimeError(
            "Value Iteration failed to converge: "
            f"gamma={gamma}, "
            f"tolerance={tolerance}, "
            f"max_iterations={VI_MAX_ITERATIONS}"
        )

    final_q = (
        immediate_reward
        + gamma
        * np.einsum(
            "san,n->sa",
            continuation,
            values,
            optimize=True,
        )
    )

    policy = np.zeros(
        n_states,
        dtype=np.int64,
    )

    for state in range(n_states):
        policy[state] = deterministic_argmax(
            final_q[state]
        )

    return (
        values,
        policy,
        iteration,
        final_residual,
    )


# ============================================================
# Exact 200-step evaluation
# ============================================================

def evaluate_policy_exact_200(
    policy,
    continuation,
    immediate_reward,
    start_state,
):
    """
    Exact model-based success probability for a STATIONARY policy
    under the real 200-action deadline.

    No simulation is used here.

    V_k(s) = R_pi(s) + sum_s' T_pi(s,s') V_{k-1}(s')
    """
    policy = np.asarray(
        policy,
        dtype=np.int64,
    )

    n_states = immediate_reward.shape[0]

    if policy.shape != (n_states,):
        raise AssertionError(
            f"Expected policy shape {(n_states,)}, got {policy.shape}."
        )

    if np.any(policy < 0) or np.any(policy > 3):
        raise AssertionError(
            "Policy contains an invalid action."
        )

    states = np.arange(n_states)

    value = np.zeros(
        n_states,
        dtype=np.float64,
    )

    for _ in range(MAX_STEPS):

        chosen_reward = immediate_reward[
            states,
            policy,
        ]

        chosen_continuation = continuation[
            states,
            policy,
            :,
        ]

        value = (
            chosen_reward
            + np.einsum(
                "sn,n->s",
                chosen_continuation,
                value,
                optimize=True,
            )
        )

    if not np.all(np.isfinite(value)):
        raise AssertionError(
            "Exact policy evaluation generated NaN/inf."
        )

    success_probability = float(
        value[start_state]
    )

    if (
        success_probability < -PROBABILITY_TOL
        or success_probability > 1.0 + PROBABILITY_TOL
    ):
        raise AssertionError(
            f"Invalid success probability {success_probability}."
        )

    return min(
        1.0,
        max(
            0.0,
            success_probability,
        ),
    )


# ============================================================
# Gamma comparison
# ============================================================

def evaluate_gamma_candidates(
    continuation,
    immediate_reward,
    start_state,
):
    """
    Run legal Value Iteration for every candidate gamma.

    IMPORTANT:
    Gamma is selected by exact 200-step model performance,
    NOT by the empirical 1000-episode score.

    This prevents seed overfitting and cherry-picking.
    """
    results = []

    for gamma in GAMMAS:

        (
            values,
            policy,
            iterations,
            residual,
        ) = value_iteration(
            continuation=continuation,
            immediate_reward=immediate_reward,
            gamma=gamma,
            tolerance=VI_TOL,
        )

        exact_200 = evaluate_policy_exact_200(
            policy=policy,
            continuation=continuation,
            immediate_reward=immediate_reward,
            start_state=start_state,
        )

        results.append(
            {
                "gamma": gamma,
                "values": values,
                "policy": policy,
                "iterations": iterations,
                "residual": residual,
                "vi_start_value": float(values[start_state]),
                "exact_200": exact_200,
            }
        )

    # Highest exact 200-step probability wins.
    #
    # If two gamma values produce exactly the same competition
    # objective, use the smaller gamma deterministically.
    results.sort(
        key=lambda item: (
            -item["exact_200"],
            item["gamma"],
        )
    )

    return results


# ============================================================
# Empirical competition-style evaluation
# ============================================================

def evaluate_policy_empirical(
    policy,
    episode_seeds,
):
    """
    Evaluate exactly one episode for every fixed seed.

    For the supplied configuration this is exactly 1000 episodes.

    Each episode is explicitly reseeded so the complete test set
    is deterministic and reproducible.
    """
    if len(episode_seeds) != TEST_EPISODES:
        raise AssertionError(
            f"Expected exactly {TEST_EPISODES} seeds, "
            f"got {len(episode_seeds)}."
        )

    env = build_environment()

    total_reward = 0.0
    successes = 0
    maximum_steps_observed = 0

    try:
        for episode_number, seed in enumerate(
            episode_seeds,
            start=1,
        ):

            state, _ = env.reset(
                seed=int(seed)
            )

            terminated = False
            truncated = False
            episode_reward = 0.0
            steps = 0

            while not (
                terminated
                or truncated
            ):

                if steps >= MAX_STEPS:
                    raise AssertionError(
                        "Episode exceeded MAX_STEPS before "
                        "Gymnasium signaled truncation."
                    )

                action = int(
                    policy[
                        int(state)
                    ]
                )

                if not 0 <= action <= 3:
                    raise AssertionError(
                        f"Invalid policy action {action}."
                    )

                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                steps += 1
                episode_reward += float(reward)

            if steps > MAX_STEPS:
                raise AssertionError(
                    f"Episode {episode_number} used "
                    f"{steps} steps."
                )

            if episode_reward not in (0.0, 1.0):
                raise AssertionError(
                    f"Unexpected episode reward "
                    f"{episode_reward}."
                )

            total_reward += episode_reward

            if episode_reward == 1.0:
                successes += 1

            maximum_steps_observed = max(
                maximum_steps_observed,
                steps,
            )

        average_reward = (
            total_reward / TEST_EPISODES
        )

        return {
            "average_reward": float(average_reward),
            "successes": successes,
            "episodes": TEST_EPISODES,
            "max_steps": maximum_steps_observed,
        }

    finally:
        env.close()


# ============================================================
# Additional validation
# ============================================================

def validate_policy(policy, n_states):
    policy = np.asarray(policy)

    if policy.shape != (n_states,):
        raise AssertionError(
            f"Policy shape should be {(n_states,)}, "
            f"got {policy.shape}."
        )

    if not np.issubdtype(
        policy.dtype,
        np.integer,
    ):
        raise AssertionError(
            "Policy must contain integer actions."
        )

    if np.any(policy < 0):
        raise AssertionError(
            "Policy contains an action below 0."
        )

    if np.any(policy > 3):
        raise AssertionError(
            "Policy contains an action above 3."
        )


def validate_policy_determinism(
    continuation,
    immediate_reward,
    gamma,
):
    """
    Run VI twice and ensure exactly the same policy is generated.
    """
    _, policy_a, _, _ = value_iteration(
        continuation,
        immediate_reward,
        gamma,
        VI_TOL,
    )

    _, policy_b, _, _ = value_iteration(
        continuation,
        immediate_reward,
        gamma,
        VI_TOL,
    )

    if not np.array_equal(
        policy_a,
        policy_b,
    ):
        raise AssertionError(
            "Value Iteration policy generation "
            "was not deterministic."
        )


def validate_empirical_reproducibility(
    policy,
):
    """
    The same fixed 1000 seeds must give exactly the same result.
    """
    first = evaluate_policy_empirical(
        policy,
        EVAL_SEEDS,
    )

    second = evaluate_policy_empirical(
        policy,
        EVAL_SEEDS,
    )

    if first != second:
        raise AssertionError(
            "Seeded 1000-episode evaluation "
            "was not reproducible."
        )

    return first


# ============================================================
# Policy display
# ============================================================

def print_policy(desc, policy):
    """
    Print the final stationary policy as an 8x8 grid.
    """
    print("\nFinal stationary policy:")

    for row in range(8):

        output_row = []

        for col in range(8):

            state = (
                row * 8 + col
            )

            tile = desc[
                row,
                col,
            ]

            if tile in ("H", "G"):
                output_row.append(tile)

            else:
                output_row.append(
                    ACTION_NAMES[
                        int(policy[state])
                    ]
                )

        print(
            " ".join(output_row)
        )


# ============================================================
# Strict Kaggle CSV generation
# ============================================================

def write_submission(empirical_average_reward):
    """
    Competition requirement:

        header + exactly 2 data rows

        Id,Expected
        1,<average reward>
        2,<average reward>

    IMPORTANT:
    Expected is the EMPIRICAL average reward over the fixed
    1000 evaluation episodes.

    It is NOT:
      - AE
      - exact model probability
      - best individual seed
      - average of multiple cherry-picked evaluations
    """
    empirical_average_reward = float(
        empirical_average_reward
    )

    if not np.isfinite(
        empirical_average_reward
    ):
        raise AssertionError(
            "Submission reward is not finite."
        )

    if not 0.0 <= empirical_average_reward <= 1.0:
        raise AssertionError(
            "Submission reward must be in [0,1]."
        )

    with open(
        SUBMISSION_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Id",
                "Expected",
            ]
        )

        # EXACTLY TWO DATA ROWS.
        writer.writerow(
            [
                1,
                f"{empirical_average_reward:.6f}",
            ]
        )

        writer.writerow(
            [
                2,
                f"{empirical_average_reward:.6f}",
            ]
        )

    # --------------------------------------------------------
    # Strict read-back validation
    # --------------------------------------------------------

    with open(
        SUBMISSION_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.reader(file)

        rows = list(reader)

    # 1 header + exactly 2 data rows
    if len(rows) != 3:
        raise AssertionError(
            f"CSV must contain exactly 3 lines "
            f"(header + 2 rows), found {len(rows)}."
        )

    if rows[0] != ["Id", "Expected"]:
        raise AssertionError(
            f"Incorrect CSV header: {rows[0]}"
        )

    if rows[1][0] != "1":
        raise AssertionError(
            "First CSV data Id must be 1."
        )

    if rows[2][0] != "2":
        raise AssertionError(
            "Second CSV data Id must be 2."
        )

    value_1 = float(
        rows[1][1]
    )

    value_2 = float(
        rows[2][1]
    )

    if value_1 != value_2:
        raise AssertionError(
            "Both required submission rows must "
            "contain the same averaged reward."
        )

    return rows


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Build and validate environment
    # --------------------------------------------------------

    env = build_environment()

    try:
        (
            desc,
            start_state,
        ) = validate_environment(env)

        P = env.unwrapped.P

        n_states = env.observation_space.n
        n_actions = env.action_space.n

        validate_transition_model(
            P=P,
            desc=desc,
            n_states=n_states,
            n_actions=n_actions,
        )

        (
            continuation,
            immediate_reward,
        ) = build_model_arrays(
            P=P,
            n_states=n_states,
            n_actions=n_actions,
        )

    finally:
        env.close()

    print("Environment validation: OK")
    print(
        f"Environment: {ENV_NAME}"
    )
    print(
        "Dynamics: is_slippery=True"
    )
    print(
        f"Episode limit: {MAX_STEPS}"
    )
    print(
        f"Evaluation episodes: {TEST_EPISODES}"
    )

    # --------------------------------------------------------
    # Run Value Iteration candidates
    # --------------------------------------------------------

    gamma_results = evaluate_gamma_candidates(
        continuation=continuation,
        immediate_reward=immediate_reward,
        start_state=start_state,
    )

    best = gamma_results[0]
    second_best = gamma_results[1]

    selected_gamma = best["gamma"]
    selected_policy = best["policy"]

    validate_policy(
        selected_policy,
        n_states,
    )

    validate_policy_determinism(
        continuation=continuation,
        immediate_reward=immediate_reward,
        gamma=selected_gamma,
    )

    # --------------------------------------------------------
    # Print complete gamma diagnostics
    # --------------------------------------------------------

    print("\nValue Iteration gamma comparison:")

    for result in gamma_results:

        print(
            f"  gamma={result['gamma']:<7} "
            f"exact_H200={result['exact_200']:.12f} "
            f"VI_V(start)={result['vi_start_value']:.12f} "
            f"iterations={result['iterations']:<5d} "
            f"residual={result['residual']:.3e}"
        )

    # --------------------------------------------------------
    # Explicitly report top two
    # --------------------------------------------------------

    print("\nTwo best VI candidates:")

    print(
        "  #1 "
        f"gamma={best['gamma']}  "
        f"exact 200-step success="
        f"{best['exact_200']:.12f}"
    )

    print(
        "  #2 "
        f"gamma={second_best['gamma']}  "
        f"exact 200-step success="
        f"{second_best['exact_200']:.12f}"
    )

    print(
        f"\nChosen gamma: {selected_gamma}"
    )

    print(
        "Algorithm: Value Iteration"
    )

    print(
        "Policy type: stationary policy[state]"
    )

    print(
        f"DP convergence: "
        f"{best['iterations']} iterations, "
        f"final Bellman residual="
        f"{best['residual']:.3e}, "
        f"tolerance={VI_TOL:.1e}"
    )

    print(
        "Exact/model-based 200-step "
        f"success probability: "
        f"{best['exact_200']:.12f}"
    )

    # --------------------------------------------------------
    # Empirical 1000-episode evaluation
    # --------------------------------------------------------

    empirical = (
        validate_empirical_reproducibility(
            selected_policy
        )
    )

    empirical_reward = empirical[
        "average_reward"
    ]

    empirical_ae = abs(
        empirical_reward - 1.0
    )

    print(
        f"\nEmpirical {TEST_EPISODES}-episode reward: "
        f"{empirical_reward:.6f}"
    )

    print(
        f"Successful episodes: "
        f"{empirical['successes']}/"
        f"{empirical['episodes']}"
    )

    print(
        f"Empirical AE: "
        f"{empirical_ae:.6f}"
    )

    print(
        f"Maximum episode length observed: "
        f"{empirical['max_steps']}"
    )

    print(
        "Reproducibility check: OK"
    )

    # --------------------------------------------------------
    # Policy grid
    # --------------------------------------------------------

    print_policy(
        desc,
        selected_policy,
    )

    # --------------------------------------------------------
    # Competition CSV
    # --------------------------------------------------------

    csv_rows = write_submission(
        empirical_reward
    )

    print(
        f"\nSubmission written to {SUBMISSION_FILE}"
    )

    print(
        "CSV validation: "
        "header + exactly 2 data rows: OK"
    )

    print(
        "\nFinal rewards.csv:"
    )

    for row in csv_rows:
        print(
            ",".join(row)
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The CSV Expected value is the averaged reward "
        "from exactly 1000 fixed-seed episodes."
    )

    print(
        "The exact analytical probability is used for "
        "gamma/policy selection and validation only."
    )


if __name__ == "__main__":
    main()