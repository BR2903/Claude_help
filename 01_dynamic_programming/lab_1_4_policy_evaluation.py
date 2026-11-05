"""
Lab 1.4 (DP) - Policy Evaluation Algorithm
FrozenLake / Gymnasium

This file implements:
1) Deterministic FrozenLake baseline
2) Stochastic (slippery) FrozenLake
3) Iteration counting
4) gamma = 0.5 vs gamma = 0.99
5) Different policies: uniform, always RIGHT, always LEFT
6) 8x8 map
7) theta sensitivity: 1e-3, 1e-6, 1e-9

It also includes an optional human-render demonstration.

Install dependencies if needed:
    pip install gymnasium numpy

Run:
    python lab_1_4_policy_evaluation.py
"""

import time
import numpy as np
import gymnasium as gym


# ------------------------------------------------------------
# Constants for FrozenLake actions
# ------------------------------------------------------------
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTION_NAMES = {
    LEFT: "LEFT",
    DOWN: "DOWN",
    RIGHT: "RIGHT",
    UP: "UP",
}


# ------------------------------------------------------------
# Policy builders
# ------------------------------------------------------------
def build_uniform_policy(nS, nA):
    """
    Build a stochastic uniform-random policy.

    For every state s:
        pi(a|s) = 1 / nA

    Example with 4 actions:
        [0.25, 0.25, 0.25, 0.25]
    """
    return np.ones((nS, nA), dtype=float) / nA


def build_fixed_action_policy(nS, nA, preferred_action):
    """
    Build a deterministic policy that always chooses one action.

    If preferred_action = RIGHT (=2), every row is:
        [0, 0, 1, 0]

    Therefore:
        pi(RIGHT|s) = 1
        pi(other_action|s) = 0
    """
    if not 0 <= preferred_action < nA:
        raise ValueError(
            f"preferred_action must be between 0 and {nA - 1}"
        )

    policy = np.zeros((nS, nA), dtype=float)
    policy[:, preferred_action] = 1.0
    return policy


# ------------------------------------------------------------
# Policy Evaluation
# ------------------------------------------------------------
def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    """
    Iterative Policy Evaluation.

    Bellman expectation equation:

        V_pi(s)
          = sum_a pi(a|s)
                sum_{s'} P(s'|s,a)
                    [ r + gamma * V_pi(s') ]

    For a terminal transition, there is no future return after
    the terminal state, so the continuation term is set to 0.

    Parameters
    ----------
    env : gymnasium environment
        FrozenLake environment.

    policy : np.ndarray, shape (nS, nA)
        policy[s, a] = pi(a|s)

    gamma : float
        Discount factor, normally 0 <= gamma < 1.

    theta : float
        Convergence threshold.

    Returns
    -------
    V : np.ndarray, shape (nS,)
        Estimated state-value function for the given policy.

    iterations : int
        Number of full sweeps over all states before convergence.
    """

    nS = env.observation_space.n
    nA = env.action_space.n

    if policy.shape != (nS, nA):
        raise ValueError(
            f"Policy shape must be {(nS, nA)}, got {policy.shape}"
        )

    # V(s) = 0 for all states initially
    V = np.zeros(nS, dtype=float)

    iterations = 0

    while True:
        # Delta measures the largest state-value change
        # during this complete sweep.
        delta = 0.0
        iterations += 1

        # For every state s
        for state in range(nS):
            old_value = V[state]
            new_value = 0.0

            # For every action a
            for action in range(nA):
                pi_a_given_s = policy[state, action]

                # If the policy can never choose this action,
                # we can skip unnecessary work.
                if pi_a_given_s == 0.0:
                    continue

                action_value = 0.0

                # env.unwrapped.P[state][action] contains possible
                # transitions:
                #
                # (probability, next_state, reward, terminated)
                #
                # There may be one outcome in deterministic FrozenLake
                # or multiple outcomes in slippery FrozenLake.
                for transition in env.unwrapped.P[state][action]:
                    probability, next_state, reward, terminated = transition[:4]

                    if terminated:
                        continuation_value = 0.0
                    else:
                        continuation_value = gamma * V[next_state]

                    action_value += probability * (
                        reward + continuation_value
                    )

                # Weight this action by the probability that
                # the policy chooses it in the current state.
                new_value += pi_a_given_s * action_value

            # In-place Bellman update
            V[state] = new_value

            # Track the largest update in this iteration
            delta = max(delta, abs(old_value - new_value))

        # Convergence condition
        if delta < theta:
            break

    return V, iterations


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def print_value_grid(V, env, title=None):
    """
    Print V(s) in map-shaped form.
    """
    rows, cols = env.unwrapped.nrow, env.unwrapped.ncol
    grid = V.reshape(rows, cols)

    if title:
        print("\n" + title)

    print(np.array2string(
        grid,
        precision=6,
        suppress_small=True,
        floatmode="fixed"
    ))


def print_policy(policy, title="Policy"):
    """
    Print the policy matrix.
    """
    print(f"\n{title}:")
    print(np.array2string(
        policy,
        precision=2,
        suppress_small=True,
        floatmode="fixed"
    ))


def evaluate_configuration(
    *,
    map_name="4x4",
    is_slippery=False,
    gamma=0.9,
    theta=1e-6,
    policy_kind="uniform",
    fixed_action=None,
    show_policy=False,
):
    """
    Create an environment, build a policy, evaluate it,
    print results, and return them.
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery
    )

    nS = env.observation_space.n
    nA = env.action_space.n

    if policy_kind == "uniform":
        policy = build_uniform_policy(nS, nA)
        policy_description = "Uniform random"

    elif policy_kind == "fixed":
        if fixed_action is None:
            raise ValueError(
                "fixed_action must be given when policy_kind='fixed'"
            )
        policy = build_fixed_action_policy(nS, nA, fixed_action)
        policy_description = f"Always {ACTION_NAMES[fixed_action]}"

    else:
        env.close()
        raise ValueError(
            "policy_kind must be either 'uniform' or 'fixed'"
        )

    V, iterations = policy_evaluation(
        env,
        policy,
        gamma=gamma,
        theta=theta
    )

    print("\n" + "=" * 72)
    print(f"Map:          {map_name}")
    print(f"Slippery:     {is_slippery}")
    print(f"Policy:       {policy_description}")
    print(f"Gamma:        {gamma}")
    print(f"Theta:        {theta}")
    print(f"Iterations:   {iterations}")

    if show_policy:
        print_policy(policy)

    print_value_grid(
        V,
        env,
        title="Estimated state values V^pi(s):"
    )

    env.close()

    return {
        "V": V,
        "iterations": iterations,
        "policy": policy,
        "map_name": map_name,
        "is_slippery": is_slippery,
        "gamma": gamma,
        "theta": theta,
        "policy_description": policy_description,
    }


def demonstrate_policy(
    *,
    map_name="4x4",
    is_slippery=False,
    policy_kind="uniform",
    fixed_action=None,
    delay=0.75,
):
    """
    Optional visual demonstration.

    IMPORTANT:
    This does NOT perform learning.
    It simply runs one episode by sampling actions from
    an already-defined policy.

    Set RUN_HUMAN_DEMO = True at the bottom if you want to use it.
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="human"
    )

    nS = env.observation_space.n
    nA = env.action_space.n

    if policy_kind == "uniform":
        policy = build_uniform_policy(nS, nA)
    elif policy_kind == "fixed":
        if fixed_action is None:
            env.close()
            raise ValueError("fixed_action is required")
        policy = build_fixed_action_policy(nS, nA, fixed_action)
    else:
        env.close()
        raise ValueError("Unknown policy_kind")

    state, info = env.reset()
    done = False

    while not done:
        action = np.random.choice(
            env.action_space.n,
            p=policy[state]
        )

        state, reward, terminated, truncated, info = env.step(action)
        time.sleep(delay)

        done = terminated or truncated

    env.close()


# ------------------------------------------------------------
# Lab tests
# ------------------------------------------------------------
def run_lab_tests():
    """
    Run all tasks requested in Lab 1.4.
    """

    # ========================================================
    # TEST 1: Deterministic FrozenLake (baseline)
    # ========================================================
    print("\n\nTEST 1 - DETERMINISTIC FROZENLAKE BASELINE")

    deterministic_baseline = evaluate_configuration(
        map_name="4x4",
        is_slippery=False,
        gamma=0.9,
        theta=1e-6,
        policy_kind="uniform",
        show_policy=True,
    )

    # ========================================================
    # TEST 2: Stochastic / slippery FrozenLake
    # ========================================================
    print("\n\nTEST 2 - STOCHASTIC / SLIPPERY FROZENLAKE")

    stochastic_baseline = evaluate_configuration(
        map_name="4x4",
        is_slippery=True,
        gamma=0.9,
        theta=1e-6,
        policy_kind="uniform",
    )

    # ========================================================
    # TEST 3: Iteration counting
    # ========================================================
    print("\n\nTEST 3 - ITERATION COUNT")
    print(
        "Deterministic baseline converged in "
        f"{deterministic_baseline['iterations']} iterations."
    )
    print(
        "Stochastic baseline converged in "
        f"{stochastic_baseline['iterations']} iterations."
    )

    # ========================================================
    # TEST 4: Gamma comparison
    # ========================================================
    print("\n\nTEST 4 - DISCOUNT FACTOR COMPARISON")

    gamma_05 = evaluate_configuration(
        map_name="4x4",
        is_slippery=False,
        gamma=0.5,
        theta=1e-6,
        policy_kind="uniform",
    )

    gamma_099 = evaluate_configuration(
        map_name="4x4",
        is_slippery=False,
        gamma=0.99,
        theta=1e-6,
        policy_kind="uniform",
    )

    print("\nGamma interpretation:")
    print(
        "- gamma=0.5 discounts distant rewards strongly, so states "
        "far from the goal usually receive smaller values."
    )
    print(
        "- gamma=0.99 keeps distant rewards important, so reward "
        "information propagates much farther backward through the map."
    )

    # ========================================================
    # TEST 5: Different policies
    # ========================================================
    print("\n\nTEST 5 - DIFFERENT POLICIES")

    uniform_policy_result = evaluate_configuration(
        map_name="4x4",
        is_slippery=False,
        gamma=0.9,
        theta=1e-6,
        policy_kind="uniform",
    )

    always_right_result = evaluate_configuration(
        map_name="4x4",
        is_slippery=False,
        gamma=0.9,
        theta=1e-6,
        policy_kind="fixed",
        fixed_action=RIGHT,
    )

    always_left_result = evaluate_configuration(
        map_name="4x4",
        is_slippery=False,
        gamma=0.9,
        theta=1e-6,
        policy_kind="fixed",
        fixed_action=LEFT,
    )

    print("\nPolicy comparison:")
    print(
        "- Uniform policy chooses every action with probability 0.25."
    )
    print(
        "- Always-RIGHT is deterministic: pi(RIGHT|s)=1."
    )
    print(
        "- Always-LEFT is deterministic: pi(LEFT|s)=1."
    )
    print(
        "- Policy evaluation does not decide which one is optimal; "
        "it only computes the value function for each fixed policy."
    )

    # ========================================================
    # TEST 6: 8x8 map
    # ========================================================
    print("\n\nTEST 6 - 8x8 FROZENLAKE")

    map_8x8_result = evaluate_configuration(
        map_name="8x8",
        is_slippery=False,
        gamma=0.9,
        theta=1e-6,
        policy_kind="uniform",
    )

    # ========================================================
    # TEST 7: Theta sensitivity
    # ========================================================
    print("\n\nTEST 7 - CONVERGENCE THRESHOLD SENSITIVITY")

    theta_results = []

    for theta in [1e-3, 1e-6, 1e-9]:
        result = evaluate_configuration(
            map_name="4x4",
            is_slippery=False,
            gamma=0.9,
            theta=theta,
            policy_kind="uniform",
        )
        theta_results.append((theta, result["iterations"]))

    print("\nTheta vs iterations:")
    for theta, iterations in theta_results:
        print(f"theta={theta:<8g} -> {iterations} iterations")

    print(
        "\nSmaller theta means stricter convergence: the algorithm "
        "usually needs more iterations, but the final estimate is "
        "more numerically precise."
    )

    # Return everything in case this file is imported from a notebook.
    return {
        "deterministic_baseline": deterministic_baseline,
        "stochastic_baseline": stochastic_baseline,
        "gamma_05": gamma_05,
        "gamma_099": gamma_099,
        "uniform_policy": uniform_policy_result,
        "always_right": always_right_result,
        "always_left": always_left_result,
        "map_8x8": map_8x8_result,
        "theta_results": theta_results,
    }


# ------------------------------------------------------------
# Discussion question answers
# ------------------------------------------------------------
DISCUSSION_ANSWERS = r"""
LAB 1.4 - REVIEW / DISCUSSION

Q1: Why does policy evaluation require the environment model P?

Policy evaluation with Dynamic Programming computes an expectation over
every possible next-state transition. To calculate

    sum_{s'} P(s'|s,a) [r + gamma V(s')]

the algorithm must know which next states are possible and the probability
of each one. Therefore DP policy evaluation is model-based.

------------------------------------------------------------

Q2: How does is_slippery change values compared to deterministic?

With is_slippery=False, the selected action has a predictable result.
For example, choosing RIGHT normally moves RIGHT.

With is_slippery=True, the environment is stochastic. An attempted move
may lead in another direction. Therefore, near a hole, an action that
looks safe in the deterministic environment may have some probability
of slipping into the hole.

Policy evaluation includes all of these outcomes, weighted by their
transition probabilities. Consequently, risky states/actions can receive
lower expected values.

------------------------------------------------------------

Q3: How does gamma affect V(s) for states far from the goal?

gamma is the discount factor.

A future reward received k steps later is multiplied approximately by

    gamma^k

When gamma is small, such as 0.5, distant rewards shrink rapidly.
Therefore states far from the goal have relatively small values.

When gamma is close to 1, such as 0.99, future rewards retain much more
importance, so states farther from the goal can still have substantial
value.

------------------------------------------------------------

Q4: If a policy is deterministic, how does evaluation simplify?

For a deterministic policy, exactly one action has probability 1 in each
state and every other action has probability 0.

Therefore

    sum_a pi(a|s) Q(s,a)

reduces to only the chosen action:

    V(s) = Q(s, chosen_action)

The algorithm can still use the same general code; the zero-probability
actions simply contribute nothing.

------------------------------------------------------------

Q5: How would policy_evaluation handle continuing non-episodic tasks?

In a continuing task there may be no natural terminal state.

The basic discounted Bellman update still works when gamma < 1:

    V(s) = sum_a pi(a|s)
           sum_{s'} P(s'|s,a)
           [r + gamma V(s')]

The special "if terminated then future value = 0" logic would normally
not be triggered because there are no terminal transitions.

For truly continuing problems with gamma=1, ordinary discounted return
may not remain finite. Such problems often require an average-reward
formulation or another appropriate continuing-task objective.

------------------------------------------------------------

Q6: Why stop when max change Delta < theta? What does theta control?

The values are updated repeatedly and approach a fixed point.

Delta records the largest absolute change among all states during one
sweep:

    Delta = max_s |V_old(s) - V_new(s)|

When Delta becomes smaller than theta, no state is changing by more than
the chosen tolerance, so the values are considered converged.

theta controls the precision/computation tradeoff:

    larger theta -> earlier stop, fewer iterations, lower precision
    smaller theta -> later stop, more iterations, higher precision

------------------------------------------------------------

Q7: Difference between policy evaluation and value iteration?

Policy Evaluation:
    Evaluates ONE fixed policy pi.
    It answers:
        "How good is each state if I keep following this policy?"

Its update averages over actions according to pi(a|s).

Value Iteration:
    Searches toward the optimal value function.
    Instead of averaging actions according to a fixed policy, it selects
    the best action value:

        V(s) <- max_a sum_{s'} P(s'|s,a)
                        [r + gamma V(s')]

Therefore policy evaluation evaluates a policy, while value iteration
is an optimization algorithm used to find optimal behavior.
"""


def print_discussion_answers():
    print("\n\n" + "=" * 72)
    print(DISCUSSION_ANSWERS)


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
if __name__ == "__main__":
    results = run_lab_tests()
    print_discussion_answers()

    # Set this to True only if you want a GUI window showing
    # one FrozenLake episode.
    RUN_HUMAN_DEMO = True

    if RUN_HUMAN_DEMO:
        demonstrate_policy(
            map_name="4x4",
            is_slippery=False,
            policy_kind="uniform",
            delay=0.75,
        )
