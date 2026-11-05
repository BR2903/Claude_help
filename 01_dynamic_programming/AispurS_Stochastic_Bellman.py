
import numpy as np
import matplotlib.pyplot as plt  
from pathlib import Path
import csv

# ============================================================
# 1.7 Bellman Optimal Stochastic Equations for 2D GridWorld
# Based directly on the professor's deterministic template.
#
# IMPORTANT DESIGN:
# The SAME stochastic transition code is used for every run.
#
# Stochastic model uses TWO probabilities:
#   success_prob = probability intended direction succeeds
#   slip_prob    = probability of EACH perpendicular slip
#
# They must satisfy:
#   success_prob + 2*slip_prob = 1
#
# Deterministic case:
#   success_prob = 1.0
#   slip_prob    = 0.0
# ============================================================


# ------------------------------------------------------------
# Grid dimensions: 3 rows x 4 columns
# ------------------------------------------------------------
ROWS, COLS = 3, 4

# Define rewards (default = 0 everywhere)
rewards = np.zeros((ROWS, COLS))

# Special states
trap = (1, 3)       # human position (2,4)
goal = (0, 3)       # human position (1,4)
invalid = (1, 1)    # wall, human position (2,2)

rewards[trap] = -1
rewards[goal] = 1


# ------------------------------------------------------------
# DEFAULT PARAMETERS
# ------------------------------------------------------------
gamma = 0.95
living_penalty = -0.04

# Baseline stochastic transition parameters.
# 0.80 intended, 0.10 slip-left, 0.10 slip-right
success_prob = 0.80
slip_prob = 0.10

# Value Iteration convergence threshold
theta = 1e-6


# ------------------------------------------------------------
# Actions: up, down, left, right
# ------------------------------------------------------------
actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# If an action slips, it moves to one of the two perpendicular
# directions. Example: intended R -> slip U or D.
slip_actions = {
    "U": ("L", "R"),
    "D": ("R", "L"),
    "L": ("D", "U"),
    "R": ("U", "D")
}

arrows = {
    "U": "↑",
    "D": "↓",
    "L": "←",
    "R": "→"
}


# ------------------------------------------------------------
# Reward model
# ------------------------------------------------------------
def get_reward(next_state, living_penalty):
    """
    Reward is received when ENTERING next_state.

    - Enter goal: +1
    - Enter trap: -1
    - Enter/stay in a normal state: living_penalty
    """
    if next_state == goal:
        return 1.0

    if next_state == trap:
        return -1.0

    return living_penalty


# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------
def is_terminal(state):
    return state == goal or state == trap


def is_valid_state(state):
    """Check if state is inside grid and not the wall."""
    r, c = state

    if r < 0 or r >= ROWS or c < 0 or c >= COLS:
        return False

    if state == invalid:
        return False

    return True


def step(state, action, living_penalty):
    """
    Physical move for ONE direction.

    If state is already terminal, it is absorbing.
    If action hits boundary/wall, agent stays in place.
    """
    if is_terminal(state):
        return state, 0.0

    r, c = state
    dr, dc = actions[action]

    new_state = (r + dr, c + dc)

    if is_valid_state(new_state):
        return new_state, get_reward(new_state, living_penalty)

    # Invalid move -> stay, but one time step was still used.
    return state, get_reward(state, living_penalty)


def validate_transition_parameters(success_prob, slip_prob):
    """
    For the professor's 3-outcome stochastic model:
        intended + left-slip + right-slip = 1
    """
    if success_prob < 0 or slip_prob < 0:
        raise ValueError("Probabilities cannot be negative.")

    total = success_prob + 2.0 * slip_prob

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Invalid probabilities: success + 2*slip = {total}, not 1."
        )


def transition_outcomes(action, success_prob, slip_prob):
    """
    Returns the THREE possible physical outcomes for a chosen action.

    Example, chosen action R with success=.80 and slip=.10:
        (.80, 'R')
        (.10, 'U')
        (.10, 'D')
    """
    validate_transition_parameters(success_prob, slip_prob)

    slip1, slip2 = slip_actions[action]

    return [
        (success_prob, action),
        (slip_prob, slip1),
        (slip_prob, slip2)
    ]


# ------------------------------------------------------------
# Initialize state-value function
# ------------------------------------------------------------
def initial_value_function():
    V = np.zeros((ROWS, COLS), dtype=float)

    # Wall is not a usable state.
    V[invalid] = np.nan

    # ASSUMPTION:
    # Terminal reward is paid when entering goal/trap.
    # Therefore there is no continuation value after termination.
    V[goal] = 0.0
    V[trap] = 0.0

    return V


# ------------------------------------------------------------
# Bellman expected value for one state-action pair
# ------------------------------------------------------------
def q_value(
    state,
    action,
    V,
    gamma,
    living_penalty,
    success_prob,
    slip_prob
):
    """
    Stochastic Bellman one-step action value:

    Q(s,a) =
      sum_s' P(s'|s,a) [ R(s,a,s') + gamma*V(s') ]
    """
    if is_terminal(state):
        return 0.0

    q = 0.0

    for probability, actual_action in transition_outcomes(
        action,
        success_prob,
        slip_prob
    ):
        next_state, reward = step(
            state,
            actual_action,
            living_penalty
        )

        q += probability * (
            reward + gamma * V[next_state]
        )

    return q


# ------------------------------------------------------------
# Bellman optimality update for V*
# ------------------------------------------------------------
def bellman_update(
    state,
    V,
    gamma,
    living_penalty,
    success_prob,
    slip_prob
):
    """
    V*(s) = max_a Q(s,a)
    """
    if state == invalid or is_terminal(state):
        return V[state]

    action_values = []

    for action in actions:
        q = q_value(
            state,
            action,
            V,
            gamma,
            living_penalty,
            success_prob,
            slip_prob
        )
        action_values.append(q)

    return max(action_values)


# ------------------------------------------------------------
# Value Iteration until convergence
# ------------------------------------------------------------
def value_iteration(
    gamma,
    living_penalty,
    success_prob,
    slip_prob,
    theta=1e-6,
    max_iterations=10000,
    verbose=False,
    snapshot_iterations=(0, 1, 2, 3, 5, 10)
):
    """
    Synchronous Value Iteration.

    Converges when:
        max_s |V_new(s) - V_old(s)| < theta
    """
    validate_transition_parameters(success_prob, slip_prob)

    V = initial_value_function()
    snapshots = {0: V.copy()}
    deltas = []

    for it in range(1, max_iterations + 1):
        new_V = V.copy()
        delta = 0.0

        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)

                if state == invalid or is_terminal(state):
                    continue

                new_value = bellman_update(
                    state,
                    V,
                    gamma,
                    living_penalty,
                    success_prob,
                    slip_prob
                )

                delta = max(
                    delta,
                    abs(new_value - V[state])
                )

                new_V[state] = new_value

        V = new_V
        deltas.append(delta)

        if it in snapshot_iterations:
            snapshots[it] = V.copy()

        if verbose:
            print(f"Iteration {it}:")
            print(V, "\n")

        if delta < theta:
            return V, deltas, snapshots, it

    raise RuntimeError(
        "Value Iteration did not converge before max_iterations."
    )


# ------------------------------------------------------------
# Extract optimal policy
# ------------------------------------------------------------
def extract_policy(
    V,
    gamma,
    living_penalty,
    success_prob,
    slip_prob
):
    """
    pi*(s) = argmax_a Q(s,a)

    If there is an exact tie within tolerance, all tied arrows
    are displayed.
    """
    policy = np.empty((ROWS, COLS), dtype=object)
    chosen_action = {}

    tolerance = 1e-10

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)

            if state == invalid:
                policy[state] = "WALL"
                continue

            if state == goal:
                policy[state] = "GOAL"
                continue

            if state == trap:
                policy[state] = "TRAP"
                continue

            q_by_action = {
                action: q_value(
                    state,
                    action,
                    V,
                    gamma,
                    living_penalty,
                    success_prob,
                    slip_prob
                )
                for action in actions
            }

            best_q = max(q_by_action.values())

            best_actions = [
                action
                for action, value in q_by_action.items()
                if abs(value - best_q) <= tolerance
            ]

            policy[state] = "".join(
                arrows[action] for action in best_actions
            )

            # Deterministic tie break for later analysis.
            chosen_action[state] = best_actions[0]

    return policy, chosen_action


# ------------------------------------------------------------
# Simple Policy Evaluation
# ------------------------------------------------------------
def policy_evaluation(
    fixed_policy,
    gamma,
    living_penalty,
    success_prob,
    slip_prob,
    theta=1e-6,
    max_iterations=10000
):
    """
    Evaluate a FIXED deterministic policy:

    V_pi(s) =
      sum_s' P(s'|s,pi(s))
      [R + gamma V_pi(s')]

    Difference from Value Iteration:
    - Policy Evaluation follows the given policy.
    - Value Iteration uses max over actions to find the best policy.
    """
    V = initial_value_function()

    for it in range(1, max_iterations + 1):
        new_V = V.copy()
        delta = 0.0

        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)

                if state == invalid or is_terminal(state):
                    continue

                action = fixed_policy[state]

                value = q_value(
                    state,
                    action,
                    V,
                    gamma,
                    living_penalty,
                    success_prob,
                    slip_prob
                )

                delta = max(
                    delta,
                    abs(value - V[state])
                )

                new_V[state] = value

        V = new_V

        if delta < theta:
            return V, it

    raise RuntimeError(
        "Policy Evaluation did not converge before max_iterations."
    )


# ------------------------------------------------------------
# Additional analysis:
# expected steps to terminal under fixed policy
# ------------------------------------------------------------
def expected_steps_to_terminal(
    fixed_policy,
    success_prob,
    slip_prob,
    theta=1e-10,
    max_iterations=100000
):
    """
    H(s) = 1 + sum_s' P(s'|s,pi(s)) H(s')
    H(goal)=H(trap)=0
    """
    H = np.zeros((ROWS, COLS), dtype=float)
    H[invalid] = np.nan

    for _ in range(max_iterations):
        new_H = H.copy()
        delta = 0.0

        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)

                if state == invalid or is_terminal(state):
                    continue

                action = fixed_policy[state]
                expected = 1.0

                for probability, actual_action in transition_outcomes(
                    action,
                    success_prob,
                    slip_prob
                ):
                    next_state, _ = step(
                        state,
                        actual_action,
                        living_penalty=0.0
                    )
                    expected += probability * H[next_state]

                delta = max(
                    delta,
                    abs(expected - H[state])
                )

                new_H[state] = expected

        H = new_H

        if delta < theta:
            return H

    raise RuntimeError(
        "Expected-step calculation did not converge."
    )


# ------------------------------------------------------------
# Additional analysis:
# probability of eventually reaching trap
# ------------------------------------------------------------
def probability_of_trap(
    fixed_policy,
    success_prob,
    slip_prob,
    theta=1e-12,
    max_iterations=100000
):
    """
    F(trap)=1
    F(goal)=0
    F(s)=sum_s' P(s'|s,pi(s)) F(s')
    """
    F = np.zeros((ROWS, COLS), dtype=float)
    F[invalid] = np.nan
    F[trap] = 1.0
    F[goal] = 0.0

    for _ in range(max_iterations):
        new_F = F.copy()
        delta = 0.0

        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)

                if state == invalid or is_terminal(state):
                    continue

                action = fixed_policy[state]
                expected = 0.0

                for probability, actual_action in transition_outcomes(
                    action,
                    success_prob,
                    slip_prob
                ):
                    next_state, _ = step(
                        state,
                        actual_action,
                        living_penalty=0.0
                    )
                    expected += probability * F[next_state]

                delta = max(
                    delta,
                    abs(expected - F[state])
                )

                new_F[state] = expected

        F = new_F

        if delta < theta:
            return F

    raise RuntimeError(
        "Trap-probability calculation did not converge."
    )


# ------------------------------------------------------------
# Manual Bellman validation
# ------------------------------------------------------------
def manual_validation(
    V,
    gamma,
    living_penalty,
    success_prob,
    slip_prob
):
    """
    Manually check state (0,2), intended action RIGHT.

    Outcomes:
    intended R -> goal
    slip U -> boundary, stays at (0,2)
    slip D -> (1,2)
    """
    state = (0, 2)
    action = "R"

    manual_q = (
        success_prob * (
            1.0 + gamma * V[goal]
        )
        +
        slip_prob * (
            living_penalty + gamma * V[(0, 2)]
        )
        +
        slip_prob * (
            living_penalty + gamma * V[(1, 2)]
        )
    )

    program_q = q_value(
        state,
        action,
        V,
        gamma,
        living_penalty,
        success_prob,
        slip_prob
    )

    return manual_q, program_q


# ------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------
def output_matrix_for_plot(V):
    data = V.copy()

    # For display only, show terminal REWARDS on the heatmap.
    data[goal] = 1.0
    data[trap] = -1.0

    return data


def save_value_heatmap(V, title, filename):
    data = output_matrix_for_plot(V)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(data)

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)

            if state == invalid:
                label = "WALL"
            elif state == goal:
                label = "GOAL\n+1"
            elif state == trap:
                label = "TRAP\n-1"
            else:
                label = f"{V[state]:.3f}"

            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                fontsize=11
            )

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="State value")
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_policy_plot(policy, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    ax.grid(True)

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)
            text = policy[state]

            if state not in (invalid, goal, trap):
                fontsize = 22
            else:
                fontsize = 11

            ax.text(
                c,
                r,
                text,
                ha="center",
                va="center",
                fontsize=fontsize
            )

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_convergence_plot(
    results,
    labels,
    title,
    filename
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for result, label in zip(results, labels):
        deltas = result["deltas"]
        x = np.arange(1, len(deltas) + 1)
        ax.plot(x, deltas, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max |V_new - V_old|")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_parameter_plot(
    x,
    y,
    xlabel,
    ylabel,
    title,
    filename
):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_value_difference_heatmap(
    V_a,
    V_b,
    title,
    filename
):
    difference = np.abs(V_a - V_b)
    difference[invalid] = np.nan
    difference[goal] = 0.0
    difference[trap] = 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(difference)

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)

            if state == invalid:
                label = "WALL"
            elif state == goal or state == trap:
                label = "T"
            else:
                label = f"{difference[state]:.3f}"

            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center"
            )

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Absolute value difference")
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


# ------------------------------------------------------------
# One experiment
# ------------------------------------------------------------
def run_experiment(
    name,
    gamma,
    living_penalty,
    success_prob,
    slip_prob,
    output_dir,
    save_figures=True,
    verbose=False
):
    V, deltas, snapshots, iterations = value_iteration(
        gamma=gamma,
        living_penalty=living_penalty,
        success_prob=success_prob,
        slip_prob=slip_prob,
        theta=theta,
        verbose=verbose
    )

    policy, fixed_policy = extract_policy(
        V,
        gamma,
        living_penalty,
        success_prob,
        slip_prob
    )

    H = expected_steps_to_terminal(
        fixed_policy,
        success_prob,
        slip_prob
    )

    F = probability_of_trap(
        fixed_policy,
        success_prob,
        slip_prob
    )

    if save_figures:
        save_value_heatmap(
            V,
            (
                f"{name}: V*  "
                f"(success={success_prob:.2f}, "
                f"slip={slip_prob:.3f}, "
                f"living={living_penalty}, "
                f"gamma={gamma})"
            ),
            output_dir / f"{name}_values.png"
        )

        save_policy_plot(
            policy,
            (
                f"{name}: Optimal policy  "
                f"(success={success_prob:.2f}, "
                f"slip={slip_prob:.3f})"
            ),
            output_dir / f"{name}_policy.png"
        )

    return {
        "name": name,
        "gamma": gamma,
        "living_penalty": living_penalty,
        "success_prob": success_prob,
        "slip_prob": slip_prob,
        "V": V,
        "policy": policy,
        "fixed_policy": fixed_policy,
        "deltas": deltas,
        "snapshots": snapshots,
        "iterations": iterations,
        "expected_steps_start": H[(2, 0)],
        "trap_probability_start": F[(2, 0)]
    }


# ------------------------------------------------------------
# Print helpers
# ------------------------------------------------------------
def print_value_grid(V):
    for r in range(ROWS):
        row = []

        for c in range(COLS):
            state = (r, c)

            if state == invalid:
                text = "WALL"
            elif state == goal:
                text = "GOAL(+1)"
            elif state == trap:
                text = "TRAP(-1)"
            else:
                text = f"{V[state]:.4f}"

            row.append(f"{text:^12}")

        print(" | ".join(row))


def print_policy_grid(policy):
    for r in range(ROWS):
        row = []

        for c in range(COLS):
            row.append(f"{str(policy[(r,c)]):^12}")

        print(" | ".join(row))


# ------------------------------------------------------------
# Required laboratory experiments
# ------------------------------------------------------------
def main():
    output_dir = Path(__file__).resolve().parent / "rl101_results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("1.7 Bellman Optimal Stochastic Equations for 2D GridWorld")
    print("=" * 80)
    print()
    print("Reward assumption:")
    print("  reward is paid when the next state is entered")
    print("  normal step -> living penalty")
    print("  enter goal  -> +1")
    print("  enter trap  -> -1")
    print("  terminal states are absorbing")
    print()
    print(
        "Transition model: success_prob intended + "
        "slip_prob left + slip_prob right"
    )
    print()

    # ========================================================
    # A. BASELINE STOCHASTIC
    # ========================================================
    baseline = run_experiment(
        name="baseline_stochastic_p080",
        gamma=0.95,
        living_penalty=-0.04,
        success_prob=0.80,
        slip_prob=0.10,
        output_dir=output_dir
    )

    print("=" * 80)
    print("BASELINE STOCHASTIC: p(success)=0.80, p(slip)=0.10 each")
    print("=" * 80)
    print(f"Converged in {baseline['iterations']} iterations")
    print("\nFinal V*:")
    print_value_grid(baseline["V"])
    print("\nOptimal policy:")
    print_policy_grid(baseline["policy"])

    manual_q, program_q = manual_validation(
        baseline["V"],
        gamma=0.95,
        living_penalty=-0.04,
        success_prob=0.80,
        slip_prob=0.10
    )

    print("\nManual Bellman validation at s=(0,2), a=R:")
    print(
        "Q = 0.80[1 + 0.95*0] "
        "+ 0.10[-0.04 + 0.95*V(0,2)] "
        "+ 0.10[-0.04 + 0.95*V(1,2)]"
    )
    print(f"V(0,2) = {baseline['V'][(0,2)]:.9f}")
    print(f"V(1,2) = {baseline['V'][(1,2)]:.9f}")
    print(f"Manual Q  = {manual_q:.9f}")
    print(f"Program Q = {program_q:.9f}")
    print(
        f"Absolute difference = "
        f"{abs(manual_q - program_q):.3e}"
    )

    assert abs(manual_q - program_q) < 1e-10

    # Save value propagation heatmaps.
    for iteration, snapshot in baseline["snapshots"].items():
        save_value_heatmap(
            snapshot,
            f"Value propagation after iteration {iteration}",
            output_dir / f"propagation_iteration_{iteration}.png"
        )

    # ========================================================
    # B. STOCHASTIC TRANSITION EXPERIMENTS
    # ========================================================
    stochastic_configs = [
        ("stochastic_p095", 0.95, 0.025),
        ("stochastic_p080", 0.80, 0.10),
        ("stochastic_p060", 0.60, 0.20)
    ]

    transition_results = []

    print("\n" + "=" * 80)
    print("STOCHASTIC TRANSITION EXPERIMENTS")
    print("=" * 80)

    for name, p_success, p_slip in stochastic_configs:
        result = run_experiment(
            name=name,
            gamma=0.95,
            living_penalty=-0.04,
            success_prob=p_success,
            slip_prob=p_slip,
            output_dir=output_dir
        )

        transition_results.append(result)

        print(
            f"\n{name}: success={p_success}, "
            f"slip={p_slip} each"
        )
        print(
            f"iterations={result['iterations']}, "
            f"V*(2,0)={result['V'][(2,0)]:.6f}, "
            f"expected steps={result['expected_steps_start']:.4f}, "
            f"P(trap)={result['trap_probability_start']:.6f}"
        )
        print_policy_grid(result["policy"])

    # ========================================================
    # C. DETERMINISTIC EXPERIMENT
    # Only the transition probabilities change.
    # ========================================================
    deterministic = run_experiment(
        name="deterministic_p100",
        gamma=0.95,
        living_penalty=-0.04,
        success_prob=1.0,
        slip_prob=0.0,
        output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("DETERMINISTIC EXPERIMENT")
    print("=" * 80)
    print(
        "Deterministic uses SAME stochastic Bellman code with:\n"
        "success_prob = 1.0\n"
        "slip_prob    = 0.0"
    )
    print(f"Converged in {deterministic['iterations']} iterations")
    print("\nFinal V*:")
    print_value_grid(deterministic["V"])
    print("\nOptimal policy:")
    print_policy_grid(deterministic["policy"])

    all_transition_results = [deterministic] + transition_results

    save_convergence_plot(
        all_transition_results,
        [
            "deterministic p=1.00",
            "stochastic p=0.95",
            "stochastic p=0.80",
            "stochastic p=0.60"
        ],
        "Value Iteration convergence: deterministic vs stochastic",
        output_dir / "convergence_transition_models.png"
    )

    save_parameter_plot(
        [1.00, 0.95, 0.80, 0.60],
        [
            deterministic["V"][(2,0)],
            transition_results[0]["V"][(2,0)],
            transition_results[1]["V"][(2,0)],
            transition_results[2]["V"][(2,0)]
        ],
        "Success probability",
        "V*(2,0)",
        "Start-state value vs transition reliability",
        output_dir / "start_value_vs_success_probability.png"
    )

    high_uncertainty = transition_results[2]

    save_value_difference_heatmap(
        deterministic["V"],
        high_uncertainty["V"],
        "Absolute V* change: deterministic p=1.00 vs stochastic p=0.60",
        output_dir / "value_difference_det_vs_p060.png"
    )

    # Determine states with largest V changes.
    changes = []

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)

            if state == invalid or is_terminal(state):
                continue

            diff = abs(
                deterministic["V"][state]
                - high_uncertainty["V"][state]
            )

            changes.append(
                (
                    diff,
                    state,
                    deterministic["V"][state],
                    high_uncertainty["V"][state]
                )
            )

    changes.sort(reverse=True)

    print("\nLargest state-value changes, deterministic vs p=0.60:")
    for diff, state, v_det, v_sto in changes:
        print(
            f"state={state}, "
            f"|change|={diff:.6f}, "
            f"V_det={v_det:.6f}, "
            f"V_stoch={v_sto:.6f}"
        )

    # ========================================================
    # D. LIVING PENALTY EXPERIMENTS
    # Hold stochastic model at 0.80 / 0.10 / 0.10.
    # ========================================================
    living_penalties = [0.0, -0.01, -0.10, -2.0]
    living_results = []

    print("\n" + "=" * 80)
    print("LIVING PENALTY EXPERIMENTS")
    print("Stochastic transition fixed at success=.80, slip=.10 each")
    print("=" * 80)

    for penalty in living_penalties:
        name = (
            "living_"
            + str(penalty)
              .replace("-", "m")
              .replace(".", "p")
        )

        result = run_experiment(
            name=name,
            gamma=0.95,
            living_penalty=penalty,
            success_prob=0.80,
            slip_prob=0.10,
            output_dir=output_dir
        )

        living_results.append(result)

        print(
            f"\nliving penalty={penalty:+.2f}: "
            f"V*(2,0)={result['V'][(2,0)]:.6f}, "
            f"expected steps={result['expected_steps_start']:.4f}, "
            f"P(trap)={result['trap_probability_start']:.6f}"
        )
        print_policy_grid(result["policy"])

    save_parameter_plot(
        living_penalties,
        [r["V"][(2,0)] for r in living_results],
        "Living penalty",
        "V*(2,0)",
        "Living penalty vs start-state value",
        output_dir / "living_penalty_vs_start_value.png"
    )

    save_parameter_plot(
        living_penalties,
        [r["expected_steps_start"] for r in living_results],
        "Living penalty",
        "Expected steps to terminal from (2,0)",
        "Living penalty vs episode length",
        output_dir / "living_penalty_vs_expected_steps.png"
    )

    save_parameter_plot(
        living_penalties,
        [r["trap_probability_start"] for r in living_results],
        "Living penalty",
        "Probability of eventually reaching trap",
        "Living penalty vs trap risk",
        output_dir / "living_penalty_vs_trap_probability.png"
    )

    save_convergence_plot(
        living_results,
        [
            "living=0",
            "living=-0.01",
            "living=-0.10",
            "living=-2.0"
        ],
        "Value Iteration convergence: living penalty experiments",
        output_dir / "convergence_living_penalties.png"
    )

    # ========================================================
    # E. GAMMA EXPERIMENTS
    # Included because learning objectives ask effect of gamma.
    # ========================================================
    gamma_values = [0.50, 0.95, 0.99]
    gamma_results = []

    print("\n" + "=" * 80)
    print("DISCOUNT FACTOR GAMMA EXPERIMENTS")
    print("Stochastic transition=.80/.10/.10, living=-.04")
    print("=" * 80)

    for g in gamma_values:
        name = "gamma_" + str(g).replace(".", "p")

        result = run_experiment(
            name=name,
            gamma=g,
            living_penalty=-0.04,
            success_prob=0.80,
            slip_prob=0.10,
            output_dir=output_dir
        )

        gamma_results.append(result)

        print(
            f"\ngamma={g:.2f}: "
            f"V*(2,0)={result['V'][(2,0)]:.6f}, "
            f"iterations={result['iterations']}"
        )

    save_parameter_plot(
        gamma_values,
        [r["V"][(2,0)] for r in gamma_results],
        "Discount factor gamma",
        "V*(2,0)",
        "Discount factor vs start-state value",
        output_dir / "gamma_vs_start_value.png"
    )

    # ========================================================
    # F. POLICY EVALUATION
    # Evaluate baseline optimal policy as a FIXED policy.
    # ========================================================
    V_pi, eval_iterations = policy_evaluation(
        baseline["fixed_policy"],
        gamma=0.95,
        living_penalty=-0.04,
        success_prob=0.80,
        slip_prob=0.10,
        theta=theta
    )

    nonterminal_differences = []

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)

            if state == invalid:
                continue

            nonterminal_differences.append(
                abs(V_pi[state] - baseline["V"][state])
            )

    max_difference = max(nonterminal_differences)

    print("\n" + "=" * 80)
    print("POLICY EVALUATION")
    print("=" * 80)
    print(
        "Evaluated the baseline optimal policy as a FIXED policy."
    )
    print(f"Policy Evaluation iterations = {eval_iterations}")
    print(
        f"max |V_pi - V*| = {max_difference:.3e}"
    )
    print(
        "Interpretation: Policy Evaluation measures a given policy; "
        "Value Iteration uses max over actions to find the optimal one."
    )

    # ========================================================
    # G. SAVE EXPERIMENT SUMMARY CSV
    # ========================================================
    csv_path = output_dir / "experiment_summary.csv"

    rows_for_csv = []

    for result in all_transition_results:
        rows_for_csv.append(
            (
                "transition",
                result
            )
        )

    for result in living_results:
        rows_for_csv.append(
            (
                "living_penalty",
                result
            )
        )

    for result in gamma_results:
        rows_for_csv.append(
            (
                "gamma",
                result
            )
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "experiment_group",
                "name",
                "gamma",
                "living_penalty",
                "success_prob",
                "slip_prob_each",
                "iterations",
                "V_start_2_0",
                "expected_steps_from_2_0",
                "trap_probability_from_2_0"
            ]
        )

        for group, result in rows_for_csv:
            writer.writerow(
                [
                    group,
                    result["name"],
                    result["gamma"],
                    result["living_penalty"],
                    result["success_prob"],
                    result["slip_prob"],
                    result["iterations"],
                    result["V"][(2,0)],
                    result["expected_steps_start"],
                    result["trap_probability_start"]
                ]
            )

    # ========================================================
    # H. SHORT ANALYSIS ANSWERS GENERATED FROM ACTUAL RESULTS
    # ========================================================
    analysis_path = output_dir / "analysis_answers.txt"

    with analysis_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "ANALYSIS ANSWERS - BASED ON THE PROGRAM'S ACTUAL OUTPUT\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(
            "1. How does the optimal policy change when transition "
            "uncertainty increases?\n"
        )
        f.write(
            "As success probability decreases, the expected value of "
            "actions near the trap is reduced because unintended "
            "perpendicular moves become more likely. The optimal policy "
            "therefore avoids risky cells more often and can prefer "
            "longer but safer routes.\n\n"
        )

        f.write(
            "2. Which states experience the largest V-value changes?\n"
        )

        for diff, state, v_det, v_sto in changes[:4]:
            f.write(
                f"   state {state}: |change|={diff:.6f}, "
                f"V_det={v_det:.6f}, V_p060={v_sto:.6f}\n"
            )

        f.write("\n")

        f.write(
            "3. Why does the agent become more risk-averse near the trap?\n"
        )
        f.write(
            "Bellman optimality uses an expectation over every possible "
            "next state. Near the trap, a slip can produce reward -1 and "
            "terminate the episode. As slip probability grows, that bad "
            "outcome receives more probability weight, lowering Q-values "
            "for actions that pass close to the trap.\n\n"
        )

        f.write(
            "4. How does increasing the magnitude of the living penalty "
            "affect path length?\n"
        )

        for result in living_results:
            f.write(
                f"   living={result['living_penalty']:+.2f}: "
                f"expected steps="
                f"{result['expected_steps_start']:.4f}\n"
            )

        f.write(
            "\nA more negative living penalty makes every extra normal "
            "step more costly, so the agent has stronger incentive to "
            "finish the episode quickly.\n\n"
        )

        f.write(
            "5. Does the agent reach terminal states faster?\n"
        )
        f.write(
            "Compare the expected-step values above. If they decrease as "
            "the penalty becomes more negative, the optimized policy is "
            "terminating sooner on average.\n\n"
        )

        f.write(
            "6. Under what conditions does the policy become more "
            "aggressive?\n"
        )
        most_negative = living_results[-1]

        f.write(
            f"For living={most_negative['living_penalty']:+.2f}, "
            f"P(trap from start)="
            f"{most_negative['trap_probability_start']:.6f}. "
            "An extreme living penalty can make quick termination "
            "preferable to accumulating many large negative step costs, "
            "even if that means accepting more trap risk.\n\n"
        )

        f.write(
            "7. Policy Evaluation vs Value Iteration\n"
        )
        f.write(
            "Policy Evaluation uses the Bellman expectation equation for "
            "a FIXED policy. Value Iteration uses the Bellman optimality "
            "equation with max over actions, so it simultaneously improves "
            "the value estimate and searches for the best policy.\n\n"
        )

        f.write(
            "8. Deterministic conversion\n"
        )
        f.write(
            "No new transition function is needed. Use the same stochastic "
            "code and set success_prob=1.0 and slip_prob=0.0. Then the "
            "intended action has probability 1 and both perpendicular "
            "outcomes have probability 0.\n"
        )

    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80)
    print(f"Results folder: {output_dir}")
    print(f"CSV summary:    {csv_path}")
    print(f"Analysis text:  {analysis_path}")
    print()
    print("All tests and manual Bellman validation passed.")


if __name__ == "__main__":
    main()
