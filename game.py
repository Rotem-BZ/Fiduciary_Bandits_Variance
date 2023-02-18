# this file implements the game, with controllable parameters and experiment scripts
import random
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from distributions import RandomVariable, Arm
from mechanisms import get_mechanism, Mechanism, MECHANISM_NAMES


def set_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def run_game(upper_bound: int, num_arms: int, meta_variance: float, num_agents: int,
             mechanism_name: str, seed: Optional[int] = None, verbose: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    # meta_variance is the variance of variance values
    assert num_arms < upper_bound, "not a necessary constraint"
    expectations = list(range((upper_bound + num_arms) // 2, (upper_bound - num_arms) // 2, -1))
    app_variances = np.random.normal(7.0, meta_variance, size=num_arms)
    random_variables = RandomVariable.generate_game_with_approx_variance(upper_bound, expectations, app_variances)
    arms = [Arm.from_RV(rv) for rv in random_variables]

    mechanism: Mechanism
    mechanism = get_mechanism(mechanism_name, actions=random_variables, upper_bound=upper_bound, num_agents=num_agents)
    current_belief = np.array(expectations)
    beta = -1
    social_welfare = 0  # total reward
    EAIR_violations = 0
    P_exploit_list = []
    actions_taken = []

    for agent_idx in range(num_agents):
        action_portfolio = mechanism.choose_action(agent_idx)
        assert len(action_portfolio) == num_arms
        assert np.all(action_portfolio >= 0.0)
        assert np.isclose(np.sum(action_portfolio), 1.0)
        P_exploit = sum(prob * mechanism.actions[action_index].P_greater_than(beta-1) for action_index, prob in
                        enumerate(action_portfolio))
        action_idx = np.random.choice(num_arms, p=action_portfolio)
        reward = arms[action_idx].sample()
        mechanism.update_knowledge(agent_idx, action_idx, reward)

        if current_belief[0] > action_portfolio.dot(current_belief) + 1e-4:
            # added a small constant for numerical accuracy
            EAIR_violations += 1
        P_exploit_list.append(P_exploit)
        current_belief[action_idx] = reward
        beta = max(beta, reward)
        social_welfare += reward
        actions_taken.append(action_idx)

    if verbose:
        print(f"""
        mechanism: {mechanism_name}\n
        actions taken: {actions_taken}\n
        total reward (social welfare): {social_welfare}\n
        number of EAIR violations: {EAIR_violations}\n
        probabilities of exploitation: {P_exploit_list}\n
        lowest probability of exploitation: {min(P_exploit_list)}\n 
        """)
    return actions_taken, social_welfare, EAIR_violations, P_exploit_list


def average_metrics(N_iterations: int, **game_kwargs):
    # run many games and average the results
    assert N_iterations > 0
    if 'seed' in game_kwargs:
        del game_kwargs['seed']
    social_welfare_list = []
    EAIR_violation_list = []
    min_P_exploit_list = []
    for seed in range(N_iterations):
        _, social_welfare, EAIR_violations, P_explit_list = run_game(**game_kwargs, seed=seed)
        if game_kwargs['mechanism_name'] == 'FEE' and EAIR_violations > 0:
            print("EAIR VIOLATION WITH FEE!!")
        social_welfare_list.append(social_welfare)
        EAIR_violation_list.append(EAIR_violations)
        min_P_exploit_list.append(min(P_explit_list))
    return sum(social_welfare_list) / N_iterations, sum(EAIR_violation_list) / N_iterations, sum(min_P_exploit_list) / N_iterations


def variance_experiment(mechanism_name: str, metric_to_plot: str, ax: plt.Axes = None, show_plot: bool = True,
                        do_xlabel: bool = True):
    assert metric_to_plot in ['social_welfare', 'EAIR_violations', 'min_P_exploit']
    if ax is None:
        ax = plt.gca()
    N_iterations = 50 if mechanism_name == 'FEE' else 500
    meta_variances = np.linspace(0, 10, 20)
    social_welfare_list = []
    EAIR_violation_list = []
    min_P_exploit_list = []
    game_kwargs = dict(upper_bound=40, num_arms=10, num_agents=30, mechanism_name=mechanism_name, verbose=False)

    for meta_variance in tqdm(meta_variances):
        social_welfare, EAIR_violations, min_P_exploit = average_metrics(N_iterations, meta_variance=meta_variance, **game_kwargs)
        social_welfare_list.append(social_welfare)
        EAIR_violation_list.append(EAIR_violations)
        min_P_exploit_list.append(min_P_exploit)

    y = {'social_welfare': social_welfare_list,
         'EAIR_violations': EAIR_violation_list,
         'min_P_exploit': min_P_exploit_list}[metric_to_plot]
    title = {'social_welfare': "social welfare",
             'EAIR_violations': "No. EAIR violations",
             'min_P_exploit': "Approximated trust metric"}[metric_to_plot]
    ax.plot(meta_variances, y, label=mechanism_name)
    ax.set_title(title)
    if do_xlabel:
        ax.set_xlabel("meta-variance")

    if show_plot:
        plt.show()


def plot_variance_comparison(metric_to_plot='min_P_exploit', exclude_FULL_EXPLORATION=True):
    mechanism_names = ['GREEDY', 'FEE'] if exclude_FULL_EXPLORATION else MECHANISM_NAMES
    ax = plt.gca()
    for mechanism_name in mechanism_names:
        variance_experiment(mechanism_name, metric_to_plot=metric_to_plot, ax=ax, show_plot=False)
    ax.legend()
    plt.show()


def main():
    # run_game(upper_bound=20, num_arms=8, meta_variance=4.0, num_agents=20,
    #          mechanism_name='FEE_Variance', seed=3, verbose=True)

    # experiment 1 - FULL-EXPLORATION performance
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    variance_experiment(MECHANISM_NAMES[1], metric_to_plot='social_welfare', ax=ax1, show_plot=False, do_xlabel=False)
    variance_experiment(MECHANISM_NAMES[1], metric_to_plot='EAIR_violations', ax=ax2, show_plot=False, do_xlabel=False)
    fig.suptitle("FULL-EXPLORATION metrics as function of meta-variance")
    fig.supxlabel("meta-variance")
    plt.show()

    # experiment 2 - trust metric vs meta-variance
    plot_variance_comparison(metric_to_plot='min_P_exploit', exclude_FULL_EXPLORATION=True)


if __name__ == '__main__':
    main()
