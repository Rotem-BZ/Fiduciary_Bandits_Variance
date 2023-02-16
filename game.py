# this file implements the game, with controllable parameters and experiment scripts
import random

import numpy as np

from distributions import RandomVariable, Arm
from mechanisms import get_mechanism, Mechanism


def run_game(upper_bound: int, num_arms: int, app_variance: float, num_agents: int, mechanism_name: str):
    assert num_arms < upper_bound, "not a necessary constraint"
    expectations = list(range(num_arms, 0, -1))
    random_variables = RandomVariable.generate_game_with_approx_variance(upper_bound, expectations, app_variance)
    arms = [Arm.from_RV(rv) for rv in random_variables]

    mechanism = get_mechanism(mechanism_name, actions=random_variables, upper_bound=upper_bound, num_agents=num_agents)
    current_belief = np.array(expectations)
    social_welfare = 0  # total reward
    EAIR_violations = 0
    actions_taken = []

    for agent_idx in range(num_agents):
        action_portfolio = mechanism.choose_action(agent_idx)
        assert len(action_portfolio) == num_arms
        assert np.all(action_portfolio >= 0.0)
        assert np.isclose(np.sum(action_portfolio), 1.0)
        action_idx = np.random.choice(num_arms, p=action_portfolio)
        reward = arms[action_idx].sample()
        mechanism.update_knowledge(agent_idx, action_idx, reward)

        if current_belief[0] > action_portfolio.dot(current_belief):
            EAIR_violations += 1
        current_belief[action_idx] = reward
        social_welfare += reward
        actions_taken.append(action_idx)

    print(f"""
    mechanism: {mechanism_name}\n
    actions taken: {actions_taken}\n
    total reward (social welfare): {social_welfare}\n
    number of EAIR violations: {EAIR_violations}\n
    """)


def main():
    random.seed(17)
    run_game(upper_bound=10, num_arms=4, app_variance=3.0, num_agents=60, mechanism_name='FEE')


if __name__ == '__main__':
    main()
