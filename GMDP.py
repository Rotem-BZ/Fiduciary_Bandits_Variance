# this file implements states, rewards and optimal policy calculation for the primary exploration phase of FEE.
from typing import List, Optional, Tuple

import numpy as np

from distributions import RandomVariable


class TwoActionPolicy:
    def __init__(self, i: int, mu_i: int, r: int, mu_r: int, alpha: int, num_arms: int):
        self.i = i
        # self.mu_i = mu_i
        self.r = r
        # self.mu_r = mu_r
        # self.alpha = alpha
        # self.num_arms = num_arms

        self.portfolio = self.make_portfolio(i, mu_i, r, mu_r, alpha, num_arms)
        self.p_i = self.portfolio[i]
        self.p_r = self.portfolio[r]

    @staticmethod
    def make_portfolio(i: int, mu_i: int, r: int, mu_r: int, alpha: int, num_arms: int):
        # generate portfolio according to the p_{ir}^{\alpha} definition in the paper
        portfolio = np.zeros(num_arms, dtype=float)
        if i == r:
            portfolio[i] = 1.0
        else:
            di = abs(alpha - mu_i)
            dr = abs(alpha - mu_r)
            portfolio[i] = dr / (di + dr)
            portfolio[r] = di / (di + dr)
        return portfolio

    @classmethod
    def deterministic_policy(cls, i: int, num_arms: int):
        mu, alpha = 1, 2    # arbitrary numbers
        return cls(i, mu, i, mu, alpha, num_arms)


class State:
    random_variables: List[RandomVariable] = None
    W_dict = dict()     # keys: state_vec (tuple), items: W value (float)

    def __init__(self, state_vec: Tuple[int, None]):
        self.state_vec = state_vec  # K-sized vector, with None for unobserved actions and the reward for observed ones.
        self.alpha = state_vec[0]
        self.beta = max(r for r in self.state_vec if r is not None)
        self.legal_actions: Optional[List[TwoActionPolicy]] = None

    def is_initial_state(self):
        return all(entry is None for entry in self.state_vec)

    def _find_legal_actions(self):
        if self.is_initial_state():
            self.legal_actions = [TwoActionPolicy.deterministic_policy(0, len(self.random_variables))]
        if self.beta > self.alpha:
            self.legal_actions = []
        raise NotImplementedError()

    def get_legal_actions(self):
        if self.legal_actions is None:
            self._find_legal_actions()
        return self.legal_actions

    def is_terminal(self):
        return len(self.get_legal_actions()) == 0

    def transition_function(self, action_idx: int):
        assert self.state_vec[action_idx] is None   # must choose unobserved arms
        rv = self.random_variables[action_idx]
        state1 = self.state_vec[:action_idx] + (rv.lower,) + self.state_vec[action_idx + 1:]
        if rv.lower == rv.upper:
            return [(state1, 1.0)]
        state2 = self.state_vec[:action_idx] + (rv.upper,) + self.state_vec[action_idx + 1:]
        return [(state1, rv.lower_probability), (state2, 1.0 - rv.lower_probability)]

    def reward(self):
        assert self.is_terminal()
        raise NotImplementedError()

    def optimal_policy(self):
        legal_actions = self.get_legal_actions()
        if self.is_initial_state():
            return legal_actions[0]
        if self.is_terminal():
            raise ValueError("requested policy in terminal state")
        best_value = None
        argbest_value = None
        for policy in legal_actions:
            next_states_after_i = self.transition_function(policy.i)
            assert all(state in self.W_dict for state, _ in next_states_after_i)
            i_expression = policy.p_i * sum(prob * self.W_dict[state] for state, prob in next_states_after_i)
            if policy.i == policy.r:
                r_expression = 0
            else:
                next_states_after_r = self.transition_function(policy.r)
                assert all(state in self.W_dict for state, _ in next_states_after_r)
                r_expression = policy.p_r * sum(prob * self.W_dict[state] for state, prob in next_states_after_r)
            value = i_expression + r_expression
            if best_value is None or value > best_value:
                best_value = value
                argbest_value = policy
        return argbest_value, best_value

    @classmethod
    def recursive_W_dict_step(cls, current_state: Tuple[int, None]):
        if current_state in cls.W_dict:
            return cls.W_dict[current_state]
        obj = cls(current_state)
        if obj.is_terminal():
            reward = obj.reward()
            cls.W_dict[current_state] = reward
            return reward
        # recursive step
        # operates in two phases:
        # 1. recursively make sure that all states reachable from this one already have the W value calculated
        legal_actions = obj.get_legal_actions()
        all_next_states = set()
        for policy in legal_actions:
            next_states = obj.transition_function(policy.i) + obj.transition_function(policy.r)
            next_states = map(lambda item: item[0], next_states)
            all_next_states.update(next_states)
        for next_state in all_next_states:
            _ = cls.recursive_W_dict_step(next_state)
        # 2. find optimal policy and derive this state's W value
        _optimal_policy, W_value = obj.optimal_policy()
        cls.W_dict[current_state] = W_value
        return W_value

    @classmethod
    def calculate_W_dict(cls, random_variables: List[RandomVariable]):
        # makes the first call to the recursive function
        cls.random_variables = random_variables
        cls.W_dict = dict()


