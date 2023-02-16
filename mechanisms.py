# this file contains the mechanism superclass and implementations for GREEDY, FULL_EXPLORE and FEE
from abc import ABC, abstractmethod
from typing import List, Set, Optional

import numpy as np

from distributions import RandomVariable
from GMDP import TwoActionPolicy, State


class Mechanism(ABC):
    @abstractmethod
    def __init__(self, actions: List[RandomVariable], upper_bound: int, num_agents: int):
        self.actions = actions
        self.K = len(actions)
        self.H = upper_bound
        self.n = num_agents
        self.observed_arm_indices: Set[int] = set()
        self.unobserved_arm_indices: Set[int] = set(range(self.K))

    @abstractmethod
    def choose_action(self, agent_idx: int) -> np.ndarray:
        # output should be non-negative and sum to 1.0
        pass

    @abstractmethod
    def update_knowledge(self, agent_idx: int, action_idx: int, reward: int):
        # This code updates the RandomVariable object and moves the relevant action idx from unobserved to observed.
        self.actions[action_idx].update_reward(reward)
        if action_idx not in self.observed_arm_indices:
            self.observed_arm_indices.add(action_idx)
            self.unobserved_arm_indices.remove(action_idx)

    def onehot_encoding(self, action_idx: int):
        portfolio = np.zeros(self.K, dtype=float)
        portfolio[action_idx] = 1.0
        return portfolio


class Greedy(Mechanism):
    def __init__(self, actions: List[RandomVariable], upper_bound: int, num_agents: int):
        super(Greedy, self).__init__(actions, upper_bound, num_agents)
        self.expectations = [rv.expectation for rv in self.actions]

    def choose_action(self, agent_idx: int) -> np.ndarray:
        argmax = np.argmax(self.expectations)
        return self.onehot_encoding(int(argmax))

    def update_knowledge(self, agent_idx: int, action_idx: int, reward: int):
        super(Greedy, self).update_knowledge(agent_idx, action_idx, reward)
        self.expectations[action_idx] = reward


class FullExploration(Mechanism):
    def __init__(self, actions: List[RandomVariable], upper_bound: int, num_agents: int):
        super(FullExploration, self).__init__(actions, upper_bound, num_agents)
        self.max_reward = None
        self.argmax_reward = None
        self.exploration_idx = -1

    def choose_action(self, agent_idx: int) -> np.ndarray:
        if self.exploration_idx < self.K - 1:
            self.exploration_idx += 1
            return self.onehot_encoding(self.exploration_idx)
        # exploitation phase
        return self.onehot_encoding(self.argmax_reward)

    def update_knowledge(self, agent_idx: int, action_idx: int, reward: int):
        super(FullExploration, self).update_knowledge(agent_idx, action_idx, reward)
        if self.max_reward is None or reward > self.max_reward:
            self.max_reward = reward
            self.argmax_reward = action_idx


class FEE(Mechanism):
    def __init__(self, actions: List[RandomVariable], upper_bound: int, num_agents: int):
        super(FEE, self).__init__(actions, upper_bound, num_agents)
        State.calculate_W_dict(actions)     # calculates W(pi^*, s) for every state s reachable from s_0.
        self.current_state = State((None,) * self.K)
        self.phase = 'primary_exploration'

        # for secondary exploration
        self.arms_worth_exploring: Optional[List[int]] = None

    def choose_action(self, agent_idx: int) -> np.ndarray:
        alpha = self.current_state.alpha
        beta = self.current_state.beta
        if self.phase == 'primary_exploration':
            optimal_policy, _W = self.current_state.optimal_policy()
            optimal_policy: TwoActionPolicy
            return optimal_policy.portfolio
        if self.phase == 'secondary_exploration':
            beta_arm = self.current_state.state_vec.index(beta)
            explored_arm = self.arms_worth_exploring[0]     # arm to be explored
            explored_arm_expectation = self.actions[explored_arm].expectation
            if explored_arm_expectation >= alpha:
                return self.onehot_encoding(explored_arm)
            portfolio = np.zeros(self.K, dtype=float)
            portfolio[explored_arm] = (beta - alpha) / (beta - explored_arm_expectation)
            portfolio[beta_arm] = 1.0 - portfolio[explored_arm]
            return portfolio
        if self.phase == 'exploitation':
            beta_arm = self.current_state.state_vec.index(beta)
            return self.onehot_encoding(beta_arm)
        raise ValueError(f"impossible phase {self.phase=}")

    def update_knowledge(self, agent_idx: int, action_idx: int, reward: int):
        super(FEE, self).update_knowledge(agent_idx, action_idx, reward)
        new_vec = self.current_state.transition_given_reward(action_idx, reward)
        self.current_state = State(new_vec)
        if self.phase == 'primary_exploration' and self.current_state.is_terminal():
            if self.current_state.beta > self.current_state.alpha:
                self.phase = 'secondary_exploration'
                self.arms_worth_exploring = [i for i in self.unobserved_arm_indices if
                                             self.actions[i].P_greater_than(self.current_state.beta) > 0]
            else:
                self.phase = 'exploitation'
        elif self.phase == 'secondary_exploration':
            if action_idx == self.arms_worth_exploring[0]:
                self.arms_worth_exploring = self.arms_worth_exploring[1:]
                if len(self.arms_worth_exploring) == 0:
                    self.phase = 'exploitation'


def get_mechanism(mechanism_name: str, **kwargs) -> Mechanism:
    mechanism_cls = {'GREEDY': Greedy, 'FULL_EXPORATION': FullExploration, 'tbd': Mechanism}[mechanism_name]
    obj = mechanism_cls(**kwargs)
    return obj


def debug_main():
    pass


if __name__ == '__main__':
    debug_main()
