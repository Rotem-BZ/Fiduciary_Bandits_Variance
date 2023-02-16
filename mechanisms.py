# this file contains the mechanism superclass and implementations for GREEDY, FULL_EXPLORE and FEE
from abc import ABC, abstractmethod
from typing import List, Set

import numpy as np

from distributions import RandomVariable


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
        pass

    @abstractmethod
    def update_knowledge(self, agent_idx: int, action_idx: int, reward: int):
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


def get_mechanism(mechanism_name: str, **kwargs) -> Mechanism:
    mechanism_cls = {'GREEDY': Greedy, 'FULL_EXPORATION': FullExploration, 'tbd': Mechanism}[mechanism_name]
    obj = mechanism_cls(**kwargs)
    return obj
