# this file implements the random variables, and allows controlling variance
import random
from typing import List

import numpy as np


def possible_variances(upper_bound: int, expectation: int):
    variances = dict()
    for lower in range(1, expectation+1):
        for upper in range(expectation, upper_bound + 1):
            var = 2 * (expectation - lower) * (upper - expectation) / (upper - lower) if upper != lower else 0
            variances[var] = (lower, upper)
    return variances


class RandomVariable:
    def __init__(self, upper_bound: int, expectation: int, variance: int, lower_value: int, upper_value: int):
        self.upper_bound = upper_bound
        self.expectation = expectation
        self.variance = variance
        self.lower = lower_value
        self.upper = upper_value
        if self.lower == self.upper:
            self.lower_probability = 1.0
        else:
            self.lower_probability = (self.upper - self.expectation) / (self.upper - self.lower)

        self.reward = None  # once realized, every future pull will be deterministic

    @classmethod
    def initialize_with_desired_variance(cls, upper_bound: int, expectation: int, desired_variance: int):
        # If possible, this returns a RandomVariable object with the given upper bound, expectation and variance.
        # If not possible, raises a ValueError.
        possible_variances_ = possible_variances(upper_bound, expectation)
        if desired_variance not in possible_variances_:
            raise ValueError(f"The given {desired_variance=} cannot be created with {expectation=} and {upper_bound=}")
        lower, upper = possible_variances_[desired_variance]
        return cls(upper_bound=upper_bound, expectation=expectation,
                   variance=desired_variance, lower_value=lower, upper_value=upper)

    # def value_or_expectation(self):
    #     # if the variable is realized, returns the known reward. Otherwise, returns the expectation.
    #     return self.reward or self.expectation

    def update_reward(self, reward: int):
        # will be used by the mechanism after the arm has been observed.
        assert self.reward is None or self.reward == reward
        self.reward = reward

    def P_greater_than(self, value: int):
        # returns P(X > value)
        if self.reward is None:
            return self.lower_probability * (self.lower > value) + (1 - self.lower_probability) * (self.upper > value)
        return float(self.reward > value)

    # def P_lesser_than(self, value: int):
    #     # returns P(X <= value)
    #     if self.reward is None:
    #         return self.lower_probability * (self.lower <= value) + (1 - self.lower_probability) * (self.upper <= value)
    #     return float(self.reward <= value)

    def lesser_than_vector(self):
        # returns a (H+1)-sized vector, where the i'th entry contains P(X <= i)
        result = np.zeros(self.upper_bound + 1, dtype=float)
        result[self.lower:] += self.lower_probability
        result[self.upper:] += 1.0 - self.lower_probability
        return result

    @classmethod
    def generate_game_with_approx_variance(cls, upper_bound: int, expectations: List[int], app_variances: List[float]):
        """
        Return a list of ranodm variables with the given expectations and upper_bounds, and variance as close to
        the matching given app_variance as possible.
        """
        rvs = []
        for expectation, app_variance in zip(expectations, app_variances):
            possible_variances_ = possible_variances(upper_bound, expectation)
            variances = np.array(list(possible_variances_.keys()))
            closest_variance = variances[np.argmin(np.abs(variances - app_variance))]
            lower, upper = possible_variances_[closest_variance]
            rvs.append(cls(upper_bound=upper_bound, expectation=expectation,
                           variance=closest_variance, lower_value=lower, upper_value=upper))
        return rvs


class Arm(RandomVariable):
    # The Arm class is the same as RandomVariable, but also implements sampling. Arm will be used by the game class, and
    # RandomVariable will be used by the mechanism, to prevent the mechanism from sampling illegaly.
    def __init__(self, upper_bound: int, expectation: int, variance: int, lower_value: int, upper_value: int):
        super(Arm, self).__init__(upper_bound, expectation, variance, lower_value, upper_value)

    def sample(self):
        if self.reward is None:
            rand = random.random()
            if rand <= self.lower_probability:
                self.reward = self.lower
            else:
                self.reward = self.upper
        return self.reward

    @classmethod
    def from_RV(cls, rv: RandomVariable):
        return cls(upper_bound=rv.upper_bound, expectation=rv.expectation, variance=rv.variance,
                   lower_value=rv.lower, upper_value=rv.upper)
