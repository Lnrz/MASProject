from grid_agent.data_structs.simple_data import Action
from abc import ABC, abstractmethod
from typing import override
import math

class MarkovTransitionDensity(ABC):

    @abstractmethod
    def __call__(self, chosen_action: Action, action: Action) -> float:
        ...

class DiscreteDistributionMarkovTransitionDensity(MarkovTransitionDensity):

    def __init__(self, chosen_action_probability: float = 0.9, right_action_probability: float = 0.05, opposite_action_probability: float = 0.0, left_action_probability: float = 0.05):
        self.__action_distribution: list[float] = [
            chosen_action_probability,
            right_action_probability,
            opposite_action_probability,
            left_action_probability
        ]
        self.__check_for_errors()

    def __check_for_errors(self) -> None:
        if not math.isclose(sum(self.__action_distribution), 1):
            raise ValueError("The given probabilities didn't sum to a number close to 1\n"
                              f"They were: {self.__action_distribution[0]}, {self.__action_distribution[1]}, {self.__action_distribution[2]}, {self.__action_distribution[3]}")
        
        if any(probability < 0.0 for probability in self.__action_distribution):
            raise ValueError("The given probabilities should be positive\n"
                              f"They were: {self.__action_distribution[0]}, {self.__action_distribution[1]}, {self.__action_distribution[2]}, {self.__action_distribution[3]}")

    @override
    def __call__(self, chosen_action: Action, action: Action) -> float:
        return self.__action_distribution[(action - chosen_action) % Action.MAX_EXCLUSIVE]
