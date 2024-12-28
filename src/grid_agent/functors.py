from grid_agent.data_structs import Action, State, Policy, ValidStateSpace, Vec2D
from abc import ABC, abstractmethod
from typing import override
import random as rnd
import math

class PolicyFun(ABC):

    @abstractmethod
    def __call__(self, state: State) -> Action:
        ...

class UniformPolicy(PolicyFun):

    @override
    def __call__(self, state: State) -> Action:
        return Action(rnd.randrange(Action.MAX_EXCLUSIVE))

class AgentPolicy(PolicyFun):

    def __init__(self, policy: Policy, valid_state_space: ValidStateSpace) -> None:
        self.__policy: Policy = policy
        self.__valid_state_space: ValidStateSpace = valid_state_space

    @override
    def __call__(self, state: State) -> Action:
        return self.__policy.get_action(self.__valid_state_space.get_valid_index(state))



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



def manhattan_distance(point_a: Vec2D, point_b: Vec2D) -> int:
    return abs(point_a.x - point_b.x) + abs(point_a.y - point_b.y)

class RewardFunction(ABC):

    @abstractmethod
    def __call__(self, state: State, next_state: State) -> float:
        ...

class DenseRewardFunction(RewardFunction):

    @override
    def __call__(self, state: State, next_state: State) -> float:
        if state.agent_pos == state.target_pos:
            return 1.0
        if state.agent_pos == state.opponent_pos:
            return -1.0
        if next_state.agent_pos == next_state.target_pos:
            return 0.25
        if next_state.agent_pos == next_state.opponent_pos:
            return -0.25
        if manhattan_distance(next_state.agent_pos, next_state.opponent_pos) == 1:
            return -0.1
        return -0.01

class SparseRewardFunction(RewardFunction):

    @override
    def __call__(self, state: State, next_state: State) -> float:
        if state.agent_pos == state.target_pos:
            return 1.0
        if state.agent_pos == state.opponent_pos:
            return -1.0
        return 0.0