from grid_agent.data_structs import Action, State, Policy, ValidStateSpace, Vec2D
from abc import ABC, abstractmethod
from typing import override
import random as rnd

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

class SimpleMarkovTransitionDensity(MarkovTransitionDensity):

    __action_distribution: list[float] = [0.9, 0.05, 0.0, 0.05]

    @override
    def __call__(self, chosen_action: Action, action: Action) -> float:
        return self.__action_distribution[(action - chosen_action) % Action.MAX_EXCLUSIVE]



def manhattan_distance(point_a: Vec2D, point_b: Vec2D) -> int:
    return abs(point_a.x - point_b.x) + abs(point_a.y - point_b.y)

class RewardFunction(ABC):

    @abstractmethod
    def __call__(self, state: State, next_state: State) -> float:
        ...

class SimpleRewardFunction(RewardFunction):

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