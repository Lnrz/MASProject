from grid_agent.data_structs import Vec2D, Action, State, MapSize
from abc import ABC, abstractmethod
from math import exp
import random as rnd

class ActionSelector(ABC):

    @abstractmethod
    def __call__(self) -> Action:
        ...

class SimpleActionSelector(ActionSelector):

    def __call__(self) -> Action:
        return Action(rnd.randrange(Action.MAX_EXCLUSIVE))



class MarkovTransitionDensity(ABC):

    @abstractmethod
    def __call__(self, chosen_action: Action, action: Action) -> float:
        ...

class SimpleMarkovTransitionDensity(MarkovTransitionDensity):

    __action_distribution: list[float] = [0.9, 0.05, 0.0, 0.05]

    def __call__(self, chosen_action: Action, action: Action) -> float:
        return self.__action_distribution[(action.value - chosen_action.value) % Action.MAX_EXCLUSIVE.value]



class RewardFunction(ABC):

    @abstractmethod
    def __call__(self, state: State, next_state: State, map_size: MapSize) -> float:
        ...

class SimpleRewardFunction(RewardFunction):

    def __call__(self, state: State, next_state: State, map_size: MapSize) -> float:
        if state.agent_pos == state.target_pos:
            return 1
        if state.agent_pos == state.opponent_pos:
            return -1
        if next_state.agent_pos == next_state.target_pos:
            return 1
        if next_state.agent_pos == next_state.opponent_pos:
            return -1
        if state.agent_pos == next_state.agent_pos:
            return -0.25
        return -0.05