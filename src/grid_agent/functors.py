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
            return -0.5
        return -0.05

def manhattan_distance(v1: Vec2D, v2: Vec2D) -> int:
    return abs(v1.x - v2.x) + abs(v1.y - v2.y)
    
class ExpRewardFunction(RewardFunction):

    def __call__(self, state: State, next_state: State, map_size: MapSize) -> float:
        target_dist: int = manhattan_distance(next_state.agent_pos, next_state.target_pos)
        opponent_dist: int = manhattan_distance(next_state.agent_pos, next_state.opponent_pos)
        max_dist: int = map_size.N + map_size.M - 2
        alpha: float = 10
        target_influence: float = exp(-alpha * (target_dist / max_dist) ** 2)
        opponent_influence: float = exp(-alpha * (opponent_dist / max_dist) ** 2)
        target_scale: float = 1
        opponent_scale: float = 0.75
        wall_demerit: float = 0
        if state == next_state:
            wall_demerit = 0.1
        return target_influence * target_scale - opponent_influence * opponent_scale - wall_demerit