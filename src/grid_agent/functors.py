from grid_agent.data_structs import Vec2D, Action, State, MapSize
from abc import ABC, abstractmethod
from copy import copy
from math import exp
import random as rnd

class ActionSelector(ABC):

    @abstractmethod
    def get_action(self) -> Action:
        ...

class STDActionSelector(ActionSelector):

    def get_action(self) -> Action:
        return Action(rnd.randrange(Action.MaxExclusive))



class NextPosSelector(ABC):

    @abstractmethod
    def get_next_pos(self, pos: Vec2D, action: Action) -> Vec2D:
        ...

    @abstractmethod
    def get_action_prob(self, chosen_action: Action, action: Action) -> float:
        ...

class STDNextPosSelector(NextPosSelector):

    __action_distribution: list[float] = [0.9, 0.05, 0.0, 0.05]

    def get_next_pos(self, pos: Vec2D, action: Action) -> Vec2D:
        next_pos: Vec2D = copy(pos)
        selected_action: Action = Action((rnd.choices(range(Action.MaxExclusive.value), self.__action_distribution)[0] + action.value) % Action.MaxExclusive.value)
        next_pos.move(selected_action)
        return next_pos
    
    def get_action_prob(self, chosen_action: Action, action: Action) -> float:
        return self.__action_distribution[(action.value - chosen_action.value) % Action.MaxExclusive.value]


class RewardFunction(ABC):

    @abstractmethod
    def calculate_reward(self, next_state: State, map_size: MapSize) -> float:
        ...

class STDRewardFunction(RewardFunction):

    def calculate_reward(self, next_state: State, map_size: MapSize) -> float:
        if next_state.agent_pos == next_state.target_pos:
            return 1
        if next_state.agent_pos == next_state.opponent_pos:
            return -1
        return -0.05

def manhattan_distance(v1: Vec2D, v2: Vec2D) -> int:
    return abs(v1.x - v2.x) + abs(v1.y - v2.y)
    
class ExpRewardFunction(RewardFunction):

    def calculate_reward(self, next_state: State, map_size: MapSize) -> float:
        target_dist: int = manhattan_distance(next_state.agent_pos, next_state.target_pos)
        opponent_dist: int = manhattan_distance(next_state.agent_pos, next_state.opponent_pos)
        max_dist: int = map_size.N + map_size.M - 2
        target_influence: float = exp(-10 * (target_dist ** 2) / max_dist)
        opponent_influence: float = exp(-10 * (opponent_dist ** 2) / max_dist)
        return target_influence - opponent_influence