from grid_agent.data_structs.simple_data import Vec2D
from grid_agent.data_structs.state import State
from abc import ABC, abstractmethod
from typing import override

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