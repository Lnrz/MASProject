from abc import ABC, abstractmethod
from grid_agent.data_structs import Vec2D, Action
from copy import copy
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

class STDNextPosSelector(NextPosSelector):

    __action_distribution: list[float] = [0.8, 0.1, 0.0, 0.1]

    def get_next_pos(self, pos: Vec2D, action: Action) -> Vec2D:
        next_pos: Vec2D = copy(pos)
        selected_action: Action = Action((rnd.choices(range(Action.MaxExclusive.value), self.__action_distribution)[0] + action.value) % Action.MaxExclusive.value)
        match selected_action:
            case Action.Up:
                next_pos.y += 1
            case Action.Right:
                next_pos.x += 1
            case Action.Down:
                next_pos.y -= 1
            case Action.Left:
                next_pos.x -=1
        return next_pos