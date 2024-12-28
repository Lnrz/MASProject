from grid_agent.data_structs.valid_state_space import ValidStateSpace
from grid_agent.entities.moving_entity import MovingEntity
from grid_agent.data_structs.state import State, Action
from grid_agent.configs.game_configs import GameConfigs

from dataclasses import dataclass, field
from collections.abc import Callable
from copy import copy, deepcopy
from enum import IntEnum

@dataclass
class GameData:
    state: State = field(default_factory=lambda: State())
    agent_action: Action = Action.MAX_EXCLUSIVE
    target_action: Action = Action.MAX_EXCLUSIVE
    opponent_action: Action = Action.MAX_EXCLUSIVE

class Result(IntEnum):
    FAIL = 0,
    SUCCESS = 1,
    WAITING_FOR_RESULT = 2

class GameManager:
    
    def __init__(self, game_configuration: GameConfigs) -> None:
        game_configuration.validate()
        self.__valid_state_space: ValidStateSpace = game_configuration.valid_state_space
        self.__state: State = State(copy(game_configuration.agent_start), copy(game_configuration.opponent_start), copy(game_configuration.target_start))
        self.__agent: MovingEntity = MovingEntity(self.__state.agent_pos, game_configuration.agent_policy, game_configuration.agent_markov_transition_density)
        self.__target: MovingEntity = MovingEntity(self.__state.target_pos, game_configuration.target_policy, game_configuration.target_markov_transition_density)
        self.__opponent: MovingEntity = MovingEntity(self.__state.opponent_pos, game_configuration.opponent_policy, game_configuration.opponent_markov_transition_density)
        self.__res: Result = Result.WAITING_FOR_RESULT
        self.__gamedata: GameData = GameData()
        self.__callback: Callable[[GameData], None] = lambda g: None

    def register_callback(self, callback: Callable[[GameData], None]) -> None:
        self.__callback = callback

    def start(self) -> Result:
        while self.__res == Result.WAITING_FOR_RESULT:
            self.__gamedata.state = deepcopy(self.__state)
            self.__next_iteration()
            self.__callback(copy(self.__gamedata))
        self.__gamedata = GameData()
        self.__gamedata.state = deepcopy(self.__state)
        self.__callback(copy(self.__gamedata))
        return self.__res

    def __next_iteration(self) -> None:
        self.__gamedata.agent_action = self.__agent.move(self.__state, self.__valid_state_space)
        if self.__check_for_result():
            return
        self.__gamedata.target_action = self.__target.move(self.__state, self.__valid_state_space)
        self.__gamedata.opponent_action = self.__opponent.move(self.__state, self.__valid_state_space)
        self.__check_for_result()

    def __check_for_result(self) -> bool:
        match self.__state.agent_pos:
            case self.__state.target_pos:
                self.__res = Result.SUCCESS
                return True
            case self.__state.opponent_pos:
                self.__res = Result.FAIL
                return True
            case _:
                return False