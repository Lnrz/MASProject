from grid_agent.data_structs.valid_state_space import ValidStateSpace
from grid_agent.entities.moving_entity import MovingEntity
from grid_agent.data_structs.simple_data import Action
from grid_agent.data_structs.state import State
from grid_agent.configs.game_configs import GameConfigs

from dataclasses import dataclass, field
from collections.abc import Callable
from copy import copy, deepcopy
from enum import IntEnum

@dataclass
class GameData:
    """Struct containing the necessary data to view a frame of the game session.
    
    It contains of:
    - ``state``: the state of the game in the current frame, containing the positions of the agent, target and opponent.
    - ``agent_action``: the ``Action`` the agent chose for the next frame.
    - ``target_action``: the ``Action`` the target chose for the next frame.
    - ``opponent_action``: the ``Action`` the opponent chose for the next frame.
    """
    state: State = field(default_factory=lambda: State())
    agent_action: Action = Action.MAX_EXCLUSIVE
    target_action: Action = Action.MAX_EXCLUSIVE
    opponent_action: Action = Action.MAX_EXCLUSIVE

class Result(IntEnum):
    """Enum enumerating the possible states of the game."""
    FAIL = 0,
    SUCCESS = 1,
    WAITING_FOR_RESULT = 2

class GameManager:
    """Manager of game sessions.
    
    It provides the method ``register_callback`` to which the user can pass a ``Callable[[GameData], None]``
    that will be called between each game iteration with a copy of the updated ``GameData``.
    
    This makes possible to implement viewers for game sessions.
    """

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
        """Register the ``callback`` to call between each game iteration."""
        self.__callback = callback

    def start(self) -> Result:
        """Start the game session."""
        while self.__res == Result.WAITING_FOR_RESULT:
            self.__gamedata.state = deepcopy(self.__state)
            self.__next_iteration()
            self.__callback(copy(self.__gamedata))
        self.__gamedata = GameData()
        self.__gamedata.state = deepcopy(self.__state)
        self.__callback(copy(self.__gamedata))
        return self.__res

    def __next_iteration(self) -> None:
        """Perform an iteration of the game session."""
        self.__gamedata.agent_action = self.__agent.move(self.__state, self.__valid_state_space)
        if self.__check_for_result():
            return
        self.__gamedata.target_action = self.__target.move(self.__state, self.__valid_state_space)
        self.__gamedata.opponent_action = self.__opponent.move(self.__state, self.__valid_state_space)
        self.__check_for_result()

    def __check_for_result(self) -> bool:
        """Return ``True`` if the game session ended and set the result accordingly. Otherwise return ``False``."""
        match self.__state.agent_pos:
            case self.__state.target_pos:
                self.__res = Result.SUCCESS
                return True
            case self.__state.opponent_pos:
                self.__res = Result.FAIL
                return True
            case _:
                return False