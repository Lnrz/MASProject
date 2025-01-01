from grid_agent.entities.game_manager import GameManager
from grid_agent.configs.base_configs import BaseConfigs
from grid_agent.configs.game_configs import GameConfigs
from grid_agent.data_structs.simple_data import Action
from grid_agent.views.ascii_view import ASCIIView
from grid_agent.functors.policy import PolicyFun
from grid_agent.data_structs.state import State
from typing import override
from enum import Enum
import random as rnd

# Enum to enumerate the possible entities
class Entity(Enum):
    AGENT = 0
    TARGET = 1
    OPPONENT = 2

# Policy that make an entity go toward the Opponent
class GoToOpponent(PolicyFun):

    def __init__(self, entity: Entity) -> None:
        # Initialize by passing the entity to which the policy is applied
        self.__entity: Entity = entity

    @override
    def __call__(self, state: State) -> Action:
        # Choose with more probability the action that shortens the longest distance to the opponent
        x_diff: int
        y_diff: int
        match self.__entity:
            case Entity.TARGET:
                x_diff = state.opponent_pos.x - state.target_pos.x
                y_diff = state.opponent_pos.y - state.target_pos.y
            case Entity.AGENT:
                x_diff = state.opponent_pos.x - state.agent_pos.x
                y_diff = state.opponent_pos.y - state.agent_pos.y
            case _:
                raise ValueError("GoToOpponent policy should not be used for Opponent")
        x_action: Action = Action.LEFT if x_diff <= 0 else Action.RIGHT
        y_action: Action = Action.DOWN if y_diff <= 0 else Action.UP
        const_weight: float = 0.1
        x_weight: float = abs(x_diff) + const_weight
        y_weight: float = abs(y_diff) + const_weight
        weights_sum: float = x_weight + y_weight
        x_weight /= weights_sum
        y_weight /= weights_sum
        return rnd.choices(population=[x_action, y_action], weights=[x_weight, y_weight])[0]

def line_processing_extension(configs: BaseConfigs, line: list[str]) -> None:
    # Applicable only for game sessions
    if isinstance(configs, GameConfigs):
        match(line):
            case ["gto", entity]:
                if entity == "agent":
                    configs.agent_policy_factory = lambda c: GoToOpponent(Entity.AGENT)
                elif entity == "target":
                    configs.target_policy_factory = lambda c: GoToOpponent(Entity.TARGET)

def main() -> None:
    configs: GameConfigs = GameConfigs()
    configs.configs_file_path = r"..\configs\extension_example.cfg"
    configs.line_processing_extension = line_processing_extension
    manager: GameManager = GameManager(configs)
    view: ASCIIView = ASCIIView(configs.map_size, configs.obstacles)
    manager.register_callback(view.get_callback())
    manager.start()
    view.start_auto(1.5)

if __name__ == "__main__":
    main()