from grid_agent.functors import PolicyFun, AgentPolicy, UniformPolicy, MarkovTransitionDensity, SimpleMarkovTransitionDensity
from grid_agent.data_structs import  Vec2D, PolicySequential, ValidStateSpace, ValidStateSpaceSequential
from grid_agent.base_configs import BaseConfigs, ConfigArgument

from collections.abc import Callable

class GameConfigs(BaseConfigs):

    def __init__(self) -> None:
        super().__init__()
        self.__agent_start: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self.__target_start: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self.__opponent_start: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self.__agent_policy_factory: ConfigArgument[Callable[[GameConfigs], PolicyFun]] = ConfigArgument(lambda c: AgentPolicy(PolicySequential.from_file(c.policy_file_path), c.valid_state_space))
        self.__target_policy_factory: ConfigArgument[Callable[[GameConfigs], PolicyFun]] = ConfigArgument(lambda c: UniformPolicy())
        self.__opponent_policy_factory: ConfigArgument[Callable[[GameConfigs], PolicyFun]] = ConfigArgument(lambda c: UniformPolicy())
        self.__target_markov_transition_density_factory: ConfigArgument[Callable[[GameConfigs], MarkovTransitionDensity]] = ConfigArgument(lambda c: SimpleMarkovTransitionDensity())
        self.__opponent_markov_transition_density_factory: ConfigArgument[Callable[[GameConfigs], MarkovTransitionDensity]] = ConfigArgument(lambda c: SimpleMarkovTransitionDensity())
    
    @property
    def agent_start(self) -> Vec2D:
        return self.__agent_start.value

    @agent_start.setter
    def agent_start(self, start: Vec2D) -> None:
        self.__agent_start.set_and_freeze(start)

    @property
    def target_start(self) -> Vec2D:
        return self.__target_start.value

    @target_start.setter
    def target_start(self, start: Vec2D) -> None:
        self.__target_start.set_and_freeze(start)

    @property
    def opponent_start(self) -> Vec2D:
        return self.__opponent_start.value

    @opponent_start.setter
    def opponent_start(self, start: Vec2D) -> None:
        self.__opponent_start.set_and_freeze(start)

    @property
    def agent_policy_factory(self) -> Callable[["GameConfigs"], PolicyFun]:
        return self.__agent_policy_factory.value

    @agent_policy_factory.setter
    def agent_policy_factory(self, factory: Callable[["GameConfigs"], PolicyFun]) -> None:
        self.__agent_policy_factory.set_and_freeze(factory)

    @property
    def target_policy_factory(self) -> Callable[["GameConfigs"], PolicyFun]:
        return self.__target_policy_factory.value

    @target_policy_factory.setter
    def target_policy_factory(self, factory: Callable[["GameConfigs"], PolicyFun]) -> None:
        self.__target_policy_factory.set_and_freeze(factory)

    @property
    def opponent_policy_factory(self) -> Callable[["GameConfigs"], PolicyFun]:
        return self.__opponent_policy_factory.value

    @opponent_policy_factory.setter
    def opponent_policy_factory(self, factory: Callable[["GameConfigs"], PolicyFun]) -> None:
        self.__opponent_policy_factory.set_and_freeze(factory)

    @property
    def target_markov_transition_density_factory(self) -> Callable[["GameConfigs"], MarkovTransitionDensity]:
        return self.__target_markov_transition_density_factory.value

    @target_markov_transition_density_factory.setter
    def target_markov_transition_density_factory(self, factory: Callable[["GameConfigs"], MarkovTransitionDensity]) -> None:
        self.__target_markov_transition_density_factory.set_and_freeze(factory)

    @property
    def opponent_markov_transition_density_factory(self) -> Callable[["GameConfigs"], MarkovTransitionDensity]:
        return self.__opponent_markov_transition_density_factory.value

    @opponent_markov_transition_density_factory.setter
    def opponent_markov_transition_density_factory(self, factory: Callable[["GameConfigs"], MarkovTransitionDensity]) -> None:
        self.__opponent_markov_transition_density_factory.set_and_freeze(factory)

    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["agent", start_x, start_y]:
                self.__agent_start.set_if_not_frozen(Vec2D(int(start_x), int(start_y)))
            case ["target", start_x, start_y]:
                self.__target_start.set_if_not_frozen(Vec2D(int(start_x), int(start_y)))
            case ["opponent", start_x, start_y]:
                self.__opponent_start.set_if_not_frozen(Vec2D(int(start_x), int(start_y)))
            case _:
                return False
        return True

    def _check_helper(self) -> None:
        self.__check_for_same_starting_position("Agent", self.agent_start, "Target", self.target_start)
        self.__check_for_same_starting_position("Agent", self.agent_start, "Opponent", self.opponent_start)
        self.__check_for_same_starting_position("Target", self.target_start, "Opponent", self.opponent_start)
        self.__check_for_out_of_bounds("Agent", self.agent_start)
        self.__check_for_out_of_bounds("Target", self.target_start)
        self.__check_for_out_of_bounds("Opponent", self.opponent_start)
        self.__check_for_collision_with_obstacles("Agent", self.agent_start)
        self.__check_for_collision_with_obstacles("Target", self.target_start)
        self.__check_for_collision_with_obstacles("Opponent", self.opponent_start)
    
    def __check_for_same_starting_position(self, name1: str, pos1: Vec2D, name2: str, pos2: Vec2D) -> None:
        if pos1 == pos2:
            raise ValueError(f"{name1} and {name2} should start at different positions.\n"
                            + f"{name1}: ({pos1.x}, {pos1.y})\n"
                            + f"{name2}: ({pos2.x}, {pos2.y})")
    
    def __check_for_out_of_bounds(self, name: str, pos: Vec2D) -> None:
        if (pos.x < 0 or pos.x >= self.map_size.x or
            pos.y < 0 or pos.y >= self.map_size.y):
            raise ValueError(f"{name} is out of bounds:\n"
                             + f"Map was {self.map_size.x}x{self.map_size.y}\n"
                             + f"{name}'s position was ({pos.x},{pos.y})")
    
    def __check_for_collision_with_obstacles(self, name: str, pos: Vec2D) -> None:
        for obstacle in self.obstacles:
            if obstacle.is_inside(pos):
                raise ValueError(f"{name} is colliding with an obstacle.\n"
                                 + f"{name}: ({pos.x}, {pos.y})\n"
                                 + f"Obstacle: [origin: ({obstacle.origin.x}, {obstacle.origin.y}), extent: ({obstacle.extent.x}, {obstacle.extent.y})]")

    def _create_helper(self) -> None:
        self.valid_state_space: ValidStateSpace = ValidStateSpaceSequential(self.map_size, self.obstacles)
        self.target_markov_transition_density: MarkovTransitionDensity = self.target_markov_transition_density_factory(self)
        self.opponent_markov_transition_density: MarkovTransitionDensity = self.opponent_markov_transition_density_factory(self)
        self.agent_policy: PolicyFun = self.agent_policy_factory(self)
        self.target_policy: PolicyFun = self.target_policy_factory(self)
        self.opponent_policy: PolicyFun = self.opponent_policy_factory(self)