from grid_agent.data_structs import Vec2D, Obstacle, ValidStateSpace
from grid_agent.functors import PolicyFun, UniformPolicy, MarkovTransitionDensity, SimpleMarkovTransitionDensity, RewardFunction, SimpleRewardFunction, AgentPolicy
from abc import ABC, abstractmethod
from typing import Callable

class BaseConfigs(ABC):

    def __init__(self) -> None:
        self.configs_file_path: str = ""
        self.policy_file_path: str = ""
        self.map_size: Vec2D  = Vec2D()
        self.obstacles: list[Obstacle]  = []
        self.agent_markov_transition_density_factory: Callable[[BaseConfigs], MarkovTransitionDensity] = lambda c: SimpleMarkovTransitionDensity()
        self.line_processing_extension: Callable[[BaseConfigs, list[str]], None] = lambda c, l: None

    def validate(self) -> None:
        self.__apply_file()
        self.__check()
        self.__create()

    def __apply_file(self) -> None:
        if self.configs_file_path is not None:
            self.__read_configuration_file(self.configs_file_path)

    def __read_configuration_file(self, configuration_file_path: str) -> None:
        with open(configuration_file_path) as f:
            for line in f.readlines():
                if not line.isspace():
                    self.__process_line(line)

    def __process_line(self, line: str) -> None:
        if line.startswith("#"):
            return
        splitted_line: list[str] = line.split()
        splitted_line[0] = splitted_line[0].casefold()
        has_match: bool = True
        match splitted_line:
            case ["mapsize", map_x_length, map_y_length]:
                self.map_size = Vec2D(int(map_x_length), int(map_y_length))
            case ["obstacle", origin_x, origin_y, extent_x, extent_y]:
                self.obstacles.append(Obstacle(Vec2D(int(origin_x), int(origin_y)), Vec2D(int(extent_x), int(extent_y))))
            case ["policy", policy_path]:
                self.policy_file_path = policy_path
            case _:
                has_match = False
        if not has_match:
            has_match = self._process_line_helper(splitted_line)
        if not has_match:
            self.line_processing_extension(self, splitted_line)
    
    @abstractmethod
    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        ...

    def __check(self) -> None:
        self.__check_map_size()
        self.__check_obstacles()
        self._check_helper()
    
    @abstractmethod
    def _check_helper(self) -> None:
        ...
    
    def __check_map_size(self) -> None:
        if self.map_size.x * self.map_size.y < 3 or self.map_size.x < 0:
            raise ValueError(f"Map size should be positive and have at least 3 cells.\n"
                             + f"Map Size: ({self.map_size.x}, {self.map_size.y})")

    def __check_obstacles(self) -> None:
        for obstacle in self.obstacles:
            if (obstacle.origin.x < 0 or obstacle.origin.x + obstacle.extent.x - 1 >= self.map_size.x or
                obstacle.origin.y >= self.map_size.y or obstacle.origin.y - obstacle.extent.y + 1 < 0):
                raise ValueError(f"An obstacle was out of bounds\n" +
                                 f"Obstacle was [origin: ({obstacle.origin.x},{obstacle.origin.y}), extent: ({obstacle.extent.x},{obstacle.extent.y})]\n" +
                                 f"Map was {self.map_size.x}x{self.map_size.y}")

    def __create(self) -> None:
        self.valid_state_space: ValidStateSpace = ValidStateSpace(self.map_size, self.obstacles)
        self.agent_markov_transition_density: MarkovTransitionDensity = self.agent_markov_transition_density_factory(self)
        self._create_helper()

    @abstractmethod
    def _create_helper(self) -> None:
        ...

class GameConfigs(BaseConfigs):

    def __init__(self) -> None:
        super().__init__()
        self.agent_start: Vec2D = Vec2D()
        self.target_start: Vec2D = Vec2D()
        self.opponent_start: Vec2D = Vec2D()
        self.agent_policy_factory: Callable[[GameConfigs], PolicyFun] = lambda c: AgentPolicy(c.policy_file_path, c.valid_state_space)
        self.target_policy_factory: Callable[[GameConfigs], PolicyFun] = lambda c: UniformPolicy()
        self.opponent_policy_factory: Callable[[GameConfigs], PolicyFun] = lambda c: UniformPolicy()
        self.target_markov_transition_density_factory: Callable[[GameConfigs], MarkovTransitionDensity] = lambda c: SimpleMarkovTransitionDensity()
        self.opponent_markov_transition_density_factory: Callable[[GameConfigs], MarkovTransitionDensity] = lambda c: SimpleMarkovTransitionDensity()
    
    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["agent", start_x, start_y]:
                self.agent_start = Vec2D(int(start_x), int(start_y))
            case ["target", start_x, start_y]:
                self.target_start = Vec2D(int(start_x), int(start_y))
            case ["opponent", start_x, start_y]:
                self.opponent_start = Vec2D(int(start_x), int(start_y))
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
        self.target_markov_transition_density: MarkovTransitionDensity = self.target_markov_transition_density_factory(self)
        self.opponent_markov_transition_density: MarkovTransitionDensity = self.opponent_markov_transition_density_factory(self)
        self.agent_policy: PolicyFun = self.agent_policy_factory(self)
        self.target_policy: PolicyFun = self.target_policy_factory(self)
        self.opponent_policy: PolicyFun = self.opponent_policy_factory(self)

class TrainConfigs(BaseConfigs):

    def __init__(self) -> None:
        super().__init__()
        self.reward_factory: Callable[[TrainConfigs], RewardFunction] = lambda c: SimpleRewardFunction()
        self.max_iter: int = 100
        self.value_function_tolerance: float = 0.0
        self.changed_actions_tolerance: int = 0
        self.changed_actions_percentage_tolerance: float = 0.0

    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["maxiter", max_iter]:
                self.max_iter = int(max_iter)
            case ["valuetolerance", value_tol]:
                self.value_function_tolerance = float(value_tol)
            case ["actiontolerance", action_tol]:
                self.changed_actions_tolerance = int(action_tol)
            case ["actionperctolerance", action_perc_tol]:
                self.changed_actions_percentage_tolerance = float(action_perc_tol)
            case _:
                return False
        return True

    def _check_helper(self) -> None:
        if self.max_iter <= 0:
            raise ValueError(f"Maximum number of iterations should be > 0.\n" +
                             f"It was {self.max_iter}")
        self.__check_non_negativity(self.value_function_tolerance, "Value function tolerance")
        self.__check_non_negativity(self.changed_actions_tolerance, "Change actions tolerance")
        self.__check_non_negativity(self.changed_actions_percentage_tolerance, "Changed actions percentage tolerance")

    def __check_non_negativity(self, value: float | int, name: str) -> None:
        if value < 0:
            raise ValueError(f"{name} should be >= 0" +
                             f"It was {value}")

    def _create_helper(self) -> None:
        self.reward: RewardFunction = self.reward_factory(self)