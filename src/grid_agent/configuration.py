from grid_agent.data_structs import Vec2D, Obstacle, ValidStateSpace
from grid_agent.functors import PolicyFun, UniformPolicy, MarkovTransitionDensity, SimpleMarkovTransitionDensity, RewardFunction, SimpleRewardFunction, AgentPolicy
from abc import ABC, abstractmethod
from typing import Any, Callable

class BaseConfigs(ABC):

    def __init__(self) -> None:
        self.configs_file_path: str | None = None
        self.policy_file_path: str | None = None
        self.map_size: Vec2D | None = None
        self.obstacles: list[Obstacle] | None = None
        self.agent_markov_transition_density_type: type[MarkovTransitionDensity] | None  = None
        self.create_policy_extension: Callable[[BaseConfigs, type[PolicyFun]], PolicyFun | None] = lambda c, t: None
        self.create_markov_transition_density_extension: Callable[[BaseConfigs, type[MarkovTransitionDensity]], MarkovTransitionDensity | None] = lambda c, t: None
        self.create_reward_extension: Callable[[BaseConfigs, type[RewardFunction]], RewardFunction | None] = lambda c, t: None
        self.line_processing_extension: Callable[[BaseConfigs, list[str]], None] = lambda c, l: None

    def validate(self) -> None:
        self.__apply_file()
        self.__apply_default()
        self.__check()
        self.__create()

    def __apply_file(self) -> None:
        if self.configs_file_path is not None:
            self.__read_configuration_file(self.configs_file_path)

    def __read_configuration_file(self, configuration_file_path: str) -> None:
        self.__prepare_for_reading_configs()
        with open(configuration_file_path) as f:
            for line in f.readlines():
                if not line.isspace():
                    self.__process_line(line)
    
    def __prepare_for_reading_configs(self) -> None:
        self.__obstacles_was_none: bool = self.obstacles is None
        if self.__obstacles_was_none:
            self.obstacles = []
        self.__map_size_was_none: bool = self.map_size is None
        self.__policy_was_none: bool = self.policy_file_path is None
        self._prepare_for_reading_configs_helper()

    @abstractmethod
    def _prepare_for_reading_configs_helper(self) -> None:
        ...

    def __process_line(self, line: str) -> None:
        if line.startswith("#"):
            return
        splitted_line: list[str] = line.split()
        splitted_line[0] = splitted_line[0].casefold()
        no_match: bool = False
        match splitted_line:
            case ["mapsize", map_x_length, map_y_length]:
                if self.__map_size_was_none:
                    self.map_size = Vec2D(int(map_x_length), int(map_y_length))
            case ["obstacle", origin_x, origin_y, extent_x, extent_y]:
                if self.__obstacles_was_none:
                    self.obstacles.append(Obstacle(Vec2D(int(origin_x), int(origin_y)), Vec2D(int(extent_x), int(extent_y))))
            case ["policy", policy_path]:
                if self.__policy_was_none:
                    self.policy_file_path = policy_path
            case _:
                no_match = True
        if no_match:
            no_match = self._process_line_helper(splitted_line)
        if no_match:
            self.line_processing_extension(self, splitted_line)
    
    @abstractmethod
    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        ...

    def __apply_default(self) -> None:
        if self.obstacles is None:
            self.obstacles = []
        if self.agent_markov_transition_density_type is None:
            self.agent_markov_transition_density_type = SimpleMarkovTransitionDensity
        self._apply_default_helper()

    @abstractmethod
    def _apply_default_helper(self) -> None:
        ...

    def __check(self) -> None:
        self._check_not_none(self.map_size, "Map size")
        self.__check_map_size()
        self._check_not_none(self.policy_file_path, "Policy file path")
        self.__check_obstacles()
        self._check_helper()
    
    @abstractmethod
    def _check_helper(self) -> None:
        ...
    
    def _check_not_none(self, var: Any | None, name: str) -> None:
        if var is None:
            raise ValueError(f"{name} was not provided.")

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
        self.agent_markov_transition_density: MarkovTransitionDensity = self._create_markov_transition_density(self.agent_markov_transition_density_type)
        self._create_helper()

    @abstractmethod
    def _create_helper(self) -> None:
        ...

    def _create_markov_transition_density(self, mtd_type: type[MarkovTransitionDensity]) -> MarkovTransitionDensity:
        if mtd_type is SimpleMarkovTransitionDensity:
            return mtd_type()
        markov_transition_density: MarkovTransitionDensity | None = self.create_markov_transition_density_extension(self, mtd_type)
        if markov_transition_density is None:
            raise ValueError(f"Unrecognized markov transition density type: {mtd_type}")
        return markov_transition_density

    def _create_reward(self, r_type: type[RewardFunction]) -> RewardFunction:
        if r_type is SimpleRewardFunction:
            return r_type()
        reward: RewardFunction | None = self.create_reward_extension(self, r_type)
        if reward is None:
            raise ValueError(f"Unrecognized reward function type: {r_type}")
        return reward

    def _create_policy(self, p_type: type[PolicyFun]) -> PolicyFun:
        if p_type is UniformPolicy:
            return p_type()
        if p_type is AgentPolicy:
            return p_type(self.policy_file_path, self.valid_state_space)
        policy: PolicyFun | None = self.create_policy_extension(self, p_type)
        if policy is None:
            raise ValueError(f"Unrecognized policy type: {p_type}")
        return policy

class GameConfigs(BaseConfigs):

    def __init__(self) -> None:
        super().__init__()
        self.agent_start: Vec2D | None = None
        self.target_start: Vec2D | None = None
        self.opponent_start: Vec2D | None = None
        self.agent_policy_type: type[PolicyFun] | None = None
        self.target_policy_type: type[PolicyFun] | None = None
        self.opponent_policy_type: type[PolicyFun] | None = None
        self.target_markov_transition_density_type: type[MarkovTransitionDensity] | None = None
        self.opponent_markov_transition_density_type: type[MarkovTransitionDensity] | None = None
    
    def _prepare_for_reading_configs_helper(self) -> None:
        self.__agent_start_was_none: bool = self.agent_start is None
        self.__target_start_was_none: bool = self.target_start is None
        self.__opponent_start_was_none: bool = self.opponent_start is None

    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["agent", start_x, start_y]:
                if self.__agent_start_was_none:
                    self.agent_start = Vec2D(int(start_x), int(start_y))
                    return False
            case ["target", start_x, start_y]:
                if self.__target_start_was_none:
                    self.target_start = Vec2D(int(start_x), int(start_y))
                    return False
            case ["opponent", start_x, start_y]:
                if self.__opponent_start_was_none:
                    self.opponent_start = Vec2D(int(start_x), int(start_y))
                    return False
            case _:
                return True

    def _apply_default_helper(self) -> None:
        if self.agent_policy_type is None:
            self.agent_policy_type = AgentPolicy
        if self.target_policy_type is None:
            self.target_policy_type = UniformPolicy
        if self.opponent_policy_type is None:
            self.opponent_policy_type = UniformPolicy
        if self.target_markov_transition_density_type is None:
            self.target_markov_transition_density_type = SimpleMarkovTransitionDensity
        if self.opponent_markov_transition_density_type is None:
            self.opponent_markov_transition_density_type = SimpleMarkovTransitionDensity

    def _check_helper(self) -> None:
        self._check_not_none(self.agent_start, "Agent start")
        self._check_not_none(self.target_start, "Target start")
        self._check_not_none(self.opponent_start, "Opponent start")
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
        self.target_markov_transition_density: PolicyFun = self._create_markov_transition_density(self.target_markov_transition_density_type)
        self.opponent_markov_transition_density: PolicyFun = self._create_markov_transition_density(self.opponent_markov_transition_density_type)
        self.agent_policy: PolicyFun = self._create_policy(self.agent_policy_type)
        self.target_policy: PolicyFun = self._create_policy(self.target_policy_type)
        self.opponent_policy: PolicyFun = self._create_policy(self.opponent_policy_type)

class TrainConfigs(BaseConfigs):

    def __init__(self) -> None:
        super().__init__()
        self.reward_type: type[RewardFunction] | None = None
        self.max_iter: int | None = None
        self.value_function_tolerance: float | None = None
        self.changed_actions_tolerance: int | None = None
        self.changed_actions_percentage_tolerance: float | None = None

    def _prepare_for_reading_configs_helper(self) -> None:
        self.__maxiter_was_none: bool = self.max_iter is None
        self.__value_function_tolerance_was_none: bool = self.value_function_tolerance is None
        self.__changed_actions_tolerance_was_none: bool = self.changed_actions_tolerance is None
        self.__changed_actions_percentage_tolerance_was_none: bool = self.changed_actions_percentage_tolerance is None

    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["maxiter", max_iter]:
                if self.__maxiter_was_none:
                    self.max_iter = int(max_iter)
                    return False
            case ["valuetolerance", value_tol]:
                if self.__value_function_tolerance_was_none:
                    self.value_function_tolerance = float(value_tol)
                    return False
            case ["actiontolerance", action_tol]:
                if self.__changed_actions_tolerance_was_none:
                    self.changed_actions_tolerance = int(action_tol)
                    return False
            case ["actionperctolerance", action_perc_tol]:
                if self.__changed_actions_percentage_tolerance_was_none:
                    self.changed_actions_percentage_tolerance = float(action_perc_tol)
                    return False
            case _:
                return True

    def _apply_default_helper(self) -> None:
        if self.reward_type is None:
            self.reward_type = SimpleRewardFunction
        if self.max_iter is None:
            self.max_iter = 100
        if self.value_function_tolerance is None:
            self.value_function_tolerance = 0.0
        if self.changed_actions_tolerance is None:
            self.changed_actions_tolerance = 0
        if self.changed_actions_percentage_tolerance is None:
            self.changed_actions_percentage_tolerance = 0.0

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
        self.reward: RewardFunction = self._create_reward(self.reward_type)