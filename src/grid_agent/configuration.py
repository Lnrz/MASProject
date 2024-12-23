from grid_agent.data_structs import Vec2D, Obstacle, ValidStateSpace
from grid_agent.functors import PolicyFun, UniformPolicy, MarkovTransitionDensity, SimpleMarkovTransitionDensity, RewardFunction, SimpleRewardFunction, AgentPolicy
from abc import ABC, abstractmethod
from typing import Callable

class  ConfigArgument[T]:

    def __init__(self, value: T) -> None:
        self.value: T = value
        self.frozen: bool = False

class BaseConfigs(ABC):

    def __init__(self) -> None:
        self._configs_file_path: ConfigArgument[str] = ConfigArgument("")
        self._policy_file_path: ConfigArgument[str] = ConfigArgument("")
        self._map_size: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self._obstacles: ConfigArgument[list[Obstacle]]  = ConfigArgument([])
        self._agent_markov_transition_density_factory: ConfigArgument[Callable[[BaseConfigs], MarkovTransitionDensity]] = ConfigArgument(lambda c: SimpleMarkovTransitionDensity())
        self.line_processing_extension: Callable[[BaseConfigs, list[str]], None] = lambda c, l: None

    @property
    def configs_file_path(self) -> str:
        return self._configs_file_path.value

    @configs_file_path.setter
    def configs_file_path(self, path: str) -> None:
        self._configs_file_path.value = path
        self._configs_file_path.frozen = True

    @property
    def policy_file_path(self) -> str:
        return self._policy_file_path.value

    @policy_file_path.setter
    def policy_file_path(self, path: str) -> None:
        self._policy_file_path.value = path
        self._policy_file_path.frozen = True

    @property
    def map_size(self) -> Vec2D:
        return self._map_size.value

    @map_size.setter
    def map_size(self, map_size: Vec2D) -> None:
        self._map_size.value = map_size
        self._map_size.frozen = True

    @property
    def obstacles(self) -> list[Obstacle]:
        return self._obstacles.value

    @obstacles.setter
    def obstacles(self, obstacles: list[Obstacle]) -> None:
        self._obstacles.value = obstacles
        self._obstacles.frozen = True

    @property
    def agent_markov_transition_density_factory(self) -> Callable[["BaseConfigs"], MarkovTransitionDensity]:
        return self._agent_markov_transition_density_factory.value

    @agent_markov_transition_density_factory.setter
    def agent_markov_transition_density_factory(self, factory: Callable[["BaseConfigs"], MarkovTransitionDensity]) -> None:
        self._agent_markov_transition_density_factory.value = factory
        self._agent_markov_transition_density_factory.frozen = True

    def validate(self) -> None:
        self.__apply_file()
        self.__check()
        self.__create()

    def __apply_file(self) -> None:
        if not self.configs_file_path:
            return
        with open(self.configs_file_path) as f:
            for line in f.readlines():
                if line.isspace():
                    continue
                self.__process_line(line)

    def __process_line(self, line: str) -> None:
        if line.startswith("#"):
            return
        splitted_line: list[str] = line.split()
        splitted_line[0] = splitted_line[0].casefold()
        has_match: bool = True
        match splitted_line:
            case ["mapsize", map_x_length, map_y_length]:
                if not self._map_size.frozen:
                    self._map_size.value = Vec2D(int(map_x_length), int(map_y_length))
            case ["obstacle", origin_x, origin_y, extent_x, extent_y]:
                if not self._obstacles.frozen:
                    self._obstacles.value.append(Obstacle(Vec2D(int(origin_x), int(origin_y)), Vec2D(int(extent_x), int(extent_y))))
            case ["policy", policy_path]:
                if not self._policy_file_path.frozen:
                    self._policy_file_path.value = policy_path
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
        self._agent_start: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self._target_start: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self._opponent_start: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self._agent_policy_factory: ConfigArgument[Callable[[GameConfigs], PolicyFun]] = ConfigArgument(lambda c: AgentPolicy(c.policy_file_path, c.valid_state_space))
        self._target_policy_factory: ConfigArgument[Callable[[GameConfigs], PolicyFun]] = ConfigArgument(lambda c: UniformPolicy())
        self._opponent_policy_factory: ConfigArgument[Callable[[GameConfigs], PolicyFun]] = ConfigArgument(lambda c: UniformPolicy())
        self._target_markov_transition_density_factory: ConfigArgument[Callable[[GameConfigs], MarkovTransitionDensity]] = ConfigArgument(lambda c: SimpleMarkovTransitionDensity())
        self._opponent_markov_transition_density_factory: ConfigArgument[Callable[[GameConfigs], MarkovTransitionDensity]] = ConfigArgument(lambda c: SimpleMarkovTransitionDensity())
    
    @property
    def agent_start(self) -> Vec2D:
        return self._agent_start.value

    @agent_start.setter
    def agent_start(self, start: Vec2D) -> None:
        self._agent_start.value = start
        self._agent_start.frozen = True

    @property
    def target_start(self) -> Vec2D:
        return self._target_start.value

    @target_start.setter
    def target_start(self, start: Vec2D) -> None:
        self._target_start.value = start
        self._target_start.frozen = True

    @property
    def opponent_start(self) -> Vec2D:
        return self._opponent_start.value

    @opponent_start.setter
    def opponent_start(self, start: Vec2D) -> None:
        self._opponent_start.value = start
        self._opponent_start.frozen = True

    @property
    def agent_policy_factory(self) -> Callable[["GameConfigs"], PolicyFun]:
        return self._agent_policy_factory.value

    @agent_policy_factory.setter
    def agent_policy_factory(self, factory: Callable[["GameConfigs"], PolicyFun]) -> None:
        self._agent_policy_factory.value = factory
        self._agent_policy_factory.frozen = True

    @property
    def target_policy_factory(self) -> Callable[["GameConfigs"], PolicyFun]:
        return self._target_policy_factory.value

    @target_policy_factory.setter
    def target_policy_factory(self, factory: Callable[["GameConfigs"], PolicyFun]) -> None:
        self._target_policy_factory.value = factory
        self._target_policy_factory.frozen = True

    @property
    def opponent_policy_factory(self) -> Callable[["GameConfigs"], PolicyFun]:
        return self._opponent_policy_factory.value

    @opponent_policy_factory.setter
    def opponent_policy_factory(self, factory: Callable[["GameConfigs"], PolicyFun]) -> None:
        self._opponent_policy_factory.value = factory
        self._opponent_policy_factory.frozen = True

    @property
    def target_markov_transition_density_factory(self) -> Callable[["GameConfigs"], MarkovTransitionDensity]:
        return self._target_markov_transition_density_factory.value

    @target_markov_transition_density_factory.setter
    def target_markov_transition_density_factory(self, factory: Callable[["GameConfigs"], MarkovTransitionDensity]) -> None:
        self._target_markov_transition_density_factory.value = factory
        self._target_markov_transition_density_factory.frozen = True

    @property
    def opponent_markov_transition_density_factory(self) -> Callable[["GameConfigs"], MarkovTransitionDensity]:
        return self._opponent_markov_transition_density_factory.value

    @opponent_markov_transition_density_factory.setter
    def opponent_markov_transition_density_factory(self, factory: Callable[["GameConfigs"], MarkovTransitionDensity]) -> None:
        self._opponent_markov_transition_density_factory.value = factory
        self._opponent_markov_transition_density_factory.frozen = True

    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["agent", start_x, start_y]:
                if not self._agent_start.frozen:
                    self._agent_start.value = Vec2D(int(start_x), int(start_y))
            case ["target", start_x, start_y]:
                if not self._target_start.frozen:
                    self._target_start.value = Vec2D(int(start_x), int(start_y))
            case ["opponent", start_x, start_y]:
                if not self._opponent_start.frozen:
                    self._opponent_start.value = Vec2D(int(start_x), int(start_y))
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
        self._reward_factory: ConfigArgument[Callable[[TrainConfigs], RewardFunction]] = ConfigArgument(lambda c: SimpleRewardFunction())
        self._max_iter: ConfigArgument[int] = ConfigArgument(100)
        self._value_function_tolerance: ConfigArgument[float] = ConfigArgument(0.0)
        self._changed_actions_tolerance: ConfigArgument[int] = ConfigArgument(0)
        self._changed_actions_percentage_tolerance: ConfigArgument[float] = ConfigArgument(0.0)

    @property
    def reward_factory(self) -> Callable[["TrainConfigs"], RewardFunction]:
        return self._reward_factory.value
    
    @reward_factory.setter
    def reward_factory(self, factory: Callable[["TrainConfigs"], RewardFunction]) -> None:
        self._reward_factory.value = factory
        self._reward_factory.frozen = True

    @property
    def max_iter(self) -> int:
        return self._max_iter.value

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        self._max_iter.value = value
        self._max_iter.frozen = True

    @property
    def value_function_tolerance(self) -> float:
        return self._value_function_tolerance.value

    @value_function_tolerance.setter
    def value_function_tolerance(self, value: float) -> None:
        self._value_function_tolerance.value = value
        self._value_function_tolerance.frozen = True

    @property
    def changed_actions_tolerance(self) -> int:
        return self._changed_actions_tolerance.value

    @changed_actions_tolerance.setter
    def changed_actions_tolerance(self, value: int) -> None:
        self._changed_actions_tolerance.value = value
        self._changed_actions_tolerance.frozen = True

    @property
    def changed_actions_percentage_tolerance(self) -> float:
        return self._changed_actions_percentage_tolerance.value

    @changed_actions_percentage_tolerance.setter
    def changed_actions_percentage_tolerance(self, value: float) -> None:
        self._changed_actions_percentage_tolerance.value = value
        self._changed_actions_percentage_tolerance.frozen = True

    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["maxiter", max_iter]:
                if not self._max_iter.frozen:
                    self._max_iter.value = int(max_iter)
            case ["valuetolerance", value_tol]:
                if not self._value_function_tolerance.frozen:
                    self._value_function_tolerance.value = float(value_tol)
            case ["actiontolerance", action_tol]:
                if not  self._changed_actions_tolerance.frozen:
                    self._changed_actions_tolerance.value = int(action_tol)
            case ["actionperctolerance", action_perc_tol]:
                if not self._changed_actions_percentage_tolerance.frozen:
                    self._changed_actions_percentage_tolerance.value = float(action_perc_tol)
            case _:
                return False
        return True

    def _check_helper(self) -> None:
        self.__check_positivity(self.max_iter, "Maximum number of iterations")
        self.__check_non_negativity(self.value_function_tolerance, "Value function tolerance")
        self.__check_non_negativity(self.changed_actions_tolerance, "Change actions tolerance")
        self.__check_non_negativity(self.changed_actions_percentage_tolerance, "Changed actions percentage tolerance")

    def __check_positivity(self, value: int | float, name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} should be > 0.\n" +
                             f"It was {value}")

    def __check_non_negativity(self, value: float | int, name: str) -> None:
        if value < 0:
            raise ValueError(f"{name} should be >= 0" +
                             f"It was {value}")

    def _create_helper(self) -> None:
        self.reward: RewardFunction = self.reward_factory(self)