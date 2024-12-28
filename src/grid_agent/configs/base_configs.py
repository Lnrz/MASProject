from grid_agent.functors.markov_transition_density import MarkovTransitionDensity, DiscreteDistributionMarkovTransitionDensity
from grid_agent.data_structs.simple_data import Vec2D, Obstacle

from collections.abc import Callable
from abc import ABC, abstractmethod

class  ConfigArgument[T]:

    def __init__(self, value: T) -> None:
        self.value: T = value
        self.frozen: bool = False
    
    def set_and_freeze(self, value: T) -> None:
        if isinstance(value, type(self.value)):
            self.value = value
            self.frozen = True
        else:
            print(f"ConfigArgument not updated with {value}\n" +
                  f"Expected {type(self.value)}, but got {type(value)}")

    def set_if_not_frozen(self, value: T) -> None:
        if not self.frozen:
            self.value = value

class BaseConfigs(ABC):

    def __init__(self) -> None:
        self.__configs_file_path: ConfigArgument[str] = ConfigArgument("")
        self.__policy_file_path: ConfigArgument[str] = ConfigArgument("")
        self.__map_size: ConfigArgument[Vec2D]  = ConfigArgument(Vec2D())
        self.__obstacles: ConfigArgument[list[Obstacle]]  = ConfigArgument([])
        self.__agent_markov_transition_density_factory: ConfigArgument[Callable[[BaseConfigs], MarkovTransitionDensity]] = ConfigArgument(lambda c: DiscreteDistributionMarkovTransitionDensity())
        self.line_processing_extension: Callable[[BaseConfigs, list[str]], None] = lambda c, l: None

    @property
    def configs_file_path(self) -> str:
        return self.__configs_file_path.value

    @configs_file_path.setter
    def configs_file_path(self, path: str) -> None:
        self.__configs_file_path.set_and_freeze(path)

    @property
    def policy_file_path(self) -> str:
        return self.__policy_file_path.value

    @policy_file_path.setter
    def policy_file_path(self, path: str) -> None:
        self.__policy_file_path.set_and_freeze(path)

    @property
    def map_size(self) -> Vec2D:
        return self.__map_size.value

    @map_size.setter
    def map_size(self, map_size: Vec2D) -> None:
        self.__map_size.set_and_freeze(map_size)

    @property
    def obstacles(self) -> list[Obstacle]:
        return self.__obstacles.value

    @obstacles.setter
    def obstacles(self, obstacles: list[Obstacle]) -> None:
        self.__obstacles.set_and_freeze(obstacles)

    @property
    def agent_markov_transition_density_factory(self) -> Callable[["BaseConfigs"], MarkovTransitionDensity]:
        return self.__agent_markov_transition_density_factory.value

    @agent_markov_transition_density_factory.setter
    def agent_markov_transition_density_factory(self, factory: Callable[["BaseConfigs"], MarkovTransitionDensity]) -> None:
        self.__agent_markov_transition_density_factory.set_and_freeze(factory)

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
                line = line.casefold()
                self.__process_line(line)

    def __process_line(self, line: str) -> None:
        if line.startswith("#"):
            return
        splitted_line: list[str] = line.split()
        has_match: bool = True
        match splitted_line:
            case ["mapsize", map_x_length, map_y_length]:
                self.__map_size.set_if_not_frozen(Vec2D(int(map_x_length), int(map_y_length)))
            case ["obstacle", origin_x, origin_y, extent_x, extent_y]:
                if not self.__obstacles.frozen:
                    self.__obstacles.value.append(Obstacle(Vec2D(int(origin_x), int(origin_y)), Vec2D(int(extent_x), int(extent_y))))
            case ["policy", policy_path]:
                self.__policy_file_path.set_if_not_frozen(policy_path)
            case ["ddmtd", "agent", chosen_action_probability, right_action_probability, opposite_action_probability, left_action_probabilty]:
                self.__agent_markov_transition_density_factory.set_if_not_frozen(
                    lambda c: DiscreteDistributionMarkovTransitionDensity(
                        float(chosen_action_probability),
                        float(right_action_probability),
                        float(opposite_action_probability),
                        float(left_action_probabilty)
                    )
                )
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
            if not obstacle.is_inside_bounds(self.map_size):
                raise ValueError(f"An obstacle was out of bounds\n" +
                                 f"Obstacle was [origin: ({obstacle.origin.x},{obstacle.origin.y}), extent: ({obstacle.extent.x},{obstacle.extent.y})]\n" +
                                 f"Map was {self.map_size.x}x{self.map_size.y}")

    def __create(self) -> None:
        self.agent_markov_transition_density: MarkovTransitionDensity = self.agent_markov_transition_density_factory(self)
        self._create_helper()

    @abstractmethod
    def _create_helper(self) -> None:
        ...