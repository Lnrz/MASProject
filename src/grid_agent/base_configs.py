from grid_agent.data_structs import Vec2D, Obstacle, ValidStateSpace
from grid_agent.functors import MarkovTransitionDensity, SimpleMarkovTransitionDensity
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