from grid_agent.data_structs import Vec2D, Obstacle
from grid_agent.functors import ActionSelector, SimpleActionSelector, MarkovTransitionDensity, SimpleMarkovTransitionDensity, RewardFunction, SimpleRewardFunction
from abc import ABC, abstractmethod
import argparse
import re

class BaseCommandLineArguments(ABC):

    def __init__(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="Grid Agent")
        self.__add_arguments(parser)
        self.__parse(parser)

    def __add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("-settings_file", type=str, help="Specify the path to the settings file")
        parser.add_argument("-policy_file", type=str, help="Specify the path to the policy file")
        self._add_arguments_helper(parser)

    @abstractmethod
    def _add_arguments_helper(self, parser: argparse.ArgumentParser) -> None:
        ...

    def __parse(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args()
        self.settings_file_path: str | None = args.settings_file
        self.policy_file_path: str | None = args.policy_file
        self._parse_helper(args)

    @abstractmethod
    def _parse_helper(self, args: argparse.Namespace) -> None:
        ... 

class GameCommandLineArguments(BaseCommandLineArguments):
    
    def _add_arguments_helper(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("-agent_start", type=str, help="Specify the agent starting position (x,y)")
        parser.add_argument("-target_start", type=str, help="Specify the target starting position (x,y)")
        parser.add_argument("-opponent_start", type=str, help="Specify the opponent starting position (x,y)")

    def _parse_helper(self, args: argparse.Namespace) -> None:
        self.agent_start_pos: Vec2D | None = None
        if args.agent_start:
            self.agent_start_pos = self.__string_to_vec2D(args.agent_start)
        self.target_start_pos: Vec2D | None = None
        if args.target_start:
            self.target_start_pos = self.__string_to_vec2D(args.target_start)
        self.opponent_start_pos: Vec2D | None = None
        if args.opponent_start:
            self.opponent_start_pos_start_pos = self.__string_to_vec2D(args.opponent_start)
    
    def __string_to_vec2D(self, string: str) -> Vec2D:
        match: re.Match | None = re.search(r"\((\d+),(\d+)\)", string)
        if match:
            return Vec2D(int(match[1]), int(match[2]))
        else:
            raise ValueError(f"A position was ill-formed.\n" + 
                             f"It should have been (x,y) but it was {string}") 

class TrainCommandLineArguments(BaseCommandLineArguments):
    
    def _add_arguments_helper(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("-max_iter", type=int, help="Specify the maximum number of iterations")

    def _parse_helper(self, args: argparse.Namespace) -> None:
        self.max_iter: int | None = None
        if args.max_iter:
            self.max_iter = args.max_iter



class BaseSettings(ABC):

    def __init__(self, command_line_arguments: BaseCommandLineArguments) -> None:
        self.__set_default_settings()
        self.__read_settings_file(command_line_arguments.settings_file_path)
        self.__set_command_line_settings(command_line_arguments)
        self.__validate_settings()

    def __set_default_settings(self) -> None:
        self.policy_file_path: str | None = None
        self.map_size: Vec2D = Vec2D(3, 3)
        self.obstacles: list[Obstacle] = list[Obstacle]()
        self.agent_next_pos_selector: MarkovTransitionDensity = SimpleMarkovTransitionDensity()
        self._set_default_settings_helper()

    @abstractmethod
    def _set_default_settings_helper(self) -> None:
        ...

    def __read_settings_file(self, settings_file_path: str | None) -> None:
        if not settings_file_path:
            print("WARNING: The settings file was not provided, using default and/or command line settings")
            return
        with open(settings_file_path) as f:
            for line in f.readlines():
                if not line.isspace():
                    self.__process_line(line)
    
    def __process_line(self, line: str) -> None:
        if line.startswith("#"):
            return
        splitted_line: list[str] = line.split()
        splitted_line[0] = splitted_line[0].casefold()
        no_match: bool = False
        match splitted_line:
            case ["mapsize", map_x_length, map_y_length]:
                self.map_size = Vec2D(int(map_x_length), int(map_y_length))
            case ["obstacle", origin_x, origin_y, extent_x, extent_y]:
                self.obstacles.append(Obstacle(Vec2D(int(origin_x), int(origin_y)), Vec2D(int(extent_x), int(extent_y))))
            case ["policy", policy_path]:
                self.policy_file_path = policy_path
            case _:
                no_match = True
        if no_match:
            self._process_line_helper(splitted_line)

    @abstractmethod
    def _process_line_helper(self, splitted_line: list[str]):
        ...

    def __set_command_line_settings(self, command_line_arguments: BaseCommandLineArguments):
        if command_line_arguments.policy_file_path:
            self.policy_file_path = command_line_arguments.policy_file_path
        self._set_command_line_settings_helper(command_line_arguments)

    @abstractmethod
    def _set_command_line_settings_helper(self, command_line_arguments: BaseCommandLineArguments) -> None:
        ...

    def __validate_settings(self) -> None:
        self.__check_map_size()
        self._validate_settings_helper()

    @abstractmethod
    def _validate_settings_helper(self) -> None:
        ...

    def __check_map_size(self) -> None:
        if self.map_size.x * self.map_size.y < 3 or self.map_size.x < 0:
            raise ValueError(f"Map size should be positive and have at least 3 cells.\n"
                             + f"Map Size: ({self.map_size.x}, {self.map_size.y})")

class GameSettings(BaseSettings):

    def __init__(self, command_line_arguments: GameCommandLineArguments) -> None:
        super().__init__(command_line_arguments)

    def _set_default_settings_helper(self) -> None:
        self.agent_start_pos: Vec2D = Vec2D(0, 0)
        self.target_start_pos: Vec2D = Vec2D(2, 2)
        self.opponent_start_pos: Vec2D = Vec2D(2, 0)
        self.target_action_selector: ActionSelector = SimpleActionSelector()
        self.opponent_action_selector: ActionSelector = SimpleActionSelector()
        self.target_next_pos_selector: MarkovTransitionDensity = SimpleMarkovTransitionDensity()
        self.opponent_next_pos_selector: MarkovTransitionDensity = SimpleMarkovTransitionDensity()
    
    def _process_line_helper(self, splitted_line: list[str]) -> None:
        match splitted_line:
            case ["agent", start_x, start_y]:
                self.agent_start_pos = Vec2D(int(start_x), int(start_y))
            case ["target", start_x, start_y]:
                self.target_start_pos = Vec2D(int(start_x), int(start_y))
            case ["opponent", start_x, start_y]:
                self.opponent_start_pos = Vec2D(int(start_x), int(start_y))

    def _set_command_line_settings_helper(self, command_line_arguments: BaseCommandLineArguments) -> None:
        if command_line_arguments.agent_start_pos:
            self.agent_start_pos = command_line_arguments.agent_start_pos
        if command_line_arguments.target_start_pos:
            self.target_start_pos = command_line_arguments.target_start_pos
        if command_line_arguments.opponent_start_pos:
            self.opponent_start_pos = command_line_arguments.opponent_start_pos

    def _validate_settings_helper(self) -> None:
        self.__check_for_same_starting_position("Agent", self.agent_start_pos, "Target", self.target_start_pos)
        self.__check_for_same_starting_position("Agent", self.agent_start_pos, "Opponent", self.opponent_start_pos)
        self.__check_for_same_starting_position("Target", self.target_start_pos, "Opponent", self.opponent_start_pos)
        self.__check_for_collision_with_obstacles("Agent", self.agent_start_pos)
        self.__check_for_collision_with_obstacles("Target", self.target_start_pos)
        self.__check_for_collision_with_obstacles("Opponent", self.opponent_start_pos)
    
    def __check_for_same_starting_position(self, name1: str, pos1: Vec2D, name2: str, pos2: Vec2D) -> None:
        if pos1 == pos2:
            raise ValueError(f"{name1} and {name2} should start at different positions.\n"
                            + f"{name1}: ({pos1.x}, {pos1.y})\n"
                            + f"{name2}: ({pos2.x}, {pos2.y})")
        
    def __check_for_collision_with_obstacles(self, name: str, pos: Vec2D) -> None:
        for obstacle in self.obstacles:
            if obstacle.is_inside(pos):
                raise ValueError(f"{name} is colliding with an obstacle.\n"
                                 + f"{name}: ({pos.x}, {pos.y})\n"
                                 + f"Obstacle: [origin: ({obstacle.origin.x}, {obstacle.origin.y}), extent: ({obstacle.extent.x}, {obstacle.extent.y})]")
            
class TrainSettings(BaseSettings):

    def __init__(self, command_line_arguments: TrainCommandLineArguments) -> None:
        super().__init__(command_line_arguments)

    def _set_default_settings_helper(self) -> None:
        self.policy_file_path = "..\policies\policy.bin"
        self.max_iter: int = 100
        self.reward: RewardFunction = SimpleRewardFunction()
    
    def _process_line_helper(self, splitted_line: list[str]) -> None:
        match splitted_line:
            case ["maxiter", max_iter]:
                self.max_iter = int(max_iter)

    def _set_command_line_settings_helper(self, command_line_arguments: BaseCommandLineArguments) -> None:
        if command_line_arguments.max_iter:
            self.max_iter = command_line_arguments.max_iter

    def _validate_settings_helper(self) -> None:
        if self.max_iter <= 0:
            raise ValueError(f"The max number of iterations should be >= 0.\n" +
                             f"It was {self.max_iter}")