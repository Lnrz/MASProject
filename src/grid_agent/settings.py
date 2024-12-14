from grid_agent.data_structs import Vec2D, Obstacle
from grid_agent.functors import ActionSelector, STDActionSelector, NextPosSelector, STDNextPosSelector
import argparse
import re

class CommandLineSettings:
    
    def __init__(self) -> None:
        self.__parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="Grid Agent")
        self.__parser.add_argument("-settings_file", type=str, help="Specify the path to the settings file")
        self.__parser.add_argument("-policy_file", type=str, help="Specify the path to the policy file")
        self.__parser.add_argument("-agent_start", type=str, help="Specify the agent starting position (x,y)")
        self.__parser.add_argument("-target_start", type=str, help="Specify the target starting position (x,y)")
        self.__parser.add_argument("-opponent_start", type=str, help="Specify the opponent starting position (x,y)")

    def parse(self) -> None:
        args = self.__parser.parse_args()
        self.settings_file_path: str | None = args.settings_file
        self.policy_file_path: str | None = args.policy_file
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
        match: re.Match | None = re.search("((\d+),(\d+))", string)
        if match:
            return Vec2D(int(match[1], int(match[2])))
        else:
            raise ValueError(f"A position was ill-formed.\n" + 
                             f"It should have been (x,y) but it was {string}") 

class GameSettings:

    def __init__(self, command_line_settings: CommandLineSettings) -> None:
        self.__set_default_settings()
        self.__read_settings_file(command_line_settings.settings_file_path)
        self.__set_command_line_settings(command_line_settings)
        self.__validate_settings()

    def __set_default_settings(self) -> None:
        self.map_size: Vec2D = Vec2D(3, 3)
        self.obstacles: list[Obstacle] = list[Obstacle]()
        self.agent_start_pos: Vec2D = Vec2D(0, 0)
        self.target_start_pos: Vec2D = Vec2D(2, 2)
        self.opponent_start_pos: Vec2D = Vec2D(2, 0)
        self.policy_file_path: str = "policy.bin"
        self.target_action_selector: ActionSelector = STDActionSelector()
        self.opponent_action_selector: ActionSelector = STDActionSelector()
        self.agent_next_pos_selector: NextPosSelector = STDNextPosSelector()
        self.target_next_pos_selector: NextPosSelector = STDNextPosSelector()
        self.opponent_next_pos_selector: NextPosSelector = STDNextPosSelector()

    def __read_settings_file(self, settings_file_path: str | None) -> None:
        if not settings_file_path:
            return
        with open(settings_file_path) as f:
            for line in f.readlines():
                self.__process_line(line)
    
    def __process_line(self, line: str) -> None:
        if line.startswith("#"):
            return
        splitted_line = line.split()
        splitted_line[0] = splitted_line[0].casefold()
        match splitted_line:
            case ["mapsize", n, m]:
                self.map_size = Vec2D(int(n), int(m))
            case ["obstacle", xo, yo, xe, ye]:
                self.obstacles.append(Obstacle(Vec2D(int(xo), int(yo)), Vec2D(int(xe), int(ye))))
            case ["agent", xs, ys]:
                self.agent_start_pos = Vec2D(int(xs), int(ys))
            case ["target", xs, ys]:
                self.target_start_pos = Vec2D(int(xs), int(ys))
            case ["opponent", xs, ys]:
                self.opponent_start_pos = Vec2D(int(xs), int(ys))

    def __set_command_line_settings(self, command_line_settings: CommandLineSettings):
        if command_line_settings.policy_file_path:
            self.policy_file_path = command_line_settings.policy_file_path
        if command_line_settings.agent_start_pos:
            self.agent_start_pos = command_line_settings.agent_start_pos
        if command_line_settings.target_start_pos:
            self.target_start_pos = command_line_settings.target_start_pos
        if command_line_settings.opponent_start_pos:
            self.opponent_start_pos = command_line_settings.opponent_start_pos

    def __validate_settings(self) -> None:
        self.__check_map_size()
        self.__check_for_same_starting_position("Agent", self.agent_start_pos, "Target", self.target_start_pos)
        self.__check_for_same_starting_position("Agent", self.agent_start_pos, "Opponent", self.opponent_start_pos)
        self.__check_for_same_starting_position("Target", self.target_start_pos, "Opponent", self.opponent_start_pos)
        self.__check_for_collision_with_obstacles("Agent", self.agent_start_pos)
        self.__check_for_collision_with_obstacles("Target", self.target_start_pos)
        self.__check_for_collision_with_obstacles("Opponent", self.opponent_start_pos)
    
    def __check_map_size(self) -> None:
        if self.map_size.x * self.map_size.y < 3 or self.map_size.x < 0:
            raise ValueError(f"Map size should be positive and have at least 3 cells.\n"
                             + f"Map Size: ({self.map_size.x}, {self.map_size.y})")
        
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