from grid_agent.data_structs import GameData, Obstacle, Vec2D, Action, Result
from typing import Callable
from time import sleep

class ASCIIView:

    __void_char: str = " "
    __unknown_char: str = "?"
    __free_space_char: str = "O"
    __obstacle_char: str = "X"
    __agent_char: str = "A"
    __target_char: str = "T"
    __opponent_char: str = "E"
    __win_char: str = "W"
    __lose_char: str = "L"
    __up_char: str = "^"
    __right_char: str = ">"
    __down_char: str = "v"
    __left_char: str = "<"
    __horizontal_border_char: str = "="
    __vertical_border_char: str = "|"
    __grid_horizontal_factor: int = 5
    __grid_horizontal_shift: int = 2
    __grid_vertical_factor: int = 3
    __grid_vertical_shift: int = 1

    def __init__(self, map_size: Vec2D, obstacles: list[Obstacle]) -> None:
        self.__grid_size: Vec2D = Vec2D(self.__grid_horizontal_factor * map_size.x, self.__grid_vertical_factor * map_size.y)
        self.__grid: list[str] = [self.__void_char for i in range(self.__grid_size.x * self.__grid_size.y)]
        self.__gamedatas: list[GameData] = list[GameData]()
        self.__result: Result = Result.WAITING_FOR_RESULT
        self.__add_free_space(map_size)
        self.__add_obstacles(obstacles)

    def get_callback(self) -> Callable[[GameData], None]:
        return lambda gamedata: self.__get_game_data(gamedata)

    def start_manual(self) -> None:
        last_gamedata: GameData | None = None
        for gamedata in self.__gamedatas:
            self.__update_grid(gamedata, last_gamedata)
            self.__print_grid()
            last_gamedata = gamedata
            match self.__result:
                case Result.WAITING_FOR_RESULT:
                    input("Press 'Enter' to continue")
                case Result.SUCCESS:
                    print("Win!")
                case Result.FAIL:
                    print("Lost")

    def start_auto(self, time_interval: float) -> None:
        last_gamedata: GameData | None = None
        for gamedata in self.__gamedatas:
            self.__update_grid(gamedata, last_gamedata)
            self.__print_grid()
            last_gamedata = gamedata
            match self.__result:
                case Result.WAITING_FOR_RESULT:
                    sleep(time_interval)
                case Result.SUCCESS:
                    print("Win!")
                case Result.FAIL:
                    print("Lost")

    def __add_free_space(self, map_size: Vec2D) -> None:
        for x in range(map_size.x):
            for y in range(map_size.y):
                self.__grid[self.__pos_to_grid_index(Vec2D(x, y))] = self.__free_space_char

    def __add_obstacles(self, obstacles: list[Obstacle]) -> None:
        for obstacle in obstacles:
            for point in obstacle.to_pos():
                self.__grid[self.__pos_to_grid_index(point)] = self.__obstacle_char

    def __pos_to_grid_index(self, pos: Vec2D) -> int:
        return ((self.__grid_horizontal_shift + self.__grid_horizontal_factor * pos.x) +
                (self.__grid_size.y - (self.__grid_vertical_shift + self.__grid_vertical_factor * pos.y) - 1) * self.__grid_size.x)

    def __get_game_data(self, gamedata: GameData) -> None:
        self.__gamedatas.append(gamedata)

    def __update_grid(self, gamedata: GameData, last_gamedata: GameData | None) -> None:
        if last_gamedata:
            self.__grid[self.__pos_to_grid_index(last_gamedata.state.agent_pos)] = self.__free_space_char
            self.__grid[self.__pos_to_grid_index(last_gamedata.state.target_pos)] = self.__free_space_char
            self.__grid[self.__pos_to_grid_index(last_gamedata.state.opponent_pos)] = self.__free_space_char
            self.__grid[self.__action_to_grid_index(last_gamedata.state.agent_pos, last_gamedata.agent_action)] = self.__void_char
            self.__grid[self.__action_to_grid_index(last_gamedata.state.target_pos, last_gamedata.target_action)] = self.__void_char
            self.__grid[self.__action_to_grid_index(last_gamedata.state.opponent_pos, last_gamedata.opponent_action)] = self.__void_char
        if gamedata.state.agent_pos == gamedata.state.target_pos:
            self.__grid[self.__pos_to_grid_index(gamedata.state.agent_pos)] = self.__win_char
            self.__grid[self.__pos_to_grid_index(gamedata.state.opponent_pos)] = self.__opponent_char
            self.__result = Result.SUCCESS
            return
        if gamedata.state.agent_pos == gamedata.state.opponent_pos:
            self.__grid[self.__pos_to_grid_index(gamedata.state.agent_pos)] = self.__lose_char
            self.__grid[self.__pos_to_grid_index(gamedata.state.target_pos)] = self.__target_char
            self.__result = Result.FAIL
            return
        self.__grid[self.__pos_to_grid_index(gamedata.state.agent_pos)] = self.__agent_char
        self.__grid[self.__pos_to_grid_index(gamedata.state.target_pos)] = self.__target_char
        self.__grid[self.__pos_to_grid_index(gamedata.state.opponent_pos)] = self.__opponent_char
        self.__grid[self.__action_to_grid_index(gamedata.state.agent_pos, gamedata.agent_action)] = self.__get_action_character(gamedata.agent_action)
        self.__grid[self.__action_to_grid_index(gamedata.state.target_pos, gamedata.target_action)] = self.__get_action_character(gamedata.target_action)
        self.__grid[self.__action_to_grid_index(gamedata.state.opponent_pos, gamedata.opponent_action)] = self.__get_action_character(gamedata.opponent_action)

    def __action_to_grid_index(self, pos: Vec2D, action: Action) -> int:
        index: int = self.__pos_to_grid_index(pos)
        match action:
            case Action.UP:
                index -= self.__grid_size.x
            case Action.RIGHT:
                index += 2
            case Action.DOWN:
                index += self.__grid_size.x
            case Action.LEFT:
                index -= 2
        return index

    def __get_action_character(self, action: Action) -> str:
        match action:
            case Action.UP:
                return self.__up_char
            case Action.RIGHT:
                return self.__right_char
            case Action.DOWN:
                return self.__down_char
            case Action.LEFT:
                return self.__left_char
            case _:
                return self.__unknown_char

    def __print_grid(self) -> None:
        print(self.__horizontal_border_char * (self.__grid_size.x + 2))
        for i in range(self.__grid_size.y):
            start_index: int = i * self.__grid_size.x
            end_index: int = start_index + self.__grid_size.x
            line: str = "".join(self.__grid[start_index : end_index])
            print(self.__vertical_border_char + line + self.__vertical_border_char)
        print(self.__horizontal_border_char * (self.__grid_size.x + 2))