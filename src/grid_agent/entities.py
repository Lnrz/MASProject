from grid_agent.data_structs import State, Policy, MapSize, Obstacle, Vec2D, Action, Result
from grid_agent.functors import ActionSelector, NextPosSelector
from grid_agent.settings import GameSettings
from copy import copy

class MapManager:
    
    def __init__(self, map_size: Vec2D) -> None:
        self.map_size: MapSize = MapSize(map_size.x, map_size.y)
        self.__obstacles: list[Obstacle] = list[Obstacle]()

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.__obstacles.append(obstacle)

    def can_move_to(self, pos: Vec2D) -> bool:
        if self.__out_of_bounds(pos):
            return False
        return not any([obs.is_inside(pos) for obs in self.__obstacles])
    
    def __out_of_bounds(self, pos: Vec2D) -> bool:
        return (pos.x < 0 or pos.x >= self.map_size.N or
                pos.y < 0 or pos.y >= self.map_size.M)

class Agent:
    
    def __init__(self, start_pos: Vec2D, target_pos: Vec2D, opponent_pos: Vec2D, next_pos_selector: NextPosSelector, policy_file_name: str | None, map_size: MapSize) -> None:
        self.__state: State = State(agent_pos=copy(start_pos), opponent_pos=opponent_pos, target_pos=target_pos)
        self.__policy: Policy = Policy()
        if policy_file_name:
            self.__policy.read_from_file(policy_file_name, map_size.N3M3)
        else:
            print("WARNING: The policy file was not provided, using default policy")
            self.__policy.fill(Action.Up, map_size.N3M3)
        self.__next_pos_selector: NextPosSelector = next_pos_selector

    def get_pos(self) -> Vec2D:
        return self.__state.agent_pos

    def move(self, map_manager: MapManager) -> None:
        action: Action = self.__policy.get_action(self.__state, map_manager.map_size)
        next_pos: Vec2D = self.__next_pos_selector.get_next_pos(self.__state.agent_pos, action)
        if (map_manager.can_move_to(next_pos)):
            self.__state.agent_pos.copy(next_pos)

class MovingEntity:
    
    def __init__(self, start_pos: Vec2D, action_selector: ActionSelector, next_pos_selector: NextPosSelector) -> None:
        self.__pos: Vec2D = copy(start_pos)
        self.__action_selector: ActionSelector = action_selector
        self.__next_pos_selector: NextPosSelector = next_pos_selector
        self.__banned_pos: Vec2D = Vec2D(-1, -1)

    def get_pos(self) -> Vec2D:
        return self.__pos

    def set_banned_pos(self, banned_pos: Vec2D) -> None:
        self.__banned_pos: Vec2D = banned_pos

    def move(self, map_manager: MapManager) -> None:
        action: Action = self.__action_selector.get_action()
        next_pos: Vec2D = self.__next_pos_selector.get_next_pos(self.__pos, action)
        if (map_manager.can_move_to(next_pos) and next_pos != self.__banned_pos):
            self.__pos.copy(next_pos)

class GameManager:
    
    def __init__(self, game_settings: GameSettings) -> None:
        self.__map_manager: MapManager = MapManager(game_settings.map_size)
        self.__target: MovingEntity = MovingEntity(game_settings.target_start_pos, game_settings.target_action_selector, game_settings.target_next_pos_selector)
        self.__opponent: MovingEntity = MovingEntity(game_settings.opponent_start_pos, game_settings.opponent_action_selector, game_settings.opponent_next_pos_selector)
        self.__target.set_banned_pos(self.__opponent.get_pos())
        self.__opponent.set_banned_pos(self.__target.get_pos())
        self.__agent: Agent = Agent(game_settings.agent_start_pos, self.__target.get_pos(), self.__opponent.get_pos(), game_settings.agent_next_pos_selector, game_settings.policy_file_path, self.__map_manager.map_size)

    def start(self) -> Result:
        self.__res: Result = Result.WaitingForResult
        while self.__res == Result.WaitingForResult:
            self.__next_iteration()
        return self.__res

    def __next_iteration(self) -> None:
        self.__agent.move(self.__map_manager)
        if self.__check_for_result():
            return
        self.__target.move(self.__map_manager)
        self.__opponent.move(self.__map_manager)
        self.__check_for_result()

    def __check_for_result(self) -> bool:
        if self.__agent.get_pos() == self.__target.get_pos():
            self.__res = Result.Success
            return True
        if self.__agent.get_pos() == self.__opponent.get_pos():
            self.__res = Result.Fail
            return True
        return False