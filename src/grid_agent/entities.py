from grid_agent.data_structs import State, Policy, MapSize, Obstacle, Vec2D, Action, Result, ValueFunction, GameData
from grid_agent.functors import ActionSelector, NextPosSelector, RewardFunction
from grid_agent.settings import GameSettings, TrainSettings
from typing import Callable
from copy import copy, deepcopy

class MapManager:
    
    def __init__(self, map_size: Vec2D, obstacles: list[Obstacle]) -> None:
        self.map_size: MapSize = MapSize(map_size.x, map_size.y)
        self.__obstacles: list[Obstacle] = obstacles
        self.__pos : Vec2D = Vec2D()

    def move_if_possible(self, pos: Vec2D, action: Action) -> None:
        self.__pos.copy(pos)
        self.__pos.move(action)
        if self.is_pos_possible(self.__pos):
            pos.copy(self.__pos)

    def is_state_possible(self, state: State) -> bool:
        return (self.is_pos_possible(state.agent_pos) and self.is_pos_possible(state.target_pos) and self.is_pos_possible(state.opponent_pos) and
                state.target_pos != state.opponent_pos)

    def is_pos_possible(self, pos: Vec2D) -> bool:
        if self.__is_out_of_bounds(pos):
            return False
        return not any([obs.is_inside(pos) for obs in self.__obstacles])
    
    def __is_out_of_bounds(self, pos: Vec2D) -> bool:
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

    def get_state(self) -> State:
        return self.__state

    def get_pos(self) -> Vec2D:
        return self.__state.agent_pos

    def move(self, map_manager: MapManager) -> Action:
        action: Action = self.__policy.get_action(self.__state, map_manager.map_size)
        next_pos: Vec2D = self.__next_pos_selector.get_next_pos(self.__state.agent_pos, action)
        if (map_manager.is_pos_possible(next_pos)):
            self.__state.agent_pos.copy(next_pos)
        return action

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

    def move(self, map_manager: MapManager) -> Action:
        action: Action = self.__action_selector.get_action()
        next_pos: Vec2D = self.__next_pos_selector.get_next_pos(self.__pos, action)
        if (map_manager.is_pos_possible(next_pos) and next_pos != self.__banned_pos):
            self.__pos.copy(next_pos)
        return action

class GameManager:
    
    def __init__(self, game_settings: GameSettings) -> None:
        self.__map_manager: MapManager = MapManager(game_settings.map_size, game_settings.obstacles)
        self.__target: MovingEntity = MovingEntity(game_settings.target_start_pos, game_settings.target_action_selector, game_settings.target_next_pos_selector)
        self.__opponent: MovingEntity = MovingEntity(game_settings.opponent_start_pos, game_settings.opponent_action_selector, game_settings.opponent_next_pos_selector)
        self.__target.set_banned_pos(self.__opponent.get_pos())
        self.__opponent.set_banned_pos(self.__target.get_pos())
        self.__agent: Agent = Agent(game_settings.agent_start_pos, self.__target.get_pos(), self.__opponent.get_pos(), game_settings.agent_next_pos_selector, game_settings.policy_file_path, self.__map_manager.map_size)
        self.__gamedata: GameData = GameData()
        self.__callback: Callable[[GameData], None] = lambda g: None

    def register_callback(self, callback: Callable[[GameData], None]) -> None:
        self.__callback = callback

    def start(self) -> Result:
        self.__res: Result = Result.WaitingForResult
        while self.__res == Result.WaitingForResult:
            self.__gamedata.state = deepcopy(self.__agent.get_state())
            self.__next_iteration()
            self.__callback(deepcopy(self.__gamedata))
        self.__gamedata = GameData()
        self.__gamedata.state = deepcopy(self.__agent.get_state())
        self.__callback(deepcopy(self.__gamedata))
        return self.__res

    def __next_iteration(self) -> None:
        self.__gamedata.agent_action = self.__agent.move(self.__map_manager)
        if self.__check_for_result():
            return
        self.__gamedata.target_action = self.__target.move(self.__map_manager)
        self.__gamedata.opponent_action = self.__opponent.move(self.__map_manager)
        self.__check_for_result()

    def __check_for_result(self) -> bool:
        if self.__agent.get_pos() == self.__target.get_pos():
            self.__res = Result.Success
            return True
        if self.__agent.get_pos() == self.__opponent.get_pos():
            self.__res = Result.Fail
            return True
        return False

class TrainManager:
    
    def __init__(self, train_settings: TrainSettings) -> None:
        self.__map_manager: MapManager = MapManager(train_settings.map_size, train_settings.obstacles)
        self.__policy_file_path: str = train_settings.policy_file_path
        self.__reward: RewardFunction = train_settings.reward
        self.__next_pos_selector: NextPosSelector = train_settings.agent_next_pos_selector
        self.__iter: int = 0
        self.__max_iter: int = train_settings.max_iter
        map_size: MapSize = self.__map_manager.map_size
        self.__next_states: list[State] = [State() for i in range(Action.MaxExclusive.value)]
        self.__policy = Policy()
        self.__policy.fill(Action.Up, map_size.N3M3)
        self.__changed_actions = map_size.N3M3
        self.__value_function = ValueFunction()
        self.__value_function.fill(0.0, map_size.N3M3)
        self.__max_relative_value_diff: float = 0.0

    def start(self) -> None:
        while (not self.__check_stop_conditions()):
            self.__changed_actions = 0
            self.__max_relative_value_diff = 0.0
            self.__next_iteration()
            self.__iter += 1
            print(f"Iteration {self.__iter}")
            print(f"Changed {self.__changed_actions} actions")
            print(f"Max relative value diff: {self.__max_relative_value_diff}")
        self.__policy.write_to_file(self.__policy_file_path)
        print(f"Convergence after {self.__iter} iterations")

    def __check_stop_conditions(self) -> bool:
        return self.__iter >= self.__max_iter or self.__changed_actions <= 0

    def __next_iteration(self) -> None:
        self.__update_value_function()
        self.__update_policy()

    def __update_value_function(self) -> None:
        state: State = State() # first state is never possible target and opponent overlap
        while not state.next_state(self.__map_manager.map_size):
            if self.__map_manager.is_state_possible(state):
                new_value: float = self.__calculate_new_value_function_value(state, self.__policy.get_action(state, self.__map_manager.map_size))
                old_value: float = self.__value_function.get_value(state, self.__map_manager.map_size)
                if old_value == 0 and new_value != 0:
                        old_value = 1
                if old_value != 0:
                    relative_diff = abs(new_value - old_value) / old_value
                    if relative_diff > self.__max_relative_value_diff:
                        self.__max_relative_value_diff = relative_diff
                self.__value_function.set_value(state, new_value, self.__map_manager.map_size)

    def __calculate_new_value_function_value(self, state: State, chosen_action: Action) -> float:
        actions: list[Action] = [Action(i) for i in range(Action.MaxExclusive.value)]
        for next_state, action in zip(self.__next_states, actions):
            next_state.copy(state)
            self.__map_manager.move_if_possible(next_state.agent_pos, action)
        next_states_values: list[float] = [self.__value_function.get_value(next_state, self.__map_manager.map_size) for next_state in self.__next_states]
        probabilities: list[float] = [self.__next_pos_selector.get_action_prob(chosen_action, action) for action in actions]
        return (self.__reward.calculate_reward(self.__next_states[chosen_action.value], self.__map_manager.map_size) +
                sum([next_state_value * probability for next_state_value, probability in zip(next_states_values, probabilities)]))

    def __update_policy(self) -> None:
        state: State = State()
        while not state.next_state(self.__map_manager.map_size):
            if self.__map_manager.is_state_possible(state):
                new_action: Action= self.__calculate_new_policy_action(state)
                old_action: Action = self.__policy.get_action(state, self.__map_manager.map_size)
                if new_action != old_action:
                    self.__changed_actions += 1
                    self.__policy.set_action(state, new_action, self.__map_manager.map_size)

    def __calculate_new_policy_action(self, state: State) -> Action:
        actions: list[Action] = [Action(i) for i in range(Action.MaxExclusive.value)]
        return max(actions, key=lambda action: self.__calculate_new_value_function_value(state, action))