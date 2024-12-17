from grid_agent.data_structs import State, Policy, MapSize, Obstacle, Vec2D, Action, Result, ValueFunction, GameData
from grid_agent.functors import ActionSelector, NextPosSelector, RewardFunction
from grid_agent.settings import GameSettings, TrainSettings
from typing import Callable
from array import array
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
        self.__actions: list[Action] = [Action(i) for i in range(Action.MaxExclusive.value)]
        self.__actions_probabilities: list[float] = [0.0 for i in self.__actions]
        self.__next_states: list[State] = [State() for i in self.__actions]
        self.__next_states_values: list[float] = [0.0 for i in self.__actions]
        self.__possible_states_indices: array = array("Q")
        self.__possible_states_num: int = 0
        self.__state: State = State()
        while (not self.__state.next_state(self.__map_manager.map_size)):
            if self.__map_manager.is_state_possible(self.__state):
                self.__possible_states_indices.append(self.__state.to_index(self.__map_manager.map_size))
                self.__possible_states_num += 1
        self.__policy: Policy = Policy()
        self.__policy.fill(Action.Up, map_size.N3M3)
        self.__changed_actions: int = map_size.N3M3
        self.__percentage_of_changed_actions: float = 1.0
        self.__value_function = ValueFunction()
        self.__value_function.fill(0.0, map_size.N3M3)
        self.__mean_value: float = 0.0

    def start(self) -> None:
        while (not self.__check_stop_conditions()):
            self.__changed_actions = 0
            self.__mean_value = 0.0
            self.__next_iteration()
            self.__iter += 1
            print(f"Iteration {self.__iter}")
            print(f"Changed {self.__changed_actions} actions")
            print(f"Changed {self.__percentage_of_changed_actions} of actions")
            print(f"Mean value: {self.__mean_value}")
        self.__policy.write_to_file(self.__policy_file_path)
        print(f"Convergence after {self.__iter} iterations")

    def __check_stop_conditions(self) -> bool:
        return self.__iter >= self.__max_iter or self.__changed_actions <= 0

    def __next_iteration(self) -> None:
        self.__update_value_function()
        self.__update_policy()

    def __update_value_function(self) -> None:
        j: int = 0
        while j < self.__possible_states_num:
            self.__state.from_index(self.__possible_states_indices[j], self.__map_manager.map_size)
            new_value: float = self.__calculate_new_value_function_value(self.__state, self.__policy.get_action(self.__state, self.__map_manager.map_size))
            self.__mean_value += new_value
            self.__value_function.set_value(self.__state, new_value, self.__map_manager.map_size)
            j += 1
        self.__mean_value /= self.__possible_states_num

    def __calculate_new_value_function_value(self, state: State, chosen_action: Action) -> float:
        for next_state, action in zip(self.__next_states, self.__actions):
            next_state.copy(state)
            next_state.agent_pos.move(action)
            if not self.__binary_search_index(next_state.to_index(self.__map_manager.map_size)):
                next_state.copy(state)
            self.__next_states_values[action.value] = self.__value_function.get_value(next_state, self.__map_manager.map_size)
            self.__actions_probabilities[action.value] = self.__next_pos_selector.get_action_prob(chosen_action, action)
        return (self.__reward.calculate_reward(self.__next_states[chosen_action.value], self.__map_manager.map_size) +
                sum([next_state_value * probability for next_state_value, probability in zip(self.__next_states_values, self.__actions_probabilities)]))

    def __binary_search_index(self, state_idx: int) -> bool:
        i: int = 0
        j: int = self.__possible_states_num - 1
        while i < j:
            k: int = (i + j) // 2
            idx: int = self.__possible_states_indices[k]
            if idx == state_idx:
                return True
            if idx < state_idx:
                i = k + 1
            else:
                j = k - 1

    def __update_policy(self) -> None:
        j: int = 0
        while j < self.__possible_states_num:
            self.__state.from_index(self.__possible_states_indices[j], self.__map_manager.map_size)
            new_action: Action= self.__calculate_new_policy_action(self.__state)
            old_action: Action = self.__policy.get_action(self.__state, self.__map_manager.map_size)
            if new_action != old_action:
                self.__changed_actions += 1
                self.__policy.set_action(self.__state, new_action, self.__map_manager.map_size)
            j += 1
        self.__percentage_of_changed_actions = self.__changed_actions / self.__possible_states_num

    def __calculate_new_policy_action(self, state: State) -> Action:
        return max(self.__actions, key=lambda action: self.__calculate_new_value_function_value(state, action))