from grid_agent.data_structs import State, Policy, MapSize, Obstacle, Vec2D, Action, Result, ValueFunction, GameData, TrainData
from grid_agent.functors import ActionSelector, MarkovTransitionDensity, RewardFunction
from grid_agent.settings import GameSettings, TrainSettings
from typing import Callable
from array import array
from copy import copy, deepcopy
import random as rnd

class MapManager:
    
    def __init__(self, map_size: Vec2D, obstacles: list[Obstacle]) -> None:
        self.map_size: MapSize = MapSize(map_size.x, map_size.y)
        self.__obstacles: list[Obstacle] = obstacles
        self.__pos : Vec2D = Vec2D()

    def move_if_possible(self, pos: Vec2D, action: Action) -> bool:
        self.__pos.copy(pos)
        self.__pos.move(action)
        if self.is_pos_possible(self.__pos):
            pos.copy(self.__pos)
            return True
        return False

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
    
    def __init__(self, start_pos: Vec2D, target_pos: Vec2D, opponent_pos: Vec2D, markov_transition_density: MarkovTransitionDensity, policy_file_name: str | None, map_size: MapSize) -> None:
        self.__state: State = State(agent_pos=copy(start_pos), opponent_pos=opponent_pos, target_pos=target_pos)
        self.__policy: Policy = Policy()
        if policy_file_name:
            self.__policy.read_from_file(policy_file_name, map_size.N3M3)
        else:
            print("WARNING: The policy file was not provided, using default policy")
            self.__policy.fill(Action.UP, map_size.N3M3)
        self.__markov_transition_density: MarkovTransitionDensity = markov_transition_density

    def get_state(self) -> State:
        return self.__state

    def get_pos(self) -> Vec2D:
        return self.__state.agent_pos

    def move(self, map_manager: MapManager) -> Action:
        chosen_action: Action = self.__policy.get_action(self.__state, map_manager.map_size)
        actual_action: Action = self.__get_next_action(chosen_action)
        map_manager.move_if_possible(self.__state.agent_pos, actual_action)
        return chosen_action
    
    def __get_next_action(self, chosen_action: Action) -> Action:
        actions: list[Action] = [Action(i) for i in range(Action.MAX_EXCLUSIVE.value)]
        probabilities: list[float] = [self.__markov_transition_density(chosen_action, action) for action in actions]
        return rnd.choices(actions, probabilities)[0]

class MovingEntity:
    
    def __init__(self, start_pos: Vec2D, action_selector: ActionSelector, markov_transition_density: MarkovTransitionDensity) -> None:
        self.__pos: Vec2D = copy(start_pos)
        self.__action_selector: ActionSelector = action_selector
        self.__markov_transition_density: MarkovTransitionDensity = markov_transition_density
        self.__banned_pos: Vec2D = Vec2D(-1, -1)

    def get_pos(self) -> Vec2D:
        return self.__pos

    def set_banned_pos(self, banned_pos: Vec2D) -> None:
        self.__banned_pos = banned_pos

    def move(self, map_manager: MapManager) -> Action:
        chosen_action: Action = self.__action_selector()
        actual_action: Action = self.__get_next_action(chosen_action)
        map_manager.move_if_possible(self.__pos, actual_action)
        if self.__pos == self.__banned_pos:
            self.__pos.undo(actual_action)
        return chosen_action
    
    def __get_next_action(self, chosen_action: Action) -> Action:
        actions: list[Action] = [Action(i) for i in range(Action.MAX_EXCLUSIVE.value)]
        probabilities: list[float] = [self.__markov_transition_density(chosen_action, action) for action in actions]
        return rnd.choices(actions, probabilities)[0]



class GameManager:
    
    def __init__(self, game_settings: GameSettings) -> None:
        self.__map_manager: MapManager = MapManager(game_settings.map_size, game_settings.obstacles)
        self.__target: MovingEntity = MovingEntity(game_settings.target_start_pos, game_settings.target_action_selector, game_settings.target_markov_transition_density)
        self.__opponent: MovingEntity = MovingEntity(game_settings.opponent_start_pos, game_settings.opponent_action_selector, game_settings.opponent_markov_transition_density)
        self.__target.set_banned_pos(self.__opponent.get_pos())
        self.__opponent.set_banned_pos(self.__target.get_pos())
        self.__agent: Agent = Agent(game_settings.agent_start_pos, self.__target.get_pos(), self.__opponent.get_pos(), game_settings.agent_markov_transition_density, game_settings.policy_file_path, self.__map_manager.map_size)
        self.__gamedata: GameData = GameData()
        self.__callback: Callable[[GameData], None] = lambda g: None

    def register_callback(self, callback: Callable[[GameData], None]) -> None:
        self.__callback = callback

    def start(self) -> Result:
        self.__res: Result = Result.WAITING_FOR_RESULT
        while self.__res == Result.WAITING_FOR_RESULT:
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
            self.__res = Result.SUCCESS
            return True
        if self.__agent.get_pos() == self.__opponent.get_pos():
            self.__res = Result.FAIL
            return True
        return False



class TrainManager:
    
    def __init__(self, train_settings: TrainSettings) -> None:
        self.__map_manager: MapManager = MapManager(train_settings.map_size, train_settings.obstacles)
        self.__policy_file_path: str = train_settings.policy_file_path
        self.__reward: RewardFunction = train_settings.reward
        self.__markov_transition_density: MarkovTransitionDensity = train_settings.agent_markov_transition_density
        self.__traindata: TrainData = TrainData()
        self.__traindata.changed_actions_number = 1
        self.__callback: Callable[[TrainData], None] = lambda t: None
        self.__max_iter: int = train_settings.max_iter
        map_size: MapSize = self.__map_manager.map_size
        self.__actions: list[Action] = [Action(i) for i in range(Action.MAX_EXCLUSIVE.value)]
        self.__actions_probabilities: list[float] = [0.0 for i in self.__actions]
        self.__next_states: list[State] = [State() for i in self.__actions]
        self.__next_states_values: list[float] = [0.0 for i in self.__actions]
        self.__possible_states_indices: array = array("Q")
        self.__possible_states_num: int = 0
        self.__state: State = State()
        while (not self.__state.next_state(map_size)):
            if self.__map_manager.is_state_possible(self.__state):
                self.__possible_states_indices.append(self.__state.to_index(map_size))
                self.__possible_states_num += 1
        self.__policy: Policy = Policy()
        self.__policy.fill(Action.UP, map_size.N3M3)
        self.__value_function = ValueFunction()
        self.__value_function.fill(0.0, map_size.N3M3)
        self.__discount_factor: float = 0.5

    def register_callback(self, callback: Callable[[TrainData], None]) -> None:
        self.__callback = callback

    def start(self) -> None:
        while (not self.__check_stop_conditions()):
            self.__prepare_traindata()
            print(f"{self.__traindata.iteration_number}-th iteration")
            self.__next_iteration()
            self.__callback(copy(self.__traindata))
        self.__policy.write_to_file(self.__policy_file_path)

    def __check_stop_conditions(self) -> bool:
        return (self.__traindata.iteration_number >= self.__max_iter or
                self.__traindata.changed_actions_number <= 0)

    def __prepare_traindata(self) -> None:
        self.__traindata.iteration_number += 1
        self.__traindata.changed_actions_number = 0
        self.__traindata.mean_value = 0

    def __next_iteration(self) -> None:
        self.__update_value_function()
        self.__update_policy()

    def __update_value_function(self) -> None:
        j: int = 0
        while j < self.__possible_states_num:
            self.__state.from_index(self.__possible_states_indices[j], self.__map_manager.map_size)
            new_value: float = self.__calculate_new_value_function_value(self.__state, self.__policy.get_action(self.__state, self.__map_manager.map_size))
            self.__traindata.mean_value += new_value
            self.__value_function.set_value(self.__state, new_value, self.__map_manager.map_size)
            j += 1
        self.__traindata.mean_value /= self.__possible_states_num

    def __calculate_new_value_function_value(self, state: State, chosen_action: Action) -> float:
        for next_state, action in zip(self.__next_states, self.__actions):
            next_state.copy(state)
            next_state.agent_pos.move(action)
            if not self.__binary_search_index(next_state.to_index(self.__map_manager.map_size)):
                next_state.copy(state)
            self.__next_states_values[action.value] = self.__value_function.get_value(next_state, self.__map_manager.map_size)
            self.__actions_probabilities[action.value] = self.__markov_transition_density(chosen_action, action)
        return (self.__reward(state, self.__next_states[chosen_action.value]) +
                self.__discount_factor * sum([next_state_value * probability for next_state_value, probability in zip(self.__next_states_values, self.__actions_probabilities)]))

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
        return False

    def __update_policy(self) -> None:
        j: int = 0
        while j < self.__possible_states_num:
            self.__state.from_index(self.__possible_states_indices[j], self.__map_manager.map_size)
            new_action: Action= self.__calculate_new_policy_action(self.__state)
            old_action: Action = self.__policy.get_action(self.__state, self.__map_manager.map_size)
            if new_action != old_action:
                self.__traindata.changed_actions_number += 1
                self.__policy.set_action(self.__state, new_action, self.__map_manager.map_size)
            j += 1
        self.__traindata.changed_actions_percentage = self.__traindata.changed_actions_number / self.__possible_states_num

    def __calculate_new_policy_action(self, state: State) -> Action:
        return max(self.__actions, key=lambda action: self.__calculate_new_value_function_value(state, action))