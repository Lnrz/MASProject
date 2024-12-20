from grid_agent.data_structs import State, Policy, Vec2D, Action, Result, ValueFunctionsContainer, GameData, TrainData, ValidStateSpace
from grid_agent.functors import PolicyFun, MarkovTransitionDensity, RewardFunction, AgentPolicy
from grid_agent.settings import GameSettings, TrainSettings
from typing import Callable
from copy import copy, deepcopy
import random as rnd



class MovingEntity:
    
    def __init__(self, start_pos: Vec2D, policy: PolicyFun, markov_transition_density: MarkovTransitionDensity) -> None:
        self.__pos: Vec2D = start_pos
        self.__policy: PolicyFun = policy
        self.__markov_transition_density: MarkovTransitionDensity = markov_transition_density

    def move(self, state: State, valid_state_space: ValidStateSpace) -> Action:
        chosen_action: Action = self.__policy(state)
        actual_action: Action = self.__get_next_action(chosen_action)
        self.__pos.move(actual_action)
        if not valid_state_space.is_valid(state):
            self.__pos.undo(actual_action)
        return chosen_action
    
    def __get_next_action(self, chosen_action: Action) -> Action:
        actions: list[Action] = [Action(i) for i in range(Action.MAX_EXCLUSIVE.value)]
        probabilities: list[float] = [self.__markov_transition_density(chosen_action, action) for action in actions]
        return rnd.choices(actions, probabilities)[0]



class GameManager:
    
    def __init__(self, game_settings: GameSettings) -> None:
        self.__valid_state_space: ValidStateSpace = ValidStateSpace(game_settings.map_size, game_settings.obstacles)
        self.__state: State = State(copy(game_settings.agent_start_pos), copy(game_settings.opponent_start_pos), copy(game_settings.target_start_pos))
        self.__agent: MovingEntity = MovingEntity(self.__state.agent_pos, AgentPolicy(game_settings.policy_file_path, self.__valid_state_space), game_settings.agent_markov_transition_density)
        self.__target: MovingEntity = MovingEntity(self.__state.target_pos, game_settings.target_action_selector, game_settings.target_markov_transition_density)
        self.__opponent: MovingEntity = MovingEntity(self.__state.opponent_pos, game_settings.opponent_action_selector, game_settings.opponent_markov_transition_density)
        self.__res: Result = Result.WAITING_FOR_RESULT
        self.__gamedata: GameData = GameData()
        self.__callback: Callable[[GameData], None] = lambda g: None

    def register_callback(self, callback: Callable[[GameData], None]) -> None:
        self.__callback = callback

    def start(self) -> Result:
        while self.__res == Result.WAITING_FOR_RESULT:
            self.__gamedata.state = deepcopy(self.__state)
            self.__next_iteration()
            self.__callback(deepcopy(self.__gamedata))
        self.__gamedata = GameData()
        self.__gamedata.state = deepcopy(self.__state)
        self.__callback(deepcopy(self.__gamedata))
        return self.__res

    def __next_iteration(self) -> None:
        self.__gamedata.agent_action = self.__agent.move(self.__state, self.__valid_state_space)
        if self.__check_for_result():
            return
        self.__gamedata.target_action = self.__target.move(self.__state, self.__valid_state_space)
        self.__gamedata.opponent_action = self.__opponent.move(self.__state, self.__valid_state_space)
        self.__check_for_result()

    def __check_for_result(self) -> bool:
        match self.__state.agent_pos:
            case self.__state.target_pos:
                self.__res = Result.SUCCESS
                return True
            case self.__state.opponent_pos:
                self.__res = Result.FAIL
                return True
            case _:
                return False



class TrainManager:
    
    def __init__(self, train_settings: TrainSettings) -> None:
        self.__policy_file_path: str = train_settings.policy_file_path
        self.__reward: RewardFunction = train_settings.reward
        self.__markov_transition_density: MarkovTransitionDensity = train_settings.agent_markov_transition_density
        self.__traindata: TrainData = TrainData(changed_actions_number=1)
        self.__callback: Callable[[TrainData], None] = lambda t: None
        self.__max_iter: int = train_settings.max_iter
        self.__actions: list[Action] = [Action(i) for i in range(Action.MAX_EXCLUSIVE.value)]
        self.__actions_probabilities: list[float] = [0.0 for i in self.__actions]
        self.__next_states: list[State] = [State() for i in self.__actions]
        self.__next_states_values: list[float] = [0.0 for i in self.__actions]
        self.__valid_states_space: ValidStateSpace = ValidStateSpace(train_settings.map_size, train_settings.obstacles)
        self.__policy: Policy = Policy.from_action(self.__valid_states_space.space_size)
        self.__value_functions_container: ValueFunctionsContainer = ValueFunctionsContainer(self.__valid_states_space.space_size)
        self.__discount_factor: float = 0.55

    def register_callback(self, callback: Callable[[TrainData], None]) -> None:
        self.__callback = callback

    def start(self) -> None:
        while (not self.__check_stop_conditions()):
            self.__prepare_traindata()
            print(f"{self.__traindata.iteration_number}-th iteration", end="\r")
            self.__next_iteration()
            self.__callback(copy(self.__traindata))
        print()
        self.__policy.write_to_file(self.__policy_file_path)

    def __check_stop_conditions(self) -> bool:
        return (self.__traindata.iteration_number >= self.__max_iter or
                self.__traindata.changed_actions_number == 0)

    def __prepare_traindata(self) -> None:
        self.__traindata.iteration_number += 1
        self.__traindata.changed_actions_number = 0
        self.__traindata.mean_value = 0

    def __next_iteration(self) -> None:
        self.__update_value_function()
        self.__update_policy()

    def __update_value_function(self) -> None:
        for (index, state) in self.__valid_states_space:
            action: Action = self.__policy.get_action(index)
            new_value: float = self.__calculate_new_value_function_value(state, action)
            self.__traindata.mean_value += new_value
            self.__value_functions_container.set_next_value(index, new_value)
        self.__value_functions_container.swap_value_functions()
        self.__traindata.mean_value /= self.__valid_states_space.space_size

    def __calculate_new_value_function_value(self, state: State, chosen_action: Action) -> float:
        for next_state, action in zip(self.__next_states, self.__actions):
            next_state.copy(state)
            next_state.agent_pos.move(action)
            if not self.__valid_states_space.is_valid(next_state):
                next_state.agent_pos.undo(action)
            self.__next_states_values[action.value] = self.__value_functions_container.get_current_value(self.__valid_states_space.get_index(next_state))
            self.__actions_probabilities[action.value] = self.__markov_transition_density(chosen_action, action)
        return (self.__reward(state, self.__next_states[chosen_action.value]) +
                self.__discount_factor * sum([next_state_value * probability for next_state_value, probability in zip(self.__next_states_values, self.__actions_probabilities)]))
    
    def __update_policy(self) -> None:
        for (index, state) in self.__valid_states_space:
            new_action: Action = self.__calculate_new_policy_action(state)
            old_action: Action = self.__policy.get_action(index)
            if new_action != old_action:
                self.__traindata.changed_actions_number += 1
                self.__policy.set_action(index, new_action)
        self.__traindata.changed_actions_percentage = self.__traindata.changed_actions_number / self.__valid_states_space.space_size

    def __calculate_new_policy_action(self, state: State) -> Action:
        return max(self.__actions, key=lambda action: self.__mask_actions(state, action))
    
    def __mask_actions(self, state: State, action: Action) -> float:
        state.agent_pos.move(action)
        is_valid: bool = self.__valid_states_space.is_valid(state)
        state.agent_pos.undo(action)
        if not is_valid:
            return float("-inf")
        return self.__calculate_new_value_function_value(state, action)