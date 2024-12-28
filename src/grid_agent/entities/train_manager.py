from grid_agent.data_structs.value_functions_container import ValueFunctionsContainer
from grid_agent.functors.markov_transition_density import MarkovTransitionDensity
from grid_agent.data_structs.valid_state_space import ValidStateSpace
from grid_agent.data_structs.simple_data import Action, c_floats
from grid_agent.entities.parallel_train import ProcessSharedData
import grid_agent.entities.parallel_train as parallel_train
from grid_agent.configs.train_configs import TrainConfigs
from grid_agent.functors.reward import RewardFunction
from grid_agent.data_structs.policy import Policy
from grid_agent.data_structs.state import State

from multiprocessing.synchronize import Event, Semaphore
from multiprocessing.context import DefaultContext
from multiprocessing.sharedctypes import RawArray
from multiprocessing import Process
import multiprocessing as mp

from collections.abc import Callable
from dataclasses import dataclass
from copy import copy
import math

@dataclass
class TrainData:
    iteration_number: int = 0
    mean_value: float = 0.0
    max_value_diff: float = 0.0
    changed_actions_number: int = 0
    changed_actions_percentage: float = 0.0

class TrainManager:
    
    def __init__(self, train_configuration: TrainConfigs) -> None:
        train_configuration.validate()
        self.__traindata: TrainData = TrainData(
            iteration_number=0,
            changed_actions_number=train_configuration.valid_state_space.space_size,
            changed_actions_percentage=1.0,
            max_value_diff=math.inf
        )
        self.__policy_file_path = train_configuration.policy_file_path
        self.__max_iter = train_configuration.max_iter
        self.__value_function_tolerance = train_configuration.value_function_tolerance
        self.__changed_actions_tolerance = train_configuration.changed_actions_tolerance
        self.__changed_actions_percentage_tolerance = train_configuration.changed_actions_percentage_tolerance
        self.__callback: Callable[[TrainData], None] = lambda t: None
        self.__processes_number: int = train_configuration.processes_number
        self.__is_dry_run: bool = train_configuration.is_dry_run
        if train_configuration.processes_number == 1:
            self.__init_sequential(train_configuration)
        else:
            self.__init_parallel(train_configuration)

    def __init_sequential(self, train_configuration: TrainConfigs) -> None:
        self.__discount_factor: float = train_configuration.discount_factor
        self.__valid_states_space: ValidStateSpace = train_configuration.valid_state_space
        self.__policy: Policy = train_configuration.policy
        self.__value_functions_container: ValueFunctionsContainer = train_configuration.value_functions_container
        self.__reward: RewardFunction = train_configuration.reward
        self.__markov_transition_density: MarkovTransitionDensity = train_configuration.agent_markov_transition_density
        self.__actions: list[Action] = [Action(i) for i in range(Action.MAX_EXCLUSIVE)]
        self.__actions_probabilities: list[float] = [0.0 for i in self.__actions]
        self.__next_states: list[State] = [State() for i in self.__actions]
        self.__next_states_values: list[float] = [0.0 for i in self.__actions]

    def __init_parallel(self, train_configuration: TrainConfigs) -> None:
        context: DefaultContext = mp.get_context()
        value_type: type[c_floats] = train_configuration.value_functions_container.get_type()
        self.__valid_state_space_size: int = train_configuration.valid_state_space.space_size
        self.__shared_data: ProcessSharedData = ProcessSharedData(
            valid_state_space= train_configuration.valid_state_space,
            policy= train_configuration.policy,
            value_functions_container= train_configuration.value_functions_container,
            reward= train_configuration.reward,
            markov_transition_density= train_configuration.agent_markov_transition_density,
            discount_rate= train_configuration.discount_factor,
            actions= [Action(i) for i in range(Action.MAX_EXCLUSIVE)],
            next_states= [State() for _ in range(Action.MAX_EXCLUSIVE)],
            next_states_values= [0.0 for _ in range (Action.MAX_EXCLUSIVE)],
            probabilities= [0.0 for _ in range(Action.MAX_EXCLUSIVE)],
            policy_event= Event(ctx=context),
            value_event= Event(ctx=context),
            semaphore= Semaphore(value=0, ctx=context),
            max_differences= RawArray(value_type, self.__processes_number),
            partial_values_sums= RawArray(value_type, self.__processes_number),
            partial_changed_actions= RawArray(train_configuration.valid_state_space.type, self.__processes_number)
        )
        intervals: list[tuple[int, int]] = self.__get_processes_intervals()
        self.__processes = [
            Process(
                target=parallel_train.process_main,
                args=[self.__shared_data, index, interval]
            )
            for index, interval in zip(range(self.__processes_number), intervals)            
        ]
        if self.__valid_state_space_size < self.__processes_number:
            print("WARNING: there are more processes than valid states")

    def __get_processes_intervals(self) -> list[tuple[int, int]]:
        states_per_process: int = self.__valid_state_space_size // self.__processes_number
        remainder: int = self.__valid_state_space_size % self.__processes_number
        intervals: list[tuple[int, int]] = []
        last_end: int = 0
        for _ in range(self.__processes_number):
            end: int = last_end + states_per_process
            if remainder > 0:
                end += 1
                remainder -= 1
            intervals.append((last_end, end))
            last_end = end
        return intervals

    def register_callback(self, callback: Callable[[TrainData], None]) -> None:
        self.__callback = callback

    def start(self) -> None:
        if self.__processes_number == 1:
            self.__start_sequential()
        else:
            self.__start_parallel()

    def __start_sequential(self) -> None:
        while (not self.__check_stop_conditions()):
            self.__prepare_traindata()
            print(f"{self.__traindata.iteration_number}-th iteration", end="\r")
            self.__next_iteration()
            self.__callback(copy(self.__traindata))
        print()
        if not self.__is_dry_run:
            self.__policy.write_to_file(self.__policy_file_path)

    def __check_stop_conditions(self) -> bool:
        return (self.__traindata.iteration_number >= self.__max_iter or
                self.__traindata.max_value_diff <= self.__value_function_tolerance or
                self.__traindata.changed_actions_number <= self.__changed_actions_tolerance or
                self.__traindata.changed_actions_percentage <= self.__changed_actions_percentage_tolerance)

    def __prepare_traindata(self) -> None:
        self.__traindata.iteration_number += 1
        self.__traindata.changed_actions_number = 0
        self.__traindata.changed_actions_percentage = 0.0
        self.__traindata.mean_value = 0.0
        self.__traindata.max_value_diff = 0.0

    def __next_iteration(self) -> None:
        self.__update_value_function()
        self.__update_policy()

    def __update_value_function(self) -> None:
        for index, state in enumerate(self.__valid_states_space):
            action: Action = self.__policy.get_action(index)
            new_value: float = self.__calculate_new_value_function_value(state, index, action)
            old_value: float = self.__value_functions_container.get_current_value(index)
            diff: float = abs(new_value - old_value)
            if diff > self.__traindata.max_value_diff:
                self.__traindata.max_value_diff = diff
            self.__traindata.mean_value += new_value
            self.__value_functions_container.set_next_value(index, new_value)
        self.__value_functions_container.swap_value_functions()
        self.__traindata.mean_value /= self.__valid_states_space.space_size
    
    def __update_policy(self) -> None:
        for index, state in zip(range(self.__valid_states_space.space_size -1, -1, -1), reversed(self.__valid_states_space)):
            new_action: Action = max(self.__actions, key=lambda action: self.__mask_actions(state, index, action))
            old_action: Action = self.__policy.get_action(index)
            if new_action != old_action:
                self.__traindata.changed_actions_number += 1
                self.__policy.set_action(index, new_action)
        self.__traindata.changed_actions_percentage = self.__traindata.changed_actions_number / self.__valid_states_space.space_size
    
    def __calculate_new_value_function_value(self, state: State, state_index:int, chosen_action: Action, knows_chosen_action_is_valid: bool = False) -> float:
        for next_state, action in zip(self.__next_states, self.__actions):
            self.__actions_probabilities[action] = self.__markov_transition_density(chosen_action, action)
            if self.__actions_probabilities[action] == 0.0:
                continue
            next_state.copy(state)
            next_state.move_checking_bounds(next_state.agent_pos, action, self.__valid_states_space.map_size)
            if (knows_chosen_action_is_valid and action == chosen_action) or self.__valid_states_space.is_state_outside_obstacles(next_state):
                self.__next_states_values[action] = self.__value_functions_container.get_current_value(self.__valid_states_space.get_valid_index(next_state))
            else:
                self.__next_states_values[action] = self.__value_functions_container.get_current_value(state_index)
        return (self.__reward(state, self.__next_states[chosen_action]) +
                self.__discount_factor * sum([next_state_value * probability for next_state_value, probability in zip(self.__next_states_values, self.__actions_probabilities)]))

    def __mask_actions(self, state: State, state_index: int, action: Action) -> float:
        is_valid: bool = state.move_checking_bounds(state.agent_pos, action, self.__valid_states_space.map_size)
        if not is_valid:
            return -math.inf
        is_valid = self.__valid_states_space.is_state_outside_obstacles(state)
        state.agent_pos.undo(action)
        if not is_valid:
            return -math.inf
        return self.__calculate_new_value_function_value(state, state_index, action, is_valid)
    

    def __start_parallel(self) -> None:
        for process in self.__processes:
            process.start()
        while not self.__check_stop_conditions():
            self.__prepare_traindata()
            print(f"{self.__traindata.iteration_number}-th iteration", end="\r")
            self.__evaluate_policy_parallel()
            self.__improve_policy_parallel()
            self.__callback(copy(self.__traindata))
        print()
        for process in self.__processes:
            process.terminate()
            process.join()
        if not self.__is_dry_run:
            self.__shared_data.policy.write_to_file(self.__policy_file_path)
    
    def __evaluate_policy_parallel(self) -> None:
        self.__shared_data.value_event.set()
        for _ in range(self.__processes_number):
            self.__shared_data.semaphore.acquire()
        self.__shared_data.value_event.clear()
        self.__shared_data.value_functions_container.swap_value_functions()
        self.__traindata.mean_value = sum(self.__shared_data.partial_values_sums) / self.__valid_state_space_size
        self.__traindata.max_value_diff = max(self.__shared_data.max_differences)

    def __improve_policy_parallel(self) -> None:
        self.__shared_data.policy_event.set()
        for _ in range(self.__processes_number):
            self.__shared_data.semaphore.acquire()
        self.__shared_data.policy_event.clear()
        self.__traindata.changed_actions_number = sum(self.__shared_data.partial_changed_actions)
        self.__traindata.changed_actions_percentage = self.__traindata.changed_actions_number / self.__valid_state_space_size