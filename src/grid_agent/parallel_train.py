from grid_agent.data_structs import State, Action, ValidStateSpace, ValueFunctionsContainer, Policy
from grid_agent.functors import RewardFunction, MarkovTransitionDensity

from ctypes import c_ubyte, c_ushort, c_ulong, c_ulonglong, c_float, c_double, Array
from multiprocessing.synchronize import Event, Semaphore
from dataclasses import dataclass
import math

@dataclass
class ProcessSharedData:
    reward: RewardFunction
    markov_transition_density: MarkovTransitionDensity
    discount_rate: float
    actions: list[Action]
    probabilities: list[float]
    next_states: list[State]
    next_states_values: list[float]
    valid_state_space: ValidStateSpace
    value_functions_container: ValueFunctionsContainer
    policy: Policy
    max_differences: Array[c_float | c_double]
    partial_values_sums: Array[c_float | c_double]
    partial_changed_actions: Array[c_ubyte | c_ushort | c_ulong | c_ulonglong]
    value_event: Event
    policy_event: Event
    semaphore: Semaphore



def process_main(shared_data: ProcessSharedData, process_index: int, indices: tuple[int, int]) -> None:
    start_index, end_index = indices
    while True:
        shared_data.value_event.wait()
        evaluate_policy(shared_data, process_index, start_index, end_index)
        shared_data.semaphore.release()
        shared_data.policy_event.wait()
        improve_policy(shared_data, process_index, start_index, end_index)
        shared_data.semaphore.release()

def evaluate_policy(shared_data: ProcessSharedData, process_index:int, start_index: int, end_index: int) -> None:
    state: State = State()
    value_sum: float = 0.0
    max_diff: float = 0.0
    for index in range(start_index, end_index):
        shared_data.valid_state_space.copy_valid_state_to(state, index)
        action: Action = shared_data.policy.get_action(index)
        new_value: float = calculate_new_value_function_value(state, index, action, False, shared_data)
        old_value: float = shared_data.value_functions_container.get_current_value(index)
        diff: float = abs(new_value - old_value)
        if diff > max_diff:
            max_diff = diff
        shared_data.value_functions_container.set_next_value(index, new_value)
        value_sum += new_value
    shared_data.partial_values_sums[process_index] = value_sum
    shared_data.max_differences[process_index] = max_diff

def improve_policy(shared_data: ProcessSharedData, process_index: int, start_index: int, end_index: int) -> None:
    state: State = State()
    changed_actions: int = 0
    for index in range(end_index - 1, start_index - 1, -1):
        shared_data.valid_state_space.copy_valid_state_to(state, index)
        new_action: Action = calculate_new_policy_action(state, index, shared_data)
        old_action: Action = shared_data.policy.get_action(index)
        if new_action != old_action:
            changed_actions += 1
            shared_data.policy.set_action(index, new_action)
    shared_data.partial_changed_actions[process_index] = changed_actions

def calculate_new_value_function_value(state: State, state_index: int, chosen_action: Action, knows_chosen_action_is_valid: bool, shared_data: ProcessSharedData) -> float:
    for next_state, action in zip(shared_data.next_states, shared_data.actions):
        shared_data.probabilities[action] = shared_data.markov_transition_density(action, chosen_action)
        if shared_data.probabilities[action] == 0.0:
            continue
        next_state.copy(state)
        next_state.move_checking_bounds(next_state.agent_pos, action, shared_data.valid_state_space.map_size)
        if (knows_chosen_action_is_valid and action == chosen_action) or shared_data.valid_state_space.is_state_outside_obstacles(next_state):
            shared_data.next_states_values[action] = shared_data.value_functions_container.get_current_value(shared_data.valid_state_space.get_valid_index(next_state))
        else:
            shared_data.next_states_values[action] = shared_data.value_functions_container.get_current_value(state_index)
    return (shared_data.reward(state, shared_data.next_states[chosen_action]) + 
            shared_data.discount_rate * sum([next_state_value * probability for next_state_value, probability in zip(shared_data.next_states_values, shared_data.probabilities)]))

def calculate_new_policy_action(state: State, state_index: int, shared_data: ProcessSharedData) -> Action:
    return max(shared_data.actions, key=lambda action: mask_actions(state, state_index, action, shared_data))

def mask_actions(state: State, state_index: int, action: Action, shared_data: ProcessSharedData) -> float:
    is_valid: bool = state.move_checking_bounds(state.agent_pos, action, shared_data.valid_state_space.map_size)
    if not is_valid:
        return -math.inf
    is_valid = shared_data.valid_state_space.is_state_outside_obstacles(state)
    state.agent_pos.undo(action)
    if not is_valid:
        return -math.inf
    return calculate_new_value_function_value(state, state_index, action, is_valid, shared_data)