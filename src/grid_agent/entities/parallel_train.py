from grid_agent.data_structs.value_functions_container import ValueFunctionsContainer
from grid_agent.functors.markov_transition_density import MarkovTransitionDensity
from grid_agent.data_structs.simple_data import Action, c_floats, c_uints
from grid_agent.data_structs.valid_state_space import ValidStateSpace
from grid_agent.functors.reward import RewardFunction
from grid_agent.data_structs.policy import Policy
from grid_agent.data_structs.state import State

from multiprocessing.synchronize import Event, Semaphore
from dataclasses import dataclass
from ctypes import Array
import math

@dataclass
class ProcessSharedData:
    """Struct containing all the data a process need to work.
    
    It contains:
    - ``reward``: the ``RewardFunction`` the process will use for learning.
    - ``markov_transition_density``: the ``MarkovTransitionDensity`` the process will use for learning.
    - ``discount_rate``: the discount rate the process will use for learning.
    - ``actions``: a list of all ``Action``s.
    - ``probabilities``: a list on which the process will compute the probabilities of ``actions``.
    - ``next_states``: a list on which the process will compute the next ``State``s.
    - ``next_states_values``: a list on which the process will compute the values of ``next_states``.
    - ``value_functions_container``: the ``ValueFunctionsContainer`` on which the process will work.
    - ``policy``: the ``Policy`` on which the process will work.
    - ``max_differences``: a shared ``Array`` on which the process will put the maximum change of value of its valid ``State``s.
    - ``partial_values_sums``: a shared ``Array`` on which the process will put the sum of the values of its valid ``State``s. 
    - ``partial_changed_actions``: a shared ``Array`` on which the process will put the number of ``Action``s it has changed for its valid ``State``s.
    - ``value_event``:  an ``Event`` on which the process will wait before executing the policy evaluation step.
    - ``policy_event``: an ``Event`` on which the process will wait before executing the policy improvement step.
    - ``semaphore``: a ``Semaphore`` the process will signal after a step of the policy iteration algorithm. 
    """
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
    max_differences: Array[c_floats]
    partial_values_sums: Array[c_floats]
    partial_changed_actions: Array[c_uints]
    value_event: Event
    policy_event: Event
    semaphore: Semaphore



def process_main(shared_data: ProcessSharedData, process_index: int, indices: tuple[int, int]) -> None:
    """Starting function of a process.
    
    ``shared_data``:
      a ``ProcessSharedData`` struct containing all the data the process need to work.
    ``process_index``:
      the index of the process.
    ``indices``:
      the interval of valid states on which the process wil work.
    """
    start_index, end_index = indices
    while True:
        shared_data.value_event.wait()
        evaluate_policy(shared_data, process_index, start_index, end_index)
        shared_data.semaphore.release()
        shared_data.policy_event.wait()
        improve_policy(shared_data, process_index, start_index, end_index)
        shared_data.semaphore.release()

def evaluate_policy(shared_data: ProcessSharedData, process_index:int, start_index: int, end_index: int) -> None:
    """Policy evaluation step of a process."""
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
    """Policy improvement step of a process."""
    state: State = State()
    changed_actions: int = 0
    for index in range(end_index - 1, start_index - 1, -1):
        shared_data.valid_state_space.copy_valid_state_to(state, index)
        new_action: Action = max(shared_data.actions, key=lambda action: mask_actions(state, index, action, shared_data))
        old_action: Action = shared_data.policy.get_action(index)
        if new_action != old_action:
            changed_actions += 1
            shared_data.policy.set_action(index, new_action)
    shared_data.partial_changed_actions[process_index] = changed_actions

def calculate_new_value_function_value(state: State, state_index: int, chosen_action: Action, knows_chosen_action_is_valid: bool, shared_data: ProcessSharedData) -> float:
    """Return the value of ``state``, of index ``state_index``, given that the current ``Policy`` returned ``chosen_action``.
    
    ``knows_chosen_action_is_valid`` is a ``bool`` used to speed up the processing if the user already knows that ``chosen_action`` brings to a valid ``State``.
    """
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

def mask_actions(state: State, state_index: int, action: Action, shared_data: ProcessSharedData) -> float:
    """Return the value of performing ``action`` when the state is ``state``, of index ``state_index``.
        
    If ``action`` brings to an invalid ``State`` return ``-inf``.
    """
    is_valid: bool = state.move_checking_bounds(state.agent_pos, action, shared_data.valid_state_space.map_size)
    if not is_valid:
        return -math.inf
    is_valid = shared_data.valid_state_space.is_state_outside_obstacles(state)
    state.agent_pos.undo(action)
    if not is_valid:
        return -math.inf
    return calculate_new_value_function_value(state, state_index, action, is_valid, shared_data)