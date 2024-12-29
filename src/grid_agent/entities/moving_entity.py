from grid_agent.functors.markov_transition_density import MarkovTransitionDensity
from grid_agent.data_structs.valid_state_space import ValidStateSpace
from grid_agent.data_structs.simple_data import Vec2D, Action
from grid_agent.functors.policy import PolicyFun
from grid_agent.data_structs.state import State

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
        if not valid_state_space.is_state_within_bounds(state) or not valid_state_space.is_state_outside_obstacles(state):
            self.__pos.undo(actual_action)
        return chosen_action
    
    def __get_next_action(self, chosen_action: Action) -> Action:
        actions: list[Action] = [Action(i) for i in range(Action.MAX_EXCLUSIVE)]
        probabilities: list[float] = [self.__markov_transition_density(chosen_action, action) for action in actions]
        return rnd.choices(actions, probabilities)[0]