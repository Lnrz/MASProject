from grid_agent.data_structs.valid_state_space import ValidStateSpace
from grid_agent.data_structs.simple_data import Action
from grid_agent.data_structs.policy import Policy
from grid_agent.data_structs.state import State
from abc import ABC, abstractmethod
from typing import override
import random as rnd

class PolicyFun(ABC):
    """Interface for the policy functors."""

    @abstractmethod
    def __call__(self, state: State) -> Action:
        """Return an ``Action`` based on ``state``."""
        ...

class UniformPolicy(PolicyFun):
    """``PolicyFun`` giving to each ``Action`` the same probability of being chosen."""

    @override
    def __call__(self, state: State) -> Action:
        return Action(rnd.randrange(Action.MAX_EXCLUSIVE))

class AgentPolicy(PolicyFun):
    """``PolicyFun`` that returns ``Action``s based on the given ``Policy`` and ``ValidStateSpace``."""

    def __init__(self, policy: Policy, valid_state_space: ValidStateSpace) -> None:
        self.__policy: Policy = policy
        self.__valid_state_space: ValidStateSpace = valid_state_space

    @override
    def __call__(self, state: State) -> Action:
        return self.__policy.get_action(self.__valid_state_space.get_valid_index(state))