from grid_agent.data_structs.simple_data import Action
from ctypes import c_ubyte, Array
import multiprocessing as mp
from itertools import repeat
from array import array
from abc import ABC

class Policy(ABC):
    
    def __init__(self) -> None:
        self._arr: array[int] | Array[c_ubyte]

    def get_action(self, index: int) -> Action:
        return Action(self._arr[index])
    
    def set_action(self, index: int, action: Action) -> None:
        self._arr[index] = action.value

    def write_to_file(self, policy_file_name: str) -> None:
        with open(policy_file_name, "wb") as f:
            f.write(self._arr)

class PolicySequential(Policy):

    @classmethod
    def from_action(cls, policy_size: int, action: Action = Action.UP) -> "PolicySequential":
        p: PolicySequential = PolicySequential()
        p._arr = array("B", repeat(action.value, policy_size))
        return p

    @classmethod
    def from_file(cls, policy_file_name: str) -> "PolicySequential":
        p: PolicySequential = PolicySequential()
        p._arr = array("B")
        with open(policy_file_name, "rb") as f:
            policy_size: int = f.seek(0, 2)
            f.seek(0, 0)
            p._arr.fromfile(f, policy_size)
        return p

class PolicyParallel(Policy):

    @classmethod
    def from_action(cls, policy_size: int, action: Action = Action.UP) -> "PolicyParallel":
        p: PolicyParallel = PolicyParallel()
        p._arr = mp.RawArray(c_ubyte, [action.value for _ in range(policy_size)]) 
        return p
