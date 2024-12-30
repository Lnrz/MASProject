from grid_agent.data_structs.simple_data import Action
from ctypes import c_ubyte, Array
import multiprocessing as mp
from itertools import repeat
from array import array
from abc import ABC

class Policy(ABC):
    """Contain ``Action``s associated to indices of ``State``s."""
    def __init__(self) -> None:
        self._arr: array[int] | Array[c_ubyte]

    def get_action(self, index: int) -> Action:
        """Return the ``Action`` associated with ``index``."""
        return Action(self._arr[index])
    
    def set_action(self, index: int, action: Action) -> None:
        """Associate the ``action`` to ``index``."""
        self._arr[index] = action.value

    def write_to_file(self, policy_file_name: str) -> None:
        """Write the ``Policy`` as a binary file in the path specified by ``policy_file_name``."""
        with open(policy_file_name, "wb") as f:
            f.write(self._arr)

class PolicySequential(Policy):
    """``Policy`` specialized for sequential learning."""
    @classmethod
    def from_action(cls, policy_size: int, action: Action = Action.UP) -> "PolicySequential":
        """Make a ``Policy`` filled with ``action`` of length ``policy_size``."""
        p: PolicySequential = PolicySequential()
        p._arr = array("B", repeat(action.value, policy_size))
        return p

    @classmethod
    def from_file(cls, policy_file_name: str) -> "PolicySequential":
        """Make a ``Policy`` and load into it the ``Action``s stored in the binary file found in the path specified by ``policy_file_name``."""
        p: PolicySequential = PolicySequential()
        p._arr = array("B")
        with open(policy_file_name, "rb") as f:
            policy_size: int = f.seek(0, 2)
            f.seek(0, 0)
            p._arr.fromfile(f, policy_size)
        return p

class PolicyParallel(Policy):
    """``Policy`` specialized for parallel training."""
    @classmethod
    def from_action(cls, policy_size: int, action: Action = Action.UP) -> "PolicyParallel":
        """Make a ``Policy`` filled with ``action`` with length ``policy_size``."""
        p: PolicyParallel = PolicyParallel()
        p._arr = mp.RawArray(c_ubyte, [action.value for _ in range(policy_size)]) 
        return p
