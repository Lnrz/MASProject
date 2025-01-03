from grid_agent.data_structs.simple_data import c_floats, c_float_types
from ctypes import c_bool, c_float, c_double, Array
from typing import override
from abc import ABC, abstractmethod
import multiprocessing as mp
from itertools import repeat
from array import array

class ValueFunctionsContainer(ABC):
    """A container storing the value functions of the current and next policies."""
    @abstractmethod
    def get_type(self) -> c_float_types:
        """Return the type of the values stored in the value functions."""
        ...

    @abstractmethod
    def get_current_value(self, index: int) -> float:
        """Return the value the current ``Policy`` associates with ``index``."""
        ...

    @abstractmethod
    def set_next_value(self, index: int, value: float) -> None:
        """Set the value the next ``Policy`` will associate with ``index``."""
        ...

    @abstractmethod
    def swap_value_functions(self) -> None:
        """Swap the value functions in order to write on the previous one the values of the next."""
        ...

class ValueFunctionsContainerSequential(ValueFunctionsContainer):
    """``ValueFunctionsContainer`` specialized for sequential learning."""
    def __init__(self, size: int, start_value: float = 0.0, use_double: bool = True) -> None:
        """Initialize the value functions with ``start_value`` and with the specified ``size`` .
        
        If ``use_double`` is ``True`` the values will be stored as ``double``, otherwise as ``float``.
        """
        self.__type: c_float_types = c_double if use_double else c_float
        type_char: str = "d" if use_double else "f"
        self.__old_values: array[float] = array(type_char, repeat(start_value, size))
        self.__new_values: array[float] = array(type_char, repeat(start_value, size))

    @override
    def get_type(self) -> c_float_types:
        return self.__type

    @override
    def get_current_value(self, index: int) -> float:
        return self.__old_values[index]

    @override
    def set_next_value(self, index: int, value: float) -> None:
        self.__new_values[index] = value

    @override
    def swap_value_functions(self) -> None:
        self.__old_values, self.__new_values = self.__new_values, self.__old_values

class ValueFunctionsContainerParallel(ValueFunctionsContainer):
    """``ValueFunctionsContainer`` specialized for parallel learning."""
    def __init__(self, size: int, start_value: float = 0.0, use_double: bool = True) -> None:
        """Initialize the value functions with ``start_value`` and with the specified ``size`` .
        
        If ``use_double`` is ``True`` the values will be stored as ``double``, otherwise as ``float``.
        """
        self.__type: c_float_types = c_double if use_double else c_float
        values: list[float] = [value for value in repeat(start_value, size)]
        self.__array_a: Array[c_floats] = mp.RawArray(self.__type, values)
        self.__array_b: Array[c_floats] = mp.RawArray(self.__type, values)
        self.__swapped: c_bool = mp.RawValue(c_bool, False)

    @override
    def get_type(self) -> c_float_types:
        return self.__type

    @override
    def get_current_value(self, index: int) -> float:
        if not self.__swapped.value:
            return self.__array_a[index]
        else:
            return self.__array_b[index]
        
    @override
    def set_next_value(self, index: int, value: float) -> None:
        if not self.__swapped.value:
            self.__array_b[index] = value
        else:
            self.__array_a[index] = value
    
    @override
    def swap_value_functions(self) -> None:
        self.__swapped.value = not self.__swapped.value