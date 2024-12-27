from collections.abc import Iterable, Sequence
from  collections import OrderedDict

from ctypes import c_bool, c_ubyte, c_ushort, c_ulong, c_ulonglong, c_float, c_double, Array
from typing import Self, Protocol, overload, override
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import multiprocessing as mp
from itertools import repeat
from enum import IntEnum
from array import array

type c_uints = c_ubyte | c_ushort | c_ulong | c_ulonglong
type c_uint_types = type[c_uints]
type c_floats = c_float | c_double
type c_float_types = type[c_floats]

class Result(IntEnum):
    FAIL = 0,
    SUCCESS = 1,
    WAITING_FOR_RESULT = 2

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    MAX_EXCLUSIVE = LEFT + 1

@dataclass
class Vec2D:
    x : int = 0
    y : int = 0

    def copy(self, oth: Self) -> None:
        self.x = oth.x
        self.y = oth.y

    def move(self, action: Action) -> None:
        match action:
            case Action.UP:
                self.y += 1
            case Action.RIGHT:
                self.x += 1
            case Action.DOWN:
                self.y -= 1
            case Action.LEFT:
                self.x -= 1
    
    def undo(self, action: Action) -> None:
        match action:
            case Action.UP:
                self.y -= 1
            case Action.RIGHT:
                self.x -= 1
            case Action.DOWN:
                self.y += 1
            case Action.LEFT:
                self.x += 1

@dataclass
class Obstacle:
    origin : Vec2D = field(default_factory=lambda: Vec2D())
    extent : Vec2D = field(default_factory=lambda: Vec2D())

    def is_inside(self, pos: Vec2D) -> bool:
        return (pos.x >= self.origin.x and pos.x <= self.origin.x + self.extent.x - 1 and
                pos.y <= self.origin.y and pos.y >= self.origin.y - self.extent.y + 1)
    
    def to_pos(self) -> list[Vec2D]:
        return [Vec2D(x, y)
                for x in range(self.origin.x, self.origin.x + self.extent.x)
                for y in range(self.origin.y - self.extent.y + 1, self.origin.y + 1)]

@dataclass
class MapSize:
    N: int
    N2: int
    N3: int
    M: int
    M2: int
    M3: int
    NM: int
    N2M: int
    N2M2: int
    N3M2: int
    N3M3: int

    def __init__(self, N: int, M: int) -> None:
        self.N = N
        self.N2 = N * N
        self.N3 = self.N2 * N
        self.M = M
        self.M2 = M * M
        self.M2 = self.M2 * M
        self.NM = N * M
        self.N2M = self.NM * N
        self.N2M2 = self.N2M * M
        self.N3M2 = self.N2M2 * N
        self.N3M3 = self.N3M2 * M

@dataclass
class State:
    agent_pos : Vec2D = field(default_factory=lambda: Vec2D())
    opponent_pos : Vec2D = field(default_factory=lambda: Vec2D())
    target_pos : Vec2D = field(default_factory=lambda: Vec2D())

    def to_index(self, map_size: MapSize) -> int:
        return (self.agent_pos.x + self.agent_pos.y * map_size.N +
                self.opponent_pos.x * map_size.NM + self.opponent_pos.y * map_size.N2M +
                self.target_pos.x *map_size.N2M2 + self.target_pos.y * map_size.N3M2)

    def from_index(self, index: int, map_size: MapSize) -> None:
        self.agent_pos.x = index % map_size.N
        self.agent_pos.y = (index % map_size.NM) // map_size.N
        self.opponent_pos.x = (index % map_size.N2M) // map_size.NM
        self.opponent_pos.y = (index % map_size.N2M2) // map_size.N2M
        self.target_pos.x = (index % map_size.N3M2) // map_size.N2M2
        self.target_pos.y = index // map_size.N3M2

    def next_state(self, map_size: MapSize) -> bool:
        if self.__next_pos(self.agent_pos, map_size):
            return True
        if self.__next_pos(self.opponent_pos, map_size):
            return True
        return self.__next_pos(self.target_pos, map_size)

    def move_checking_bounds(self, pos: Vec2D, action: Action, map_size: MapSize) -> bool:
        pos.move(action)
        is_within_bounds: bool
        match action:
            case Action.UP:
                is_within_bounds = pos.y < map_size.M
            case Action.RIGHT:
                is_within_bounds = pos.x < map_size.N
            case Action.DOWN:
                is_within_bounds = pos.y > -1
            case Action.LEFT:
                is_within_bounds = pos.x > -1
        if not is_within_bounds:
            pos.undo(action)
        return is_within_bounds

    def __next_pos(self, pos: Vec2D, map_size: MapSize) -> bool:
        pos.x += 1
        if pos.x < map_size.N:
            return True
        pos.x = 0
        pos.y += 1
        if pos.y < map_size.M:
            return True
        pos.y = 0
        return False
    
    def copy(self, oth: Self) -> None:
        self.agent_pos.copy(oth.agent_pos)
        self.target_pos.copy(oth.target_pos)
        self.opponent_pos.copy(oth.opponent_pos)

@dataclass
class GameData:
    state: State = field(default_factory=lambda: State())
    agent_action: Action = Action.MAX_EXCLUSIVE
    target_action: Action = Action.MAX_EXCLUSIVE
    opponent_action: Action = Action.MAX_EXCLUSIVE

@dataclass
class TrainData:
    iteration_number: int = 0
    mean_value: float = 0.0
    max_value_diff: float = 0.0
    changed_actions_number: int = 0
    changed_actions_percentage: float = 0.0

class ValidStateSpaceArray(Protocol):

    @overload
    def __getitem__(self, index: int) -> int:
        ...
    
    @overload
    def __getitem__(self, slice: slice) -> Sequence[int]:
        ...

class ValidStateSpaceIterator:

    def __init__(self, states: ValidStateSpaceArray, space_size: int, map_size: MapSize, reversed: bool) -> None:
        self.__array: ValidStateSpaceArray = states
        self.__space_size: int = space_size
        self.__map_size: MapSize = map_size
        self.__reversed: bool = reversed
        self.__current_index: int = space_size - 1 if reversed else 0
        self.__state: State = State()

    def __iter__(self) -> "ValidStateSpaceIterator":
        return self

    def __next__(self) -> State:
        if ((not self.__reversed and self.__current_index == self.__space_size) or
            (self.__reversed and self.__current_index == -1)):
            raise StopIteration()
        self.__state.from_index(self.__array[self.__current_index], self.__map_size)
        if not self.__reversed:
            self.__current_index += 1
        else:
            self.__current_index -= 1
        return self.__state

class ValidStateSpace(ABC):

    def __init__(self, map_size: Vec2D, obstacles: Iterable[Obstacle]) -> None:
        self.map_size: MapSize = MapSize(map_size.x, map_size.y)
        self.space_size: int = 0
        self.__valid_cache: OrderedDict[int, int] = OrderedDict()
        self.__not_valid_cache: OrderedDict[int, int] = OrderedDict()
        self.__max_cache_length: int = 3 * map_size.x
        index_list: list[int] = []
        state: State = State()
        state_index: int = 0
        while state.next_state(self.map_size):
            state_index += 1
            if self.__is_state_valid(state, obstacles):
                index_list.append(state_index)
                self.space_size += 1
        types: tuple[str, c_uint_types] = self.__select_type(self.space_size)
        self.type: c_uint_types = types[1]
        self.__array: ValidStateSpaceArray = self._get_collection(index_list, types)

    @abstractmethod
    def _get_collection(self, indices: list[int], types: tuple[str, c_uint_types]) -> ValidStateSpaceArray:
        ...

    def get_valid_index(self, state: State) -> int:
        state_index: int = state.to_index(self.map_size)
        valid_index: int | None = self.__valid_cache.get(state_index)
        if valid_index is not None:
            return valid_index
        is_valid: bool
        is_valid, valid_index = self.__binary_search(state_index)
        if not is_valid:
            raise ValueError("Should not happen")
        self.__add_to_valid_cache(state_index, valid_index)
        return valid_index
    
    def is_state_outside_obstacles(self, state: State) -> bool:
        # assume that state is inside bounds
        state_index: int = state.to_index(self.map_size)
        if state_index in self.__valid_cache:
            return True
        if state_index in self.__not_valid_cache:
            return False
        is_valid: bool
        valid_index: int
        is_valid, valid_index = self.__binary_search(state_index)
        if is_valid:
            self.__add_to_valid_cache(state_index, valid_index)
        else:
            self.__add_to_not_valid_cache(state_index, valid_index)
        return is_valid

    def __add_to_valid_cache(self, state_index: int, valid_state_index: int) -> None:
        self.__valid_cache[state_index] = valid_state_index
        self.__load_near_states_to_cache(state_index, valid_state_index, True)
        if len(self.__valid_cache) == self.__max_cache_length:
            self.__valid_cache.popitem(last=False)

    def __add_to_not_valid_cache(self, state_index: int, last_smaller_valid_index: int) -> None:
        self.__not_valid_cache[state_index] = last_smaller_valid_index
        self.__load_near_states_to_cache(state_index, last_smaller_valid_index, False)
        if len(self.__not_valid_cache) == self.__max_cache_length:
            self.__not_valid_cache.popitem(last=False)

    def __load_near_states_to_cache(self, state_index: int, valid_state_index: int, is_state_valid: bool) -> None:
        prev_valid_state_index: int = valid_state_index - 1 if is_state_valid else valid_state_index
        prev_state_index: int = state_index - 1
        next_valid_state_index: int = valid_state_index + 1
        next_state_index: int = state_index + 1
        
        if prev_valid_state_index > -1:
            prev_state_index_found: int = self.__array[prev_valid_state_index]
            if prev_state_index_found == prev_state_index:
                self.__valid_cache[prev_state_index] = prev_valid_state_index
            else:
                self.__valid_cache[prev_state_index_found] = prev_valid_state_index
                self.__not_valid_cache[prev_state_index_found + 1] = prev_valid_state_index
                self.__not_valid_cache[prev_state_index] = prev_valid_state_index
        else:
            self.__not_valid_cache[prev_state_index] = prev_valid_state_index
        
        if next_valid_state_index < self.space_size:
            next_state_index_found: int = self.__array[next_valid_state_index]
            if next_state_index_found == next_state_index:
                self.__valid_cache[next_state_index] = next_valid_state_index
            else:
                self.__valid_cache[next_state_index_found] = next_valid_state_index
                self.__not_valid_cache[next_state_index_found - 1] = valid_state_index
                self.__not_valid_cache[next_state_index] = valid_state_index
        else:
            self.__not_valid_cache[next_state_index] = valid_state_index
        
        while len(self.__valid_cache) > self.__max_cache_length:
            self.__valid_cache.popitem(last=False)
        while len(self.__not_valid_cache) > self.__max_cache_length:
            self.__not_valid_cache.popitem(last=False)

    def copy_valid_state_to(self, state: State, index: int) -> None:
        state.from_index(self.__array[index], self.map_size)

    def __select_type(self, number_of_states: int) -> tuple[str, c_uint_types]:
        match number_of_states:
            case n if n <= 2 ** 8:
                return ("B",  c_ubyte)
            case n if n <= 2 ** 16:
                return ("H", c_ushort)
            case n if n <= 2 ** 32:
                return ("L", c_ulong)
            case _:
                return ("Q", c_ulonglong)
    
    def __binary_search(self, state_index: int) -> tuple[bool, int]:
        i: int = 0
        j: int = self.space_size - 1
        while i <= j:
            k: int = (i + j) // 2
            retrieved_index: int = self.__array[k]
            if retrieved_index == state_index:
                return (True, k)
            elif retrieved_index < state_index:
                i = k + 1
            else:
                j = k - 1
        return (False, j)

    def __is_state_valid(self, state: State, obstacles: Iterable[Obstacle]) -> bool:
        if state.target_pos == state.opponent_pos:
            return False
        for obstacle in obstacles:
            if (obstacle.is_inside(state.agent_pos) or
                obstacle.is_inside(state.opponent_pos) or
                obstacle.is_inside(state.target_pos)):
                return False
        return True
    
    def is_state_within_bounds(self, state: State) -> bool:
        return (self.__is_pos_within_bounds(state.agent_pos) and
                self.__is_pos_within_bounds(state.opponent_pos) and
                self.__is_pos_within_bounds(state.target_pos))

    def __is_pos_within_bounds(self, pos: Vec2D) -> bool:
        return (pos.x > -1 and pos.x < self.map_size.N and
                pos.y > -1 and pos.y < self.map_size.M)

    def __iter__(self) -> ValidStateSpaceIterator:
        return ValidStateSpaceIterator(self.__array, self.space_size, self.map_size, False)
    
    def __reversed__(self) -> ValidStateSpaceIterator:
        return ValidStateSpaceIterator(self.__array, self.space_size, self.map_size, True)

    def __len__(self) -> int:
        return self.space_size
    
    def __getitem__(self, index: int | slice) -> State | Sequence[State]:
        state_indices: int | Sequence[int] = self.__array[index]
        if isinstance(state_indices, int):
            state: State = State()
            state.from_index(state_indices, self.map_size)
            return state
        states: list[State] = [State() for _ in state_indices]
        for state, index in zip(states, state_indices):
            state.from_index(index, self.map_size)
        return states

    def __contains__(self, obj: int | State) -> bool:
        if isinstance(obj, int):
            return self.__binary_search(obj)[0]
        if isinstance(obj, State):
            return self.is_state_within_bounds(obj) and self.is_state_outside_obstacles(obj)
        return False

class ValidStateSpaceSequential(ValidStateSpace):

    @override
    def _get_collection(self, indices: list[int], types: tuple[str, c_uint_types]) -> ValidStateSpaceArray:
        return array(types[0], indices)

class ValidStateSpaceParallel(ValidStateSpace):

    @override
    def _get_collection(self, indices: list[int], types: tuple[str, c_uint_types]) -> ValidStateSpaceArray:
        return mp.RawArray(types[1], indices)

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

class ValueFunctionsContainer(ABC):

    @abstractmethod
    def get_type(self) -> type[c_float | c_double]:
        ...

    @abstractmethod
    def get_current_value(self, index: int) -> float:
        ...

    @abstractmethod
    def set_next_value(self, index: int, value: float) -> None:
        ...

    @abstractmethod
    def swap_value_functions(self) -> None:
        ...

class ValueFunctionsContainerSequential(ValueFunctionsContainer):

    def __init__(self, size: int, start_value: float = 0.0, use_double: bool = True) -> None:
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

    def __init__(self, size: int, start_value: float = 0.0, use_double: bool = True) -> None:
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