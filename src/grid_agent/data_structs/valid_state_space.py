from grid_agent.data_structs.simple_data import Vec2D, MapSize, Obstacle, c_uint_types
from grid_agent.data_structs.state import State

from collections.abc import Iterable, Sequence
from collections import OrderedDict

from ctypes import c_ubyte, c_ushort, c_ulong, c_ulonglong
from typing import Protocol, overload, override
from abc import ABC, abstractmethod
import multiprocessing as mp
from array import array

class ValidStateSpaceArray(Protocol):
    """Protocol for the container used by ``ValidStateSpace``."""
    @overload
    def __getitem__(self, index: int) -> int:
        ...
    
    @overload
    def __getitem__(self, slice: slice) -> Sequence[int]:
        ...

class ValidStateSpaceIterator:
    """Iterator for ``ValidStateSpace``."""
    def __init__(self, states_indices: ValidStateSpaceArray, space_size: int, map_size: MapSize, reversed: bool) -> None:
        """Make an iterator for ``states_indices`` of size ``space_size``.
        
        ``map_size`` is needed for converting indices back to ``State``s.

        If ``reversed`` is ``True`` the order of iteration is reversed.
        """
        self.__array: ValidStateSpaceArray = states_indices
        self.__space_size: int = space_size
        self.__map_size: MapSize = map_size
        self.__reversed: bool = reversed
        self.__current_index: int = space_size - 1 if reversed else 0
        self.__state: State = State()

    def __iter__(self) -> "ValidStateSpaceIterator":
        return self

    def __next__(self) -> State:
        """Return the next valid ``State``.
        
        Throw ``StopIteration`` if the iteration is finished.
        """
        if (self.__current_index == self.__space_size or
            self.__current_index == -1):
            raise StopIteration()
        self.__state.from_index(self.__array[self.__current_index], self.__map_size)
        self.__current_index = self.__current_index - 1 if self.__reversed else self.__current_index + 1
        return self.__state

class ValidStateSpace(ABC):
    """The space of valid ``State``s."""
    def __init__(self, map_size: Vec2D, obstacles: Iterable[Obstacle]) -> None:
        """Given ``map_size`` and ``obstacles`` initialize the space of valid ``State``s."""
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
        """Return the container that ``ValidStateSpace`` will use to store the indices of valid ``State``s."""
        ...

    def get_valid_index(self, state: State) -> int:
        """Return the valid index of ``state``.
        
        The valid index is the index of ``state`` if counting only valid ``State``s.

        Return -1 if ``state`` is not valid.

        WARNING
          It doesn't check if the positions of ``state`` are out of bounds.

          Use ``is_state_within_bounds`` method if not sure.
        """
        state_index: int = state.to_index(self.map_size)
        valid_index: int | None = self.__valid_cache.get(state_index)
        if valid_index is not None:
            return valid_index
        is_valid: bool
        is_valid, valid_index = self.__binary_search(state_index)
        if not is_valid:
            return -1
        self.__add_to_valid_cache(state_index, valid_index)
        return valid_index
    
    def is_state_outside_obstacles(self, state: State) -> bool:
        """Return ``True`` if ``state`` doesn't contain any positions in collision with obstacles.
        
        WARNING
          It doesn't check if the positions of ``state`` are out of bounds.
        
          Use ``is_state_within_bounds`` method if not sure.
        """
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
        """Add to the cache of indices of valid ``State``s ``state_index`` associated with ``valid_state_index``."""
        self.__valid_cache[state_index] = valid_state_index
        self.__load_near_states_to_cache(state_index, valid_state_index, True)
        if len(self.__valid_cache) == self.__max_cache_length:
            self.__valid_cache.popitem(last=False)

    def __add_to_not_valid_cache(self, state_index: int, last_smaller_valid_index: int) -> None:
        """Add to the cache of indices of not valid ``State``s ``state_index`` associated with ``last_smaller_valid_index``."""
        self.__not_valid_cache[state_index] = last_smaller_valid_index
        self.__load_near_states_to_cache(state_index, last_smaller_valid_index, False)
        if len(self.__not_valid_cache) == self.__max_cache_length:
            self.__not_valid_cache.popitem(last=False)

    def __load_near_states_to_cache(self, state_index: int, valid_state_index: int, is_state_valid: bool) -> None:
        """Add into the caches the indices of the states near the state of index ``state_index``.
        
        ``is_state_valid`` is ``bool`` signalling if ``state_index`` was put into the cache of indices
        of valid states or into the cache of indices of not valid states.

        ``valid_state_index`` is the valid index associated with ``state_index`` if ``is_state_valid`` is ``True``
        otherwise is the index of the last smaller index of a valid ``State`` respect to ``state_index``. 
        """
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
        """Copy into ``state`` the ``State`` found at ``index``."""
        state.from_index(self.__array[index], self.map_size)

    def __select_type(self, number_of_states: int) -> tuple[str, c_uint_types]:
        """Select the smallest type of unsigned integer able to index ``number_of_states``."""
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
        """Perform a binary search into the indices of valid ``State``s.
        
        Return
          ``True`` and the valid index of ``state_index`` if found.

          ``False`` and the valid index of the last valid ``State`` whose index is smaller than ``state_index``.
        """
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
        """Return ``True`` if ``state`` is valid, given ``obstacles``."""
        if state.target_pos == state.opponent_pos:
            return False
        for obstacle in obstacles:
            if (obstacle.is_inside(state.agent_pos) or
                obstacle.is_inside(state.opponent_pos) or
                obstacle.is_inside(state.target_pos)):
                return False
        return True
    
    def is_state_within_bounds(self, state: State) -> bool:
        """Return ``True`` if ``state`` is within the bounds contained by ``ValidStateSpace``."""
        return (self.__is_pos_within_bounds(state.agent_pos) and
                self.__is_pos_within_bounds(state.opponent_pos) and
                self.__is_pos_within_bounds(state.target_pos))

    def __is_pos_within_bounds(self, pos: Vec2D) -> bool:
        """Return ``True`` if ``pos`` is within the bounds contained by ``ValidStateSpace``."""
        return (pos.x > -1 and pos.x < self.map_size.N and
                pos.y > -1 and pos.y < self.map_size.M)

    def __iter__(self) -> ValidStateSpaceIterator:
        """Return an iterator of ``ValidStateSpace``."""
        return ValidStateSpaceIterator(self.__array, self.space_size, self.map_size, False)
    
    def __reversed__(self) -> ValidStateSpaceIterator:
        """Return a reversed iterator of ``ValidStateSpace``."""
        return ValidStateSpaceIterator(self.__array, self.space_size, self.map_size, True)

    def __len__(self) -> int:
        """Return the length of ``ValidStateSpace``."""
        return self.space_size
    
    def __getitem__(self, index: int | slice) -> State | Sequence[State]:
        """Return the ``State``(or a ``Sequence`` of ``State``) associated with ``index``(or indices)."""
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
        """Return ``True`` if ``obs`` is a valid ``State`` or the index of a valid ``State``."""
        if isinstance(obj, int):
            return self.__binary_search(obj)[0]
        if isinstance(obj, State):
            return self.is_state_within_bounds(obj) and self.is_state_outside_obstacles(obj)
        return False

class ValidStateSpaceSequential(ValidStateSpace):
    """``ValidStateSpace`` specialized for sequential learning."""
    @override
    def _get_collection(self, indices: list[int], types: tuple[str, c_uint_types]) -> ValidStateSpaceArray:
        return array(types[0], indices)

class ValidStateSpaceParallel(ValidStateSpace):
    """``ValidStateSpace`` specialized for parallel learning."""
    @override
    def _get_collection(self, indices: list[int], types: tuple[str, c_uint_types]) -> ValidStateSpaceArray:
        return mp.RawArray(types[1], indices)