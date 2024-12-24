from enum import IntEnum
from dataclasses import dataclass, field
from array import array
from typing import Self, final

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
            return False
        if self.__next_pos(self.opponent_pos, map_size):
            return False
        return not self.__next_pos(self.target_pos, map_size)

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

class ValidStateSpaceIterator:

    def __init__(self, states: array[int], space_size: int, map_size: MapSize) -> None:
        self.__array: array[int] = states
        self.__space_size: int = space_size;
        self.__map_size: MapSize = map_size
        self.__current_index: int = -1
        self.__state: State = State()

    def __next__(self) -> tuple[int, State]:
        self.__current_index += 1
        if self.__current_index == self.__space_size:
            raise StopIteration()
        self.__state.from_index(self.__array[self.__current_index], self.__map_size)
        return (self.__current_index, self.__state)

class ValidStateSpace:

    def __init__(self, map_size: Vec2D, obstacles: list[Obstacle]) -> None:
        self.__map_size: MapSize = MapSize(map_size.x, map_size.y)
        self.space_size: int = 0
        index_list: list[int] = []
        state: State = State()
        while (not state.next_state(self.__map_size)):
            if self.__is_state_valid(state, obstacles):
                index_list.append(state.to_index(self.__map_size))
                self.space_size += 1
        self.__pick_smallest_array(self.space_size)
        self.__arr.fromlist(index_list)

    def get_index(self, state: State) -> int:
        state_index: int = state.to_index(self.__map_size)
        return self.__binary_search(state_index)

    def is_valid(self, state: State) -> bool:
        state_index: int = state.to_index(self.__map_size)
        return self.__binary_search(state_index) != -1 and self.__is_state_within_bounds(state)

    def __pick_smallest_array(self, number_of_states: int) -> None:
        type_char: str
        if number_of_states <= 2 ** 8:
            type_char = "B"
        elif number_of_states <= 2 ** 16:
            type_char = "H"
        elif number_of_states <= 2 ** 32:
            type_char = "L"
        else:
            type_char = "Q"
        self.__arr : array[int] = array(type_char)

    def __binary_search(self, state_index: int) -> int:
        i: int = 0
        j: int = self.space_size - 1
        while i <= j:
            k: int = (i + j) // 2
            retrieved_index: int = self.__arr[k]
            if retrieved_index == state_index:
                return k
            elif retrieved_index < state_index:
                i = k + 1
            else:
                j = k - 1
        return -1

    def __is_state_valid(self, state: State, obstacles: list[Obstacle]) -> bool:
        if state.target_pos == state.opponent_pos:
            return False
        for obstacle in obstacles:
            if (obstacle.is_inside(state.agent_pos) or
                obstacle.is_inside(state.opponent_pos) or
                obstacle.is_inside(state.target_pos)):
                return False
        return True
    
    def __is_state_within_bounds(self, state: State) -> bool:
        return (self.__is_pos_within_bounds(state.agent_pos) and
                self.__is_pos_within_bounds(state.opponent_pos) and
                self.__is_pos_within_bounds(state.target_pos))

    def __is_pos_within_bounds(self, pos: Vec2D) -> bool:
        return (pos.x > -1 and pos.x < self.__map_size.N and
                pos.y > -1 and pos.y < self.__map_size.M)

    def __iter__(self) -> ValidStateSpaceIterator:
        return ValidStateSpaceIterator(self.__arr, self.space_size, self.__map_size)
    
    def __len__(self) -> int:
        return self.space_size
    
    def __getitem__(self, index: int | slice) -> State | list[State]:
        state_indices: int | array[int] = self.__arr[index]
        if isinstance(state_indices, int):
            state: State = State()
            state.from_index(state_indices, self.__map_size)
            return state
        states: list[State] = [State() for _ in state_indices]
        for state, index in zip(states, state_indices):
            state.from_index(index, self.__map_size)
        return states

    def __contains__(self, obj: int | State) -> bool:
        if isinstance(obj, int):
            return self.__binary_search(obj) != -1
        if isinstance(obj, State):
            return self.is_valid(obj)
        raise TypeError(f"ValidStateSpace expected int or State, but was {type(obj)}")

class Policy:

    @classmethod
    def from_file(cls, policy_file_name: str) -> "Policy":
        p: Policy = Policy()
        with open(policy_file_name, "rb") as f:
            policy_size: int = f.seek(0, 2)
            f.seek(0, 0)
            p.__arr.fromfile(f, policy_size)
        return p

    @classmethod
    def from_action(cls, policy_size: int, action: Action = Action.UP) -> "Policy":
        p: Policy = Policy()
        p.__arr.fromlist([action.value for i in range(policy_size)])
        return p

    def __init__(self) -> None:
        self.__arr : array[int] = array("B")

    def get_action(self, index: int) -> Action:
        return Action(self.__arr[index])
    
    def set_action(self, index: int, action: Action) -> None:
        self.__arr[index] = action.value

    def write_to_file(self, policy_file_name: str) -> None:
        with open(policy_file_name, "wb") as f:
            self.__arr.tofile(f)

class ValueFunctionsContainer:

    def __init__(self, size: int, start_value: float = 0.0):
        self.__old_values : array[float] = array("d")
        self.__new_values : array[float] = array("d")
        values: list[float] = [start_value for i in range(size)]
        self.__old_values.fromlist(values)
        self.__new_values.fromlist(values)

    def get_current_value(self, index: int) -> float:
        return self.__old_values[index]

    def set_next_value(self, index: int, value: float) -> None:
        self.__new_values[index] = value

    def swap_value_functions(self) -> None:
        self.__old_values, self.__new_values = self.__new_values, self.__old_values