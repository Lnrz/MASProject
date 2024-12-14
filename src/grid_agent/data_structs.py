from enum import IntEnum
from dataclasses import dataclass, field
from array import array
from typing import Self

class Result(IntEnum):
    Fail = 0,
    Success = 1,
    WaitingForResult = 2

class Action(IntEnum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3
    MaxExclusive = Left + 1

@dataclass
class Vec2D:
    x : int = 0
    y : int = 0

    def copy(self, oth: Self) -> None:
        self.x = oth.x
        self.y = oth.y

@dataclass
class Obstacle:
    origin : Vec2D = field(default_factory=lambda: Vec2D())
    extent : Vec2D = field(default_factory=lambda: Vec2D())

    def is_inside(self, pos: Vec2D) -> bool:
        return (pos.x >= self.origin.x and pos.x <= self.origin.x + self.extent.x and
                pos.y <= self.origin.y and pos.y >= self.origin.y + self.extent.y)

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

@dataclass
class Policy:
    __arr : array = field(default_factory=lambda: array("b"))

    def get_action(self, state: State, map_size: MapSize) -> Action:
        return Action(self.__arr[state.to_index(map_size)])
    
    def set_action(self, state: State, action: Action, map_size: MapSize) -> None:
        self.__arr[state.to_index(map_size)] = action.value

    def load_from_file(self, policy_file_name: str, policy_size: int) -> None:
        self.__arr.clear()
        with open(policy_file_name, "rb") as f:
            self.__arr.fromfile(f, policy_size)

    def fill(self, action: Action, policy_size: int) -> None:
        self.__arr.clear()
        self.__arr.fromlist([action.value for i in range(policy_size)])

@dataclass
class ValueFunction:
    __arr : array = field(default_factory=lambda: array("d"))

    def get_value(self, state: State, map_size: MapSize) -> float:
        return self.__arr[state.to_index(map_size)]

    def set_value(self, state: State, value: float, map_size: MapSize) -> None:
        self.__arr[state.to_index(map_size)] = value