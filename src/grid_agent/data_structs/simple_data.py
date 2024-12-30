from ctypes import c_ubyte, c_ushort, c_ulong, c_ulonglong, c_float, c_double
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Self

type c_uints = c_ubyte | c_ushort | c_ulong | c_ulonglong
type c_uint_types = type[c_uints]
type c_floats = c_float | c_double
type c_float_types = type[c_floats]

class Action(IntEnum):
    """An enum listing the possible actions."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    MAX_EXCLUSIVE = LEFT + 1
    """Constant to be used for the enumeration of actions.
    
    For example:
      ``[Action(i) for i in range(Action.MAX_EXCLUSIVE)]``
    """

@dataclass
class Vec2D:
    """A two-dimensional vector of integers."""
    x : int = 0
    y : int = 0

    def copy(self, oth: Self) -> None:
        """Copy the values of another ``Vec2D`` into itself."""
        self.x = oth.x
        self.y = oth.y

    def move(self, action: Action) -> None:
        """Change the values of the ``Vec2D`` based on ``action``."""
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
        """Change the values of the ``Vec2D`` to undo ``action``.
        
        If the ``Vec2D`` was (2,1) passing ``Action.UP`` would change it to (1,1).
        """
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
    """An obstacle with a certain ``origin`` and ``extent``.
    
    ``origin``
      the leftmost-lowermost point occupied by the obstacle.

    ``extent``
      specifies the width and height of the obstacle, both in the positive direction of the corresponding axis.
    """
    origin : Vec2D = field(default_factory=lambda: Vec2D())
    extent : Vec2D = field(default_factory=lambda: Vec2D())

    def is_inside(self, pos: Vec2D) -> bool:
        """Return ``True`` if ``pos`` is colliding with the ``Obstacle``."""
        return (pos.x >= self.origin.x and pos.x < self.origin.x + self.extent.x and
                pos.y >= self.origin.y and pos.y < self.origin.y + self.extent.y)
    
    def to_pos(self) -> list[Vec2D]:
        """Return a list of ``Vec2D`` containing the points occupied by the ``Obstacle``."""
        return [Vec2D(x, y)
                for x in range(self.origin.x, self.origin.x + self.extent.x)
                for y in range(self.origin.y, self.origin.y + self.extent.y)]
    
    def is_inside_bounds(self, map_size: Vec2D) -> bool:
        """Return ``True`` if the ``Obstacle`` is inside the bounds defined by the ``map_size``.
        
        ``map_size``
          is a ``Vec2D`` containing the extent of the map.
        """
        return (self.origin.x > -1 and self.origin.x + self.extent.x <= map_size.x and
                self.origin.y > -1 and self.origin.y + self.extent.y <= map_size.y)

@dataclass
class MapSize:
    """A simple object storing useful values.
    
    The values are just products and exponetiation of the width and height of the map.
    """
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