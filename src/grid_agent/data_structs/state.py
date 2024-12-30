from grid_agent.data_structs.simple_data import Vec2D, MapSize, Action
from dataclasses import dataclass, field
from typing import Self

@dataclass
class State:
    """Represent the state as a triplet of agent position, opponent position and target position."""
    agent_pos : Vec2D = field(default_factory=lambda: Vec2D())
    opponent_pos : Vec2D = field(default_factory=lambda: Vec2D())
    target_pos : Vec2D = field(default_factory=lambda: Vec2D())

    def to_index(self, map_size: MapSize) -> int:
        """Return the index of the ``State``.
        
        If restricted to valid ``State``s it's injective.

        WARNING
          If the ``State`` is invalid, due to negative positions or out of bounds positions,
          the returned index will most likely be shared with a valid ``State``.
        """
        return (self.agent_pos.x + self.agent_pos.y * map_size.N +
                self.opponent_pos.x * map_size.NM + self.opponent_pos.y * map_size.N2M +
                self.target_pos.x *map_size.N2M2 + self.target_pos.y * map_size.N3M2)

    def from_index(self, index: int, map_size: MapSize) -> None:
        """Copy into the ``State`` the positions of the valid ``State`` associated with ``index``.
        
        WARNING
          It is assumed that ``index`` is valid relative to ``map_size``.
        """
        self.agent_pos.x = index % map_size.N
        self.agent_pos.y = (index % map_size.NM) // map_size.N
        self.opponent_pos.x = (index % map_size.N2M) // map_size.NM
        self.opponent_pos.y = (index % map_size.N2M2) // map_size.N2M
        self.target_pos.x = (index % map_size.N3M2) // map_size.N2M2
        self.target_pos.y = index // map_size.N3M2

    def next_state(self, map_size: MapSize) -> bool:
        """Change the ``State`` to the next valid ``State`` according to ``map_size``.
        
        Return
          ``True``, if the enumeration hasn't ended.
          
          ``False``, if the ``State`` was reversed to its initial state.
        """
        if self.__next_pos(self.agent_pos, map_size):
            return True
        if self.__next_pos(self.opponent_pos, map_size):
            return True
        return self.__next_pos(self.target_pos, map_size)

    def move_checking_bounds(self, pos: Vec2D, action: Action, map_size: MapSize) -> bool:
        """Move ``pos`` according to ``action`` only if the resulting ``State``, relative to ``map_size``, is valid.
        
        WARNING
          It is assumed that ``pos`` is contained by the ``State`` on which the method is called.
        """
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
        """Advance ``pos`` to the next one according to ``map_size``.
        
        Return
          ``True``, if the enumeration hasn't ended.

          ``False``, if ``pos`` was reversed to its initial values.
        """
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
        """Copy the positions of the passed ``State`` into itself."""
        self.agent_pos.copy(oth.agent_pos)
        self.target_pos.copy(oth.target_pos)
        self.opponent_pos.copy(oth.opponent_pos)