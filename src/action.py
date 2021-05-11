from enum import Enum, auto


class Action(Enum):
    NORTH = auto()
    NORTH_EAST = auto()
    EAST = auto()
    SOUTH_EAST = auto()
    SOUTH = auto()
    SOUTH_WEST = auto()
    WEST = auto()
    NORTH_WEST = auto()
    STAIR_UP = auto()
    STAIR_DOWN = auto()
    OPEN = auto()
    CLOSE = auto()
    SEARCH = auto()
    LOOK = auto()
    MORE = auto()
    YES = auto()
    NO = auto()
    LOOT = auto()
    UNTRAP = auto()
    PRAY = auto()


KEY_ACTIONS = {
    Action.NORTH: 'k',
    Action.NORTH_EAST: 'u',
    Action.EAST: 'l',
    Action.SOUTH_EAST: 'n',
    Action.SOUTH: 'j',
    Action.SOUTH_WEST: 'b',
    Action.WEST: 'h',
    Action.NORTH_WEST: 'y',
    Action.STAIR_UP: '<',
    Action.STAIR_DOWN: '>',
    Action.OPEN: 'o',
    Action.CLOSE: 'c',
    Action.SEARCH: 's',
    Action.LOOK: ':',
    Action.MORE: '\n',
    Action.YES: 'y',
    Action.NO: 'n'
}

HASH_ACTIONS = {
    Action.LOOT: '#loot',
    Action.UNTRAP: '#untrap',
    Action.PRAY: '#pray'
}

MENU_ACTIONS = [
    Action.MORE,
    Action.YES,
    Action.NO
]

MOVE_ACTIONS = [
    Action.NORTH,
    Action.NORTH_EAST,
    Action.EAST,
    Action.SOUTH_EAST,
    Action.SOUTH,
    Action.SOUTH_WEST,
    Action.WEST,
    Action.NORTH_WEST,
    Action.STAIR_UP,
    Action.STAIR_DOWN,
    # Action.SEARCH
]

map_act_int = {
    Action.NORTH: 0,
    Action.NORTH_EAST: 1,
    Action.EAST: 2,
    Action.SOUTH_EAST: 3,
    Action.SOUTH: 4,
    Action.SOUTH_WEST: 5,
    Action.WEST: 6,
    Action.NORTH_WEST: 7,
    Action.STAIR_UP: 8,
    Action.STAIR_DOWN: 9,
}
map_act_int2 = {
    Action.MORE: 10,
    Action.YES: 11,
    Action.NO: 12,
    Action.LOOT: 13,
    Action.UNTRAP: 14,
    Action.PRAY: 15,
    Action.OPEN: 16,
    Action.CLOSE: 17,
    Action.SEARCH: 18,
    Action.LOOK: 19,
}
print(map_act_int[MOVE_ACTIONS[0]])
