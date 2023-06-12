from enum import Enum

strategies = {}

class Strategy(Enum):
    REWARD = 1
    STATE = 2
    ACTION_MAP = 3
    NETWORK_STRUCTURE = 4
    TRANSFORM_OBSERVATION = 5