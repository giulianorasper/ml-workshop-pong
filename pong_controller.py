import threading
from abc import ABC, abstractmethod

import pygame

import strategy_pattern
from observer_pattern import Observer
from pong_action import PongAction
from strategy_pattern import Strategy


class PongController(ABC):

    @abstractmethod
    def get_pong_action(self):
        raise NotImplementedError("Subclasses must implement get_pong_action()")


class KeyboardPongController(PongController, ABC):

    @staticmethod
    def left_controller():
        return KeyboardPongController(pygame.K_w, pygame.K_s)

    @staticmethod
    def right_controller():
        return KeyboardPongController(pygame.K_UP, pygame.K_DOWN)

    def __init__(self, up_key: int, down_key: int):
        super().__init__()

        self.up_key = up_key
        self.down_key = down_key

    def get_pong_action(self):
        keys = pygame.key.get_pressed()
        up_pressed = keys[self.up_key]
        down_pressed = keys[self.down_key]
        return PongAction(up_pressed, down_pressed)


class RLPongController(PongController, Observer):

    def __init__(self, agent):
        super().__init__()

        self.agent = agent
        self.agent.register_observer(self)
        self.next_action = PongAction(False, False)
        self.next_action_lock = threading.Lock()

    @staticmethod
    def get_action_map():
        if Strategy.ACTION_MAP in strategy_pattern.strategies:
            return strategy_pattern.strategies[Strategy.ACTION_MAP]()
        pong_action_map = {
            0: PongAction(False, False),
            1: PongAction(True, False),
            2: PongAction(False, True)
        }
        return pong_action_map

    def update(self, action_id):
        pong_action = RLPongController.get_action_map()[action_id]
        with self.next_action_lock:
            self.next_action = pong_action

    def get_pong_action(self):
        with self.next_action_lock:
            return self.next_action

