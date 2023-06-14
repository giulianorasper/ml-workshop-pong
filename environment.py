from concurrent.futures import Future
from multiprocessing import Lock

from observer_pattern import Observer
from pong import Pong
from pong_action import PongAction


class PongEnvironment(Observer):

    def __init__(self):
        super().__init__()

        self.pong = Pong(left_agent=self)
        self.pong.register_observer(self)
        self.next_action = PongAction(False, False)
        self.action_lock: Lock = Lock()
        self.current_obs_lock: Lock = Lock()
        self.current_observation = Future()

    def reset(self):
        self.pong.reset()
        # forward initial state to RL-agent
        return self.__get_state()

    def step(self, action):
        pong_act_map = {
            0: PongAction(False, False),
            1: PongAction(True, False),
            2: PongAction(False, True)
        }
        pong_action = pong_act_map[action]

        with self.action_lock:
            self.next_action = pong_action
        return self.__get_state(), 0

    def get_action(self):
        with self.action_lock:
            return self.next_action

    def __get_state(self):
        with self.current_obs_lock:
            current_observation = self.current_observation
        pong_state = current_observation.result()
        with self.current_obs_lock:
            self.current_observation = Future()

        return self.__get_state_with_pong_state(pong_state)

    def __get_state_with_pong_state(self, pong_state: Pong):
        return pong_state.left_paddle_y, pong_state.ball_x, pong_state.ball_y, pong_state.ball_dx, pong_state.ball_dy

    def update(self, data):
        with self.current_obs_lock:
            if self.current_observation.done():
                self.current_observation = Future()
        self.current_observation.set_result(data)

    def get_action_space(self):
        nope, up, down = 0, 1, 2
        actions = [nope, up, down]
        return actions