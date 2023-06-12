from enum import Enum

import pong
import printing
import strategy_pattern
import visualisation
from pong_action import PongAction
from tuples import Transition, PongObservationUpdate
from observer_pattern import Observer, Observable
from pong import Pong
from pong_controller import RLPongController
from strategy_pattern import Strategy


class Player(Enum):
    LEFT = 1
    RIGHT = 2


class PongObserver(Observer, Observable):

    def __init__(self, observed_game: Pong, observed_player):
        super().__init__()
        observed_game.register_observer(self)
        self.observed_player = observed_player
        self.observation: Pong = None
        self.should_train = 0

        self.ball_touches = []

    def update(self, data):
        observation: Pong = self.observation
        if observation and self.observed_player == Player.RIGHT:
            if Strategy.TRANSFORM_OBSERVATION in strategy_pattern.strategies:
                observation = strategy_pattern.strategies[Strategy.TRANSFORM_OBSERVATION](observation)

        next_observation: Pong = data

        # TODO: throwing away the first observation might not be ideal (no action is chosen by agent)
        if observation is None:
            self.observation = next_observation
            return
        if next_observation is None:
            return

        if self.observed_player == Player.LEFT:
            action_id = observation.last_action_taken_left
        else:
            action_id = observation.last_action_taken_right

        # TODO: generate mapping at initialisation instead?
        action_to_action_id = PongObserver.__reverse_mapping(RLPongController.get_action_map())

        state = self.get_state(observation)
        # print(f"state: {state}")
        action = action_to_action_id[action_id]
        next_state = self.get_state(next_observation)

        # TODO: add done attribute to Pong?
        reward = self.get_reward(self.observed_player, observation, next_observation)
        sparse = reward == 0
        if reward != 0:
            printing.special_print(printing.SpecialPrint.REWARD, f"{reward}")
        if strategy_pattern.Strategy.REWARD not in strategy_pattern.strategies:
            reward = (reward + 20) / 70


        # TODO: LEFT AGENT!
        if observation.resets < next_observation.resets:
            next_state = None
        if observation.right_score < next_observation.right_score:
            next_state = None

        transition: Transition = Transition(state, action, next_state, reward)

        self.log_metrics(self.observed_player, observation, next_observation)

        if self.should_train <= 0 or not sparse:
            # yes we want to train
            self.should_train = 10
            update: PongObservationUpdate = PongObservationUpdate(transition, next_state)
            self.notify_observers(update)
        else:
            # no we don't want to train
            self.should_train -= 1
            update = PongObservationUpdate(None, next_state)
            self.notify_observers(update)

            # this return is required so that we do not override the current state
            return


        # if next_state:
        #     self.agent.select_action(next_state, convert_to_tensor=True)

        self.observation: Pong = data

    @staticmethod
    def __reverse_mapping(mapping):
        reversed_mapping = {}
        for key, value in mapping.items():
            reversed_mapping[value] = key
        return reversed_mapping

    @staticmethod
    def get_state(observation: Pong):
        if Strategy.STATE in strategy_pattern.strategies:
            return strategy_pattern.strategies[Strategy.STATE](observation)
        return observation.left_paddle_y / 400, observation.ball_x / pong.width, observation.ball_y / pong.height, observation.ball_dx / 6, observation.ball_dy / 6, observation.ticks_this_ball_exchange / 1000

    @staticmethod
    def get_reward(observed_player: Player, observation: Pong, next_observation: Pong):
        if Strategy.REWARD in strategy_pattern.strategies:
            return strategy_pattern.strategies[Strategy.STATE](observed_player, observation, next_observation)

        reward = 0

        if observation.right_score < next_observation.right_score:
            ball_y = observation.ball_y
            y = observation.left_paddle_y
            height = observation.paddle_height
            middle = y + height / 2
            board_height = pong.height
            relative_distance = abs(middle - ball_y) / (board_height - height / 2)
            reward += - 100 * relative_distance
        elif observation.ball_dx < 0 < next_observation.ball_dx:
            reward += 20

        # if observation.last_action_taken_left != next_observation.last_action_taken_left:
        #     reward += -1

        # if observation.left_paddle_y == next_observation.left_paddle_y and next_observation.last_action_taken_left != PongAction(False, False):
        #     reward += -1
        return reward

    def log_metrics(self, observed_player: Player, observation: Pong, next_observation: Pong):
        # TODO: also log ball touches for right player
        if observation.resets < next_observation.resets:
            self.ball_touches.append(observation.ball_touches_left)

    def plot_ball_touches(self):
        # TODO: also plot ball touches for right player
        self.ball_touches.append(self.observation.ball_touches_left)
        visualisation.plot_results(self.ball_touches, title="Ball touches per Episode", xlabel="Episode",
                                   ylabel="Ball touches")

    @staticmethod
    def get__n_obs_n_act():
        pong = Pong()
        pong.reset()
        n_obs = len(PongObserver.get_state(pong))
        n_act = len(RLPongController.get_action_map())
        return n_obs, n_act
