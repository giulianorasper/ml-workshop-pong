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

    def __init__(self, observed_player=Player.LEFT):
        """
        :param observed_player: the Player to observe
        :note: Observing the right player requires an adequate implementation
        of the transform strategy, which swaps the sides of an observation.
        """
        super().__init__()
        if Strategy.TRANSFORM_OBSERVATION not in strategy_pattern.strategies and observed_player == Player.RIGHT:
            printing.warn("Observing the right player without a transform strategy!")
        self.observation: Pong = None
        self.observed_player = observed_player
        self.should_train = 0

        self.ball_touches = []

    def update(self, data):
        observation: Pong = self.observation
        next_observation: Pong = data


        if next_observation and self.observed_player == Player.RIGHT:
            if Strategy.TRANSFORM_OBSERVATION in strategy_pattern.strategies:
                next_observation = strategy_pattern.strategies[Strategy.TRANSFORM_OBSERVATION](next_observation)

        # note: we throw away the first observation, because we don't have a transition yet
        # this also causes the first action to be random in evaluation mode

        if observation is None:
            self.observation = next_observation
            return

        if next_observation is None:
            return

        try:
            action_id = observation.last_action_taken_left

            action_to_action_id = PongObserver.__reverse_mapping(RLPongController.get_action_map())

            state = self.get_state(observation)
            # print(f"state: {state}")
            action = action_to_action_id[action_id]
            next_state = self.get_state(next_observation)

            reward = self.get_reward(observation, next_observation)

            # we expect the students to define neutral reward as 0
            # we classify each non neutral-reward as sparse to selectively
            # use more transitions with non-sparse rewards for training
            # otherwise the learning is too slow for a workshop
            sparse = reward != 0
            if reward != 0:
                printing.special_print(printing.PrintFlag.REWARD, f"{reward}")
            if strategy_pattern.Strategy.REWARD not in strategy_pattern.strategies:
                # scaling the default rewards into range [0, 1]
                reward = (reward + 20) / 70

            # we transform the observations such that the end of each ball exchange is a terminal state
            if observation.resets < next_observation.resets:
                next_state = None
            if observation.right_score < next_observation.right_score:
                next_state = None
            if observation.left_score < next_observation.left_score:
                next_state = None

            transition: Transition = Transition(state, action, next_state, reward)

            self.log_metrics(observation, next_observation)

            # we only forward every NON_SPARSE_TRAINING-th transition to the replay buffer
            NON_SPARSE_TRAINING = 10
            if self.should_train <= 0 or sparse:
                # yes we want to train
                self.should_train = NON_SPARSE_TRAINING
                # we add the transition to the replay buffer and request an action recommendation
                update: PongObservationUpdate = PongObservationUpdate(transition, next_state)
                self.notify_observers(update)
            else:
                # no we don't want to train
                self.should_train -= 1
                # we do not add the transition to the replay buffer but request an action recommendation
                # i.e. we also skip training in this case
                update = PongObservationUpdate(None, next_state)
                self.notify_observers(update)

                # this return is required so that we do not override the current state
                return

            # if next_state:
            #     self.agent.select_action(next_state, convert_to_tensor=True)

        finally:
            self.observation: Pong = data

    def __transform_observation(observation: Pong) -> Pong:
        """
        :param observation: the observation to transform
        :return: Observation with the locations of the paddles,
        the ball and the ball's velocity swapped (reflected on vertical axis).
        This function may be used as strategy for using an agent
        trained on the left hand side also on the right hand side.
        Note: This is not enough for using the existing framework to train
        a right hand side agent from scratch.
        Exercise: Why is that? Extend this function such that it also enables training!
        """
        observation.left_paddle_y, observation.right_paddle_y = observation.right_paddle_y, observation.left_paddle_y
        width = pong.width
        observation.ball_x = width - observation.ball_x
        observation.ball_dx = -observation.ball_dx
        return observation

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
    def get_reward(observation: Pong, next_observation: Pong):
        if Strategy.REWARD in strategy_pattern.strategies:
            return strategy_pattern.strategies[Strategy.REWARD](observation, next_observation)

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

    def log_metrics(self, observation: Pong, next_observation: Pong):
        if observation.resets < next_observation.resets:
            self.ball_touches.append(observation.ball_touches_left)

    def plot_ball_touches(self):
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
