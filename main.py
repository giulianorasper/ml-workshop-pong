# This is a sample Python script.
import time

import numpy as np

import printing
import strategy_pattern
from agent import DQNAgent
from pong import Pong
from pong_controller import RLPongController
from pong_observer import PongObserver, Player


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def train(episodes=100, training_time=None):
    printing.info("Enabled strategies: " + str(strategy_pattern.strategies.keys()))
    if training_time:
        episodes = 0
    else:
        training_time = 0

    graphical = True
    pong: Pong = Pong(right_invincible=True, ticks_per_second=6000, graphical=graphical, win_screen_time_in_ticks=10)
    dqn_agent = DQNAgent()
    pong_observer: PongObserver = PongObserver()
    pong.register_observer(pong_observer)
    pong_observer.register_observer(dqn_agent)
    pong_controller = RLPongController(dqn_agent)
    pong.set_left_agent(pong_controller)

    episode = 0
    start_time = time.time()
    end_time = time.time() + training_time

    while episode <= episodes or time.time() < end_time:
        episode += 1
        printing.info(f"Episode {episode} / {int(time.time() - start_time)} seconds")
        pong.reset()
        pong.run()
        dqn_agent.next_episode()

    dqn_agent.save_model("model.ckpt")

    if printing.log_level == printing.LogLevel.ANALYSIS:
        dqn_agent.plot_results()
        pong_observer.plot_ball_touches()

    pong.quit()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    printing.print_flags = [] # [printing.SpecialPrint.REWARD]
    train(training_time=60)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
