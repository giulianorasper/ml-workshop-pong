from agent import DQNAgent
from pong import Pong
from pong_controller import RLPongController, KeyboardPongController
from pong_observer import PongObserver, Player


def play(ai_enemy=False, invincible_enemy=False, debug=False, swap_players=False):
    pong: Pong = Pong(right_invincible=invincible_enemy, left_invincible=False, ticks_per_second=60)

    try:
        ai_side = Player.LEFT
        if swap_players:
            ai_side = Player.RIGHT

        if debug:
            pong.register_observer(PongObserver(Player.LEFT))
        elif ai_enemy:
            dqn_agent = DQNAgent(eval_mode=True)
            pong_observer: PongObserver = PongObserver(observed_player=ai_side)
            pong.register_observer(pong_observer)
            pong_observer.register_observer(dqn_agent)
            pong_controller = RLPongController(dqn_agent)
            if swap_players:
                pong.set_right_agent(pong_controller)
            else:
                pong.set_left_agent(pong_controller)
            dqn_agent.load_model("model.ckpt")

        pong.reset()
        pong.run()

    finally:
        pong.quit()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    play(ai_enemy=True)
