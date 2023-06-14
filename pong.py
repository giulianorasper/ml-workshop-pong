import copy
import math
import random
import sys
import time
from multiprocessing import Process
from threading import Thread

import numpy as np
import pygame
from PIL import Image
from pygame.locals import *
from observer_pattern import Observable
from pong_controller import PongController, KeyboardPongController
from pong_action import PongAction

width, height = 640, 480

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)  #


class Pong(Observable):

    def __init__(self, left_agent=None, right_agent=None, left_invincible=False, right_invincible=False,
                 ticks_per_second=60, graphical=True, win_screen_time_in_ticks=60):
        super().__init__()

        # set the graphical flag
        self.graphical = graphical

        # set the win screen time in ticks
        self.win_screen_time_in_ticks = win_screen_time_in_ticks

        # set the pong controllers
        self.left_agent: PongController = left_agent
        self.right_agent: PongController = right_agent

        # set the invincibility flags -> not trained agent is perfect player
        self.left_invincible = left_invincible
        self.right_invincible = right_invincible

        self.ticks_per_second = ticks_per_second

        # meta knowledge about the game
        self.ball_touches_left = None
        self.ball_touches_right = None
        self.resets = 0
        self.ticks_this_game = None
        self.ticks_this_ball_exchange = None

        # if no pong controllers are given, fallback to KeyboardController
        if self.left_agent is None:
            self.left_agent = KeyboardPongController.left_controller()
        if self.right_agent is None:
            self.right_agent = KeyboardPongController.right_controller()

        # declare position and velocity variables
        self.ball_x: int = None
        self.ball_y: int = None
        self.ball_dx: int = None
        self.ball_dy: int = None

        # declare paddles positions and dimensions variables
        self.paddle_width: int = None
        self.paddle_height: int = None
        self.paddle_speed: int = None
        self.left_paddle_x: int = None
        self.left_paddle_y: int = None
        self.right_paddle_x: int = None
        self.right_paddle_y: int = None

        # Declare the score counters
        self.left_score: int = None
        self.right_score: int = None

        # Game loop variable declarations
        self.running: bool = None
        self.clock: pygame.time.Clock = None
        self.game_over: bool = None
        # controlling the window closing after a short time period after a player wins
        self.close_timer: int = None

        # Initialize Pygame
        pygame.init()
        self.font = pygame.font.Font(None, 36)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pong")

        # log the last action taken
        self.last_action_taken_left = PongAction(False, False)
        self.last_action_taken_right = PongAction(False, False)

        # concurrency control
        self.process: Thread = None

    def set_left_agent(self, agent: PongController):
        self.left_agent = agent

    def set_right_agent(self, agent: PongController):
        self.right_agent = agent

    def reset(self):
        # we do not touch anything until the current episode is finished

        # reset meta knowledge about the game
        self.ball_touches_left = 0
        self.ball_touches_right = 0
        self.resets += 1
        self.ticks_this_game = 0
        self.ticks_this_ball_exchange = 0

        # Set up the ball's initial position and velocity
        self._reset_ball()

        # Set up the paddles' initial positions and dimensions
        self.paddle_width = 10
        self.paddle_height = 100
        self.paddle_speed = 5
        self.left_paddle_x = 0
        self.left_paddle_y = height // 2 - self.paddle_height // 2
        self.right_paddle_x = width - self.paddle_width
        self.right_paddle_y = height // 2 - self.paddle_height // 2

        # Set up the score counters
        self.left_score = 0
        self.right_score = 0

        # Game loop variable initialization
        self.running = True
        self.clock = pygame.time.Clock()
        self.game_over = False
        self.close_timer = self.win_screen_time_in_ticks

        # start the game
        # self.process = Process(target=self.run)
        # self.process.start()
        # self.process = Thread(target=self.run)
        # self.process.start()
        self.notify_observers(copy.copy(self))
        return copy.copy(self)

    def quit(self):
        self.running = False
        pygame.quit()

    def transform_board(self):
        # reset meta knowledge about the game
        self.ball_touches_left, self.ball_touches_right = self.ball_touches_right, self.ball_touches_left

        # Set up the ball's initial position and velocity
        self.ball_x, self.ball_y = width - self.ball_x, height - self.ball_y
        self.ball_dx, self.ball_dy = -self.ball_dx, -self.ball_dy

        self.left_paddle_x, self.right_paddle_x = self.right_paddle_x, self.left_paddle_x
        self.left_paddle_y, self.right_paddle_y = self.right_paddle_y, self.left_paddle_y

        # Set up the score counters
        self.left_score, self.right_score = self.right_score, self.left_score

    def run(self):
        while self.running:
            # print(f"ball_x: {self.ball_x}, ball_y: {self.ball_y}, ball_dx: {self.ball_dx}, ball_dy: {self.ball_dy}")
            if self.game_over:
                self.notify_observers(None)
            else:
                self.notify_observers(copy.copy(self))

            self.clock.tick(self.ticks_per_second)  # Limit the frame rate to 60 FPS
            self.ticks_this_game += 1
            self.ticks_this_ball_exchange += 1

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

            # Move the paddles up and down
            if not self.game_over:

                left_up, left_down = self.left_agent.get_pong_action()
                right_up, right_down = self.right_agent.get_pong_action()
                self.last_action_taken_left = PongAction(left_up, left_down)
                self.last_action_taken_right = PongAction(right_up, right_down)

                if left_up and self.left_paddle_y > 0:
                    self.left_paddle_y -= self.paddle_speed
                if left_down and self.left_paddle_y < height - self.paddle_height:
                    self.left_paddle_y += self.paddle_speed
                if right_up and self.right_paddle_y > 0:
                    self.right_paddle_y -= self.paddle_speed
                if right_down and self.right_paddle_y < height - self.paddle_height:
                    self.right_paddle_y += self.paddle_speed

                # Update the ball's position
                self.ball_x += self.ball_dx
                self.ball_y += self.ball_dy

                # Check for collision with paddles
                if self.ball_x <= self.left_paddle_x + self.paddle_width:
                    if self.left_paddle_y <= self.ball_y <= self.left_paddle_y + self.paddle_height or self.left_invincible:
                        self.ball_dx = abs(self.ball_dx)
                        self.ball_touches_left += 1
                        self.alter_velocity()
                if self.ball_x >= self.right_paddle_x - self.paddle_width:
                    if self.right_paddle_y <= self.ball_y <= self.right_paddle_y + self.paddle_height or self.right_invincible:
                        self.ball_dx = -abs(self.ball_dx)
                        self.ball_touches_right += 1
                        self.alter_velocity()

                # Check for collision with walls
                if self.ball_y <= 0 or self.ball_y >= height:
                    self.ball_dy = -self.ball_dy

                # Check if the ball went off the screen
                if self.ball_x < 0:
                    self.right_score += 1
                    self._reset_ball()
                    self.ticks_this_ball_exchange = 0
                elif self.ball_x > width:
                    self.left_score += 1
                    self._reset_ball()
                    self.ticks_this_ball_exchange = 0

                # Check for winning condition
                if self.left_score >= 5 or self.right_score >= 5:
                    self.game_over = True

            if self.graphical:
                self.refresh_display()

            if self.game_over:
                self.close_timer -= 1

                if self.close_timer <= 0:
                    self.running = False
                    # Close the game window
                    # print("Game Over")
                    return

    def refresh_display(self):
        # Clear the screen
        self.screen.fill(BLACK)

        if not self.game_over:
            # Draw the paddles, ball, and scores
            pygame.draw.rect(self.screen, WHITE,
                             (self.left_paddle_x, self.left_paddle_y, self.paddle_width, self.paddle_height))
            pygame.draw.rect(self.screen, WHITE,
                             (self.right_paddle_x, self.right_paddle_y, self.paddle_width, self.paddle_height))
            pygame.draw.circle(self.screen, WHITE, (self.ball_x, self.ball_y), 10)
            left_score_text = self.font.render(str(self.left_score), True, WHITE)
            right_score_text = self.font.render(str(self.right_score), True, WHITE)
            self.screen.blit(left_score_text, (width // 4 - left_score_text.get_width() // 2, 10))
            self.screen.blit(right_score_text, (width // 4 * 3 - right_score_text.get_width() // 2, 10))
        else:
            # Display winning message
            if self.left_score >= 5:
                win_text = self.font.render("Player 1 Wins!", True, WHITE)
            else:
                win_text = self.font.render("Player 2 Wins!", True, WHITE)
            self.screen.blit(win_text,
                             (width // 2 - win_text.get_width() // 2, height // 2 - win_text.get_height() // 2))

        # Update the display
        pygame.display.flip()

    def _reset_ball(self):
        self.ball_dx, self.ball_dy = 3, 3
        self.ball_x, self.ball_y = width // 2, height // 2
        dx, dy = self.ball_dx, self.ball_dy

        # ball flies to the right
        range1 = (-math.pi / 4, math.pi / 4)
        # ball flies to the left
        range2 = (3 * math.pi / 4, 5 * math.pi / 4)

        if random.choice([True, False]):
            radians = random.uniform(range2[0], range2[1])
        else:
            radians = random.uniform(range1[0], range1[1])

        new_dx = dx * math.cos(radians) + dy * math.cos(radians)
        new_dy = dx * math.sin(radians) + dy * math.cos(radians)
        self.ball_dx = new_dx
        self.ball_dy = new_dy

    def alter_velocity(self):
        flies_to_right = self.ball_dx > 0
        self.ball_dx, self.ball_dy = 3, 3

        dx, dy = self.ball_dx, self.ball_dy

        # ball flies to the right
        range1 = (-math.pi / 4, math.pi / 4)
        # ball flies to the left
        range2 = (3 * math.pi / 4, 5 * math.pi / 4)

        if flies_to_right:
            radians = random.uniform(range1[0], range1[1])
        else:
            radians = random.uniform(range2[0], range2[1])

        new_dx = dx * math.cos(radians) + dy * math.cos(radians)
        new_dy = dx * math.sin(radians) + dy * math.cos(radians)
        self.ball_dx = new_dx
        self.ball_dy = new_dy

    def copy(self):
        return copy.deepcopy(self)

    def get_image(self):
        # Take a screenshot of the screen surface
        screenshot = pygame.surfarray.array3d(self.screen)

        # Convert the screenshot to a PIL image
        pil_image = Image.fromarray(np.uint8(screenshot))

        # Rotate the image by 90 degrees clockwise
        pil_image = pil_image.rotate(360 - 90, expand=True)
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Display the PIL image
        return pil_image


def do_reset(game):
    time.sleep(5)
    game.reset()


if __name__ == '__main__':
    game = Pong()
    game.reset()
    game.run()
