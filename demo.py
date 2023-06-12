import pygame
from pygame.locals import *

def run():
    # Initialize Pygame
    pygame.init()

    # Set up the game window
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")

    # Define colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # Set up the ball's initial position and velocity
    ball_x = width // 2
    ball_y = height // 2
    ball_dx = 3
    ball_dy = 3

    # Set up the paddles' initial positions and dimensions
    paddle_width = 10
    paddle_height = 60
    paddle_speed = 5
    left_paddle_x = 0
    left_paddle_y = height // 2 - paddle_height // 2
    right_paddle_x = width - paddle_width
    right_paddle_y = height // 2 - paddle_height // 2

    # Set up the score counters
    left_score = 0
    right_score = 0
    font = pygame.font.Font(None, 36)

    # Game loop
    running = True
    clock = pygame.time.Clock()
    game_over = False
    close_timer = 0

    while running:
        clock.tick(60)  # Limit the frame rate to 60 FPS

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # Move the paddles up and down
        if not game_over:
            keys = pygame.key.get_pressed()

            if keys[K_w] and left_paddle_y > 0:
                left_paddle_y -= paddle_speed
            if keys[K_s] and left_paddle_y < height - paddle_height:
                left_paddle_y += paddle_speed
            if keys[K_UP] and right_paddle_y > 0:
                right_paddle_y -= paddle_speed
            if keys[K_DOWN] and right_paddle_y < height - paddle_height:
                right_paddle_y += paddle_speed

            # Update the ball's position
            ball_x += ball_dx
            ball_y += ball_dy

            # Check for collision with paddles
            if ball_x <= left_paddle_x + paddle_width and left_paddle_y <= ball_y <= left_paddle_y + paddle_height:
                ball_dx = abs(ball_dx)
            if ball_x >= right_paddle_x - paddle_width and right_paddle_y <= ball_y <= right_paddle_y + paddle_height:
                ball_dx = -abs(ball_dx)

            # Check for collision with walls
            if ball_y <= 0 or ball_y >= height:
                ball_dy = -ball_dy

            # Check if the ball went off the screen
            if ball_x < 0:
                right_score += 1
                ball_x, ball_y = width // 2, height // 2
            elif ball_x > width:
                left_score += 1
                ball_x, ball_y = width // 2, height // 2

            # Check for winning condition
            if left_score >= 5 or right_score >= 5:
                game_over = True

        # Clear the screen
        screen.fill(BLACK)

        if not game_over:
            # Draw the paddles, ball, and scores
            pygame.draw.rect(screen, WHITE, (left_paddle_x, left_paddle_y, paddle_width, paddle_height))
            pygame.draw.rect(screen, WHITE, (right_paddle_x, right_paddle_y, paddle_width, paddle_height))
            pygame.draw.circle(screen, WHITE, (ball_x, ball_y), 10)
            left_score_text = font.render(str(left_score), True, WHITE)
            right_score_text = font.render(str(right_score), True, WHITE)
            screen.blit(left_score_text, (width // 4 - left_score_text.get_width() // 2, 10))
            screen.blit(right_score_text, (width // 4 * 3 - right_score_text.get_width() // 2, 10))
        else:
            # Display winning message
            if left_score >= 5:
                win_text = font.render("Player 1 Wins!", True, WHITE)
            else:
                win_text = font.render("Player 2 Wins!", True, WHITE)
            screen.blit(win_text, (width // 2 - win_text.get_width() // 2, height // 2 - win_text.get_height() // 2))

            if close_timer == 0:
                close_timer = 5 * 60  # 5 seconds (60 FPS)

            close_timer -= 1

            if close_timer == 0:
                # Close the game window
                pygame.quit()
                return


        # Update the display
        pygame.display.flip()

if __name__ == '__main__':
    run()