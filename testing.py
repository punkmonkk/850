# -*- coding: utf-8 -*-
import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the game window
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pong")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define game constants
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
BALL_RADIUS = 10
FPS = 60

# Define game variables
paddle_speed = 5
ball_x_speed = 3
ball_y_speed = 3

# Create the paddles
player_paddle = pygame.Rect(0, height // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
opponent_paddle = pygame.Rect(width - PADDLE_WIDTH, height // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

# Create the ball
ball = pygame.Rect(width // 2 - BALL_RADIUS // 2, height // 2 - BALL_RADIUS // 2, BALL_RADIUS, BALL_RADIUS)

# Set up the game clock
clock = pygame.time.Clock()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the paddles
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and player_paddle.y > 0:
        player_paddle.y -= paddle_speed
    if keys[pygame.K_s] and player_paddle.y < height - PADDLE_HEIGHT:
        player_paddle.y += paddle_speed

    # Move the ball
    ball.x += ball_x_speed
    ball.y += ball_y_speed

    # Check collisions with paddles
    if ball.colliderect(player_paddle) or ball.colliderect(opponent_paddle):
        ball_x_speed *= -1

    # Check collisions with walls
    if ball.y <= 0 or ball.y >= height - BALL_RADIUS:
        ball_y_speed *= -1

    # Draw the game elements
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player_paddle)
    pygame.draw.rect(screen, WHITE, opponent_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (width // 2, 0), (width // 2, height))

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(FPS)

# Quit the game
pygame.quit()