import pygame
import sys

# Initialize pygame
pygame.init()

# Create window
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Update UI Example")

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Example state
x = 50
y = 50
speed = 5

# Game loop (like Tkinter's after loop)
running = True
while running:
    # 1. Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 2. Update game state
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        x += speed
    if keys[pygame.K_LEFT]:
        x -= speed
    if keys[pygame.K_DOWN]:
        y += speed
    if keys[pygame.K_UP]:
        y -= speed

    # 3. Redraw
    screen.fill((0, 0, 0))  # clear screen
    pygame.draw.rect(screen, (0, 255, 0), (x, y, 50, 50))  # draw player
    pygame.display.flip()  # update display

    # Control frame rate (like scheduling updates)
    clock.tick(60)  # 60 FPS

pygame.quit()
sys.exit()
