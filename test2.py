from ursina import *

app = Ursina()

player = Entity(model='cube', color=color.orange, scale=(1,1,1))

def update():
    speed = 0.05
    if held_keys['w']:
        player.z -= speed
    if held_keys['s']:
        player.z += speed
    if held_keys['a']:
        player.x -= speed
    if held_keys['d']:
        player.x += speed

camera.position = (0, 5, -15)
camera.rotation_x = 30

app.run()
