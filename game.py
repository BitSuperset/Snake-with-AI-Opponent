from enum import Enum
import numpy as np
class Direction(Enum):
        Right=1
        Left=2
        Up=3
        Down=4

class Objects(Enum):
     Border=1
     Empty=0
     Fruit=2
     Snake=3

class Snake():
    def __init__(self,x,y,direction=Direction.Right):
        from collections import namedtuple
        Point=namedtuple('Point','x,y')
        self.head=Point(x,y)
        self.original_point=Point(x,y)
        self.direction=direction
        # self.border=
    def reset(self):
        self.head=self.original_point
    def move(self,direction):
         pass
    def show_direction(self):
         return self.direction
        

class Arena():
    def __init__(self,width,height):
        self.width=width
        self.height=height
        self.matrix=np.zeros(shape=(width,height))
        self.generate_border()

    
    def generate_border(self):
        """Generates boarder"""
        self.matrix[0,:]=1
        self.matrix[-1,:]=1
        self.matrix[:,0]=1
        self.matrix[:,-1]=1

    def printArena(self):
        """Displays the whole matrix. Testing purpose"""
        print(self.matrix)


        

class SnakeGameAITron():
    """Creates a Snake game class. Reqires pygames, collection and enum. Give height and width"""
    def __init__(self,width,height):
        import pygame
        from enum import Enum
        self.pygame=pygame
        self.pygame.init()
        self.pygame.display.set_caption("Snake Ai Game")
        self.display=self.pygame.display.set_mode((self.width,self.height))
        self.clock=self.pygame.time.Clock()

        self.width=width
        self.height=height
        self.player_snake=Snake()
        self.ai_snake=Snake()

mat=Arena(20,20)
mat.printArena()