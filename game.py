from enum import Enum
import numpy as np
from collections import namedtuple
SPACESIZE= 25
Point=namedtuple('Point','x,y')
import pygame
import random


WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

SPEED = 40


class Direction(Enum):
        RIGHT=1
        LEFT=2
        UP=3
        DOWN=4
        NoDIR=5

class Objects(Enum):
     EMPTY=0
     BORDER=1
     NORMALFRUIT=2
     SNAKEPLAYER=3
     BONUSFRUIT=4
     POSIONFRUIT=5
     SNAKEAI=6
     SNAKEPLAYER2=7

class Flags(Enum):
    EmptyGround=0
    HitItselfOrBorder=1
    HitAnotherSnake=2
    AteNormalFruit=4
    AteBonusFruit=5
    AtePoisonusFruit=6

class Arena():
    def __init__(self,width,height):
        self.width=width
        self.height=height
        self.matrix=np.zeros(shape=(height//SPACESIZE,width//SPACESIZE))
        self.generate_border()
        

    
    def generate_border(self):
        """Generates boarder"""
        self.matrix[0,:]=1
        self.matrix[-1,:]=1
        self.matrix[:,0]=1
        self.matrix[:,-1]=1

    def translateCoords(self,x,y):
        x//=SPACESIZE
        y//=SPACESIZE
        return x,y

    def updateArena(self,x,y,value,spacecoords=True):
        if spacecoords:
            x,y=self.translateCoords(x,y)
        self.matrix[x][y]=value

    def printArena(self):
        """Displays the whole matrix. Testing purpose"""
        print(self.matrix)

    def returnObjectAtValue(self,x,y):
        """Returns the value in the matrix. Not the Objects class"""
        x,y=self.translateCoords(x,y)
        return self.matrix[x][y]
    
    def generateCoordRandomEmpty(self):
        """return the Screen coordinates x,y value for the empty space in arena"""
        emptyListX,emptyListY= np.where(self.matrix==Objects.EMPTY.value)
        if len(emptyListX)==0:
            return None
        idx=random.randint(0,len(emptyListX)-1)
        x,y=emptyListX[idx]*SPACESIZE,emptyListY[idx]*SPACESIZE
        return x,y
    
class Snake():
    def __init__(self,x,y,direction=Direction.RIGHT,id=Objects.SNAKEPLAYER):
        
        self.head=Point(x,y)
        self.original_point=Point(x,y)
        self.direction=direction
        self.body=[]
        self.body.append((self.head,self.direction))
        self.id=id
        # self.border=

    def reset(self):
        self.head=self.original_point
        self.bodyLength=1
        
    def move(self,direction:Direction,arena:Arena):
        """Takes the Direction Enum and return the Flag enum, and Arena"""
        self.direction=direction
        x=self.head.x
        y=self.head.y
        if self.direction == Direction.RIGHT:
            x += SPACESIZE
        elif self.direction == Direction.LEFT:
            x -= SPACESIZE
        elif self.direction == Direction.DOWN:
            y += SPACESIZE
        elif self.direction == Direction.UP:
            y -= SPACESIZE
        
        print(x,y)
        flags=self.obstacle_checker(obstacle=arena.returnObjectAtValue(x,y))
        if flags!=Flags.HitAnotherSnake and flags!=Flags.HitItselfOrBorder:
            self.updateSnake(flags,x,y,arena=arena)
        return flags

    
    def obstacle_checker(self,obstacle:int)->Flags:
        """Return the Flag of event after the snake move"""
        if obstacle==Objects.BORDER.value or obstacle==self.id.value:
            return Flags.HitItselfOrBorder
        elif obstacle==Objects.NORMALFRUIT.value:
            return Flags.AteNormalFruit
        elif obstacle ==Objects.BONUSFRUIT.value:
            return Flags.AteBonusFruit
        elif obstacle==Objects.POSIONFRUIT.value:
            return Flags.AtePoisonusFruit
        elif obstacle==Objects.EMPTY.value:
            return Flags.EmptyGround
        else:
            return Flags.HitAnotherSnake

    def updateSnake(self,flags:Flags,x,y,arena:Arena):
        """Update the snake body for movement"""
        self.head = Point(x, y)
        self.body.insert(0,(self.head,self.direction))
        if flags==Flags.EmptyGround or flags==Flags.AtePoisonusFruit:
            temp,direction=self.body.pop()
            arena.updateArena(temp.x,temp.y,Objects.EMPTY.value)
        arena.updateArena(self.head.x,self.head.y,value=self.id.value)

    def updateUi(self,pygame:pygame,display:pygame.display):
        for i, (pt,direction) in enumerate(self.body):
            color = BLUE1 if i > 0 else RED   # head different
            pygame.draw.rect(display, color, pygame.Rect(pt.x, pt.y, SPACESIZE, SPACESIZE))

    def show_direction(self):
        """Returns the direction of snake. Used for when player has not clicked the input"""
        return self.direction
    
        

class Fruits():
    class BaseFruit:
        def __init__(self,arena:Arena):
            self.x,self.y=arena.generateCoordRandomEmpty()
            self.reward=10
            self.color=RED
        def reset(self,arena:Arena):
            self.x,self.y=arena.generateCoordRandomEmpty()
        def updateUi(self,pygame:pygame,display:pygame.display):
            pygame.draw.rect(display, self.color, pygame.Rect(self.x, self.y, SPACESIZE, SPACESIZE))
    class NormalFruit(BaseFruit):
        def __init__(self, arena:Arena):
            super().__init__(arena)
    class BonusFruit():
        pass
    class PosionusFruit():
        pass

class SnakeGameAITron():
    """Creates a Snake game class. Reqires pygames, collection and enum. Give height and width"""
    def __init__(self,width,height):
        import pygame
        from enum import Enum
        self.pygame=pygame
        self.pygame.init()
        self.width=width
        self.height=height
        self.pygame.display.set_caption("Snake Ai Game")
        self.display=self.pygame.display
        self.display_mode=self.display.set_mode((self.height,self.width))
        self.clock=self.pygame.time.Clock()

        self.arena=Arena(self.width,self.height)
        self.player_snake=Snake(50,50,id=Objects.SNAKEPLAYER)
        # self.ai_snake=Snake()
    
    def Sync(self):
        direction= self.get_input()
        if direction == Direction.NoDIR:
            direction=self.player_snake.show_direction()
        print(direction)
        self.player_snake.move(direction=direction,arena=self.arena)
        self.player_snake.updateUi(pygame=self.pygame,display=self.display_mode)
        
        

    def get_input(self)->Direction:
        keys = self.pygame.key.get_pressed()
        if keys[self.pygame.K_w]:
            return Direction.UP
        elif keys[self.pygame.K_d]:
            return Direction.RIGHT
        elif keys[self.pygame.K_s]:
            return Direction.DOWN
        elif keys[self.pygame.K_a]:
            return Direction.LEFT
        elif keys[self.pygame.K_ESCAPE]:
            pass
        else:
            return Direction.NoDIR
        

game=SnakeGameAITron(width=500,height=1000)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    game.display_mode.fill(BLACK)
    game.Sync()
    game.display.flip()
    game.clock.tick(10)
