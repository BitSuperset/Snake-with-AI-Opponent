from enum import Enum
import numpy as np
from collections import namedtuple
SPACESIZE=27
Point=namedtuple('Point','x,y')
import pygame
import random
from aiAgent import SnakeAIAgent
from directions import Direction,Flags
from typing import Tuple
from torch import Tensor

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN=(0,255,0)
GOLDEN=(255,215,0)
DARKBLUE=(0,0,139)
DARKPURPLE=(48,25,52)
PURPLE=(128,0,128)
BORDER_COLOR = (100, 100, 100)
THICKNESS = 22  # thickness of the border


class Objects(Enum):
     EMPTY=0
     BORDER=1
     NORMALFRUIT=2
     SNAKEPLAYER=3
     BONUSFRUIT=4
     POSIONFRUIT=5
     SNAKEAI=6
     SNAKEPLAYER2=7
     NORMALFRUIT2=8
     NORMALFRUIT3=9
     NORMALFRUIT4=10
     NORMALFRUIT5=11


class Arena():
    def __init__(self,width,height,hudSpace=0):
        self.width=width
        self.hudSpace=hudSpace
        self.height=height-hudSpace*SPACESIZE
        self.matrix=np.zeros(shape=(self.width//SPACESIZE,self.height//SPACESIZE))
        self.generate_border()

    
    def generate_border(self):
        """Generates boarder"""
        self.matrix[0,:]=Objects.BORDER.value
        self.matrix[-1,:]=Objects.BORDER.value
        self.matrix[:,0]=Objects.BORDER.value
        self.matrix[:,-1]=Objects.BORDER.value

    def updateArena(self,x,y,value):

        self.matrix[x][y]=value

    def printArena(self):
        """Displays the whole matrix. Testing purpose"""
        print(self.matrix)

    def returnObjectAtValue(self,x,y):
        """Returns the value in the matrix. Not the Objects class"""
        return self.matrix[x][y]
    
    def generateCoordRandomEmpty(self):
        """return the Screen coordinates x,y value for the empty space in arena"""
        emptyListX,emptyListY= np.where(self.matrix==Objects.EMPTY.value)
        if len(emptyListX)==0:
            return None
        idx=random.randint(0,len(emptyListX)-1)
        x,y=emptyListX[idx],emptyListY[idx]
        return x,y

    def getArena(self):
        """returns the arena matrix as numpy array"""
        return self.matrix
    
class Snake():
    def __init__(self,x,y,arena:Arena,direction=Direction.RIGHT,id=Objects.SNAKEPLAYER,hudSpace=0,headcolor=DARKBLUE,bodycolor=BLUE1):
        
        self.head=Point(x,y)
        self.direction=direction
        self.body=[]
        self.body.append((self.head,self.direction))
        self.id=id
        arena.updateArena(x,y,value=self.id.value)
        self.alive=True
        self.last_spawn=0
        self.last_tailpoint_for_bonus=()
        self.score=0
        self.hudSpace=hudSpace
        self.headColor=headcolor
        self.bodyColor=bodycolor


    def reset(self,arena:Arena):
        x,y=arena.generateCoordRandomEmpty()
        arena.updateArena(x,y,value=self.id.value)
        self.head=Point(x,y)
        for segment, _ in self.body:
            arena.updateArena(segment.x, segment.y, Objects.EMPTY.value)
        del self.body
        self.body=[]
        self.score=0
        # self.direction=Direction.RIGHT
        self.body.append((self.head,self.direction))
        self.alive=True
        
        
    def move(self,direction:Direction,arena:Arena)->Flags:
        """Takes the Direction Enum and return the Flag enum, and Arena"""
        self.direction=direction
        x=self.head.x
        y=self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        

        flags=self.obstacle_checker(obstacle=arena.returnObjectAtValue(x,y))
        if flags!=Flags.HitPLAYERSNAKE1 and flags!=Flags.HitItselfOrBorder and flags!=Flags.HitPLAYERSNAKE2 and flags!=Flags.HitAISNAKE:
            self.updateSnake(flags,x,y,arena=arena)
        else:
            self.alive=False
        return flags

    
    def obstacle_checker(self,obstacle:int)->Flags:
        """Return the Flag of event after the snake move"""
        if obstacle==Objects.BORDER.value or obstacle==self.id.value:
            return Flags.HitItselfOrBorder
        elif obstacle==Objects.NORMALFRUIT.value:
            return Flags.AteNormalFruit
        elif obstacle==Objects.NORMALFRUIT2.value:
            return Flags.AteNormalFruit2
        elif obstacle==Objects.NORMALFRUIT3.value:
            return Flags.AteNormalFruit3
        elif obstacle ==Objects.BONUSFRUIT.value:
            return Flags.AteBonusFruit
        elif obstacle==Objects.POSIONFRUIT.value:
            return Flags.AtePoisonusFruit
        elif obstacle==Objects.EMPTY.value:
            return Flags.EmptyGround
        elif obstacle==Objects.SNAKEPLAYER.value:
            return Flags.HitPLAYERSNAKE1
        elif obstacle==Objects.SNAKEPLAYER2.value:
            return Flags.HitPLAYERSNAKE2
        elif obstacle==Objects.SNAKEAI.value:
            return Flags.HitAISNAKE
        
        

    def updateSnake(self,flags:Flags,x,y,arena:Arena):
        """Update the snake body for movement"""
        self.head = Point(x, y)
        self.body.insert(0,(self.head,self.direction))
        if flags==Flags.EmptyGround or flags==Flags.AtePoisonusFruit:
            self.last_tailpoint_for_bonus=self.body.pop()
            temp,direction=self.last_tailpoint_for_bonus
            arena.updateArena(temp.x,temp.y,Objects.EMPTY.value)
            if flags==Flags.AtePoisonusFruit and len(self.body)!=1:
                self.last_tailpoint_for_bonus=self.body.pop()
                temp,direction=self.last_tailpoint_for_bonus
                arena.updateArena(temp.x,temp.y,Objects.EMPTY.value)
        elif flags== Flags.AteBonusFruit:
            self.body.append(self.last_tailpoint_for_bonus)
            temp,direction=self.last_tailpoint_for_bonus
            arena.updateArena(temp.x,temp.y,value=self.id.value)
        arena.updateArena(self.head.x,self.head.y,value=self.id.value)

    def updateUi(self,pygame:pygame,display:pygame.display):
        if self.alive:
            for i, (pt,direction) in enumerate(self.body):
                color = self.bodyColor if i > 0 else self.headColor   # head different
                pygame.draw.rect(display, color, pygame.Rect(pt.x*SPACESIZE, (pt.y+self.hudSpace)*SPACESIZE, SPACESIZE, SPACESIZE))

    def show_direction(self):
        """Returns the direction of snake. Used for when player has not clicked the input"""
        return self.direction
    
    def update_score(self,value):
        """Increment the score by value parameter"""
        self.score+=value
    
    def updateLastspawnTime(seld,value):
        seld.last_spawn=value
        

class Fruits():
    class BaseFruit:
        def __init__(self,arena:Arena,hudSpace=0):
            self.x,self.y=arena.generateCoordRandomEmpty()
            self.reward=10
            self.color=RED
            self.id=Objects.NORMALFRUIT
            self.alive=True
            self.last_spawned=0
            self.hudSpace=hudSpace
        def spawn(self,arena:Arena):
            self.x,self.y=arena.generateCoordRandomEmpty()
            self.alive=True
            arena.updateArena(self.x,self.y,value=self.id.value)
        def updateUi(self,pygame:pygame,display:pygame.display):
            if self.alive:
                pygame.draw.rect(display, self.color, pygame.Rect(self.x*SPACESIZE, (self.y+self.hudSpace)*SPACESIZE, SPACESIZE, SPACESIZE))

    class NormalFruit(BaseFruit):
        def __init__(self, arena:Arena,hudSpace,id=Objects.NORMALFRUIT):
            super().__init__(arena,hudSpace=hudSpace)
            self.id=id
            arena.updateArena(self.x,self.y,value=self.id.value)
    class BonusFruit(BaseFruit):
        def __init__(self, arena,hudSPace):
            super().__init__(arena,hudSpace=hudSPace)
            self.reward=50
            self.color=GOLDEN
            self.id=Objects.BONUSFRUIT
            self.alive=False

    class PosionusFruit(BaseFruit):
        def __init__(self, arena,hudSpace):
            super().__init__(arena,hudSpace=hudSpace)
            self.reward=-10
            self.color=GREEN
            self.id=Objects.POSIONFRUIT
            self.alive=False

class SnakeGameAITron():
    """Creates a Snake game class. Reqires pygames, collection and enum. Give height and width"""
    def __init__(self,width=None,height=None,hudSpace=50):
        self.pygame=pygame
        self.pygame.init()
        self.display=self.pygame.display
        self.hudSpace=hudSpace
        if width==None and height==None:
            self.width=self.display.Info().current_w
            self.height=self.display.Info().current_h
        else:
            self.width=width
            self.height=height
        self.display.set_caption("Snake Ai Game")
        self.display_mode=self.display.set_mode((self.width,self.height))
        self.clock=self.pygame.time.Clock()

        self.arena=Arena(self.width,self.height,hudSpace=self.hudSpace)
        self.player_snake=Snake(5,5,id=Objects.SNAKEPLAYER,arena=self.arena,hudSpace=self.hudSpace)
        self.normal_fruit=Fruits.NormalFruit(self.arena,hudSpace=self.hudSpace)
        self.normal_fruit2=Fruits.NormalFruit(self.arena,hudSpace=self.hudSpace,id=Objects.NORMALFRUIT2)
        self.normal_fruit3=Fruits.NormalFruit(self.arena,hudSpace=self.hudSpace,id=Objects.NORMALFRUIT3)
        self.posionFruit=Fruits.PosionusFruit(self.arena,self.hudSpace)
        self.bonusFruit=Fruits.BonusFruit(self.arena,self.hudSpace)

        self.AiSnake=Snake(15,15,arena=self.arena,id=Objects.SNAKEAI,hudSpace=self.hudSpace,headcolor=DARKPURPLE,bodycolor=PURPLE)
        self.agent=SnakeAIAgent()
        self.NotFirstTime=False
        self.oldreward=0
        self.oldaction=None
        self.oldMat=None
        self.oldFlag=None

        self.letAiTrain=False
    
        # self.ai_snake=Snake()

        self.SNAKESPEED = 60
        self.SPAWNINGOFPOSIONFRUIT=5000
        self.POSIONLIFE=5000

        self.SPAWININGBONUSFRUIT=5000
        self.BONUSLIFE=5000 


    def uiUpdate(self):
        """Update the games ui"""
        self.display_mode.fill(BLACK)
        self.pygame.draw.rect(self.display_mode, BORDER_COLOR, pygame.Rect(0, self.hudSpace*SPACESIZE, self.width, self.height-self.hudSpace*SPACESIZE), SPACESIZE)    
        self.player_snake.updateUi(pygame=self.pygame,display=self.display_mode)
        self.normal_fruit.updateUi(pygame=self.pygame,display=self.display_mode)
        self.normal_fruit2.updateUi(pygame=self.pygame,display=self.display_mode)
        self.normal_fruit3.updateUi(pygame=self.pygame,display=self.display_mode)
        self.posionFruit.updateUi(pygame=self.pygame,display=self.display_mode)
        self.bonusFruit.updateUi(pygame=self.pygame,display=self.display_mode)
        self.AiSnake.updateUi(pygame=self.pygame,display=self.display_mode)
        playerBluescore_text = self.pygame.font.Font('arial.ttf', 25).render("Player Blue Score: " + str(self.player_snake.score), True, WHITE)
        playerRedscore_text = self.pygame.font.Font('arial.ttf', 25).render("Player Purple Score: " + str(self.AiSnake.score), True, WHITE)
        self.display_mode.blit(playerBluescore_text, [10, 10])
        self.display_mode.blit(playerRedscore_text, [1000, 10])

    def SnakeLogicupdate(self,time,snake:Snake,direction:Direction=None)->Tuple[int,Tensor,Flags]:
        action=Tensor([0,1,0])
        reward=0
        flag=None
        if time-snake.last_spawn>=self.SNAKESPEED:
            if direction == Direction.NoDIR:
                direction=snake.show_direction()
            elif direction==Direction.AICALL:
                direction,action=self.AIsnakeInput(direct=self.get_input())
                self.letAiTrain=True
            flag=snake.move(direction=direction,arena=self.arena)
            reward=self.snake_flag_response(flag=flag,time=time,snake=snake)
            snake.updateLastspawnTime(time)
            
        return reward,action,flag

    def trainAI(self,oldMat,newMat,action,reward,flag):
        if flag==Flags.HitItselfOrBorder or flag==Flags.HitPLAYERSNAKE1 or flag==Flags.HitPLAYERSNAKE2:
            self.agent.train_long_memory()
        self.agent.shortTermMemory(oldMat,newMat,action=action,reward=reward)
    def PosionFruitUpdate(self,time):
        if time-self.posionFruit.last_spawned>self.SPAWNINGOFPOSIONFRUIT:
            if self.posionFruit.alive==False:
                self.posionFruit.spawn(self.arena)
                self.posionFruit.last_spawned = time 
            else:
                if(time-self.posionFruit.last_spawned)>self.POSIONLIFE:
                    self.posionFruit.alive=False
                    self.arena.updateArena(self.posionFruit.x,self.posionFruit.y,value=Objects.EMPTY.value)
                    self.posionFruit.last_spawned=time
                        
    def BonusFruitUpdate(self,time):
        if time-self.bonusFruit.last_spawned>self.SPAWININGBONUSFRUIT:
            if self.bonusFruit.alive==False:
                self.bonusFruit.spawn(self.arena)
                self.bonusFruit.last_spawned = time 
            else:
                if(time-self.bonusFruit.last_spawned)>self.BONUSLIFE:
                    self.bonusFruit.alive=False
                    self.arena.updateArena(self.bonusFruit.x,self.bonusFruit.y,value=Objects.EMPTY.value)
                    self.bonusFruit.last_spawned=time

    
    
    def snake_flag_response(self,flag:Flags,time,snake:Snake)->int:
        reward=0
        if flag==Flags.AteNormalFruit:
            snake.update_score(self.normal_fruit.reward)
            reward=self.normal_fruit.reward
            self.normal_fruit.spawn(arena=self.arena)
        elif flag==Flags.AteNormalFruit2:
            snake.update_score(self.normal_fruit.reward)
            reward=self.normal_fruit2.reward
            self.normal_fruit2.spawn(arena=self.arena)
        elif flag==Flags.AteNormalFruit3:
            snake.update_score(self.normal_fruit.reward)
            reward=self.normal_fruit3.reward
            self.normal_fruit3.spawn(arena=self.arena)
        elif flag==Flags.HitItselfOrBorder:
            reward=-100
            snake.reset(arena=self.arena)
        elif flag==Flags.AtePoisonusFruit:
            reward=self.posionFruit.reward
            snake.update_score(self.posionFruit.reward)
            self.posionFruit.alive=False
            self.posionFruit.last_spawned=time
        elif flag==Flags.AteBonusFruit:
            reward=self.bonusFruit.reward
            snake.update_score(self.normal_fruit.reward)
            self.bonusFruit.alive=False
            self.bonusFruit.last_spawned=time
        elif flag==Flags.HitPLAYERSNAKE1:
            reward=-150
            snake.reset(arena=self.arena)
            self.player_snake.update_score(80)
        elif flag==Flags.HitAISNAKE:
            reward=-150
            snake.reset(arena=self.arena)
            self.AiSnake.update_score(80)
        return reward


    def get_input(self)->Direction:
        keys = self.pygame.key.get_pressed()

        if keys[self.pygame.K_w] and self.player_snake.show_direction()!=Direction.DOWN:
            return Direction.UP
        elif keys[self.pygame.K_d] and self.player_snake.show_direction()!=Direction.LEFT:
            return Direction.RIGHT
        elif keys[self.pygame.K_s]and self.player_snake.show_direction()!=Direction.UP:
            return Direction.DOWN
        elif keys[self.pygame.K_a]and self.player_snake.show_direction()!=Direction.RIGHT:
            return Direction.LEFT
        elif keys[self.pygame.K_ESCAPE]:
            return Direction.NoDIR
            pass
        elif keys[self.pygame.K_q]:
            self.pygame.quit()
            exit()
        else:
            return Direction.NoDIR
        
    def get_inputforPlayer2(self)->Direction:
        keys = self.pygame.key.get_pressed()
        if keys[self.pygame.K_UP] and self.player_snake.show_direction()!=Direction.DOWN:
            return Direction.UP
        elif keys[self.pygame.K_RIGHT] and self.player_snake.show_direction()!=Direction.LEFT:
            return Direction.RIGHT
        elif keys[self.pygame.K_DOWN]and self.player_snake.show_direction()!=Direction.UP:
            return Direction.DOWN
        elif keys[self.pygame.K_LEFT]and self.player_snake.show_direction()!=Direction.RIGHT:
            return Direction.LEFT
        elif keys[self.pygame.K_ESCAPE]:
            return Direction.NoDIR
        
        elif keys[self.pygame.K_q]:
            print("working")
            self.agent.dumpHistory()
            self.pygame.quit()
            exit()
        else:
            return Direction.NoDIR
    
    def AIsnakeInput(self,direct:Direction)->Tuple[Direction,Tensor]:
        return self.agent.response(self.arena.getArena(),self.AiSnake.show_direction(),direct=direct)

    
    def Sync(self,time)->int:
        # self.SnakeLogicupdate(time,snake=self.player_snake,direction=self.get_input())
        if self.NotFirstTime and self.letAiTrain:
            self.trainAI(oldMat=self.oldMat,newMat=self.arena.getArena(),action=self.oldaction,reward=self.oldreward,flag=self.oldFlag)
            self.letAiTrain=False
            pass
        self.oldMat=self.arena.getArena()
        self.oldreward,self.oldaction,self.oldFlag=self.SnakeLogicupdate(time,snake=self.AiSnake,direction=Direction.AICALL)
        self.PosionFruitUpdate(time)
        self.BonusFruitUpdate(time=time)
        self.uiUpdate()
        self.NotFirstTime=True

game=SnakeGameAITron(hudSpace=2)
last_time=game.pygame.time.get_ticks()
while True:
    for event in game.pygame.event.get():
        if event.type == game.pygame.QUIT:
            game.pygame.quit()
            exit()
    last_time=game.Sync(game.pygame.time.get_ticks())
    game.display.flip()
    game.clock.tick(80)
