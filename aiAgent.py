import torch
from torch import nn,Tensor
from directions import Direction
import numpy as np
import os
import pickle
from collections import deque
import random
from typing import Tuple
import atexit
device='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=1000
HISTORY_SIZE=10000


class SnakeAIModel(nn.Module):
    def __init__(self, input_shape,hidden_shape,out_shape):
        super().__init__()
        self.input=input_shape
        self.hidden_shape=hidden_shape
        self.out_shape=out_shape
        self.layer_stack=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=hidden_shape,kernel_size=3,stride=1,padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(hidden_shape*420,40),
            nn.LayerNorm(40),
            nn.Linear(40,hidden_shape),
            nn.Linear(hidden_shape,out_features=3)
        )

    def forward(self,x:Tensor)->Tensor:
        return self.layer_stack(x)




class SnakeAIAgent():
    def __init__(self):
        self.model=SnakeAIModel(input_shape=1,
                                hidden_shape=10,
                                out_shape=3).to(device=device)
        if os.path.exists('model.pth'):
            self.model.load_state_dict(torch.load('model.pth'))
        self.optim=torch.optim.Adam(params=self.model.parameters(),lr=1e-3)
        self.lsfn=nn.MSELoss()
        self.gamma=0.8
        self.oldMatrix=None
        self.highscore=0
        self.history=deque(maxlen=HISTORY_SIZE)
        self.num_action=0
        self.debug=0

        if os.path.getsize("memory.pkl")>0:
            with open("memory.pkl",mode="rb") as f:
                self.history,self.highscore=pickle.load(f)
                self.num_action=len(self.history)
        atexit.register(self.dumpHistory)
    
    def response(self,matrix,last_dir,direct:Direction)->Tuple[Direction,Tensor]:
        matrix=Tensor(matrix).to(device=device).unsqueeze(0).unsqueeze(0)
        final_move=[0,0,0]
        self.epsilon = max(0.01, 1.0 - (self.num_action * 0.0007))
        if random.random()<self.epsilon:
            if direct==Direction.NoDIR:
                direct=last_dir
            move=self.reverse_translate(last_dir=last_dir,new_dir=direct)
        else:
            prediction=self.model(matrix)
            move=torch.argmax(prediction).item()
        final_move[move]=1
        return self.translate(last_dir,action=final_move),final_move
    
    def translate(self,direction,action)->Direction:
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        return new_dir

    def reverse_translate(self, last_dir: Direction, new_dir: Direction)->int:
        """
        Given last direction and new direction, return the action array.
        [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(last_dir)
        new_idx = clock_wise.index(new_dir)

        # no change
        if new_idx == idx:
            return 0
        # right turn
        elif new_idx == (idx + 1) % 4:
            return 1
        # left turn
        elif new_idx == (idx - 1) % 4:
            return 2
        else:
            return 0
        
    def training(self, actions: Tensor, states: Tensor, next_states: Tensor, rewards: Tensor, dones: Tensor = None):
        """
        actions:     (B, A)  one-hot (or float mask) OR can adapt if using indices
        states:      (B, C, H, W)
        next_states: (B, C, H, W)
        rewards:     (B,)    float
        dones:       (B,)    bool (True if episode terminated after that step)
        """
        pred_q = self.model(states)
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]  

        # 3) If dones not provided, assume none are terminal
        if dones is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool)
        target_q = rewards + self.gamma * next_q * (~dones)

        # 5) Extract the Q-value predicted for the ACTION actually taken
        #    If actions are one-hot (B, A), elementwise multiply and sum across actions -> (B,)
        chosen_q = torch.sum(pred_q * actions, dim=1)
        loss = self.lsfn(chosen_q, target_q)  

        # 7) Standard backward + optimizer step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 8) Return scalar loss for logging
        return loss.item()

    def shortTermMemory(self,state,next_state,action,reward):
        if reward>self.highscore:
            self.highscore=reward
            torch.save(self.model.state_dict(),'model.pth')
        self.num_action+=1
        action = torch.as_tensor(action, dtype=torch.long, device=device)
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        self.history.append((state.unsqueeze(0),next_state.unsqueeze(0),action,reward))
        self.training(action=action.unsqueeze(0),state=state.unsqueeze(0).unsqueeze(0),next_state=next_state.unsqueeze(0).unsqueeze(0),reward=reward.unsqueeze(0))
        if len(self.history) < HISTORY_SIZE and self.num_action % BATCH_SIZE == 0:
            print("cond1")
            self.train_long_memory()
        elif len(self.history) == HISTORY_SIZE and self.num_action % BATCH_SIZE == 0:
            print("cond2",self.num_game)
            self.train_long_memory()

    def dumpHistory(self):
        """Dumps the history to a generated pkl file"""
        print("Dumped")
        torch.save(self.model.state_dict(),'model.pth')
        with open('memory.pkl', "wb") as f:
            pickle.dump((self.history,self.highscore), f)

    def train_long_memory(self):
            if len(self.history)>=BATCH_SIZE:
                self.debug=0
                mini_sample =random.sample(self.history,BATCH_SIZE)
                states,next_states,actions,rewards =  zip(*mini_sample)
                actions=torch.stack(actions).to(device)
                next_states=torch.stack(next_states).to(device)
                states=torch.stack(states).to(device)
                rewards=torch.stack(rewards).to(device)
                self.training(action=actions,state=states,next_state=next_states,reward=rewards)
    
    def incrementNumAction(self):
        self.num_action+=1
            