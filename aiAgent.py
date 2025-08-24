import torch
from torch import nn,Tensor
from game import Direction
import numpy as np
device='gpu' if torch.cuda.is_available() else 'cpu'
class SnakeAIModel(nn.Module):
    def __init__(self, input_shape,hidden_shape,out_shape):
        super().__init__()
        self.input=input_shape
        self.hidden_shape=hidden_shape
        self.out_shape=out_shape
        self.layer_stack=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=hidden_shape,kernel_size=1,stride=1,padding=1 ),
            nn.Linear(input_shape,hidden_shape),
            nn.Conv2d(in_channels=1,out_channels=hidden_shape,kernel_size=1,stride=1,padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(hidden_shape,hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape,hidden_shape),
            nn.Linear(hidden_shape,out_features=out_shape)
        )

    def forward(self,x:Tensor)->Tensor:
        return self.layer_stack(x)
    
class AIModel():
    def __init__(self):
        self.model=SnakeAIModel(1,10,3).to(device=device)
        self.optim=torch.optim.Adam(params=self.model.parameters(),lr=1e-3)
        self.lsfn=nn.L1Loss()
        self.gamma=0.8
    def response(self,matrix):
        matrix=Tensor(matrix).to(device=device)
        return self.model(matrix).argmax()
    
    def training(self,action:Tensor,state:Tensor,next_state:Tensor,reward:Tensor):
        if len(reward.shape)==0:
            state=state.unsqueeze(0)
            next_state=next_state.unsqueeze(0)
            reward=reward.unsqueeze(0)
        self.model.train()
        try:
            pred=self.model(state)
        except:
            print(f"State :{state.shape}")
        self.model.eval()
        target=pred.detach().clone()
        for i in range(len(state)):
            Q_new = reward[i]
            if reward[i]>0:
                Q_new=reward[i]+self.gamma*(self.model(next_state).max())
            target[i][action[i].argmax().item()]=Q_new
        self.model.train()
        self.optim.zero_grad()
        loss=self.lsfn(pred,target)
        print(f"loss{loss}")
        loss.backward()
        self.optim.step()
        self.model.eval()

    def shortTermMemory():
        pass
            