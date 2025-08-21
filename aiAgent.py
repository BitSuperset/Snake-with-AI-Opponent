import torch
from torch import nn,Tensor

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
            nn.Linear(hidden_shape,hidden_shape)
        )
        