import torch.nn as nn
import torch

class ElementwiseLinear(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(ElementwiseLinear, self).__init__()

        # w is the learnable weight of this layer module
        self.w = nn.Parameter(torch.rand(input_size), requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # simple elementwise multiplication
        return self.w * x

class BaseNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(BaseNet, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(0.1)
        self.net = nn.Sequential(
            self.flatten,
            self.fc1, self.relu, self.dropout,
            self.fc2, self.relu, self.dropout,
            self.fc3
        )
        
    def forward(self, x):
        x = self.net(x)
        return x