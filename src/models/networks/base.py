import torch.nn as nn
# import torch.nn.functional as F
# import torch
from src.data.quality_control import interested_sites

class BaseNet(nn.Module):
    def __init__(self, k=3, hidden_size=128):
        super(BaseNet, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(716 * (4 ** k), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, len(interested_sites))
        self.droput = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.droput(x)
        x = self.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return x