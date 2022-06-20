import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self, k=3):
        super(BaseNet, self).__init__()
        self.k = k
        self.fc1 = nn.Linear(716 * (4 ** k), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 46)
        self.droput = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 716 * (4 ** self.k))
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return x