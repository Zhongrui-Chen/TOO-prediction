import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(BaseNet, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.droput = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.droput(x)
        x = self.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return x