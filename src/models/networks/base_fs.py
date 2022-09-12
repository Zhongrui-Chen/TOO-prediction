import torch.nn as nn
import torch
from src.models.networks.autoencoder import Autoencoder

# class ElementwiseLinear(nn.Module):
#     def __init__(self, input_size: int) -> None:
#         super(ElementwiseLinear, self).__init__()

#         # w is the learnable weight of this layer module
#         self.w = nn.Parameter(torch.rand(input_size), requires_grad=True)

#     def forward(self, x: torch.tensor) -> torch.tensor:
#         # simple elementwise multiplication
#         return self.w * x

class BaseNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc21 = nn.Linear(hidden_size, hidden_size)
        # self.fc22 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(0.5)
        self.net = nn.Sequential(
            self.flatten,
            self.fc1, self.relu, self.dropout,
            self.fc2, self.relu, self.dropout,
            # self.fc21, self.relu, self.dropout,
            # self.fc22, self.relu, self.dropout,
            self.fc3
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

class EncoderNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, autoencoder):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.autoencoder = autoencoder

        # self.enc_weights = []

        # self.enc_weights.append((autoencoder_state_dict['enc1.weight'], autoencoder_state_dict['enc1.bias']))
        # self.enc_weights.append((autoencoder_state_dict['enc2.weight'], autoencoder_state_dict['enc2.bias']))

        self.dropout = nn.Dropout(0.5)
        self.net = nn.Sequential(
            self.fc1, self.relu, self.dropout,
            self.fc2, self.relu, self.dropout,
            # self.fc21, self.relu, self.dropout,
            # self.fc22, self.relu, self.dropout,
            self.fc3
        )
        
    def forward(self, x):
        x = self.flatten(x)

        # Encode

        # for w, b in self.enc_weights:
        #     x = self.relu(x @ w.t()+ b)
        # with torch.no_grad():
        x = self.autoencoder.encoder(x)
        
        x = self.net(x)
        return x