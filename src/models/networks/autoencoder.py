import torch.nn as nn
import torch
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, in_size, embed_size):
        super().__init__()

        self.relu = nn.ReLU()
        # self.sig = nn.Sigmoid()

        # hidden_size = embed_size * 2

        self.enc1 = nn.Linear(in_size, embed_size * 2)
        self.enc2 = nn.Linear(embed_size * 2, embed_size)
        self.dec1 = nn.Linear(embed_size, embed_size * 2)
        self.dec2 = nn.Linear(embed_size * 2, in_size)

        # Suggestion of writing separate modules: https://discuss.pytorch.org/t/how-to-use-only-the-decoder-part-from-an-autoencoder/135677
        self.encoder = nn.Sequential(
            nn.Linear(in_size, embed_size * 4), nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size * 2), nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2), nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size * 4), nn.ReLU(),
            nn.Linear(embed_size * 4, in_size)
        )
        
    def forward(self, x):
        # x = self.relu(self.enc1(x))
        # x = self.enc2(x)
        # x = self.relu(self.dec1(x))
        # x = self.sig(self.dec2(x))
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    # def encode(self, x):
    #     # x = self.relu(self.enc1(x))
    #     # x = self.relu(self.enc2(x))
    #     x = self.sig(self.relu(self.enc(x)))
    #     return x