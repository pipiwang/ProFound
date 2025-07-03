import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, bottleneck_dim=256):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim
        self.head = torch.nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        if type(x) == tuple:
            x = x[0]
        x = self.head(x)
        return x

